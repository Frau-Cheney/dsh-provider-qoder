/** Qoder chat request encoding, timeout lifecycle, and SSE streaming. */

import {
  attributionHeaders,
  type GenerateOptions,
  type StreamChunk,
  type TokenUsage,
} from '@deepseek-ai/dsh-llm'
import type { QoderCatalogModel } from '../catalog.ts'
import { QoderLlmError, qoderModelError, canRefreshModelCredentials, qoderRequestId } from '../errors.ts'
import type { QoderRegion } from '../region.ts'
import { getQoderChatUrl } from './endpoints.ts'
import { defaultMaxErrorBytes, readLimitedText } from './request.ts'
import { redactLogValue, type QoderLogger } from './logging.ts'
import { buildAuthHeaders, type CosyCredentials } from './wire/cosy.ts'
import { qoderEncodeBody } from './wire/encoding.ts'
import { buildQoderRequestBody } from './wire/serialize.ts'
import { parseQoderSse } from './wire/sse.ts'
import type { QoderWireMessage } from './wire/wire-types.ts'

/**
 * Suspends the queue loop for one wait.
 *
 * Injectable so a test can observe the delays a queue reply produces without
 * actually sleeping through them, which is also what makes the wait durations
 * assertable rather than only the retry count.
 */
export type QoderQueueSleep = (delayMs: number, signal?: AbortSignal) => Promise<void>

export interface QoderChatDependencies {
  fetch: typeof fetch
  logger?: QoderLogger
  region: QoderRegion
  responseHeaderTimeoutMs: number
  streamIdleTimeoutMs: number
  refreshCredentials?: (rejected: CosyCredentials, signal?: AbortSignal) => Promise<CosyCredentials>
  /**
   * Re-resolve the credentials a queued request will be sent with, without
   * treating the current ones as rejected.
   *
   * A queued attempt is not an authentication failure, so this must not evict
   * the cached token the way {@link QoderChatDependencies.refreshCredentials}
   * does. The wait can still outlive one job token, so the queue loop asks again
   * before retrying; an implementation that caches until near expiry costs
   * nothing on the common path.
   */
  renewCredentials?: (signal?: AbortSignal) => Promise<CosyCredentials>
  /** Overrides how a queued wait is performed. Defaults to a real timer. */
  queueSleep?: QoderQueueSleep
  /**
   * Overrides the jitter source for a queued wait.
   *
   * Injected only by tests, which need a fixed delay to assert a duration; the
   * default keeps the production jitter that stops several clients from
   * repeating one delay in lockstep.
   */
  queueRandom?: () => number
}

function aborted(message: string): QoderLlmError {
  return new QoderLlmError(message, 'ABORTED')
}

/** Total time one queued turn may spend waiting for a free-model slot. */
const qoderQueueBudgetMs = 2 * 60 * 1000
/** Bounds on a single queued wait, so a bad hint cannot stall a turn for hours. */
const qoderQueueMinWaitMs = 1000
const qoderQueueMaxWaitMs = 30 * 1000
/** Attempts that follow the queue's own hint before the backoff ladder starts. */
const qoderQueueGraceAttempts = 3

/**
 * How long to wait before asking again, or `undefined` when this request must
 * give up — either because the failure is not a queue, or because the budget is
 * spent.
 *
 * Qoder defers a busy free-model pool with a `10605` business code carrying
 * `retryAfterSeconds`, which `qoderModelError` already turned into a
 * `RATE_LIMIT` failure with `providerRetryAfterMs`. Reading that classification
 * rather than re-parsing the payload keeps one definition of what a queue is.
 *
 * The wait comes from `queueRetryAfterMs`, the hint the queue descriptor
 * itself carried, and never from `providerRetryAfterMs`, which is the larger
 * of that hint and any `retry-after` header. A throttling header is not a queue
 * position: waiting on a header the queue never sent would turn each attempt
 * into a much longer stall than the service asked for.
 *
 * A `RATE_LIMIT` from a plain HTTP 429 carries no business code and stays with
 * the caller's retry policy: that is throttling, not a queue position.
 *
 * Past the grace attempts the hint is no longer trusted on its own: a queue
 * that keeps repeating the same number is not making progress, so the wait
 * grows instead of hammering the pool at one request per hint forever. That
 * ladder is what keeps a hint-less queue (`retryAfterSeconds` absent, so the
 * minimum is used) from degenerating into one request per second.
 *
 * Every wait is jittered because the free pool is shared: replaying one exact
 * delay from several clients is what keeps their retries in lockstep. The
 * cost of undershooting is a single extra rejection.
 *
 * The clamp into `[min, max]` happens *before* the remaining budget caps it,
 * and deliberately so: with the floor applied last, a budget under the minimum
 * would still sleep the full minimum and the budget would be decorative.
 */
function queueWaitMs(error: unknown, waitedMs: number, attempt: number, random: () => number): number | undefined {
  if (!(error instanceof QoderLlmError)) return undefined
  if (error.code !== 'RATE_LIMIT' || error.upstreamCode !== '10605') return undefined
  const remaining = qoderQueueBudgetMs - waitedMs
  if (remaining <= 0) return undefined
  // A queue answer without a hint is still a queue: start from the minimum
  // rather than dropping the request back to the caller.
  const hinted = error.queueRetryAfterMs ?? qoderQueueMinWaitMs
  const escalated = attempt > qoderQueueGraceAttempts
    ? qoderQueueMinWaitMs * 2 ** Math.min(attempt - qoderQueueGraceAttempts, 8)
    : 0
  const target = Math.max(hinted, escalated) * (0.75 + random() * 0.5)
  const clamped = Math.max(qoderQueueMinWaitMs, Math.min(target, qoderQueueMaxWaitMs))
  return Math.min(clamped, remaining)
}

/** Wait `delayMs`, ending early when the caller aborts the request. */
function waitForQueue(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(aborted('Request was aborted.'))
  return new Promise((resolve, reject) => {
    const onAbort = (): void => {
      clearTimeout(timer)
      reject(aborted('Request was aborted.'))
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, delayMs)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}
/**
 * Chat entry point that waits out a queued model inside the transport.
 *
 * Qoder answers a busy free-model pool with `10605` plus a `retryAfterSeconds`
 * hint while holding the request in a queue. That is not a failure — the client
 * is expected to wait the hinted delay and ask again — but the failure carries
 * a `providerRetryAfterMs` above the delay any retry policy accepts (30s
 * observed against a 10s cap), so handing it straight back ends the turn on a
 * pool that would have served it moments later. Waiting here, before the first
 * frame reaches the caller, keeps the turn intact.
 *
 * Every attempt rebuilds its request instead of replaying bytes: the signature
 * covers the body and `Cosy-Date`, so a replay is rejected as a duplicate.
 * `buildQoderRequestBody` mints a fresh `request_id` while the turn tracker
 * keeps `request_set_id` and `business.id`, so every attempt still reports as
 * one agent run. Only attempts that produced no output are retried; once a
 * frame has been handed to the caller the reply is already being written, and
 * the failure is reported as-is.
 */
export async function* streamQoderChat(
  options: GenerateOptions,
  model: QoderCatalogModel | undefined,
  credentials: CosyCredentials,
  messages: QoderWireMessage[],
  dependencies: QoderChatDependencies,
): AsyncGenerator<StreamChunk> {
  let waitedMs = 0
  let activeCredentials = credentials
  const sleep = dependencies.queueSleep ?? waitForQueue
  const random = dependencies.queueRandom ?? Math.random
  for (let attempt = 1; ; attempt++) {
    if (options.signal?.aborted) throw aborted('Request was aborted.')
    let emitted = false
    // Everything this attempt costs — the handshake, the response headers, and
    // the wait below — is charged to the budget, not only the wait. Counting
    // just the wait is what makes a budget decorative: a queue that keeps
    // answering slowly would spend unbounded time without ever exhausting it.
    const attemptStartedAt = Date.now()
    try {
      for await (const chunk of streamQoderChatOnce(options, model, activeCredentials, messages, dependencies)) {
        emitted = true
        yield chunk
      }
      return
    } catch (error: unknown) {
      // A partial reply cannot be retried: the caller already holds frames.
      if (emitted) throw error
      waitedMs += Date.now() - attemptStartedAt
      const delayMs = queueWaitMs(error, waitedMs, attempt, random)
      // `undefined` covers both "not a queue" and "budget spent", and either
      // way the caller — not this loop — decides what happens next.
      if (delayMs === undefined) throw error
      waitedMs += delayMs
      if (dependencies.renewCredentials) {
        try {
          activeCredentials = await dependencies.renewCredentials(options.signal)
        } catch {
          // Keep the credentials we have; the attempt below reports the reason.
        }
      }
      dependencies.logger?.debug?.('[Qoder Stream] Model queued, waiting before retry', {
        delayMs,
        waitedMs,
        budgetMs: qoderQueueBudgetMs,
        attempt,
      })
      await sleep(delayMs, options.signal)
    }
  }
}

async function* streamQoderChatOnce(
  options: GenerateOptions,
  model: QoderCatalogModel | undefined,
  credentials: CosyCredentials,
  messages: QoderWireMessage[],
  dependencies: QoderChatDependencies,
): AsyncGenerator<StreamChunk> {
  const request = await buildQoderRequestBody(options, credentials.userID, messages, model)
  request.session_type = dependencies.region === 'china' ? 'qoderclicn' : 'qodercli'
  const encodedBody = qoderEncodeBody(JSON.stringify(request))
  const encodedBytes = Buffer.from(encodedBody, 'utf8')
  const chatUrl = getQoderChatUrl(dependencies.region)
  const requestController = new AbortController()
  let headerTimedOut = false
  let idleTimedOut = false
  let idleTimer: ReturnType<typeof setTimeout> | undefined
  let chunkCount = 0
  let reqId: ReturnType<typeof qoderRequestId> | undefined
  const startedAt = performance.now()
  let lastActivityAt = startedAt
  const onCallerAbort = (): void => requestController.abort(options.signal?.reason)
  options.signal?.addEventListener('abort', onCallerAbort, { once: true })
  let headerTimer = setTimeout(() => {
    headerTimedOut = true
    requestController.abort('response header timeout')
  }, dependencies.responseHeaderTimeoutMs)
  const resetIdleTimer = (): void => {
    lastActivityAt = performance.now()
    if (idleTimer !== undefined) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => {
      idleTimedOut = true
      requestController.abort('stream idle timeout')
    }, dependencies.streamIdleTimeoutMs)
  }

  try {
    const markedTiers = Object.entries(request.model_config.context_config ?? {})
      .filter(([, tier]) => tier.is_default === true)
    dependencies.logger?.debug?.('[Qoder Stream] Request context', redactLogValue({
      sessionId: options.sessionId ?? null,
      requestId: request.request_id,
      model: request.model_config.key,
      region: dependencies.region,
      purpose: options.purpose ?? 'chat',
      contextTier: markedTiers.length === 1 ? markedTiers[0][0] : null,
      context_length: request.parameters.context_length ?? null,
    }))
    let response: Response | undefined
    let activeCredentials = credentials
    for (let attempt = 0; attempt < 2; attempt++) {
      if (options.signal?.aborted) throw aborted('Request was aborted.')
      if (attempt > 0) {
        headerTimer = setTimeout(() => {
          headerTimedOut = true
          requestController.abort('response header timeout')
        }, dependencies.responseHeaderTimeoutMs)
      }
      response = await dependencies.fetch(chatUrl, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'accept': 'text/event-stream',
          'cache-control': 'no-cache',
          'connection': 'keep-alive',
          'accept-encoding': 'identity',
          'x-model-key': options.model || 'cmodel',
          'x-model-source': model?.source || 'system',
          ...attributionHeaders(),
          ...buildAuthHeaders(encodedBytes, chatUrl, activeCredentials),
        },
        body: encodedBytes,
        signal: requestController.signal,
      })
      clearTimeout(headerTimer)
      reqId = qoderRequestId(response.headers)
      dependencies.logger?.debug?.('[Qoder Stream] Response headers received', {
        region: dependencies.region,
        status: response.status,
        attempt: attempt + 1,
        durationMs: Math.round(performance.now() - startedAt),
        ...reqId === undefined ? {} : { requestId: reqId },
      })
      resetIdleTimer()
      if (response.ok) break

      let bodyText = ''
      try {
        bodyText = (await readLimitedText(response, defaultMaxErrorBytes, 'Qoder model error response')).trim()
      } catch (error) {
        if (requestController.signal.aborted) throw error
        // Diagnostics are optional; the HTTP status still governs recovery and errors.
      }
      const failure = qoderModelError(`Qoder upstream service returned HTTP ${response.status}`, {
        status: response.status,
        headers: response.headers,
        cause: bodyText || undefined,
        source: 'http',
      })
      // Only an HTTP authentication rejection before SSE starts can be retried.
      // Reuse the prepared envelope and turn identity; never replay a partial stream.
      if (attempt === 0 && dependencies.refreshCredentials && canRefreshModelCredentials(failure)) {
        if (idleTimer !== undefined) clearTimeout(idleTimer)
        activeCredentials = await dependencies.refreshCredentials(activeCredentials, options.signal)
        if (activeCredentials.userID !== credentials.userID) {
          throw new QoderLlmError('Qoder subscriber identity changed during authentication recovery.', 'AUTH')
        }
        continue
      }
      throw failure
    }
    if (!response) throw new QoderLlmError('Qoder response is missing.', 'EMPTY_RESPONSE')
    if (!response.body) {
      throw new QoderLlmError('Qoder response contains no readable body stream.', 'EMPTY_RESPONSE')
    }

    let firstChunkDurationMs: number | undefined
    let tokenUsage: TokenUsage | undefined
    let finishReason: string | undefined
    const streamStartedAt = performance.now()

    for await (const chunk of parseQoderSse(response.body, { onActivity: resetIdleTimer, headers: response.headers, httpStatus: response.status })) {
      chunkCount++
      if (firstChunkDurationMs === undefined) {
        firstChunkDurationMs = Math.round(performance.now() - startedAt)
        dependencies.logger?.debug?.('[Qoder Stream] First chunk received', {
          durationMs: firstChunkDurationMs,
          ...reqId === undefined ? {} : { requestId: reqId },
        })
      }
      if (chunk.type === 'usage') {
        tokenUsage = chunk.usage
      }
      if (chunk.type === 'finish') {
        finishReason = chunk.reason.kind
      }
      yield chunk
    }

    dependencies.logger?.debug?.('[Qoder Stream] Stream completed', {
      durationMs: Math.round(performance.now() - startedAt),
      streamDurationMs: Math.round(performance.now() - streamStartedAt),
      chunkCount,
      ...finishReason === undefined ? {} : { finishReason },
      ...tokenUsage === undefined ? {} : { usage: tokenUsage },
      ...reqId === undefined ? {} : { requestId: reqId },
    })
  } catch (error: unknown) {
    if (options.signal?.aborted) throw aborted('Request was aborted.')
    const elapsedMs = Math.round(performance.now() - startedAt)
    if (headerTimedOut) {
      const failure = new QoderLlmError('Qoder model request exceeded its response header timeout.', 'TIMEOUT')
      dependencies.logger?.error?.('[Qoder Stream] Request failed', redactLogValue(failure), {
        phase: 'header',
        elapsedMs,
        timeoutMs: dependencies.responseHeaderTimeoutMs,
      })
      throw failure
    }
    if (idleTimedOut) {
      const failure = new QoderLlmError('Qoder model stream exceeded its idle timeout.', 'TIMEOUT')
      dependencies.logger?.error?.('[Qoder Stream] Request failed', redactLogValue(failure), {
        phase: 'stream-idle',
        chunkCount,
        idleDurationMs: Math.round(performance.now() - lastActivityAt),
        elapsedMs,
        ...reqId === undefined ? {} : { requestId: reqId },
      })
      throw failure
    }
    if (error instanceof QoderLlmError) {
      dependencies.logger?.error?.('[Qoder Stream] Request failed', redactLogValue(error), {
        chunkCount,
        elapsedMs,
        ...reqId === undefined ? {} : { requestId: reqId },
      })
      throw error
    }
    const failure = new QoderLlmError('Qoder transport request failed.', 'TRANSPORT', { cause: error })
    dependencies.logger?.error?.('[Qoder Stream] Request failed', redactLogValue(failure), {
      chunkCount,
      elapsedMs,
      ...reqId === undefined ? {} : { requestId: reqId },
      cause: redactLogValue(error),
    })
    throw failure
  } finally {
    clearTimeout(headerTimer)
    if (idleTimer !== undefined) clearTimeout(idleTimer)
    options.signal?.removeEventListener('abort', onCallerAbort)
    requestController.abort('request complete')
  }
}

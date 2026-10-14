/** The only external seam for communication with Qoder. */

import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import type { QoderAccountInfo } from '../account.ts'
import type { QoderCatalogModel } from '../catalog.ts'
import type { QoderRegion } from '../region.ts'
import { DefaultQoderTransport, defaultStreamIdleTimeoutMs } from './default-transport.ts'
import { defaultResponseHeaderTimeoutMs } from './request.ts'
import type { QoderLogger } from './logging.ts'
import type { QoderQueueSleep } from './chat.ts'

import type { WebSearchRequest, WebSearchResult } from '@deepseek-ai/dsh-web'

export { defaultResponseHeaderTimeoutMs, defaultStreamIdleTimeoutMs }

export interface QoderTransport {
  stream(options: GenerateOptions, model?: QoderCatalogModel): AsyncIterable<StreamChunk>
  discoverModels(signal?: AbortSignal): Promise<readonly QoderCatalogModel[]>
  readAccount(options?: { force?: boolean; signal?: AbortSignal }): Promise<QoderAccountInfo>
  searchWeb(request: WebSearchRequest, signal?: AbortSignal): Promise<WebSearchResult>
}

export interface QoderTransportOptions {
  region: QoderRegion
  resolvePat: () => Promise<string>
  fetch?: typeof fetch
  logger?: QoderLogger
  streamIdleTimeoutMs?: number
  responseHeaderTimeoutMs?: number
  metadataTimeoutMs?: number
  resolveMachineId?: () => string
  attachments?: Pick<AttachmentStore, 'imageLimits' | 'readImageRequest'>
  /** Deadline for one center image publication attempt. */
  imageUploadTimeoutMs?: number
  /** Lifetime of a remembered center image URL. */
  imageUrlCacheTtlMs?: number
  /** Whether prior assistant reasoning content is preserved across turns. */
  preserveThinking?: boolean
  /**
   * Overrides how a queued model wait is performed.
   *
   * A test injects a no-op to observe the delays a queue reply produces without
   * sleeping through them; production leaves it unset and waits for real.
   */
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

export function createQoderTransport(options: QoderTransportOptions): QoderTransport {
  return new DefaultQoderTransport(options)
}

export { QoderSearchClient } from './search.ts'
export type { QoderSearchClientOptions } from './search.ts'


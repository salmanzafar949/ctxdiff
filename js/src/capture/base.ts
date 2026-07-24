/**
 * The adapter contract. An adapter is the ONLY provider-aware code: it turns a
 * provider's request kwargs into role-tagged RawBlocks and pulls usage/params
 * off the request and response. Everything downstream is provider-agnostic.
 *
 * This layer is deliberately shaped so Anthropic/Gemini adapters slot in next
 * phase: each provider ships one object implementing `Adapter`, registered in
 * trace.ts's provider map. Nothing else changes.
 */
import type { RawBlock } from "../models.js";

export interface Adapter {
  /** Provider id, e.g. "openai". Stored on each block's token method decision
   * and on the call's provider column. */
  readonly provider: string;

  /**
   * A tuple of attribute paths from the client root to each completion method
   * this adapter intercepts, e.g. `[["chat","completions","create"],
   * ["responses","create"], ["chat","completions","stream"],
   * ["responses","stream"]]`. The client Proxy walks these to know what to wrap.
   */
  readonly createPaths: string[][];

  /** Flatten the request payload into ordered context blocks. */
  extractBlocks(kwargs: Record<string, unknown>): RawBlock[];

  /** Return sampling/model params (everything except block content). */
  extractParams(kwargs: Record<string, unknown>): Record<string, unknown>;

  /** Return provider-reported token usage as a plain object, or null. */
  extractUsage(response: unknown): Record<string, unknown> | null;

  /**
   * OPTIONAL — fold any usage carried by one streamed chunk into `state`, using
   * this provider's own usage-dict key names, so `state` ends up shaped exactly
   * like what `extractUsage` would have returned from a non-streaming response.
   * Called once per chunk from inside the caller's own iteration, so it must be
   * duck-typed and never throw. An adapter that omits it simply never
   * accumulates stream usage (recorded usage stays null).
   */
  accumulateStreamUsage?(chunk: unknown, state: Record<string, unknown>): void;
}

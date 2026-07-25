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

/**
 * What ONE intercepted call turned out to be, as decided by an adapter's
 * `interpretCall`: the request payload to extract blocks/params from, and
 * whether the call's result is a stream.
 *
 * It exists for providers whose SDK does not put the operation in the method
 * NAME. The AWS SDK v3 has a single `client.send(command)` for every Bedrock
 * operation, so "which operation is this, is it even an LLM call, and where is
 * the payload" can only be answered from the ARGUMENT — see `BedrockAdapter`.
 */
export interface CallShape {
  /** The provider-shaped request payload — what `extractBlocks`/`extractParams`
   * read. For a path-per-operation SDK this is simply the first argument. */
  kwargs: Record<string, unknown>;
  /** Whether the result is a stream, and so must be recorded at stream
   * completion rather than at call time. */
  streaming: boolean;
}

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

  /**
   * OPTIONAL — the member of a streaming call's result that holds the actual
   * iterator, for a provider whose streaming call returns an ENVELOPE rather
   * than the stream itself (`{$metadata, stream}` for Bedrock's
   * `ConverseStreamCommand`). Declaring it makes trace.ts proxy only that
   * member and hand the host its envelope otherwise untouched. Absent — the
   * normal case — means the result IS the stream.
   */
  readonly streamEnvelopeKey?: string;

  /**
   * OPTIONAL — decide what one intercepted call is, from its ARGUMENTS, for a
   * provider whose SDK routes every operation through one method (Bedrock's
   * `client.send(command)`). Returns the request payload plus whether it
   * streams, or NULL for "not a call this adapter records", which makes the
   * interceptor a transparent pass-through that records nothing.
   *
   * An adapter that omits it gets the default: the first argument is the
   * request payload, and streaming is decided from the method path and a
   * `stream: true` kwarg.
   */
  interpretCall?(args: unknown[], path: string[]): CallShape | null;

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

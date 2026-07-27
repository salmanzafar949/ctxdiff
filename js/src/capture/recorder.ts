/**
 * The fail-open recording path. Turns one (request, response) pair into a
 * stored call: adapter -> token counts -> hashes -> labels -> redaction ->
 * store. Its single public method never throws — a debugging tool must not be
 * able to crash the program it is debugging. Mirrors Python `Recorder`.
 */
import type { Adapter } from "./base.js";
import type { Block, CallBlock, RawBlock } from "../models.js";
import { basicLabel, contentHash } from "../models.js";
import { countTokens } from "../tokenize.js";
import type { Awaitable, RecordCallArgs, Store } from "../store/base.js";

export type RedactHook = (block: Block) => Block;

/**
 * Turn one adapter-extracted `RawBlock` into the stored, content-addressed
 * `Block`: give it its identity and its token numbers.
 *
 * Two overrides, both defaulting to the original behavior (see `RawBlock`):
 *
 *   - IDENTITY is `contentHash(role, kind, hashInput ?? text)`. For an ordinary
 *     block that is the text itself — unchanged, so every hash this SDK has ever
 *     written stays valid. For an image block it is a digest of the image BYTES,
 *     so two copies of the same picture are one block and a 1024×768 red square
 *     and a 1024×768 blue one are not.
 *   - TOKENS come from the tokenizer over `text`, unless the adapter already
 *     computed them — which it does only for images, where the truthful cost is
 *     the provider's vision formula over the pixel dimensions and NOT the
 *     tokenization of the `[image …]` descriptor standing in for them.
 *
 * Factored out of `Recorder.build` so there is exactly ONE definition of what a
 * stored block is: the golden harness (`test/helpers/golden.ts`) calls this same
 * function to materialize its fixtures, which is what lets the committed goldens
 * be evidence about the real capture path rather than about a parallel
 * reimplementation of it. Mirrors Python `build_block`.
 */
export function buildBlock(rb: RawBlock, provider: string): Block {
  const [tokenCount, tokenMethod] =
    rb.tokenCount !== undefined
      ? ([rb.tokenCount, rb.tokenMethod ?? "estimate"] as [number, string])
      : countTokens(rb.text, provider);
  const hashInput = rb.hashInput !== undefined ? rb.hashInput : rb.text;
  return {
    contentHash: contentHash(rb.role, rb.kind, hashInput),
    role: rb.role,
    kind: rb.kind,
    text: rb.text,
    tokenCount,
    tokenMethod,
  };
}

/**
 * One call, fully prepared and detached from the host's objects, ready to be
 * written. Produced by `build()` on the HOST's tick — blocks extracted, tokens
 * counted, hashes taken, labels resolved, redaction applied — and consumed by
 * `persist()`, which does nothing but hand it to the store.
 *
 * The split exists for the networked backends: a write there is a round trip, so
 * it happens off the host's call path (see `trace.ts`'s writer), and by then the
 * host may already have mutated the `messages` array the call was made with.
 * Snapshotting into a job at call time is what makes the deferred write record
 * what was actually SENT. It is exactly Python's `build`/`persist` pair.
 */
export type PersistJob = RecordCallArgs;

/** Whether `x` is promise-like — the one branch that lets this class serve both
 * a synchronous `CTrace` and a promise-based network store without either
 * paying for the other. */
function isThenable(x: unknown): x is Promise<unknown> {
  return (
    x != null &&
    (typeof x === "object" || typeof x === "function") &&
    typeof (x as { then?: unknown }).then === "function"
  );
}

export class Recorder {
  private store: Store;
  private adapter: Adapter;
  private redact: RedactHook | null;
  // One-time latch for the record-failure warning. A store that fails one write
  // usually fails every write (a full disk, a revoked file handle, a stuck
  // lock), and this path runs once per LLM call — unlatched it floods the host's
  // logs with an identical stack per turn. Mirrors Python's `_persist_warned`.
  private recordWarned = false;

  constructor(store: Store, adapter: Adapter, redact: RedactHook | null) {
    this.store = store;
    this.adapter = adapter;
    this.redact = redact;
  }

  /**
   * Build and store one call from its request kwargs and response. Every step
   * runs inside a catch-all: any failure is logged once and swallowed, leaving
   * the host's own call path untouched (fail-open). `tagged` is a list of
   * [label, needle] pairs used to override labels. `agent`/`step`/`provider`
   * are the v2 attribution fields threaded through to the store. `quiet`
   * suppresses the warning log (used only by a stream proxy's best-effort
   * finalize). Mirrors Python `Recorder.record`.
   *
   * This is the SYNCHRONOUS path, used with the local `.ctrace` store where the
   * write is a same-tick call: build the job, persist it, done — no promise is
   * created and nothing about the existing local behavior changes. A networked
   * store's writer calls `build()` and `persist()` separately instead.
   */
  record(args: {
    seq: number;
    kwargs: Record<string, unknown>;
    response: unknown;
    latencyMs: number | null;
    error: string | null;
    tagged: [string, string][];
    agent?: string | null;
    step?: string | null;
    provider?: string | null;
    quiet?: boolean;
  }): void {
    const job = this.build(args);
    if (job === null) return;
    const result = this.persist(job, args.quiet ?? false);
    // A synchronous store returns nothing; a store that unexpectedly returns a
    // promise here still cannot surface an unhandled rejection into the host.
    if (isThenable(result)) void result.then(undefined, () => undefined);
  }

  /**
   * Turn one (request, response) pair into a ready-to-write `PersistJob`, or
   * null if anything about the extraction fails (warned at most once, then
   * silent — fail-open). Pure and synchronous: it touches the adapter, the
   * tokenizer and the redaction hook, but never the store, so it is safe to run
   * on the host's own tick even when the store is a database on another
   * continent. Mirrors Python `Recorder.build`.
   */
  build(args: {
    seq: number;
    kwargs: Record<string, unknown>;
    response: unknown;
    latencyMs: number | null;
    error: string | null;
    tagged: [string, string][];
    agent?: string | null;
    step?: string | null;
    provider?: string | null;
    quiet?: boolean;
  }): PersistJob | null {
    const {
      seq,
      kwargs,
      response,
      latencyMs,
      error,
      tagged,
      agent = null,
      step = null,
      provider: providerArg = null,
      quiet = false,
    } = args;
    try {
      const raw = this.adapter.extractBlocks(kwargs);
      const params = this.adapter.extractParams(kwargs);
      const usage = response != null ? this.adapter.extractUsage(response) : null;
      // The caller's `provider` is the ATTRIBUTION label decided at wrap()
      // time — e.g. "gemini" for an OpenAI-SDK client pointed at Gemini's
      // OpenAI-compatible endpoint — and wins when supplied; the adapter's
      // own name is only the fallback for callers that don't attribute. Safe
      // because nothing mechanical below keys off this string: extraction
      // already ran above via the adapter, and `buildBlock` treats an
      // unrecognized provider as estimate-tokenized (which is MORE honest
      // for a non-OpenAI model than gpt-tokenizer counts marked exact).
      // Mirrors Python `Recorder.build` (dogfood finding 2026-07-27).
      const provider = providerArg ?? this.adapter.provider;

      const callBlocks: CallBlock[] = [];
      raw.forEach((rb, position) => {
        // Count tokens and take the content hash for this block — see
        // `buildBlock` for the two adapter-supplied overrides (an image hashes
        // its bytes and carries a pre-computed vision estimate; everything else
        // hashes and tokenizes its text).
        let block: Block = buildBlock(rb, provider);
        // Redact AFTER hashing/counting but BEFORE storage: the hash keeps the
        // original text so identity/dedup stays stable even if redaction is
        // nondeterministic; only the stored text changes.
        if (this.redact !== null) block = this.safeRedact(block);
        const [label, labelSource] = basicLabel(rb.role, rb.kind, rb.text, tagged);
        callBlocks.push({ block, position, label, labelSource });
      });

      return { seq, params, usage, latencyMs, error, callBlocks, agent, step, provider };
    } catch (err) {
      this.warnOnce(seq, err, quiet);
      return null;
    }
  }

  /**
   * Write one prepared job to the store. Returns whatever the store returns — a
   * promise for a networked backend (which the writer awaits, one job at a time,
   * so the single connection is never used concurrently) and nothing at all for
   * the local file. Fail-open in BOTH shapes: a synchronous throw and a rejected
   * promise are each caught here and warned about at most once, so a failing
   * store degrades capture and never reaches the host. Mirrors Python
   * `Recorder.persist`.
   */
  persist(job: PersistJob, quiet = false): Awaitable<void> {
    try {
      const result = this.store.recordCall(job);
      if (isThenable(result)) {
        return result.then(
          () => undefined,
          (err: unknown) => this.warnOnce(job.seq, err, quiet),
        );
      }
    } catch (err) {
      this.warnOnce(job.seq, err, quiet);
    }
  }

  /**
   * Report a capture failure AT MOST ONCE for the run, then stay silent: the
   * first failure carries the diagnosis (message + stack) and every subsequent
   * one is dropped rather than repeating the same stack once per turn. `quiet`
   * callers (a stream's best-effort finalize) skip logging entirely.
   */
  private warnOnce(seq: number, err: unknown, quiet: boolean): void {
    if (quiet || this.recordWarned) return;
    this.recordWarned = true;
    // eslint-disable-next-line no-console
    console.warn(
      `ctxdiff: failed to record call seq=${seq} (tracing skipped); ` +
        "further record failures in this run will be silent",
      err,
    );
  }

  /** Apply the redaction hook, but never let a throwing redactor break
   * recording: on error, replace the text with a sentinel so nothing sensitive
   * leaks and the run continues. Mirrors Python `_safe_redact`. */
  private safeRedact(block: Block): Block {
    if (this.redact === null) return block;
    try {
      return this.redact(block);
    } catch {
      return { ...block, text: "[redaction-error]" };
    }
  }
}

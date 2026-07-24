/**
 * The fail-open recording path. Turns one (request, response) pair into a
 * stored call: adapter -> token counts -> hashes -> labels -> redaction ->
 * store. Its single public method never throws — a debugging tool must not be
 * able to crash the program it is debugging. Mirrors Python `Recorder`.
 */
import type { Adapter } from "./base.js";
import type { Block, CallBlock } from "../models.js";
import { basicLabel, contentHash } from "../models.js";
import { countTokens } from "../tokenize.js";
import type { CTrace } from "../store/ctrace.js";

export type RedactHook = (block: Block) => Block;

export class Recorder {
  private ct: CTrace;
  private adapter: Adapter;
  private redact: RedactHook | null;

  constructor(ct: CTrace, adapter: Adapter, redact: RedactHook | null) {
    this.ct = ct;
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
    quiet?: boolean;
  }): void {
    const { seq, kwargs, response, latencyMs, error, tagged, agent = null, step = null, quiet = false } = args;
    try {
      const raw = this.adapter.extractBlocks(kwargs);
      const params = this.adapter.extractParams(kwargs);
      const usage = response != null ? this.adapter.extractUsage(response) : null;
      const provider = this.adapter.provider;

      const callBlocks: CallBlock[] = [];
      raw.forEach((rb, position) => {
        // Count tokens for this text under the provider, then build the
        // content-addressed Block.
        const [tokenCount, tokenMethod] = countTokens(rb.text, provider);
        let block: Block = {
          contentHash: contentHash(rb.role, rb.kind, rb.text),
          role: rb.role,
          kind: rb.kind,
          text: rb.text,
          tokenCount,
          tokenMethod,
        };
        // Redact AFTER hashing/counting but BEFORE storage: the hash keeps the
        // original text so identity/dedup stays stable even if redaction is
        // nondeterministic; only the stored text changes.
        if (this.redact !== null) block = this.safeRedact(block);
        const [label, labelSource] = basicLabel(rb.role, rb.kind, rb.text, tagged);
        callBlocks.push({ block, position, label, labelSource });
      });

      this.ct.recordCall({
        seq,
        params,
        usage,
        latencyMs,
        error,
        callBlocks,
        agent,
        step,
        provider,
      });
    } catch (err) {
      // fail-open is the whole point
      if (!quiet) {
        // eslint-disable-next-line no-console
        console.warn(
          `ctxdiff: failed to record call seq=${seq} (tracing skipped)`,
          err,
        );
      }
    }
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

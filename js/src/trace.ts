/**
 * The public entry point. `init()` opens a `.ctrace` and returns a Tracer;
 * `tracer.wrap(client)` returns a transparent JS Proxy that records every
 * completion call while behaving exactly like the original client.
 *
 * FAIL-OPEN IS ABSOLUTE. Wrapping must never throw into the host, never break
 * iteration of a stream, never drop/reorder/delay a chunk, and never alter the
 * request. Recording is a strictly best-effort side channel.
 *
 * Non-streaming: the interceptor calls the real method, awaits its Promise,
 * records at call time, and returns the response untouched.
 *
 * Streaming — two shapes, both deferred to stream completion because usage
 * isn't on the stream yet at call time (only in later/final chunks as the
 * CALLER consumes them):
 *   - `create({stream:true})` returns a Promise<Stream>; once awaited, the
 *     Stream is wrapped so each chunk passes through unchanged while usage is
 *     folded, and the call is recorded once iteration completes.
 *   - the `.stream()` convenience helper (`chat.completions.stream` /
 *     `responses.stream`) returns the stream object SYNCHRONOUSLY (confirmed
 *     Phase JS1 Step 0: not a Promise, not a context-manager — unlike the
 *     Python SDK's manager) and is async-iterable directly; it is wrapped the
 *     same way. The last path segment being "stream" is the signal.
 *
 * The adapter layer is provider-agnostic below this file: adding Anthropic/
 * Gemini next phase means registering another entry in `REGISTRY` — nothing
 * here changes.
 */
import type { Adapter } from "./capture/base.js";
import { OpenAIAdapter } from "./capture/openai.js";
import { AnthropicAdapter } from "./capture/anthropic.js";
import { GeminiAdapter } from "./capture/gemini.js";
import { Recorder, type RedactHook } from "./capture/recorder.js";
import { CTrace } from "./store/ctrace.js";

/** The per-wrap recording context that travels down the client proxy tree and
 * into the stream proxies. Carries the adapter separately (the stream path
 * needs `accumulateStreamUsage` per-chunk, before any record happens). */
interface WrapContext {
  tracer: Tracer;
  recorder: Recorder;
  agent: string | null;
  provider: string;
  adapter: Adapter;
  createPaths: string[][];
}

/** A provider entry: how to recognize a client and build its adapter. The
 * registry is the single extension point — Anthropic/Gemini slot in here. */
interface ProviderEntry {
  provider: string;
  detect(client: unknown): boolean;
  make(): Adapter;
}

function ctorName(client: unknown): string | undefined {
  if (client == null || typeof client !== "object") return undefined;
  return (client as { constructor?: { name?: string } }).constructor?.name;
}

/** Recognize an OpenAI client (sync or async): by class name, or by duck-typing
 * the two completion resources. Azure's client subclasses OpenAI, so its name
 * check is covered too. Kept liberal so a lightly-wrapped or re-exported client
 * still resolves; anything else falls through to unrecognized (pass-through). */
function detectOpenAI(client: unknown): boolean {
  if (client == null || typeof client !== "object") return false;
  const name = ctorName(client);
  if (name === "OpenAI" || name === "AzureOpenAI") return true;
  const c = client as {
    chat?: { completions?: { create?: unknown } };
    responses?: unknown;
  };
  return typeof c.chat?.completions?.create === "function" && c.responses != null;
}

/** Recognize an Anthropic client: by class name, or by duck-typing
 * `messages.create` while ensuring it is NOT an OpenAI client (which has no
 * top-level `messages` resource but does have `chat`/`responses`). */
function detectAnthropic(client: unknown): boolean {
  if (client == null || typeof client !== "object") return false;
  if (ctorName(client) === "Anthropic") return true;
  const c = client as {
    messages?: { create?: unknown };
    chat?: unknown;
    responses?: unknown;
  };
  return (
    typeof c.messages?.create === "function" &&
    c.chat == null &&
    c.responses == null
  );
}

/** Recognize a @google/genai client (`GoogleGenAI`): by class name, or by
 * duck-typing `models.generateContent`. */
function detectGemini(client: unknown): boolean {
  if (client == null || typeof client !== "object") return false;
  if (ctorName(client) === "GoogleGenAI") return true;
  const c = client as { models?: { generateContent?: unknown } };
  return typeof c.models?.generateContent === "function";
}

// The single extension point. Order matters only to disambiguate overlapping
// duck-types; these three are mutually exclusive (distinct resource shapes).
const REGISTRY: ProviderEntry[] = [
  { provider: "openai", detect: detectOpenAI, make: () => new OpenAIAdapter() },
  { provider: "anthropic", detect: detectAnthropic, make: () => new AnthropicAdapter() },
  { provider: "gemini", detect: detectGemini, make: () => new GeminiAdapter() },
];

/** Options for `init`. */
export interface InitOptions {
  /** Per-block scrubber applied before storage. */
  redact?: RedactHook;
  /** Where the `.ctrace` is written (defaults to ./<project>-<8hex>.ctrace). */
  path?: string;
}

/** Options for `wrap`. */
export interface WrapOptions {
  /** Names the agent this client belongs to; stamped onto every recorded call
   * so one run can attribute calls across several agents. */
  agent?: string;
}

/** Short random hex, used only for the default `.ctrace` filename. */
function shortId(): string {
  return Math.random().toString(16).slice(2, 10).padEnd(8, "0");
}

/** Best-effort exception type name, mirroring Python's `type(exc).__name__`. */
function errName(exc: unknown): string {
  if (exc instanceof Error) return exc.constructor?.name || exc.name || "Error";
  if (exc && typeof exc === "object") {
    return (exc as { constructor?: { name?: string } }).constructor?.name || "Error";
  }
  return "Error";
}

function isThenable(x: unknown): x is Promise<unknown> {
  return (
    x != null &&
    (typeof x === "object" || typeof x === "function") &&
    typeof (x as { then?: unknown }).then === "function"
  );
}

function pathEquals(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((s, i) => s === b[i]);
}

function isCreatePath(path: string[], createPaths: string[][]): boolean {
  return createPaths.some((cp) => pathEquals(cp, path));
}

function isPrefixOfAny(path: string[], createPaths: string[][]): boolean {
  return createPaths.some(
    (cp) => path.length < cp.length && path.every((s, i) => s === cp[i]),
  );
}

/**
 * Create a Tracer for one run. `project` names the run; `opts.redact` is an
 * optional per-block scrubber applied before storage; `opts.path` is where the
 * `.ctrace` is written (defaults to ./<project>-<8hex>.ctrace in the cwd).
 */
export function init(project: string, opts: InitOptions = {}): Tracer {
  const path = opts.path ?? `${project}-${shortId()}.ctrace`;
  return new Tracer(project, opts.redact ?? null, path);
}

export class Tracer {
  readonly path: string;
  private project: string;
  private redact: RedactHook | null;
  private ct: CTrace | null = null;
  private firstRecorder: Recorder | null = null;
  private seq = 0; // monotonically increasing turn index across ALL agents
  private step: string | null = null; // sticky step label (see mark())
  private pendingTags: [string, string][] = []; // (label, needle) for next call

  constructor(project: string, redact: RedactHook | null, path: string) {
    this.project = project;
    this.redact = redact;
    this.path = path;
  }

  /**
   * Return a transparent Proxy over `client` that records every call to the
   * provider's completion methods. The store/run is created lazily on the first
   * wrap, once the provider is known. FAIL-OPEN: any setup failure (unrecognized
   * client, store error) logs and returns the ORIGINAL client unwrapped, so the
   * host keeps working with tracing simply absent — wrap never throws.
   */
  wrap(client: object, opts: WrapOptions = {}): object {
    try {
      const entry = REGISTRY.find((e) => e.detect(client));
      if (!entry) {
        console.warn(
          "ctxdiff: unrecognized client; returning it unwrapped (no tracing)",
        );
        return client;
      }
      const adapter = entry.make();
      if (this.ct === null) {
        // model is per-call, not known yet: pass "" so CTrace.create leaves
        // run.models == [] rather than seeding a bogus [""]; recordCall/
        // noteModel backfill the real model(s) as calls arrive.
        const started = new Date().toISOString();
        this.ct = CTrace.create(this.path, this.project, entry.provider, "", started);
      }
      const recorder = new Recorder(this.ct, adapter, this.redact);
      if (this.firstRecorder === null) this.firstRecorder = recorder;
      const ctx: WrapContext = {
        tracer: this,
        recorder,
        agent: opts.agent ?? null,
        provider: entry.provider,
        adapter,
        createPaths: adapter.createPaths,
      };
      return makeClientProxy(client, [], ctx);
    } catch (err) {
      console.warn("ctxdiff: wrap failed; returning unwrapped client", err);
      return client;
    }
  }

  /**
   * Buffer semantic tags for the NEXT recorded call only. Each item is reduced
   * to its text (string as-is, else a 'text'/'content' field, else String()) and
   * paired with `label`; the recorder marks any block containing that text as
   * `label`. Contrast mark(): tag() is next-call-only, mark() is sticky.
   */
  tag(label: string, items: unknown[]): void {
    for (const item of items) {
      let text: string;
      if (typeof item === "string") {
        text = item;
      } else if (item && typeof item === "object") {
        const o = item as Record<string, unknown>;
        text = (o["text"] as string) || (o["content"] as string) || "";
      } else {
        text = String(item);
      }
      if (text) this.pendingTags.push([label, text]);
    }
  }

  /**
   * Set the CURRENT step label stamped onto every subsequent recorded call —
   * across ALL agents — until changed; `mark(null)` clears it. STICKY, unlike
   * tag() which applies to the next call only.
   */
  mark(step: string | null): void {
    this.step = step;
  }

  /**
   * Interceptor callback: advance the turn counter, hand everything to the
   * wrapping proxy's recorder (with its agent and the tracer's current sticky
   * step), then clear pending tags. `seq` stays a single monotonic counter
   * across ALL agents. Never throws — record is internally fail-open, but this
   * wiring is guarded too so a replaced/broken record can't break the host.
   */
  onCreate(args: {
    kwargs: Record<string, unknown>;
    response: unknown;
    latencyMs: number | null;
    error: string | null;
    recorder: Recorder | null;
    agent: string | null;
    quiet?: boolean;
  }): void {
    const { kwargs, response, latencyMs, error, recorder, agent, quiet = false } = args;
    this.seq += 1;
    const tags = this.pendingTags;
    this.pendingTags = [];
    const step = this.step;
    if (recorder !== null) {
      try {
        recorder.record({
          seq: this.seq,
          kwargs,
          response,
          latencyMs,
          error,
          tagged: tags,
          agent,
          step,
          quiet,
        });
      } catch (err) {
        if (!quiet) {
          console.warn(
            `ctxdiff: recorder.record raised; tracing skipped for seq=${this.seq}`,
            err,
          );
        }
      }
    }
  }

  /** Close the underlying store, if one was opened. */
  close(): void {
    if (this.ct !== null) this.ct.close();
  }
}

/**
 * A transparent Proxy that forwards attribute access to the wrapped target,
 * following the adapter's create paths down to the completion methods — which it
 * replaces with an interceptor. Anything off those paths is returned as-is, so
 * the wrapped client is behaviorally identical to the original. Multi-path: any
 * segment matching ANY member of `createPaths` keeps the traversal alive, so
 * both `chat.completions.create`/`.stream` and `responses.create`/`.stream` are
 * intercepted off the same proxy tree.
 */
function makeClientProxy(target: object, path: string[], ctx: WrapContext): object {
  return new Proxy(target, {
    get(t, prop, receiver) {
      if (typeof prop === "symbol") return Reflect.get(t, prop, receiver);
      // Read against the real target (2-arg Reflect.get) so lazy resource
      // getters on the SDK run with the correct `this`.
      const value = Reflect.get(t, prop);
      const newPath = [...path, prop];

      if (isCreatePath(newPath, ctx.createPaths)) {
        // Bind the real method to its real resource object so its internal
        // `this` (private fields, etc.) stays correct.
        const bound = typeof value === "function" ? value.bind(t) : value;
        return makeInterceptor(bound as (...a: unknown[]) => unknown, newPath, ctx);
      }
      if (isPrefixOfAny(newPath, ctx.createPaths)) {
        if (value != null && (typeof value === "object" || typeof value === "function")) {
          return makeClientProxy(value as object, newPath, ctx);
        }
        return value;
      }
      return value;
    },
  });
}

/**
 * Wrap one completion method. Calls the REAL method first (never delaying or
 * altering the host's request), measures latency, then records — or, for a
 * streaming call, hands back a stream proxy that records once at completion. On
 * host error, records the failed call and re-raises the host's exception
 * unchanged. The ONLY exceptions this function lets escape are the host's own.
 */
function makeInterceptor(
  realFn: (...a: unknown[]) => unknown,
  path: string[],
  ctx: WrapContext,
): (...a: unknown[]) => unknown {
  const last = path[path.length - 1];
  // Two "this method always streams" shapes, distinguished by the last path
  // segment (a STRUCTURAL signal read at proxy-resolution time, never
  // duck-typed off the return value):
  //   - EXACTLY "stream" → a `.stream()` convenience helper (OpenAI's
  //     chat.completions/responses, Anthropic's messages) that returns the
  //     stream object synchronously (no `stream:true` kwarg, no Promise).
  //   - ends WITH "stream" but isn't exactly "stream" → a distinctly-named
  //     streaming method (Gemini's `generateContentStream`) that returns a
  //     Promise resolving to a direct async-iterable; there is no `stream`
  //     kwarg to read, so the method name is the only signal.
  const isStreamHelper = last === "stream";
  const isNamedStreamMethod =
    last !== "stream" && last.toLowerCase().endsWith("stream");

  return function interceptor(...args: unknown[]): unknown {
    const kwargs =
      args[0] && typeof args[0] === "object"
        ? (args[0] as Record<string, unknown>)
        : {};
    const start = performance.now();

    let result: unknown;
    try {
      result = realFn(...args);
    } catch (exc) {
      // Sync throw: a construction-time error, or a sync SDK failing outright.
      const latencyMs = Math.round(performance.now() - start);
      ctx.tracer.onCreate({
        kwargs,
        response: null,
        latencyMs,
        error: errName(exc),
        recorder: ctx.recorder,
        agent: ctx.agent,
      });
      throw exc;
    }

    // Streaming if: a `.stream()` helper, a named streaming method (Gemini), or
    // the caller's own `stream:true` kwarg. Any of these routes to the stream
    // proxy instead of recording immediately.
    const streaming = isStreamHelper || isNamedStreamMethod || !!kwargs["stream"];

    if (isThenable(result)) {
      // create()/create({stream:true}) — returns a Promise. Record after await.
      return result.then(
        (resolved: unknown) => {
          if (streaming) return wrapStream(resolved, kwargs, ctx, start);
          const latencyMs = Math.round(performance.now() - start);
          ctx.tracer.onCreate({
            kwargs,
            response: resolved,
            latencyMs,
            error: null,
            recorder: ctx.recorder,
            agent: ctx.agent,
          });
          return resolved;
        },
        (exc: unknown) => {
          const latencyMs = Math.round(performance.now() - start);
          ctx.tracer.onCreate({
            kwargs,
            response: null,
            latencyMs,
            error: errName(exc),
            recorder: ctx.recorder,
            agent: ctx.agent,
          });
          throw exc; // host's own async error — re-raised unchanged
        },
      );
    }

    // Synchronous return — the `.stream()` helper hands back its stream directly.
    if (streaming) return wrapStream(result, kwargs, ctx, start);

    // Non-streaming sync return (unusual for openai) — record now.
    const latencyMs = Math.round(performance.now() - start);
    ctx.tracer.onCreate({
      kwargs,
      response: result,
      latencyMs,
      error: null,
      recorder: ctx.recorder,
      agent: ctx.agent,
    });
    return result;
  };
}

/**
 * Wrap a provider stream in a transparent async-iterable Proxy: every real
 * chunk is yielded to the caller UNCHANGED and IMMEDIATELY — never buffered,
 * dropped, reordered, or delayed — while each chunk's usage is folded into a
 * running `state` via the adapter, then the call is recorded ONCE the stream
 * completes. "Complete" = the iterator running out, the caller breaking early
 * (`return()`), or the stream raising mid-generation (`throw()` / a rejected
 * next), whichever fires first; a `finalized` flag makes recording happen
 * exactly once. A mid-stream error is recorded as a FAILED call with whatever
 * usage accumulated before the failure, then the original error propagates
 * unchanged. All non-iterator properties/methods (`finalChatCompletion`, etc.)
 * forward to the real stream, so callers relying on them keep working.
 *
 * LIMITATION: a stream that is NEVER iterated at all (no consumption, no early
 * break, no error) is not recorded. JS has no `__del__`/deterministic finalizer,
 * and `FinalizationRegistry` (GC-timed, non-deterministic, unreliable at
 * shutdown) was deliberately avoided rather than record calls at an arbitrary
 * later moment. Streams that are consumed, broken out of early, errored, or
 * exhausted all DO record.
 */
function wrapStream(
  stream: unknown,
  kwargs: Record<string, unknown>,
  ctx: WrapContext,
  start: number,
): unknown {
  // If the returned value isn't actually a stream (defensive), record it as a
  // plain response and return unchanged rather than risk breaking the caller.
  if (
    stream == null ||
    typeof stream !== "object" ||
    typeof (stream as { [Symbol.asyncIterator]?: unknown })[Symbol.asyncIterator] !==
      "function"
  ) {
    const latencyMs = Math.round(performance.now() - start);
    ctx.tracer.onCreate({
      kwargs,
      response: stream,
      latencyMs,
      error: null,
      recorder: ctx.recorder,
      agent: ctx.agent,
    });
    return stream;
  }

  const state: Record<string, unknown> = {};
  let finalized = false;

  const finalize = (error: string | null): void => {
    if (finalized) return;
    finalized = true;
    try {
      // Route accumulated usage through the SAME `extractUsage` path a
      // non-streaming call uses, by presenting `state` as a synthetic response.
      // The state is exposed under BOTH `usage` (OpenAI/Anthropic's
      // `extractUsage` reads `response.usage`) AND `usageMetadata` (Gemini's
      // reads `response.usageMetadata`) — both names point at the SAME object,
      // so whichever attribute a given adapter duck-types off of, it finds the
      // right data. This mirrors the Python `_SyntheticStreamResponse` fix:
      // without the `usageMetadata` alias a Gemini stream's usage would vanish
      // at record time even though accumulation worked.
      const hasUsage = Object.keys(state).length > 0;
      const usageNs = { ...state };
      const response = hasUsage
        ? { usage: usageNs, usageMetadata: usageNs }
        : null;
      const latencyMs = Math.round(performance.now() - start);
      ctx.tracer.onCreate({
        kwargs,
        response,
        latencyMs,
        error,
        recorder: ctx.recorder,
        agent: ctx.agent,
      });
    } catch {
      // fail-open: finalize must never throw into a completion path
    }
  };

  const accumulate = (chunk: unknown): void => {
    try {
      ctx.adapter.accumulateStreamUsage?.(chunk, state);
    } catch {
      // fail-open: a chunk must reach the caller regardless
    }
  };

  const target = stream as {
    [Symbol.asyncIterator](): AsyncIterator<unknown>;
  };

  return new Proxy(target, {
    get(t, prop, receiver) {
      if (prop === Symbol.asyncIterator) {
        return function (): AsyncIterator<unknown> {
          const realIter = t[Symbol.asyncIterator]();
          return {
            async next(): Promise<IteratorResult<unknown>> {
              let r: IteratorResult<unknown>;
              try {
                r = await realIter.next();
              } catch (exc) {
                finalize(errName(exc));
                throw exc; // stream's own mid-generation failure, re-raised
              }
              if (r.done) {
                finalize(null);
                return r;
              }
              // Chunk has reached us; fold usage, then hand it back untouched.
              accumulate(r.value);
              return r;
            },
            async return(v?: unknown): Promise<IteratorResult<unknown>> {
              // Caller broke out early (or `for await` cleanup) — record now.
              finalize(null);
              if (typeof realIter.return === "function") {
                return realIter.return(v);
              }
              return { done: true, value: v };
            },
            async throw(e?: unknown): Promise<IteratorResult<unknown>> {
              finalize(errName(e));
              if (typeof realIter.throw === "function") {
                return realIter.throw(e);
              }
              throw e;
            },
          };
        };
      }
      // Everything else forwards to the real stream (finalChatCompletion, etc.).
      const value = Reflect.get(t, prop, receiver);
      if (typeof value === "function") return (value as (...a: unknown[]) => unknown).bind(t);
      return value;
    },
  });
}

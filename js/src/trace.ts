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
import { AsyncLocalStorage } from "node:async_hooks";
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
  // null when store setup failed and capture degraded fail-open: the proxy still
  // forwards every host call, `onCreate` just short-circuits and records nothing.
  recorder: Recorder | null;
  agent: string | null;
  provider: string;
  adapter: Adapter;
  createPaths: string[][];
}

/**
 * The per-async-context capture state that `tag()`/`mark()`/`step()` read and
 * write. Held in an `AsyncLocalStorage` so each concurrent async operation sees
 * its OWN pending tags + sticky step and can't cross-contaminate a sibling's
 * attribution (see `Tracer`'s ALS docstring). `pendingTags` is one-shot (drained
 * by the next recorded call); `step` is sticky within the context until changed.
 */
interface TagStore {
  pendingTags: [string, string][];
  step: string | null;
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
  /**
   * Where the project `.ctrace` is written. Defaults to a STABLE
   * `./<project>.ctrace` in the cwd (NOT a per-run unique name): the first
   * `wrap()` opens that file if it exists and APPENDS a new session, or creates
   * it if absent, so every `init(project)` accumulates one more session in the
   * same project db. An explicit `path` works the same way — it appends when the
   * file already exists.
   */
  path?: string;
}

/** Options for `wrap`. */
export interface WrapOptions {
  /** Names the agent this client belongs to; stamped onto every recorded call
   * so one run can attribute calls across several agents. */
  agent?: string;
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
 * Create a Tracer that opens the project's `.ctrace` and starts a NEW SESSION in
 * it. `project` names the project; `opts.redact` is an optional per-block
 * scrubber applied before storage; `opts.path` is where the project `.ctrace` is
 * written (defaults to a STABLE ./<project>.ctrace in the cwd — see InitOptions).
 * The first `wrap()` APPENDS a session to that file if it already exists, or
 * creates it if absent, so every `init(project)` accumulates one more session in
 * the same project db. Mirrors Python `trace.init`.
 */
export function init(project: string, opts: InitOptions = {}): Tracer {
  const path = opts.path ?? `${project}.ctrace`;
  return new Tracer(project, opts.redact ?? null, path);
}

export class Tracer {
  readonly path: string;
  private project: string;
  private redact: RedactHook | null;
  private ct: CTrace | null = null;
  private firstRecorder: Recorder | null = null;
  private seq = 0; // monotonically increasing turn index across ALL agents
  // One-time guard for the store-setup fail-open path in wrap(): if the project
  // store can't be created/opened (e.g. a persistent lock under heavy concurrent
  // session creation, or an unopenable path) we degrade fail-open and warn at
  // most once for the run rather than raising into the host.
  private setupWarned = false;
  // The matching one-way latch for the RETRY, not just the warning. `ct === null`
  // alone can't distinguish "not opened yet" from "tried and failed", so every
  // later wrap() re-entered `openOrCreateSession` and paid its full retry budget
  // again — and that budget blocks Node's single event loop synchronously (see
  // store/ctrace.ts). One failure is enough: capture is degraded for the run, so
  // the second wrap() must cost nothing.
  private setupFailed = false;

  /**
   * Per-async-context capture state — the core of the concurrency model.
   *
   * `tag()`/`mark()` used to live in instance fields, so under a `Promise.all`
   * fan-out one call's tag/step would bleed into another and MISLABEL
   * attribution — the worst failure mode for a debugger (confidently wrong).
   * They now live in an `AsyncLocalStorage` store instead: `step(label, fn)`
   * runs `fn` inside `als.run()` with a fresh store, so each concurrent branch
   * gets its OWN pending tags + sticky step and mutations never escape into a
   * sibling branch. (Node is single-threaded and `node:sqlite` writes stay
   * synchronous on the main thread, so — unlike the Python port — there is no
   * cross-thread DB race and no background writer thread is needed; the only
   * real bug shared with Python was this tag/step interleaving.)
   *
   * `rootStore` is the fallback used when no `step()` scope is active (the
   * common sequential case): `tag()`/`mark()` then behave EXACTLY as before —
   * mark() sticky, tag() next-call-only — because everything shares this one
   * store. It is per-Tracer (a Tracer is a once-per-run object), so a fresh run
   * never inherits a leftover step/tag from a previous Tracer.
   */
  private als = new AsyncLocalStorage<TagStore>();
  private rootStore: TagStore = { pendingTags: [], step: null };

  constructor(project: string, redact: RedactHook | null, path: string) {
    this.project = project;
    this.redact = redact;
    this.path = path;
  }

  /** The capture store for the CURRENT async context: the `step()` scope's
   * store if one is active, else the shared `rootStore`. Never throws. */
  private currentStore(): TagStore {
    return this.als.getStore() ?? this.rootStore;
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
      if (this.ct === null && !this.setupFailed) {
        // model is per-call, not known yet: pass "" so the store leaves
        // run.models == [] rather than seeding a bogus [""]; recordCall/
        // noteModel backfill the real model(s) as calls arrive. Canonical UTC
        // (`...Z` via toISOString) so downstream local-time rendering is
        // unambiguous — see store.parseStartedAt.
        const started = new Date().toISOString();
        // openOrCreateSession APPENDS a new session (run row) to an existing
        // project db, or creates the file — the project-scoped write path. Its
        // connection carries WAL + busy_timeout + bounded retry for safe
        // concurrent multi-writer access.
        //
        // Fail-open guard (the CRITICAL parity item): the store already sets
        // busy_timeout first and retries on a transient lock, but if it STILL
        // throws (a genuinely stuck lock under extreme concurrent creation, or an
        // unopenable path), that must NEVER escape into the host and lose the
        // whole session hard. We degrade fail-open: leave `ct` null, LATCH the
        // failure (so no later wrap() re-runs the blocking retry loop), warn
        // once, and fall through to return a proxy whose calls run normally but
        // record nothing (recorder stays null → `onCreate` short-circuits).
        try {
          this.ct = CTrace.openOrCreateSession(
            this.path,
            this.project,
            entry.provider,
            "",
            started,
          );
        } catch (err) {
          this.ct = null;
          this.setupFailed = true;
          this.warnSetupDegraded(err);
        }
      }
      const recorder = this.ct ? new Recorder(this.ct, adapter, this.redact) : null;
      if (recorder !== null && this.firstRecorder === null) {
        this.firstRecorder = recorder;
      }
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
   * Emit the capture-degradation warning AT MOST ONCE for the run when store
   * setup in `wrap()` fails and capture falls back to fail-open (record nothing).
   * A one-time flag stops repeated wraps on a broken store from spamming the
   * host's logs; the underlying setup error is included for diagnosis without
   * ever re-raising. Mirrors Python `_warn_setup_degraded`.
   */
  private warnSetupDegraded(err: unknown): void {
    if (this.setupWarned) return;
    this.setupWarned = true;
    console.warn(
      "ctxdiff: capture degraded (store setup failed); this run will not be recorded",
      err,
    );
  }

  /**
   * Buffer semantic tags for the NEXT recorded call only, IN THE CURRENT ASYNC
   * CONTEXT. Each item is reduced to its text (string as-is, else a
   * 'text'/'content' field, else String()) and paired with `label`; the recorder
   * marks any block containing that text as `label`. Contrast mark(): tag() is
   * next-call-only (drained after one call), mark() is sticky.
   *
   * Concurrency: the pending tags live in an AsyncLocalStorage store, so a tag()
   * inside one `step()` scope is visible ONLY to that scope's next recorded call
   * — never a concurrent sibling's. Fail-open: a store failure is swallowed so a
   * broken tag() can never break the host.
   */
  tag(label: string, items: unknown[]): void {
    try {
      const store = this.currentStore();
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
        if (text) store.pendingTags.push([label, text]);
      }
    } catch {
      // fail-open: a tag() must never break the host
    }
  }

  /**
   * Set the sticky step label stamped onto every subsequent recorded call IN THE
   * CURRENT ASYNC CONTEXT until changed; `mark(null)` clears it. STICKY (persists
   * until the next mark()), unlike tag() which is next-call-only.
   *
   * SEMANTICS: the step is stored per async context, not in global tracer state,
   * so "sticky" means sticky WITHIN THE CURRENT ASYNC CONTEXT. In sequential code
   * this is IDENTICAL to the old behavior. Under concurrency it is only self-
   * correct for a branch that runs inside its own `step()` scope (which gives it
   * a private store). A bare `mark()` at the synchronous top of several
   * `Promise.all` branches all mutate the SHARED root store and WILL bleed —
   * exactly the mislabeling this change fixes. For concurrent fan-out prefer the
   * scoped `step()` below, which isolates each branch. Fail-open.
   */
  mark(step: string | null): void {
    try {
      this.currentStore().step = step;
    } catch {
      // fail-open: a mark() must never break the host
    }
  }

  /**
   * Scoped, concurrency-safe phase label — the RECOMMENDED way to label phases
   * under concurrency. Two forms:
   *
   *   // callback (fully concurrency-safe — runs `fn` in its own async context)
   *   await tracer.step("retrieve", async () => { ...calls... });
   *
   *   // Disposable (TS 5.2 `using`, for sequential/single-context phases)
   *   { using _ = tracer.step("retrieve"); ...calls... }
   *
   * Every call recorded inside the scope carries `step=label`, and on exit the
   * previous step is restored. Prefer the callback form under a `Promise.all`
   * fan-out: it uses `AsyncLocalStorage.run()` to give the block a FRESH store
   * (its own pending tags + step), so `tag()`/`mark()` inside one branch are
   * isolated from every sibling branch — a branch that opens no `step()` scope
   * records `step=null`, never a sibling's leftover label. The Disposable form
   * uses `enterWith`, which is leak-proof for sequential nesting but — like a
   * bare `mark()` — can share state across branches started synchronously in the
   * same tick, so reach for the callback form when fanning out. Fully reentrant:
   * nested scopes restore the exact enclosing state. Fail-open throughout.
   */
  step<T>(label: string | null, fn: () => T): T;
  step(label: string | null): Disposable;
  step<T>(label: string | null, fn?: () => T): T | Disposable {
    // A fresh store for the scope: inherit the parent's pending tags (so a tag()
    // set just before the scope still applies) but override the step to `label`.
    // Because it's a new object, nothing written inside the scope escapes to the
    // parent or a sibling branch.
    let parentTags: [string, string][] = [];
    try {
      parentTags = [...this.currentStore().pendingTags];
    } catch {
      /* fail-open: fall back to empty */
    }
    const child: TagStore = { pendingTags: parentTags, step: label };

    if (fn) {
      // Callback form: run `fn` inside its own async context. als.run restores
      // the previous store automatically when `fn` (and its awaited promise)
      // settle — including on a synchronous throw — so nested/sibling scopes
      // never see this one's state. Let als.run's throw propagate UNCHANGED:
      // it is `fn`'s own error, and re-running `fn` in a catch would double
      // its side effects. The only fail-open guard is around store SETUP
      // above (parentTags), never around `fn` execution itself.
      return this.als.run(child, fn);
    }

    // Disposable form: swap in the child store now and restore on dispose.
    const previous = this.als.getStore();
    try {
      this.als.enterWith(child);
    } catch {
      /* fail-open: leave the store as-is */
    }
    let disposed = false;
    return {
      [Symbol.dispose]: (): void => {
        if (disposed) return;
        disposed = true;
        try {
          this.als.enterWith(previous ?? this.rootStore);
        } catch {
          /* fail-open: nothing to restore */
        }
      },
    };
  }

  /**
   * Drain the CURRENT async context's capture state for one recorded call.
   * Called SYNCHRONOUSLY by the interceptor on the calling side — before any
   * await that could cross into another context — so the returned tags/step
   * belong to the branch that made the call, not whoever happens to be running
   * when the async record later completes. Consumes the one-shot pending tags
   * (clears them so they apply to exactly this one call) and reads the sticky
   * step. Never throws: on any failure returns empty attribution (fail-open).
   */
  consumeContext(): { tags: [string, string][]; step: string | null } {
    try {
      const store = this.currentStore();
      const tags = store.pendingTags;
      store.pendingTags = [];
      return { tags, step: store.step };
    } catch {
      return { tags: [], step: null };
    }
  }

  /**
   * Interceptor callback: advance the turn counter and hand everything to the
   * wrapping proxy's recorder. `tags`/`step` are the attribution SNAPSHOTTED by
   * the interceptor via `consumeContext()` at call time (never re-read here,
   * which could run after an await in a different async context). `seq` stays a
   * single monotonic counter across ALL agents. Never throws — record is
   * internally fail-open, but this wiring is guarded too so a replaced/broken
   * record can't break the host.
   */
  onCreate(args: {
    kwargs: Record<string, unknown>;
    response: unknown;
    latencyMs: number | null;
    error: string | null;
    recorder: Recorder | null;
    agent: string | null;
    tags: [string, string][];
    step: string | null;
    quiet?: boolean;
  }): void {
    const { kwargs, response, latencyMs, error, recorder, agent, tags, step, quiet = false } = args;
    this.seq += 1;
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

    // Snapshot this call's attribution SYNCHRONOUSLY, on the calling side, before
    // the real method's promise can settle in a different async context. This is
    // what makes concurrent fan-out correct: whatever tag()/mark()/step() applied
    // in THIS branch is captured here and threaded into every deferred record —
    // never re-read after an await. Drains the one-shot pending tags.
    const { tags, step } = ctx.tracer.consumeContext();

    // Streaming if: a `.stream()` helper, a named streaming method (Gemini), or
    // the caller's own `stream:true` kwarg. Any of these routes to the stream
    // proxy, whose record is DEFERRED until the stream finishes iterating.
    const streaming = isStreamHelper || isNamedStreamMethod || !!kwargs["stream"];

    // Snapshot the request NOW for the deferred streaming record, on the SAME
    // tick as consumeContext() — before the stream is handed back. The proxy
    // records only after iteration ends, by which point the host may have
    // mutated its own live `messages` array / params object (append an
    // assistant placeholder to fill during streaming, or reuse it for the next
    // turn). Recording off the live ref would then capture blocks that were
    // never sent in THIS call. A deep clone freezes exactly what went out; the
    // cloned text is identical, so tag() needle-matching still resolves.
    // Fail-open: if the payload isn't structured-cloneable, fall back to the
    // live ref rather than throw into the host. Non-streaming needs no snapshot
    // — it records synchronously in `.then`/on return, before the host resumes.
    let streamKwargs: Record<string, unknown> = kwargs;
    if (streaming) {
      try {
        streamKwargs = structuredClone(kwargs);
      } catch {
        streamKwargs = kwargs;
      }
    }

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
        tags,
        step,
      });
      throw exc;
    }

    if (isThenable(result)) {
      // create()/create({stream:true}) — returns a Promise. Record after await,
      // but with the tags/step SNAPSHOTTED above (not re-read post-await).
      return result.then(
        (resolved: unknown) => {
          if (streaming) return wrapStream(resolved, streamKwargs, ctx, start, tags, step);
          const latencyMs = Math.round(performance.now() - start);
          ctx.tracer.onCreate({
            kwargs,
            response: resolved,
            latencyMs,
            error: null,
            recorder: ctx.recorder,
            agent: ctx.agent,
            tags,
            step,
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
            tags,
            step,
          });
          throw exc; // host's own async error — re-raised unchanged
        },
      );
    }

    // Synchronous return — the `.stream()` helper hands back its stream directly.
    if (streaming) return wrapStream(result, streamKwargs, ctx, start, tags, step);

    // Non-streaming sync return (unusual for openai) — record now.
    const latencyMs = Math.round(performance.now() - start);
    ctx.tracer.onCreate({
      kwargs,
      response: result,
      latencyMs,
      error: null,
      recorder: ctx.recorder,
      agent: ctx.agent,
      tags,
      step,
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
  tags: [string, string][],
  step: string | null,
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
      tags,
      step,
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
        tags,
        step,
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

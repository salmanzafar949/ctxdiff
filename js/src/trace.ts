/**
 * The public entry point. `init()` opens the project's store and returns a
 * Tracer; `tracer.wrap(client)` returns a transparent JS Proxy that records
 * every completion call while behaving exactly like the original client.
 *
 * FAIL-OPEN IS ABSOLUTE. Wrapping must never throw into the host, never break
 * iteration of a stream, never drop/reorder/delay a chunk, and never alter the
 * request. Recording is a strictly best-effort side channel.
 *
 * WHERE the session lands is decided ONCE, in the constructor, by
 * `resolveBackend`: an explicit `store`, an explicit `path`, `configure()`,
 * `CTXDIFF_STORE`, or — with nothing configured — the unchanged zero-config
 * `./<project>.ctrace`. Everything below that line talks to the `Store`
 * protocol, and the two kinds of backend are handled differently ON PURPOSE:
 *
 * - a FILE store opens and writes synchronously, exactly as it always has (a
 *   local open costs microseconds and every existing caller depends on the
 *   write having happened by the time `close()` returns);
 * - a NETWORKED store touches nothing on the host's call path. `wrap()` builds
 *   a `DeferredStore` + an `AsyncWriter` and returns; the writer opens the
 *   session and drains one job at a time; every wait — connect, statement,
 *   `close()` — is bounded. Node's single thread makes this non-negotiable: an
 *   unbounded await here is a program that cannot exit.
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
import type { Adapter, CallShape } from "./capture/base.js";
import { OpenAIAdapter } from "./capture/openai.js";
import { AnthropicAdapter } from "./capture/anthropic.js";
import { GeminiAdapter } from "./capture/gemini.js";
import { BedrockAdapter } from "./capture/bedrock.js";
import { buildHandler, type CtxdiffCallbackHandler } from "./capture/langchain.js";
import { Recorder, type RedactHook } from "./capture/recorder.js";
import type {
  Awaitable,
  Call,
  CallBlock,
  RecordCallArgs,
  Run,
  Session,
  Store,
  StoreBackend,
} from "./store/base.js";
import { isFileBackend, statementTimeoutOf } from "./store/base.js";
import { SQLiteStore } from "./store/sqlite.js";
import { resolve as resolveConfigured } from "./store/config.js";

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

/**
 * Recognize an `@aws-sdk/client-bedrock-runtime` client: by class name, or by
 * the resolved config's `serviceId` — which the SDK's own runtime config sets
 * to `"Bedrock Runtime"` and which therefore survives a bundler mangling the
 * class name.
 *
 * DELIBERATELY NARROW, unlike the other three detectors. Every AWS SDK v3
 * client is `{ send(command), config, middlewareStack }`, so duck-typing `send`
 * would claim S3, DynamoDB and SQS clients too — wrapping them would add a
 * proxy and a per-call `interpretCall` to code paths that can never produce a
 * context block. `serviceId` is the narrowest signal the SDK publishes, and a
 * non-Bedrock client simply falls through to unrecognized (returned unwrapped).
 */
function detectBedrock(client: unknown): boolean {
  if (client == null || typeof client !== "object") return false;
  if (ctorName(client) === "BedrockRuntimeClient") return true;
  const c = client as { send?: unknown; config?: { serviceId?: unknown } };
  return typeof c.send === "function" && c.config?.serviceId === "Bedrock Runtime";
}

// The single extension point. Order matters only to disambiguate overlapping
// duck-types; these four are mutually exclusive (distinct resource shapes).
const REGISTRY: ProviderEntry[] = [
  { provider: "openai", detect: detectOpenAI, make: () => new OpenAIAdapter() },
  { provider: "anthropic", detect: detectAnthropic, make: () => new AnthropicAdapter() },
  { provider: "gemini", detect: detectGemini, make: () => new GeminiAdapter() },
  { provider: "bedrock", detect: detectBedrock, make: () => new BedrockAdapter() },
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
  /**
   * Where this run's session lands, as a `StoreBackend` — `SQLiteStore`,
   * `PostgresStore`, `MySQLStore`. Usually you don't pass it: `configure({
   * store })` once at startup, or the `CTXDIFF_STORE` env var, applies to every
   * `init()` from then on. See `resolveBackend` for the precedence.
   */
  store?: StoreBackend;
}

// How long `close()` waits for the writer when the store publishes no bound of
// its own — the local `.ctrace`, whose writes are synchronous anyway, so this is
// only ever a safety valve.
const DEFAULT_CLOSE_TIMEOUT_MS = 30_000;

// Milliseconds added to a networked store's statement bound to get the close
// bound: the writer may be mid-statement when close() arrives (up to that
// timeout) and then still has to close the connection.
const CLOSE_TIMEOUT_MARGIN_MS = 2_000;

/**
 * How long `close()` should wait for the writer to drain, given the store it is
 * writing to.
 *
 * A networked backend publishes a `statementTimeout`: nothing on that connection
 * can legitimately be busy for longer, so waiting 30 seconds on a database that
 * has stopped answering just makes a failed deployment slower to shut down.
 * Anything without one (the local SQLite store, a test double) keeps the
 * generous default. Read by capability rather than by class, the same way
 * `Tracer` asks a backend for `pathFor`.
 */
function closeTimeoutFor(backend: StoreBackend): number {
  const seconds = statementTimeoutOf(backend);
  return seconds === null ? DEFAULT_CLOSE_TIMEOUT_MS : seconds * 1000 + CLOSE_TIMEOUT_MARGIN_MS;
}

/**
 * Decide which backend a Tracer writes to, explicit-beats-ambient:
 *
 * 1. an explicit `store` option — the caller named a backend outright;
 * 2. an explicit `path` — a filesystem path is unambiguously a local `.ctrace`,
 *    so it beats an ambient `configure()`/env-var setting rather than being
 *    silently ignored (a caller who passes a path and gets a row in someone's
 *    Postgres would rightly call that a bug);
 * 3. `configure()`, then `CTXDIFF_STORE` (both via `store/config.ts`);
 * 4. nothing configured -> `new SQLiteStore()`, i.e. `./<project>.ctrace` — the
 *    unchanged zero-config default every existing user keeps getting.
 *
 * Returns a backend, never null; it may throw (e.g. an unparseable
 * `CTXDIFF_STORE`), which is why `Tracer`'s constructor calls it inside a guard.
 */
function resolveBackend(path: string | undefined, store: StoreBackend | undefined): StoreBackend {
  if (store !== undefined) return store;
  if (path !== undefined) return new SQLiteStore({ path });
  return resolveConfigured() ?? new SQLiteStore();
}

/**
 * Stand-in for a backend that could not even be RESOLVED — e.g. a typo'd
 * `CTXDIFF_STORE=postgres`, or a `configure()`d backend whose module failed to
 * load. Holds the original error and re-throws it from `openSession()`, so the
 * failure surfaces at exactly the point `wrap()` already guards against a dead
 * store: capture degrades fail-open with one warning carrying the real cause,
 * and the host runs untouched.
 *
 * Why not just throw from `init()`: a misconfigured trace destination is still a
 * tracing problem, and tracing problems must never take down the program being
 * traced. Why not silently fall back to a local file: a user who asked for
 * Postgres and got a surprise `.ctrace` in their container's working directory
 * has been lied to.
 */
class UnavailableBackend implements StoreBackend {
  private readonly error: Error;

  constructor(error: Error) {
    this.error = error;
  }

  /** Re-throw the resolution failure into `wrap()`'s fail-open guard. */
  openSession(): never {
    throw this.error;
  }

  /** Re-throw for read-side callers (the CLI), which report it rather than
   * degrade. */
  openReader(): never {
    throw this.error;
  }
}

/** Raised by `DeferredStore` when the session could not be opened. Its own type
 * so the writer can tell "this run has no store at all" (drop the job silently —
 * the one-time warning already fired at the open) apart from "this particular
 * write failed" (warn once, keep going). */
class StoreUnavailableError extends Error {}

/**
 * A `Store` handle whose session is OPENED BY THE WRITER, on first use, rather
 * than by whoever constructed it.
 *
 * Why this exists at all: `wrap()` runs on the host's own tick, in the middle of
 * an agent doing real work, and opening a networked session is I/O — a TCP
 * connect, an authentication handshake, `CREATE TABLE IF NOT EXISTS`, an INSERT.
 * Doing that inline would make the host pay for the tracing store's health: an
 * `await` inside `wrap()` turns the host's first LLM call into a continuation
 * that cannot run until a database answers, and a database that completes its
 * handshake and then stops answering has no bound at all — a connect timeout
 * covers connect and auth and nothing after them, and a server-side statement
 * timeout cannot fire when the packets carrying it are being dropped.
 *
 * Bounding that I/O tighter would only shrink the damage. Moving it removes it:
 * `wrap()` constructs this handle (pure bookkeeping, zero I/O, zero awaits) and
 * the writer opens the real store as its FIRST act — concurrently with the
 * host's first call, with any calls made meanwhile waiting in the queue.
 *
 * Failure keeps the existing fail-open shape: the open is attempted exactly
 * once, a failure warns exactly once through `onFailure`, and every later method
 * rejects with `StoreUnavailableError` so the writer drops jobs instead of
 * retrying a store that is not there.
 */
class DeferredStore implements Store {
  private readonly openSession: () => Awaitable<Store>;
  private readonly onFailure: (err: unknown) => void;
  private store: Store | null = null;
  private opening: Promise<Store | null> | null = null;
  private settled = false;

  /**
   * Record HOW to open the session (a zero-arg callable closing over the
   * project/provider/startedAt decided at `wrap()` time, so the session still
   * carries the moment the host started tracing — not the moment the writer got
   * around to connecting) and who to tell if it fails. Nothing is opened here;
   * `open()` does that, from the writer.
   */
  constructor(openSession: () => Awaitable<Store>, onFailure: (err: unknown) => void) {
    this.openSession = openSession;
    this.onFailure = onFailure;
  }

  /**
   * Open the session, ONCE, resolving to the real store or null if it failed.
   * Called by the writer before it processes any job; caching the in-flight
   * promise makes a second call (a racing submit, a test driving it directly) a
   * no-op rather than a second session. A failure is swallowed and reported
   * through `onFailure` — this runs on the writer's loop, where throwing would
   * kill the thing that is the host's only protection from store errors.
   */
  open(): Promise<Store | null> {
    if (this.settled) return Promise.resolve(this.store);
    if (this.opening !== null) return this.opening;
    this.opening = (async () => {
      try {
        this.store = await this.openSession();
      } catch (err) {
        this.store = null;
        this.onFailure(err);
      }
      this.settled = true;
      return this.store;
    })();
    return this.opening;
  }

  /** The opened store, or throw. Opens on demand so a caller that is not the
   * writer loop (a direct `recorder.persist`, a test) still gets a working
   * handle, and throws `StoreUnavailableError` when the open failed so the
   * caller's own fail-open guard treats it as the write failure it is. */
  private async require(): Promise<Store> {
    const store = await this.open();
    if (store === null) {
      throw new StoreUnavailableError("ctxdiff: the store for this run could not be opened");
    }
    return store;
  }

  /** Persist one call through the real store (see `Store.recordCall`). */
  async recordCall(args: RecordCallArgs): Promise<string> {
    return (await this.require()).recordCall(args);
  }

  /** Roll a model id up onto the session (see `Store.noteModel`). */
  async noteModel(model: string | null | undefined): Promise<void> {
    await (await this.require()).noteModel(model);
  }

  /** Every session in the underlying store. */
  async listSessions(): Promise<Session[]> {
    return (await this.require()).listSessions();
  }

  /** One session's run row. */
  async getRun(sessionId?: string): Promise<Run> {
    return (await this.require()).getRun(sessionId);
  }

  /** One session's calls, in turn order. */
  async getCalls(sessionId?: string): Promise<Call[]> {
    return (await this.require()).getCalls(sessionId);
  }

  /** One call's blocks, in position order. */
  async getCallBlocks(callId: string): Promise<CallBlock[]> {
    return (await this.require()).getCallBlocks(callId);
  }

  /**
   * Close the underlying store if one was ever opened, and never reject — this
   * runs on the writer's way out. A store that was never opened (no call was
   * ever recorded, or the open failed) has nothing to close, so this is also
   * what stops a degraded run from connecting at shutdown just to disconnect
   * again.
   */
  async close(): Promise<void> {
    const store = this.store;
    const opening = this.opening;
    this.store = null;
    this.settled = true; // never open a session while shutting down
    if (store === null) {
      // The open may still be IN FLIGHT (close raced it, or close's own
      // deadline expired first). Whatever it eventually produces is a live
      // connection nobody owns, so it is closed on arrival rather than leaked —
      // without awaiting it here, which would let a wedged connect stall the
      // shutdown it is supposed to be bounded by.
      if (opening !== null) {
        void opening.then(
          (late) => {
            if (late !== null) void Promise.resolve(late.close()).catch(() => undefined);
          },
          () => undefined,
        );
      }
      return;
    }
    try {
      await store.close();
    } catch {
      /* close is best-effort on the way out */
    }
  }
}

/**
 * The run's single serial writer for a NETWORKED store, sitting behind a bounded
 * queue.
 *
 * Why it exists: a networked write is a round trip that can be slow or fail, and
 * one SQL connection tolerates exactly one statement at a time. So every persist
 * for a run is funnelled through ONE async loop that owns the connection:
 * `submit()` (called from the host's tick) appends a job and returns
 * IMMEDIATELY, having awaited nothing; the loop drains FIFO, awaiting each job
 * in turn. Because exactly one loop ever writes, there is never concurrent
 * connection access — and, since this loop also OPENS the store (see
 * `DeferredStore`), the connection is never even created anywhere else.
 *
 * This is the JS shape of Python's writer THREAD, and it buys the same thing for
 * the same reason. Node needs no thread — network I/O is already off the call
 * path — but it does need the serialization (one statement per connection) and
 * the bounded queue, and it needs `close()` to be the one place that waits.
 *
 * Ordering: `seq` is assigned by the caller BEFORE `submit()` (see
 * `Tracer.onCreate`), so it reflects call-COMPLETION order; the queue is FIFO
 * and the store reads back `ORDER BY seq`, so the persisted timeline is stable
 * regardless of how the loop interleaves with the host.
 *
 * Fail-open (the whole point): `submit()` never blocks the host and never
 * throws. On a full queue (backpressure) or after close, the record is DROPPED
 * and a ONE-TIME degradation warning is emitted. Each job runs inside its own
 * guard, so a single bad job can never end the loop.
 *
 * Shutdown: `close()` lets the loop finish what is already queued, then closes
 * the connection — bounded by `closeTimeout`, so a wedged store bounds shutdown
 * instead of hanging the program that is trying to exit.
 */
class AsyncWriter {
  private readonly store: DeferredStore;
  private readonly closeTimeout: number;
  private readonly maxsize: number;
  private queue: (() => Awaitable<void>)[] = [];
  private draining = false;
  private drainDone: Promise<void> = Promise.resolve();
  private closed = false;
  private warned = false;
  private persistWarned = false;
  // How many writes arrived DURING the close flush and were still recorded —
  // reported once at the end of close(), so the rare race is observable rather
  // than silent (mirrors Python's `_drain_stragglers`).
  private stragglers = 0;
  // null until the loop has tried: false means the store never opened, so jobs
  // are dropped (the warning already fired at the open).
  private storeReady: boolean | null = null;
  // The in-flight close, so a second close() awaits the FIRST one's full
  // shutdown (drain AND connection release) rather than returning early.
  private closing: Promise<void> | null = null;

  constructor(store: DeferredStore, closeTimeout: number, maxsize = 10_000) {
    this.store = store;
    this.closeTimeout = closeTimeout;
    this.maxsize = maxsize;
  }

  /**
   * Enqueue one persist job from the host's tick and return at once. How: pushes
   * `job` (a thunk that persists one call) and kicks the drain loop if it is not
   * already running; if the writer is closed or the queue is full
   * (backpressure), the job is dropped and `degrade` fires the one-time warning.
   * The whole body is wrapped so nothing — not even an unexpected error building
   * the enqueue — can ever propagate into the host call. `quiet` suppresses the
   * warning for shutdown-time callers (a stream's best-effort finalize), where
   * warning is noise at best.
   *
   * A call that lands DURING `close()`'s flush is still ACCEPTED — the drain
   * loop is running and will pick it up before it finishes. This is the JS shape
   * of Python's `_drain_stragglers`, and it matters more here: Python's
   * `close()` blocks its caller, while ours yields to the event loop for the
   * whole flush, so a host LLM call that was already in flight WILL land in that
   * window, and dropping it would lose a turn the user watched happen. It cannot
   * postpone shutdown, because `close()`'s own deadline bounds the drain
   * regardless of what arrives. Only a call landing after the drain has finished
   * is dropped, with the one-time warning.
   */
  submit(job: () => Awaitable<void>, quiet = false): void {
    try {
      if (this.closed && !this.draining) {
        this.degrade("writer already closed", quiet);
        return;
      }
      if (this.closed) this.stragglers += 1;
      if (this.queue.length >= this.maxsize) {
        this.degrade("write queue overflow", quiet);
        return;
      }
      this.queue.push(job);
      this.kick();
    } catch {
      this.degrade("enqueue failed", quiet);
    }
  }

  /**
   * Begin opening the store WITHOUT waiting for it — called by `wrap()` the
   * moment the writer exists.
   *
   * Why eagerly rather than on the first write: a run that wraps a client and
   * records nothing (an agent that errors before its first LLM call, a
   * short-circuited request) should still leave a SESSION behind, saying "this
   * ran, and captured nothing" — which is exactly what the local `.ctrace` does,
   * and what the Python SDK's writer thread does by opening before its first
   * queue read. Without this, such a run would be invisible.
   *
   * It costs the host nothing: this returns synchronously after scheduling the
   * drain, so the connect happens in a later microtask while the host gets on
   * with its first LLM call.
   */
  start(): void {
    this.kick();
  }

  /** Start the drain loop if it is not already running, remembering its promise
   * so `close()` has something to wait on. The loop never rejects, so this
   * floating promise can never become an unhandled rejection. */
  private kick(): void {
    if (this.draining) return;
    this.draining = true;
    this.drainDone = this.drain();
  }

  /**
   * The loop: OPEN the store, then drain the queue FIFO, persisting each job.
   *
   * The open comes first and happens HERE — every byte of store I/O for the run,
   * from the TCP connect onwards, belongs to this loop and never to the host's
   * call path (see `DeferredStore`). When it fails, the one-time degradation
   * warning has already fired and every job is then dropped: retrying each write
   * against a store that was never there would only produce a second class of
   * warning for the same fact.
   *
   * Termination is race-free without a lock because JS is single-threaded: there
   * is no `await` between the loop's final emptiness check and clearing
   * `draining`, so a `submit()` can never slip a job in and then find the loop
   * already gone.
   */
  private async drain(): Promise<void> {
    try {
      if (this.storeReady === null) {
        this.storeReady = (await this.store.open()) !== null;
      }
      while (this.queue.length > 0) {
        const job = this.queue.shift()!;
        // A job is skipped outright when the store never opened — the
        // degradation was already warned about once, at the open — so a
        // dead-database run produces exactly one warning rather than a second
        // one about the first write it could never have made.
        if (!this.storeReady) continue;
        try {
          await job();
        } catch (err) {
          this.warnPersist(err);
        }
      }
    } catch {
      /* the loop itself must never reject: it is the host's protection */
    } finally {
      this.draining = false;
    }
  }

  /** Emit the capture-degradation warning AT MOST ONCE for the run, then stay
   * silent — a storm of dropped calls must not spam the host's logs. */
  private degrade(reason: string, quiet: boolean): void {
    if (quiet || this.warned) return;
    this.warned = true;
    console.warn(
      `ctxdiff: capture degraded (${reason}); some calls in this run will not be recorded`,
    );
  }

  /** Warn AT MOST ONCE about a job that failed on the writer side. A store
   * failing every write would otherwise log one line per turn. */
  private warnPersist(err: unknown): void {
    if (this.persistWarned) return;
    this.persistWarned = true;
    console.warn(
      "ctxdiff: writer failed to persist a call; further writer failures in " +
        "this run will be silent (skipped)",
      err,
    );
  }

  /**
   * Flush every enqueued write, stop accepting new ones, and close the store.
   *
   * BOUNDED, and that bound is the whole point: the WHOLE of close() — the flush
   * and the connection release together — takes at most `closeTimeout` (derived
   * from the store's own statement timeout, see `closeTimeoutFor`), after which
   * close warns and returns rather than leaving a program unable to exit because
   * a database stopped answering. A store that was never opened is never
   * connected to just to be disconnected. Idempotent: a second close awaits the
   * same drain.
   */
  async close(): Promise<void> {
    // Idempotent by SHARING the first close's promise: awaiting only the drain
    // would let a second caller return while the connection was still being
    // released.
    if (this.closing === null) this.closing = this.shutdown();
    return this.closing;
  }

  /**
   * The one real close: flush, report, release. Never rejects.
   *
   * ONE deadline covers BOTH phases. Giving the flush its full budget and then
   * the release another full budget would make the real ceiling twice the
   * documented one — 24 seconds on the defaults, not 12 — for exactly the case
   * the bound exists for: a database that has stopped answering, which stalls
   * both phases. So the release gets whatever the flush left, and a flush that
   * used the whole budget leaves the release ATTEMPTED (the store is told to
   * close, and its own internal bounds still apply) but not waited on.
   */
  private async shutdown(): Promise<void> {
    this.closed = true;
    if (this.queue.length > 0) this.kick();
    const deadline = Date.now() + this.closeTimeout;
    const drained = await raceDeadline(this.drainDone, this.closeTimeout);
    if (!drained) {
      console.warn(
        `ctxdiff: writer did not drain within ${this.closeTimeout}ms on close; ` +
          "some writes may be lost",
      );
    } else if (this.stragglers > 0) {
      console.warn(
        `ctxdiff: drained ${this.stragglers} write(s) enqueued during close ` +
          "(recorded, not lost)",
      );
    }
    await raceDeadline(this.store.close(), Math.max(0, deadline - Date.now()));
  }
}

/**
 * Wait for `work`, but never longer than `ms`; resolves true if it finished and
 * false if the deadline won. Used only by `close()`.
 *
 * The timer is deliberately REFERENCED — the one place in this file that holds
 * the event loop open on purpose. A networked store's socket is detached
 * (`SqlConn.unref`) so that forgetting `close()` cannot hang a program; the
 * consequence is that during an explicit `close()` there may be no referenced
 * handle left, and Node would exit mid-flush, silently losing the writes the
 * caller explicitly asked to flush. This timer is what keeps the process alive
 * for exactly as long as the flush is allowed to take, and not one millisecond
 * longer: it is cleared the moment the work settles.
 *
 * The abandoned promise gets a no-op rejection handler so a late failure cannot
 * surface as an unhandled rejection in the host's process.
 */
function raceDeadline(work: Promise<unknown>, ms: number): Promise<boolean> {
  return new Promise<boolean>((resolveRace) => {
    const timer = setTimeout(() => resolveRace(false), ms);
    work.then(
      () => {
        clearTimeout(timer);
        resolveRace(true);
      },
      () => {
        clearTimeout(timer);
        resolveRace(true);
      },
    );
  });
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
 * Create a Tracer that opens the project's store and starts a NEW SESSION in it.
 * `project` names the project; `opts.redact` is an optional per-block scrubber
 * applied before storage; `opts.path` is where the project `.ctrace` is written
 * (defaults to a STABLE ./<project>.ctrace in the cwd — see InitOptions). The
 * first `wrap()` APPENDS a session to that file if it already exists, or creates
 * it if absent, so every `init(project)` accumulates one more session in the
 * same project db.
 *
 * Pluggable storage: `opts.store` overrides WHERE that session lands with any
 * `StoreBackend` — `SQLiteStore`, `PostgresStore`, `MySQLStore`. Usually you
 * don't pass it: `configure({ store })` once at startup, or the `CTXDIFF_STORE`
 * env var, applies to every `init()` from then on (see `resolveBackend`).
 * Mirrors Python `trace.init`.
 */
export function init(project: string, opts: InitOptions = {}): Tracer {
  return new Tracer(project, opts.redact ?? null, opts.path, opts.store);
}

export class Tracer {
  /**
   * The concrete `.ctrace` this run writes to — or null for a networked backend,
   * where "the path" is meaningless. Backend-derived (via `pathFor`) rather than
   * computed here, so a local run reports exactly what it has always reported.
   */
  readonly path: string | null;
  private backend: StoreBackend;
  private project: string;
  private redact: RedactHook | null;
  // The synchronous local store (a `CTrace`), for a file backend only. Opened
  // inline by wrap(), exactly as before.
  private ct: Store | null = null;
  // The deferred handle + serial writer, for a NETWORKED backend only. wrap()
  // creates both without touching the network; the writer opens the session.
  private deferred: DeferredStore | null = null;
  private writer: AsyncWriter | null = null;
  private firstRecorder: Recorder | null = null;
  // Recorders built per PROVIDER for capture paths that only learn the provider
  // per call (the LangChain handler) — see `recorderFor`.
  private recorders = new Map<string, Recorder>();
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

  /**
   * Store the run's static config and decide, ONCE, which backend this run
   * writes to. Resolving a backend is pure and connection-less (see
   * `resolveBackend`), so it is safe here; it is nonetheless wrapped, because a
   * bad `CTXDIFF_STORE` must become a deferred fail-open degradation rather than
   * an exception out of `trace.init()` — tracing problems never take down the
   * program being traced.
   *
   * `path` stays part of the public surface but is backend-derived: the concrete
   * `.ctrace` file for a SQLite backend (unchanged for every existing user), and
   * null for a networked one.
   */
  constructor(
    project: string,
    redact: RedactHook | null,
    path?: string,
    store?: StoreBackend,
  ) {
    try {
      this.backend = resolveBackend(path, store);
    } catch (err) {
      this.backend = new UnavailableBackend(err as Error);
    }
    this.path = isFileBackend(this.backend) ? this.backend.pathFor(project) : null;
    this.project = project;
    this.redact = redact;
  }

  /** The capture store for the CURRENT async context: the `step()` scope's
   * store if one is active, else the shared `rootStore`. Never throws. */
  private currentStore(): TagStore {
    return this.als.getStore() ?? this.rootStore;
  }

  /**
   * Return a transparent Proxy over `client` that records every call to the
   * provider's completion methods. The store/run is created lazily on the first
   * wrap, once the provider is known — inline for a local `.ctrace`, and as a
   * deferred recipe for a database, so that against a networked backend this
   * method performs NO I/O and awaits nothing (see `ensureStore`). FAIL-OPEN:
   * any setup failure (unrecognized client, store error) logs and returns the
   * ORIGINAL client unwrapped, so the host keeps working with tracing simply
   * absent — wrap never throws.
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
      this.ensureStore(entry.provider);
      const store: Store | null = this.ct ?? this.deferred;
      const recorder = store ? new Recorder(store, adapter, this.redact) : null;
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
   * Return a LangChain callback handler that records every chat-model call made
   * under it — the IDIOMATIC way to trace LangChain and LangGraph:
   *
   *   const handler = tracer.langchainHandler();
   *   const llm = new ChatOpenAI({ model: "gpt-4o", callbacks: [handler] });
   *   // or per-invocation, which is what LangGraph propagates:
   *   await graph.invoke(state, { callbacks: [handler] });
   *
   * WHY A HANDLER RATHER THAN `wrap()`: `tracer.wrap()` needs a provider SDK
   * client, and a LangChain app hands you a `ChatOpenAI`, not an `OpenAI`.
   * Reaching inside LangChain for the client it happens to hold is a bet on
   * someone else's private structure. A callback is LangChain's own extension
   * point: it fires for EVERY integration (ChatOpenAI, ChatAnthropic,
   * ChatVertexAI, ...), streaming or not, and LangGraph propagates it through
   * an entire graph, so one handler covers a whole agent.
   *
   * The blocks it records are IDENTICAL — same hashes — to what wrapping that
   * provider's SDK directly would have recorded for the same request, because
   * the handler rebuilds the provider's own wire shape and feeds it to the very
   * same adapter (see `capture/langchain.ts`). That holds ACROSS SDKs too: the
   * Python handler normalizes to the same shapes, so a `.ctrace` written here
   * dedups against one written there.
   *
   * `opts.agent` names the agent these calls belong to, exactly as
   * `wrap(client, { agent })` does. The returned object is LangChain's plain
   * `CallbackHandlerMethods` shape, so no LangChain import (or dependency) is
   * involved. Fail-open throughout, like the rest of capture.
   *
   * All four provider branches are live — ChatOpenAI, ChatAnthropic,
   * ChatGoogleGenerativeAI/ChatVertexAI and ChatBedrockConverse — so one
   * handler covers a multi-provider graph. A call whose provider this SDK
   * genuinely has no adapter for is skipped with one warning rather than
   * recorded through some other provider's adapter.
   */
  langchainHandler(opts: WrapOptions = {}): CtxdiffCallbackHandler {
    return buildHandler(this, opts.agent ?? null);
  }

  /**
   * The Recorder for `provider`, created once per provider and cached — the
   * entry point for capture paths that discover their provider per CALL rather
   * than per client (today: the LangChain handler, which sees
   * provider-agnostic messages and may serve ChatOpenAI and ChatAnthropic from
   * the same handler).
   *
   * `wrap()` can build its recorder eagerly because a client has exactly one
   * provider; a handler cannot, so this does the same steps lazily: ensure the
   * run's store exists (idempotent — the first provider to arrive still decides
   * the session's `provider`, same as the first `wrap()` does), build that
   * provider's adapter, wrap both in a Recorder.
   *
   * Returns null when this SDK has no adapter for the provider, or when store
   * setup already failed — never throws: it is called from inside a callback on
   * the host's own path, where fail-open outranks fail-loud.
   */
  recorderFor(provider: string): Recorder | null {
    try {
      const cached = this.recorders.get(provider);
      if (cached !== undefined) return cached;
      const entry = REGISTRY.find((e) => e.provider === provider);
      if (entry === undefined) return null;
      this.ensureStore(provider);
      const store: Store | null = this.ct ?? this.deferred;
      if (store === null) return null;
      const recorder = new Recorder(store, entry.make(), this.redact);
      this.recorders.set(provider, recorder);
      if (this.firstRecorder === null) this.firstRecorder = recorder;
      return recorder;
    } catch (err) {
      console.warn("ctxdiff: could not prepare a recorder; call not recorded", err);
      return null;
    }
  }

  /**
   * Create this run's store handle exactly ONCE, on the first `wrap()` — and do
   * it differently for the two kinds of backend, which is the crux of the
   * "never block the host" contract.
   *
   * FILE BACKEND (the zero-config default): the session is opened INLINE and
   * synchronously, exactly as it always has been. `node:sqlite` has no async API,
   * the cost is a local file open with no network in it, and every existing
   * caller — including a test that wraps, calls, closes and immediately reads the
   * file back — depends on the write having happened by the time `close()`
   * returns. So this path is byte-for-byte the code it was, fail-open guard
   * included: a failure leaves `ct` null, LATCHES (so no later `wrap()` re-runs
   * the blocking retry budget, which on Node's single thread is real frozen
   * time), warns once, and the returned proxy runs the host's calls while
   * recording nothing.
   *
   * NETWORKED BACKEND: no I/O happens here AT ALL — not a connect, not a DNS
   * lookup, not an `await`. Only a `DeferredStore` (a recipe) and an
   * `AsyncWriter` (an empty queue) are constructed, and the writer opens the
   * real session as its first act. A database that is slow, wedged,
   * blackholed or simply absent therefore costs the host's call path exactly
   * nothing, and the failure surfaces where every other store failure does: one
   * warning, capture degraded, host untouched.
   *
   * `startedAt` is stamped HERE either way, on the host's tick, so a session
   * records when tracing began rather than whenever the writer finished
   * connecting. `model` is left empty because a run's model is a per-CALL fact
   * `wrap()` does not know yet — seeding a placeholder would store a permanent
   * blank, and `noteModel()` backfills the real ones.
   */
  private ensureStore(provider: string): void {
    if (this.ct !== null || this.deferred !== null || this.setupFailed) return;
    // Canonical UTC (`...Z` via toISOString) so downstream local-time rendering
    // is unambiguous — see store.parseStartedAt.
    const startedAt = new Date().toISOString();
    const args = { project: this.project, provider, model: "", startedAt };

    if (isFileBackend(this.backend)) {
      try {
        this.ct = this.backend.openSession(args);
      } catch (err) {
        this.ct = null;
        this.setupFailed = true;
        this.warnSetupDegraded(err);
      }
      return;
    }

    const backend = this.backend;
    this.deferred = new DeferredStore(
      () => backend.openSession(args),
      (err) => this.warnSetupDegraded(err),
    );
    this.writer = new AsyncWriter(this.deferred, closeTimeoutFor(backend));
    // Schedules the open; awaits nothing. See `AsyncWriter.start`.
    this.writer.start();
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
    if (recorder === null) return;
    const seq = this.seq;
    try {
      if (this.writer === null) {
        // Local file: build and write in the same tick, exactly as before.
        recorder.record({
          seq,
          kwargs,
          response,
          latencyMs,
          error,
          tagged: tags,
          agent,
          step,
          quiet,
        });
        return;
      }
      // Networked store: SNAPSHOT the call here, on the host's tick (so the
      // host's own `messages` array can be mutated the moment this returns
      // without changing what gets recorded), then hand the write to the writer
      // and return immediately — nothing on the host's path awaits a database.
      const job = recorder.build({
        seq,
        kwargs,
        response,
        latencyMs,
        error,
        tagged: tags,
        agent,
        step,
        quiet,
      });
      if (job === null) return;
      this.writer.submit(() => recorder.persist(job, quiet), quiet);
    } catch (err) {
      if (!quiet) {
        console.warn(`ctxdiff: recorder.record raised; tracing skipped for seq=${seq}`, err);
      }
    }
  }

  /**
   * Close the underlying store, if one was opened.
   *
   * The local file is closed SYNCHRONOUSLY, before this returns — so the
   * existing `tracer.close(); CTrace.open(path)` shape keeps working with no
   * `await` — and the returned promise is already resolved. For a networked
   * store the promise is what matters: awaiting it flushes every queued write
   * and closes the connection, BOUNDED by the store's own statement timeout so a
   * database that has stopped answering can never stop a program from exiting.
   * Never rejects, so an un-awaited `close()` cannot produce an unhandled
   * rejection.
   */
  async close(): Promise<void> {
    if (this.ct !== null) {
      try {
        // `CTrace.close()` is synchronous AND idempotent, so this runs (and
        // finishes) before the returned promise is even created — which is what
        // keeps `tracer.close(); CTrace.open(path)` working with no await.
        await this.ct.close();
      } catch {
        /* close must never throw into the host */
      }
    }
    if (this.writer !== null) {
      try {
        // The handle is deliberately NOT cleared: `AsyncWriter.close()` is
        // itself idempotent and a second call AWAITS the first one's flush,
        // whereas dropping the reference would make a second `close()` return
        // instantly while writes were still in flight.
        await this.writer.close();
      } catch {
        /* close must never throw into the host */
      }
    }
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
    // WHAT IS THIS CALL? Two answers, and the adapter gets first refusal.
    //
    // The default (OpenAI/Anthropic/Gemini): the method PATH already named the
    // operation, so the first argument is the request payload and streaming is
    // read off the path plus a `stream: true` kwarg.
    //
    // The adapter override (`interpretCall`, today only Bedrock): the AWS SDK
    // v3 routes every operation through one `client.send(command)`, so both the
    // operation and the payload live on the argument. The adapter answers "is
    // this recordable, what is its payload, does it stream" — and a NULL answer
    // (an `InvokeModelCommand`, an embeddings call) makes this a transparent
    // pass-through: the real method runs with the host's own arguments, its
    // result and any rejection reach the host exactly as they would have, and
    // nothing at all is recorded.
    //
    // Fail-open: an `interpretCall` that THROWS is treated as that same
    // pass-through. A broken adapter costs capture for the call, never the call.
    let kwargs =
      args[0] && typeof args[0] === "object"
        ? (args[0] as Record<string, unknown>)
        : {};
    let interpretedStreaming: boolean | null = null;
    if (ctx.adapter.interpretCall !== undefined) {
      let shape: CallShape | null = null;
      try {
        shape = ctx.adapter.interpretCall(args, path);
      } catch {
        shape = null;
      }
      if (shape === null) return realFn(...args);
      kwargs = shape.kwargs;
      interpretedStreaming = shape.streaming;
    }
    const start = performance.now();

    // Snapshot this call's attribution SYNCHRONOUSLY, on the calling side, before
    // the real method's promise can settle in a different async context. This is
    // what makes concurrent fan-out correct: whatever tag()/mark()/step() applied
    // in THIS branch is captured here and threaded into every deferred record —
    // never re-read after an await. Drains the one-shot pending tags.
    const { tags, step } = ctx.tracer.consumeContext();

    // Streaming if the adapter said so (Bedrock's `ConverseStreamCommand`), or
    // else: a `.stream()` helper, a named streaming method (Gemini), or the
    // caller's own `stream:true` kwarg. Any of these routes to the stream
    // proxy, whose record is DEFERRED until the stream finishes iterating.
    const streaming =
      interpretedStreaming ?? (isStreamHelper || isNamedStreamMethod || !!kwargs["stream"]);

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
          if (streaming) return wrapStreamResult(resolved, streamKwargs, ctx, start, tags, step);
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
    if (streaming) return wrapStreamResult(result, streamKwargs, ctx, start, tags, step);

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
 * Return what the host should get back from a streaming call: either the stream
 * proxy itself, or — for a provider whose streaming call resolves to an
 * ENVELOPE around the stream — that same envelope with only its stream member
 * proxied.
 *
 * Every provider but one hands back the iterator directly, so `wrapStream` is
 * the whole story. Bedrock's `ConverseStreamCommand` does not: it resolves to
 * `{ $metadata: {...}, stream: AsyncIterable }`, and proxying THAT would wrap a
 * non-iterable — the host's `response.stream` would come back unwrapped and
 * nothing would ever be recorded. An adapter declares the member by name via
 * the optional `streamEnvelopeKey` (see `BedrockAdapter`); when it is present
 * and the result really carries that key, the envelope is SHALLOW-COPIED (never
 * mutated — the host's object is not ours to alter) with the stream member
 * replaced by its proxy, so `$metadata` and every other member reach the caller
 * untouched.
 *
 * Fail-open: anything unexpected here falls back to returning the host's own
 * result UNWRAPPED. Capture is lost for that call; the host's stream is not.
 *
 * That fallback is why the two "no proxy from an envelope" conditions are
 * SEPARATE checks rather than one. No declared key means this provider hands
 * the stream back directly, so the result IS the stream and proxying it is
 * correct. A declared key the result does NOT carry means the opposite: an
 * envelope was expected and the stream is not where it should be (an
 * error-shaped response, an SDK that renamed the member), so there is nothing
 * iterable to proxy — and wrapping the envelope anyway would hand the host a
 * stream proxy where its own next line reads `response.$metadata`. Mirrors
 * Python's `_wrap_stream_result`, including that distinction, which is the
 * fail-open fix that shipped with the Python envelope support.
 */
function wrapStreamResult(
  result: unknown,
  kwargs: Record<string, unknown>,
  ctx: WrapContext,
  start: number,
  tags: [string, string][],
  step: string | null,
): unknown {
  const key = ctx.adapter.streamEnvelopeKey;
  if (key === undefined) return wrapStream(result, kwargs, ctx, start, tags, step);
  try {
    if (result == null || typeof result !== "object" || !(key in result)) return result;
    const envelope = result as Record<string, unknown>;
    return {
      ...envelope,
      [key]: wrapStream(envelope[key], kwargs, ctx, start, tags, step),
    };
  } catch (err) {
    console.warn(
      "ctxdiff: failed to wrap a streamed response; this call will not be recorded",
      err,
    );
    return result;
  }
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

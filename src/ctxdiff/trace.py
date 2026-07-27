"""The public entry point. `trace.init()` opens a .ctrace and returns a Tracer;
`tracer.wrap(client)` returns a transparent proxy that records every completion
call while behaving exactly like the original client.

Streaming (`stream=True`): the interceptor can't record at call-time the way
it does for a plain response — the real call returns an unconsumed iterator
with no usage on it yet (that only appears in later/final chunks as the
CALLER consumes them). So the streaming path defers recording: it wraps the
returned stream in `_StreamProxy`/`_AsyncStreamProxy`, which pass every chunk
through to the caller UNCHANGED and IMMEDIATELY (never buffering, dropping,
reordering, or delaying — the fail-open contract is absolute here, since
breaking iteration would be worse than not capturing usage at all) while
folding each chunk's usage into a running dict via the adapter, then records
the call ONCE the stream is done (exhausted, closed, or its context exited —
see `_StreamProxy` for the full completion contract). Blocks/params still
come from `kwargs` exactly as before; only usage now flows from the stream
instead of a completed response. `stream=True` is a reliable signal read
straight off the request kwargs — deliberately NOT duck-typed off the
response shape, which would be fragile and provider-specific.

`.stream()` convenience helpers (Anthropic's `messages.stream`, OpenAI's
`chat.completions.stream`/`responses.stream` — the `with client.messages.
stream(...) as stream:` context-manager style the providers' own docs
recommend): these are a SEPARATE completion method per adapter (its own
`create_paths` entry ending in "stream"), resolved by `_ClientProxy` exactly
like `create` — but they return a StreamManager, not a stream, so
`_make_interceptor` wraps THAT in `_StreamManagerProxy`/
`_AsyncStreamManagerProxy` instead. The manager makes no request at all until
`__enter__`/`__aenter__`; once entered, the real stream it hands back is
wrapped in the SAME `_StreamProxy`/`_AsyncStreamProxy` used for
`create(stream=True)`, so all usage-accumulation/finalize-once/fail-open
machinery is shared, not duplicated."""
from __future__ import annotations

import contextlib
import contextvars
import inspect
import itertools
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

from ctxdiff.capture.anthropic import AnthropicAdapter
from ctxdiff.capture.bedrock import BedrockAdapter
from ctxdiff.capture.gemini import GeminiAdapter
from ctxdiff.capture.openai import OpenAIAdapter
from ctxdiff.capture.recorder import Recorder, SyntheticUsageResponse
from ctxdiff.models import Block
from ctxdiff.store import config as store_config
from ctxdiff.store.base import Store, StoreBackend
from ctxdiff.store.sqlite import SQLiteStore

_log = logging.getLogger("ctxdiff")

# How long `close()` waits for the writer thread when the store publishes no
# bound of its own — the local `.ctrace`, whose own contention budget is a 5s
# busy timeout across 6 retries, so a shorter join would abandon writes that
# were about to succeed.
_DEFAULT_CLOSE_TIMEOUT = 30.0

# Seconds added to a networked store's statement timeout to get the join bound:
# the writer may be mid-statement when close() arrives (up to that timeout) and
# then still has to close the connection.
_CLOSE_TIMEOUT_MARGIN = 2.0


def _close_timeout_for(backend: object) -> float:
    """How long `close()` should wait for the writer thread, given the store it
    is writing to.

    A networked backend publishes a `statement_timeout`: the server aborts any
    statement past it, so the writer CANNOT be legitimately busy for longer, and
    waiting 30 seconds on a database that has stopped answering just makes a
    failed deployment slower to shut down. Anything without one (the local
    SQLite store, a test double) keeps the generous default — SQLite's
    lock-contention retries genuinely can take tens of seconds and are worth
    waiting for. Read by capability rather than by isinstance, the same way
    `Tracer` asks a backend for `path_for`."""
    statement_timeout = getattr(backend, "statement_timeout", None)
    if not isinstance(statement_timeout, (int, float)) or statement_timeout <= 0:
        return _DEFAULT_CLOSE_TIMEOUT
    return float(statement_timeout) + _CLOSE_TIMEOUT_MARGIN


# Provider detection maps a client's top-level module to an adapter factory.
_ADAPTERS = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "gemini": GeminiAdapter,
             "bedrock": BedrockAdapter}

# Some SDKs don't own their top-level module root — `google-genai`'s client
# lives under `google.genai...`, but the bare root `google` is shared with
# unrelated packages (google.cloud, google.protobuf, ...) and must NOT map to
# an adapter blindly. For those, detection falls back to matching a known
# DOTTED prefix of the full module path instead of just its root, so only
# `google.genai` (and its submodules) resolve to gemini while any other
# `google.*` package still falls through to the "unrecognized" error.
_DOTTED_PREFIXES = {"google.genai": "gemini"}

# boto3's `bedrock-runtime` client (and every OTHER boto3 client — s3, ec2,
# ...) lives under the shared module root/prefix "botocore.client": botocore
# generates one generically-named `BaseClient` subclass per service at
# runtime, all living in that same module, so neither the root-map nor the
# dotted-prefix map above can distinguish them — every boto3 service would
# collide on "botocore". Detection falls back a THIRD time here, keyed on the
# client's CLASS NAME instead of its module, and only consulted once the
# module root is confirmed to be "botocore" so this narrow class-name check
# never fires for unrelated same-named classes from other packages.
_BOTOCORE_CLASSES = {"BedrockRuntime": "bedrock"}

# SDK response-wrapper hops that sit BETWEEN a resource and its `.create`
# method without changing which HTTP call gets made — e.g. LangChain's
# `ChatOpenAI._generate()` calls `self.client.with_raw_response.create(...)`
# instead of `self.client.create(...)` to get at raw headers/status before
# parsing the body. `_ClientProxy` treats a name in this set as transparent
# (see `__getattr__`) when it is encountered exactly one step before the
# create path, so the call is still tracked and recorded.
_TRANSPARENT_HOPS = ("with_raw_response", "with_streaming_response")

# google-genai's `genai.Client` mirrors its ENTIRE sync surface under a `.aio`
# namespace instead of shipping a separate async client class — the same
# client exposes both `client.models.generate_content` (sync) and
# `client.aio.models.generate_content` (async). `.aio` is a transparent hop
# like `_TRANSPARENT_HOPS` above, but structurally different: it sits at the
# very FRONT of the create path (the client root) rather than one step before
# it, so it needs its own constant and its own root-only gate in
# `__getattr__` — recognized ONLY at path == () (the client root) AND only
# for the gemini provider (whose create_path this hop actually leads to);
# gating on provider keeps an unrelated `.aio` attribute on some other SDK's
# client (which would never have a `.models.generate_content` at the end of
# it) a plain, unwrapped pass-through rather than getting swept into
# transparent-hop wrapping it doesn't belong to.
_TRANSPARENT_ROOT_HOPS = ("aio",)


def _detect_provider(client: object) -> str:
    """Infer the provider from the client's module path (e.g. an OpenAI client's
    class lives under the 'openai' package). Raises if unrecognized so wrap()
    fails loudly at setup time — not silently at record time.

    Three-stage lookup: first the module's top-level root (works for openai/
    anthropic, whose packages are dedicated to one provider). When that
    misses, try matching a dotted prefix from `_DOTTED_PREFIXES` — needed for
    SDKs like google-genai whose module root ('google') is shared across
    unrelated packages, so root-only matching would be too broad. When THAT
    also misses and the root is specifically 'botocore' (boto3's generated
    clients all share this one module regardless of AWS service — s3,
    ec2, bedrock-runtime, ... — so root/prefix matching can't tell them
    apart), fall back to a class-name check against `_BOTOCORE_CLASSES`; any
    other botocore client (e.g. S3) still falls through to the error below,
    with a hint that only bedrock-runtime is supported from boto3."""
    module = type(client).__module__ or ""
    root = module.split(".", 1)[0]
    if root in _ADAPTERS:
        return root
    for prefix, provider in _DOTTED_PREFIXES.items():
        if module == prefix or module.startswith(prefix + "."):
            return provider
    if root == "botocore":
        class_name = type(client).__name__
        if class_name in _BOTOCORE_CLASSES:
            return _BOTOCORE_CLASSES[class_name]
        raise ValueError(
            f"ctxdiff: unrecognized boto3 client class '{class_name}'; "
            f"only bedrock-runtime is supported from boto3")
    raise ValueError(
        f"ctxdiff: unrecognized client module '{module}'; "
        f"supported providers: {sorted(_ADAPTERS)}")


# Hosts whose OpenAI-COMPATIBLE endpoints we can NAME. An OpenAI-SDK client
# pointed at one of these is really talking to that vendor, and recording
# `provider=openai` for Gemini traffic is confidently-wrong attribution — the
# trust-killer class of bug for a debugger (dogfood finding 2026-07-27).
_OPENAI_COMPAT_HOSTS = {
    "generativelanguage.googleapis.com": "gemini",
    "api.anthropic.com": "anthropic",
}


def _openai_compat_label(client: object) -> str:
    """Refine the ATTRIBUTION label for an OpenAI-SDK client by its base_url
    host. The ADAPTER stays 'openai' regardless — the wire shape genuinely is
    OpenAI's, and capture mechanics key off the adapter — this only changes
    the `provider` string stamped on the run/calls (the same nullable
    attribution field the LangChain handler already sets per call).

    Mapping is deliberately conservative: hosts in `_OPENAI_COMPAT_HOSTS` get
    their vendor's name; OpenAI's own hosts and Azure OpenAI keep 'openai'
    (Azure has always recorded 'openai' — relabeling it here would churn
    existing users' traces for no diagnostic gain); any OTHER host (Ollama,
    vLLM, OpenRouter, LiteLLM proxies, ...) is labeled 'openai-compatible' —
    truthful without guessing a vendor from an address we don't recognize.
    A side benefit for the named vendors: their blocks now go through the
    estimate token path instead of tiktoken counts being marked exact for
    models tiktoken does not tokenize. Fail-open: any surprise reading
    base_url returns 'openai' unchanged."""
    try:
        base = getattr(client, "base_url", None)
        if base is None:
            return "openai"
        # openai-python exposes base_url as an httpx.URL (has .host); tolerate
        # a plain string too, since we only need the hostname either way.
        host = (getattr(base, "host", "") or "").lower()
        if not host:
            host = (urlparse(str(base)).hostname or "").lower()
        if not host:
            return "openai"
        if host in _OPENAI_COMPAT_HOSTS:
            return _OPENAI_COMPAT_HOSTS[host]
        if host == "api.openai.com" or host.endswith(".openai.com"):
            return "openai"
        if host.endswith(".openai.azure.com") or host.endswith(".azure.com"):
            return "openai"
        return "openai-compatible"
    except Exception:  # noqa: BLE001 — labeling must never break wrap()
        return "openai"


class _DeferredStore:
    """A `Store` handle whose session is OPENED ON THE WRITER THREAD, on first
    use, rather than by whoever constructed it.

    Why this exists at all: `wrap()` runs on the host's own thread, in the
    middle of an agent doing real work, and opening a session is I/O — a TCP
    connect, an authentication handshake, `CREATE TABLE IF NOT EXISTS`, an
    INSERT. Doing that inline made the host pay for the tracing store's health:
    a slow database cost the agent's first LLM call up to the full connect
    timeout, and a database that completed its handshake and then stopped
    answering (a wedged box, a hung pooler, a network partition) blocked it with
    no bound at all — a client connect timeout covers connect and auth, nothing
    after, and a server-side statement timeout cannot fire when the packets
    carrying it are being dropped.

    Bounding that I/O tighter would only shrink the damage. Moving it removes
    it: the writer thread already exists, already owns the connection, and is
    already the thread whose slowness costs the host nothing. So `wrap()` now
    constructs this handle (pure bookkeeping, no I/O) and the writer opens the
    real store as its FIRST act — concurrently with the host's first call, with
    any calls made meanwhile waiting in the queue.

    Failure keeps the existing fail-open shape: the open is attempted exactly
    once, a failure warns exactly once through `on_failure`, and every later
    method raises `_StoreUnavailable` so the writer drops jobs instead of
    retrying a store that is not there."""

    def __init__(self, open_session: Callable[[], Store],
                 on_failure: Callable[[], None] | None = None):
        """Record HOW to open the session (a zero-arg callable closing over the
        project/provider/started_at decided at `wrap()` time, so the session
        still carries the moment the host started tracing — not the moment the
        writer got around to connecting) and who to tell if it fails. Nothing
        is opened here; `open()` does that, on the writer thread."""
        self._open_session = open_session
        self._on_failure = on_failure
        self._store: Store | None = None
        self._opened = False
        self._lock = threading.Lock()

    def open(self) -> Store | None:
        """Open the session, ONCE, returning the real store or None if it
        failed. Called by the writer thread before it processes any job; the
        lock and `_opened` flag make a second call (a re-entrant close, a test
        driving it directly) a no-op rather than a second session. A failure is
        swallowed and reported through `on_failure` — this runs on the writer
        thread, where raising would kill the loop that is the host's only
        protection from store errors."""
        with self._lock:
            if self._opened:
                return self._store
            self._opened = True
            try:
                self._store = self._open_session()
            except Exception:  # noqa: BLE001 — degrade capture, never the host
                self._store = None
                if self._on_failure is not None:
                    self._on_failure()
            return self._store

    def _require(self) -> Store:
        """The opened store, or raise. Opens on demand so a caller that is not
        the writer loop (`Recorder.persist`, a test) still gets a working
        handle, and raises `_StoreUnavailable` when the open failed so the
        caller's own fail-open guard treats it as the write failure it is."""
        store = self.open()
        if store is None:
            raise _StoreUnavailable(
                "ctxdiff: the store for this run could not be opened")
        return store

    def record_call(self, seq: int, params: dict, usage: dict | None,
                    latency_ms: int | None, error: str | None,
                    call_blocks: list, agent: str | None = None,
                    step: str | None = None, provider: str | None = None) -> str:
        """Persist one call through the real store (see `Store.record_call`)."""
        return self._require().record_call(
            seq=seq, params=params, usage=usage, latency_ms=latency_ms,
            error=error, call_blocks=call_blocks, agent=agent, step=step,
            provider=provider)

    def note_model(self, model: str | None) -> None:
        """Roll a model id up onto the session (see `Store.note_model`)."""
        self._require().note_model(model)

    def list_sessions(self) -> list:
        """Every session in the underlying store."""
        return self._require().list_sessions()

    def get_run(self, session_id: str | None = None):
        """One session's run row."""
        return self._require().get_run(session_id)

    def get_calls(self, session_id: str | None = None) -> list:
        """One session's calls, in turn order."""
        return self._require().get_calls(session_id)

    def get_call_blocks(self, call_id: str) -> list:
        """One call's blocks, in position order."""
        return self._require().get_call_blocks(call_id)

    def close(self) -> None:
        """Close the underlying store if one was ever opened, and never raise —
        this runs on the writer thread's way out. A store that was never opened
        (no call was ever recorded, or the open failed) has nothing to close, so
        this is also what stops a degraded run from connecting at shutdown just
        to disconnect again."""
        store = self._store
        self._store = None
        self._opened = True          # never open a session while shutting down
        if store is None:
            return
        try:
            store.close()
        except Exception:  # noqa: BLE001 — close is best-effort on the way out
            pass


class _StoreUnavailable(RuntimeError):
    """Raised by `_DeferredStore` when the session could not be opened. Its own
    type so the writer loop can tell "this run has no store at all" (drop the
    job silently — the one-time warning already fired when the open failed)
    apart from "this particular write failed" (warn once, keep going)."""


class _Writer:
    """The run's single dedicated writer thread, sitting behind a bounded queue.

    Why it exists: SQLite connections are thread-affine and disk writes should
    not sit on the host's call path. So every persist for a run is funnelled
    onto ONE thread that owns the connection: `submit()` (called from any host
    thread/asyncio task) enqueues a zero-arg job and returns immediately; the
    thread drains the queue FIFO and runs each job. Because exactly one thread
    ever writes, there is never concurrent connection access — and, since this
    thread also OPENS the store (see `_DeferredStore`), the connection is never
    even created anywhere else.

    This matters MORE, not less, for a networked store (Postgres/MySQL): a
    write is now a round-trip that can be slow or fail, and a DB-API connection
    tolerates only one statement at a time. Off-loading every write to this one
    thread means a slow database costs the host nothing (the queue absorbs it)
    and a dead one costs it nothing either (the job fails on this thread, is
    warned about once, and is dropped) — the host call is never blocked,
    delayed or broken either way. The bounds that keep that promise honest are
    the queue's `maxsize`, the adapters' connect/statement timeouts, and
    `close()`'s join timeout.

    Ordering: `seq` is assigned by the caller BEFORE `submit()` (see
    `Tracer._on_create`), so it reflects call-COMPLETION order; the queue is
    FIFO and the store reads back `ORDER BY seq`, so the persisted timeline is
    stable regardless of how writer scheduling interleaves.

    Fail-open (the whole point): `submit()` never blocks the host meaningfully
    and never raises. On a full queue (backpressure) or a writer that is no
    longer alive, the record is DROPPED and a ONE-TIME degradation warning is
    emitted — capture degrades silently rather than ever slowing or breaking
    the host. Each job runs inside its own guard on the writer thread, so a
    single bad job can never kill the loop; the loop only ends on the close
    sentinel.

    Shutdown: `close()` enqueues a sentinel AFTER every already-queued job;
    FIFO ordering means the writer persists them all before it sees the
    sentinel, giving a true flush with no lost writes. It then closes the
    connection (on its own thread, honouring affinity) and exits, and `close()`
    joins it — for at most `close_timeout`, so a wedged store bounds shutdown
    instead of hanging the program that is trying to exit."""

    _SENTINEL = object()

    def __init__(self, ctrace: Store, maxsize: int = 10000,
                 close_timeout: float = _DEFAULT_CLOSE_TIMEOUT):
        """Start the writer thread that will own `ctrace`'s connection. How:
        creates the bounded FIFO queue (maxsize caps memory / defines the
        backpressure point), a lock guarding the one-time-warning + closed
        flags, and a daemon thread running `_run` (daemon so a host that exits
        without calling `tracer.close()` is never blocked by it). `maxsize`
        (default 10k) is generous enough that a healthy writer never hits it;
        reaching it means the writer is falling behind, which is exactly the
        degradation the drop-and-warn path is for. `close_timeout` is how long
        `close()` will wait for this thread — see `_close_timeout_for`, which
        derives it from the store's own bounds."""
        self._ct = ctrace
        self._close_timeout = close_timeout
        # Set by the thread itself before it takes its first job: False means
        # the store never opened, so jobs are dropped (the warning already fired
        # at the open). Written and read only on the writer thread.
        self._store_ready = True
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._closed = False
        self._warned = False
        self._persist_warned = False   # one-time flag for writer-side job failures
        self._thread = threading.Thread(
            target=self._run, name="ctxdiff-writer", daemon=True)
        self._thread.start()

    def submit(self, job: Callable[[], None], quiet: bool = False) -> None:
        """Enqueue one persist job from a host thread/task and return at once.
        What: hands `job` (a zero-arg callable that persists one call) to the
        writer thread. How: a non-blocking `put_nowait` so the host is never
        stalled; if the writer is already stopped or dead, or the queue is full
        (backpressure), the job is dropped and `_degrade` fires the one-time
        warning. The whole body is wrapped so nothing — not even an unexpected
        error building the enqueue — can ever propagate into the host call.
        `quiet` suppresses the degradation warning for GC/shutdown-time callers
        (a stream `__del__` finalize), where warning is noise at best."""
        try:
            if self._closed or not self._thread.is_alive():
                self._degrade("writer thread not running", quiet)
                return
            self._queue.put_nowait(job)
        except queue.Full:
            self._degrade("write queue overflow", quiet)
        except Exception:  # noqa: BLE001 — enqueue must never break the host
            self._degrade("enqueue failed", quiet)

    def _degrade(self, reason: str, quiet: bool) -> None:
        """Emit the capture-degradation warning AT MOST ONCE for the run, then
        stay silent. What: signals genuine capture loss (writer dead / queue
        overflow) — deliberately NOT fired for normal concurrency, which just
        enqueues successfully. How: a lock-guarded `_warned` flag makes the
        first genuine degradation log and every subsequent one a no-op, so a
        storm of dropped calls can't spam the host's logs. `quiet` callers
        (GC/shutdown) skip logging entirely."""
        if quiet:
            return
        with self._lock:
            if self._warned:
                return
            self._warned = True
        _log.warning("ctxdiff: capture degraded (%s); some calls in this run "
                     "will not be recorded", reason)

    def _run(self) -> None:
        """The writer thread's loop: OPEN the store, then drain the queue FIFO,
        persisting each job, until the close sentinel.

        The open comes first and happens HERE — every byte of store I/O for the
        run, from the TCP connect onwards, belongs to this thread and never to
        the host's (see `_DeferredStore`). It is attempted before the first
        `queue.get()` so a session exists even for a run that records nothing,
        and so calls made while it is still connecting simply queue up. When it
        fails, `_open_store` has already fired the one-time degradation warning
        and every job is then dropped — retrying each write against a store that
        was never there would only produce a second class of warning for the
        same fact.

        Then: blocks on `queue.get()`, and for a real job runs it inside
        `_run_job` (a guarded runner) so a single failing persist is warned-once
        and skipped rather than killing the loop (which would silently end all
        further capture). On the sentinel it first DRAINS any jobs still queued
        BEHIND it (a `submit()` that passed the `_closed` check before `close()`
        set it can land its `put_nowait` after the sentinel — see
        `_drain_stragglers`), then closes the connection on this same owning
        thread, honouring SQLite thread-affinity; that close is guarded so a
        failure still lets the thread exit cleanly."""
        self._store_ready = self._open_store()
        while True:
            job = self._queue.get()
            if job is self._SENTINEL:
                self._drain_stragglers()
                break
            self._run_job(job)
        try:
            self._ct.close()
        except Exception:  # noqa: BLE001 — close is best-effort on the way out
            pass

    def _open_store(self) -> bool:
        """Open the run's store on this thread, reporting whether capture is
        live. Only a `_DeferredStore` has anything to open (it warns once itself
        on failure); an already-open `Store` handed straight to this writer is
        taken as ready. Matched by TYPE rather than by looking for an `open`
        attribute, because `CTrace.open` is a classmethod that means something
        entirely different — duck-typing here would call it with no path and
        conclude the store was dead."""
        if not isinstance(self._ct, _DeferredStore):
            return True
        try:
            return self._ct.open() is not None
        except Exception:  # noqa: BLE001 — the writer thread must not die here
            return False

    def _run_job(self, job: Callable[[], None]) -> None:
        """Run one persist job inside a guard so a single failure can never kill
        the writer loop. A job is skipped outright when the store never opened —
        the degradation was already warned about once, at the open — so a
        dead-database run produces exactly one warning rather than a second one
        about the first write it could never have made. Any other failure is
        warned AT MOST ONCE for the run (mirrors `_degrade`) via
        `_persist_warned`: a store failing every write would otherwise log one
        line per job. Never host-facing — writer-only."""
        if not self._store_ready:
            return
        try:
            job()
        except Exception:  # noqa: BLE001 — one bad job must not kill the writer
            with self._lock:
                if self._persist_warned:
                    return
                self._persist_warned = True
            _log.warning("ctxdiff: writer failed to persist a call; further "
                         "writer failures in this run will be silent (skipped)",
                         exc_info=True)

    def _drain_stragglers(self) -> None:
        """After the sentinel is seen, process any jobs still sitting in the
        queue BEHIND it before the thread exits. Why: `submit()` reads
        `_closed==False`, then `close()` sets `_closed` and enqueues the
        sentinel, then that racing `submit()` finally `put_nowait`s its job —
        which now sits AFTER the sentinel. Without this drain that straggler
        would be silently abandoned (a lost write, no warning). How: non-blocking
        `get_nowait` until the queue is empty, running each real job through the
        same guarded `_run_job`; extra sentinels (from an idempotent double
        close) are skipped. Warns once if any straggler was found so the rare
        close-race is observable rather than silent."""
        stragglers = 0
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is self._SENTINEL:
                continue
            stragglers += 1
            self._run_job(job)
        if stragglers:
            _log.warning("ctxdiff: drained %d write(s) enqueued during close "
                         "(recorded, not lost)", stragglers)

    def close(self, timeout: float | None = None) -> None:
        """Flush every enqueued write, stop the thread, and close the store —
        blocking until done, with no lost writes. How: sets `_closed` (so any
        racing `submit()` now drops instead of enqueuing past the sentinel),
        then enqueues the sentinel with a BLOCKING put (close may block; the
        host call path may not) so it lands AFTER all already-queued jobs —
        FIFO then guarantees the writer persists them all before exiting.
        Joins the thread (which closes the connection as its last act).

        The join timeout defaults to this writer's `close_timeout`, which is
        derived from the STORE's own bounds (`_close_timeout_for`) rather than
        being a flat 30 seconds. It is a safety valve against a wedged writer:
        no store may hold the thread longer than its statement bound, so a
        longer join buys nothing and costs a host — one whose database has
        already failed it — half a minute of not being able to exit. Exceeding
        it warns rather than hanging the caller forever. Idempotent — a second
        close is a no-op."""
        if timeout is None:
            timeout = self._close_timeout
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put(self._SENTINEL)  # blocking: ensure the flush is enqueued
        except Exception:  # noqa: BLE001 — never raise out of close
            pass
        self._thread.join(timeout)
        if self._thread.is_alive():
            _log.warning("ctxdiff: writer did not drain within %ss on close; "
                         "some writes may be lost", timeout)


def init(project: str, redact: Callable[[Block], Block] | None = None,
         path: str | None = None, store: StoreBackend | None = None) -> "Tracer":
    """Create a Tracer that opens the project's store and starts a NEW SESSION
    in it. `project` names the project; `redact` is an optional per-block
    scrubber applied before storage.

    Project-scoped storage (v0.6): `path` defaults to a STABLE
    `./<project>.ctrace` in the cwd — NOT a per-run `<project>-<uuid>.ctrace`.
    The first `wrap()` opens that file if it already exists and APPENDS a new
    session (a fresh `run` row) to it, or creates it if absent; so every
    `trace.init(project)` accumulates one more session in the same project DB
    rather than scattering a file per run. An explicit `path=` works the same
    way — it appends when the file already exists.

    Pluggable storage (v0.7): `store` overrides WHERE that session lands with
    any `StoreBackend` — `SQLiteStore`, `PostgresStore(dsn=...)`,
    `MySQLStore(dsn=...)`. Usually you don't pass it: `ctxdiff.configure(store=
    ...)` once at startup, or the `CTXDIFF_STORE` env var, applies to every
    `init()` from then on. Resolution is explicit-beats-ambient — this argument,
    then `path=` (which is unambiguously a local file), then `configure()`,
    then `CTXDIFF_STORE`, and when NOTHING is configured the unchanged
    zero-config default: a local `./<project>.ctrace`."""
    return Tracer(project=project, redact=redact, path=path, store=store)


def _resolve_backend(path: str | None,
                     store: StoreBackend | None) -> StoreBackend:
    """Decide which backend a Tracer writes to, explicit-beats-ambient:

    1. an explicit `store=` argument — the caller named a backend outright;
    2. an explicit `path=` — a filesystem path is unambiguously a local
       `.ctrace`, so it beats an ambient `configure()`/env-var setting rather
       than being silently ignored (a caller who passes a path and gets a row
       in someone's Postgres would rightly call that a bug);
    3. `configure()`, then `CTXDIFF_STORE` (both via `store.config.resolve`);
    4. nothing configured -> `SQLiteStore()`, i.e. `./<project>.ctrace` — the
       unchanged zero-config default that every existing user keeps getting.

    Returns a backend, never None; it may raise (e.g. an unparseable
    `CTXDIFF_STORE`), which is why `Tracer.__init__` calls it inside a guard."""
    if store is not None:
        return store
    if path is not None:
        return SQLiteStore(path=path)
    return store_config.resolve() or SQLiteStore()


class _UnavailableBackend:
    """Stand-in for a backend that could not even be RESOLVED — e.g. a typo'd
    `CTXDIFF_STORE=postgres//host/db`, or a `configure()`d backend whose module
    failed to import. Holds the original error and re-raises it from
    `open_session()`, so the failure surfaces at exactly the point `wrap()`
    already guards against a dead store: capture degrades fail-open with one
    warning carrying the real cause, and the host runs untouched.

    Why not just raise from `init()`: a misconfigured trace destination is
    still a tracing problem, and tracing problems must never take down the
    program being traced. Why not silently fall back to a local file: a user who
    asked for Postgres and got a surprise `.ctrace` in their container's
    working directory has been lied to."""

    def __init__(self, error: Exception):
        """Keep the resolution error to re-raise later, unchanged."""
        self._error = error

    def open_session(self, *args, **kwargs):
        """Re-raise the resolution failure into `wrap()`'s fail-open guard."""
        raise self._error

    def open_reader(self):
        """Re-raise the resolution failure for read-side callers (the CLI),
        which report it rather than degrade."""
        raise self._error


class Tracer:
    """Owns the run's store handle and hands out recording proxies. The store's
    run row is created lazily on the first wrap(), when the provider becomes
    known. Which store that is — a local `.ctrace`, Postgres, MySQL — is decided
    once here and never again: everything below this line talks to the `Store`
    protocol."""

    def __init__(self, project: str, redact: Callable[[Block], Block] | None,
                 path: str | None = None, store: StoreBackend | None = None):
        """Store the run's static config and initialize empty-run state. How:
        the store handle/recorder/writer are NOT created here — they need the
        provider, which is only known once `wrap()` is called — so `_ct`/
        `_recorder`/`_writer` start as None. What IS decided here is the
        BACKEND (see `_resolve_backend`), because that is a pure, connection-
        less decision; resolving it is wrapped so a bad `CTXDIFF_STORE` becomes
        a deferred fail-open degradation rather than an exception out of
        `trace.init()`.

        `self.path` stays part of the public surface but is now backend-derived:
        the concrete `.ctrace` file for a SQLite backend (unchanged for every
        existing user), and None for a networked one, where "the path" is
        meaningless.

        Concurrency state (the core of the v0.5 model):
        - `_seq` is an `itertools.count`, not a plain int: `next(self._seq)` is
          atomic under CPython's GIL, so many threads/tasks completing calls in
          parallel each get a UNIQUE, monotonic turn index without a lock on the
          hot path.
        - pending tags and the sticky step live in `contextvars.ContextVar`s,
          NOT instance attributes. This is what makes attribution correct under
          concurrency: asyncio copies the context per Task (so `tag()`/`mark()`
          inside one gathered coroutine never leak into a sibling), and each OS
          thread has its own context (so a `ThreadPoolExecutor` fan-out is
          isolated too). The vars are per-Tracer instances (a Tracer is a
          once-per-run object, not created in a hot loop, so this is safe and
          gives clean per-run isolation) and are reset on the CONSTRUCTING
          context here so a fresh run never inherits a leftover step/tag from a
          previous Tracer on this same thread."""
        try:
            self._backend: StoreBackend = _resolve_backend(path, store)
        except Exception as exc:  # noqa: BLE001 — a bad DSN must not break init()
            self._backend = _UnavailableBackend(exc)
        # `path` is a SQLite-only concept, so it is asked for by capability
        # (`path_for`) rather than assumed: a networked backend simply has none.
        path_for = getattr(self._backend, "path_for", None)
        self.path: str | None = path_for(project) if path_for is not None else None
        self._project = project
        self._redact = redact
        self._ct: Store | None = None
        self._recorder: Recorder | None = None  # the FIRST wrap's recorder (kept
        # for backward compat: tests monkeypatch t._recorder.build to prove the
        # interceptor wiring is fail-open even when recording is broken)
        self._writer: _Writer | None = None     # single writer thread (lazy, per wrap)
        # Recorders built per PROVIDER for capture paths that only learn the
        # provider per call (the LangChain handler) — see `_recorder_for`.
        self._recorders: dict[str, Recorder] = {}
        self._recorders_lock = threading.Lock()
        # Guards the lazy store/writer creation in `_ensure_store`, so several
        # threads wrapping this tracer at once produce ONE session and ONE
        # writer rather than one of each per thread.
        self._setup_lock = threading.Lock()
        # One-time guard for the store-setup fail-open path: if the project
        # store can't be created/opened (a persistent lock under heavy
        # concurrent session creation, an unreachable database, a bad DSN) we
        # degrade fail-open and warn at most once for the run rather than
        # raising into the host. Its OWN lock, not `_setup_lock`: the warning is
        # now raised on the writer thread, which must not queue behind a host
        # thread that is still setting up.
        self._setup_warn_lock = threading.Lock()
        self._setup_warned = False
        self._seq = itertools.count(1)          # thread-safe monotonic turn index
        # Per-execution-context capture state (see docstring). Defaults: no
        # pending tags (empty tuple) and no sticky step (None).
        self._pending_tags: contextvars.ContextVar[tuple[tuple[str, str], ...]] = \
            contextvars.ContextVar("ctxdiff_pending_tags", default=())
        self._step: contextvars.ContextVar[str | None] = \
            contextvars.ContextVar("ctxdiff_step", default=None)
        # Clear any value lingering on THIS context from a prior Tracer so runs
        # don't bleed into each other on a reused (e.g. main) thread.
        self._pending_tags.set(())
        self._step.set(None)

    def wrap(self, client: object, agent: str | None = None) -> object:
        """Return a transparent proxy over `client` that records every call to
        the provider's completion method. Detects the provider, creates the
        run/store on first use, and wires a Recorder.

        `agent` names the agent this client belongs to; it is stamped onto
        every call this proxy records, so one run can attribute calls across
        several agents. Each wrap builds ITS OWN adapter and Recorder and the
        returned proxy carries them — so wrapping two clients of DIFFERENT
        providers in the same run records each through the correct adapter
        (the pre-v2 code built a single recorder from the first provider's
        adapter and mis-parsed a second provider's calls). The store/run is
        still created lazily exactly once, on the first wrap, and keeps that
        first-seen provider on `run.provider` for backward compatibility."""
        provider = _detect_provider(client)
        adapter = _ADAPTERS[provider]()
        # ATTRIBUTION label vs ADAPTER: an OpenAI-SDK client may really be
        # talking to Gemini/Anthropic/an OSS server through their OpenAI-
        # compatible endpoints. The adapter (wire mechanics) stays keyed to
        # `provider`; the label recorded on the run and stamped per call is
        # refined from the client's base_url host so the trace names the
        # vendor actually being called. Non-openai providers pass through
        # unchanged (their SDK IS the vendor).
        label = _openai_compat_label(client) if provider == "openai" else provider
        self._ensure_store(label)
        recorder = Recorder(self._ct, adapter, self._redact)
        if self._recorder is None:
            # Keep the first recorder reachable as t._recorder (see __init__).
            self._recorder = recorder
        # Resolve once, here at wrap-time: adapters with more than one
        # completion method (e.g. OpenAI's chat.completions.create AND
        # responses.create) expose plural `create_paths`; single-path
        # adapters don't define it at all, so fall back to the singular
        # `create_path` wrapped in a 1-tuple. `_ClientProxy` only ever deals
        # in the plural form from here on.
        paths = getattr(adapter, "create_paths", None) or (adapter.create_path,)
        return _ClientProxy(client, (), self, paths,
                            recorder, agent, label, adapter)

    def langchain_handler(self, agent: str | None = None):
        """Return a LangChain callback handler that records every chat-model
        call made under it — the IDIOMATIC way to trace LangChain and
        LangGraph:

            handler = tracer.langchain_handler()
            llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
            # or per-invocation, which is what LangGraph propagates:
            graph.invoke(state, config={"callbacks": [handler]})

        WHY A HANDLER RATHER THAN `wrap()`. `tracer.wrap()` needs a provider
        SDK client; a LangChain app hands you a `ChatOpenAI`, not an
        `OpenAI`. The older answer was to inject a wrapped client into
        LangChain's internals (`ChatOpenAI(client=wrapped.chat.completions,
        root_client=wrapped)`) — which still works and is still tested, but
        depends on LangChain's private structure and covers only the
        providers whose SDK object you can reach. A callback is LangChain's
        own extension point: it fires for EVERY integration (ChatOpenAI,
        ChatAnthropic, ChatVertexAI, ChatBedrockConverse, ...), streaming or
        not, and LangGraph propagates it through an entire graph, so one
        handler covers a whole agent.

        The blocks it records are IDENTICAL — same hashes — to what wrapping
        that provider's SDK directly would have recorded for the same
        request, because the handler rebuilds the provider's own wire shape
        and feeds it to the very same adapter (see
        `ctxdiff.capture.langchain`). So a LangChain trace and a direct trace
        of the same prompt dedup against each other instead of looking like
        two unrelated contexts.

        `agent` names the agent these calls belong to, exactly as
        `wrap(client, agent=...)` does — pass a different handler per agent
        to attribute a multi-agent graph. Raises ImportError (with the
        install line) if langchain-core isn't installed, since asking for a
        LangChain handler without LangChain is a setup mistake worth failing
        loudly on; everything AFTER construction is fail-open, like the rest
        of capture."""
        from ctxdiff.capture.langchain import build_handler
        return build_handler(self, agent)

    def _recorder_for(self, provider: str) -> Recorder | None:
        """The Recorder for `provider`, created once per provider and cached
        — the entry point used by capture paths that discover their provider
        per CALL rather than per client (today: the LangChain handler, which
        sees provider-agnostic messages and may serve ChatOpenAI and
        ChatAnthropic from the same handler).

        `wrap()` can build its recorder eagerly because a client has exactly
        one provider; a handler cannot, so this does the same three steps
        lazily: ensure the run's store/writer exist (idempotent — the first
        provider to arrive still decides the session's `run.provider`, same
        as the first `wrap()` does), build the provider's adapter, and wrap
        both in a Recorder. Its own lock, not `_setup_lock`, because
        `_ensure_store` takes that one.

        Returns None for an unknown provider name rather than raising: this
        is called from inside a callback on the host's own execution path,
        where fail-open outranks fail-loud."""
        adapter_cls = _ADAPTERS.get(provider)
        if adapter_cls is None:
            return None
        with self._recorders_lock:
            recorder = self._recorders.get(provider)
            if recorder is not None:
                return recorder
            self._ensure_store(provider)
            recorder = Recorder(self._ct, adapter_cls(), self._redact)
            self._recorders[provider] = recorder
            if self._recorder is None:
                # Keep the first recorder reachable as t._recorder (see
                # __init__), the same way `wrap()` does.
                self._recorder = recorder
            return recorder

    def _ensure_store(self, provider: str) -> None:
        """Create this run's store handle and writer thread, exactly ONCE,
        however many threads call `wrap()` at the same moment.

        The whole body is under `_setup_lock` because the check and the create
        must be one step. Unguarded — `if self._ct is None:` followed by the
        create — a tracer wrapped concurrently by several threads (one tracer at
        module scope, worker threads each wrapping their own client: the normal
        agent-framework shape) had every thread find it None and every thread
        make its own: N sessions for one logical run, N writer threads, N
        connections, and a `close()` that shut down only the last of them and
        leaked the rest.

        What is created is deliberately NOT a live store: `_DeferredStore` holds
        only the recipe, and the writer thread opens the real session. So this
        method does no I/O and cannot fail — the fail-open guard that used to
        wrap it now lives where the connecting happens (see `_DeferredStore.
        open` and `_Writer._run`), which is what keeps `wrap()` off the network
        entirely.

        `model` is left empty because a run's model is a per-CALL fact `wrap()`
        does not know yet — seeding a placeholder would store a permanent blank,
        and `note_model()` backfills the real ones. `started_at` is stamped HERE,
        on the host thread, so a session records when tracing began rather than
        whenever the writer thread finished connecting."""
        with self._setup_lock:
            if self._ct is not None:
                return
            # Canonical UTC-with-offset (`...+00:00`) so downstream local-time
            # rendering is always unambiguous — see store.parse_started_at.
            started = datetime.now(timezone.utc).isoformat()
            project = self._project
            backend = self._backend

            def _open() -> Store:
                """Open this run's session — run on the writer thread. Appends a
                new run row to the configured store, creating the `.ctrace` file
                or the database tables if they aren't there yet."""
                return backend.open_session(project=project, provider=provider,
                                            model="", started_at=started)

            self._ct = _DeferredStore(_open, on_failure=self._warn_setup_degraded)
            # The single writer thread that owns the connection and performs
            # every persist for the run — created exactly once, with the store
            # handle, on the first wrap (see `_Writer`).
            self._writer = _Writer(self._ct,
                                   close_timeout=_close_timeout_for(backend))

    def _warn_setup_degraded(self) -> None:
        """Emit the capture-degradation warning AT MOST ONCE for the run when
        opening the store fails and capture falls back to fail-open (record
        nothing). Called from the WRITER thread, where the open now happens.
        Mirrors `_Writer._degrade`'s one-time semantics with a lock-guarded flag
        so repeated failures on a broken store can't spam the host's logs.
        `exc_info=True` captures the underlying setup error (e.g. the stuck-lock
        OperationalError) for diagnosis without ever re-raising — it is called
        from inside the `except` that caught it."""
        with self._setup_warn_lock:
            if self._setup_warned:
                return
            self._setup_warned = True
        _log.warning("ctxdiff: capture degraded (store setup failed); this run "
                     "will not be recorded", exc_info=True)

    def tag(self, label: str, items: list) -> None:
        """Buffer semantic tags for the NEXT recorded call only, in the CURRENT
        execution context. Each item is reduced to its text (str as-is, else a
        'text'/'content' field) and paired with `label`; the recorder marks any
        block containing that text as `label`. Contrast mark(): tag() is
        next-call-only (consumed and cleared after one call), whereas mark() is
        sticky across many calls.

        How (concurrency): the pending tags live in a ContextVar, so a `tag()`
        call inside one asyncio Task or one thread is visible ONLY to that
        task/thread's next recorded call — never a sibling's. Because a
        ContextVar's default is shared, this copies-on-write: it reads the
        current tuple, appends to a fresh copy, and `set()`s that back, so no
        mutation ever escapes into another context."""
        pending = list(self._pending_tags.get())
        for item in items:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if text:
                pending.append((label, text))
        self._pending_tags.set(tuple(pending))

    def mark(self, step: str | None) -> None:
        """Set the sticky step label stamped onto every subsequent recorded call
        IN THE CURRENT EXECUTION CONTEXT until changed; `mark(None)` clears it.
        Sticky (persists until the next mark()), unlike tag() which is
        next-call-only. Use it to label phases of a run (e.g. 'plan', 'retrieve',
        'answer') so per-step views can slice the timeline.

        SEMANTICS (v0.5): the step is stored in a ContextVar rather than global
        tracer state, so "sticky" means sticky WITHIN THE CURRENT EXECUTION
        CONTEXT, not across ALL agents globally. In sequential code this is
        IDENTICAL to the old behavior. Under `asyncio.gather` or
        `asyncio.to_thread` it is also correct: each COPIES the context per task,
        so one task's mark() never relabels a sibling's calls.

        CAVEAT — raw thread pools. A ContextVar aliases the OS thread, and a raw
        `ThreadPoolExecutor` REUSES its worker threads WITHOUT resetting their
        context between tasks. So a step you mark() lingers on that worker: a
        LATER logical task that runs on the same worker and does NOT call mark()
        inherits the previous task's step. mark() is therefore only self-correct
        for a task that sets it every time. Under a raw pool, either call mark()
        at the start of EVERY task, or — better — use the scoped `step()` context
        manager below, which resets on exit and so cannot leak across tasks even
        on a reused worker (and remains correct under asyncio)."""
        self._step.set(step)

    @contextlib.contextmanager
    def step(self, label: str | None):
        """Scoped, concurrency-safe phase label — the RECOMMENDED way to label
        phases under concurrency. Use as `with tracer.step("retrieve"): ...`;
        every call recorded inside the block carries `step=label`, and on exit
        the previous step is restored.

        Why prefer this over `mark()`: it `set()`s the step ContextVar on enter,
        saving the returned token, and `reset()`s via that token on exit — so the
        label is cleared BEFORE the worker thread can be handed the next task.
        That makes it leak-proof even under a raw `ThreadPoolExecutor` that
        reuses workers (unlike sticky `mark()`, whose value lingers on the
        worker — see mark()'s CAVEAT), while remaining correct under asyncio
        (each task has its own context, so the set/reset is task-local). A task
        that does NOT open a `step()` block therefore records `step=None`, never
        a sibling task's leftover label. Fully reentrant: nested `step()` blocks
        restore the exact enclosing label, since each holds its own token."""
        token = self._step.set(label)
        try:
            yield
        finally:
            # Reset to the value the ContextVar held before this block, clearing
            # the label off this context (and, crucially, off a pooled worker)
            # before it can be reused by another logical task.
            self._step.reset(token)

    def _on_create(self, kwargs: dict, response: object | None,
                   latency_ms: int | None, error: str | None,
                   recorder: Recorder | None, agent: str | None,
                   provider: str | None, quiet: bool = False) -> None:
        """Interceptor callback, run on the HOST's thread/task at call
        completion. What: assigns this call's seq, reads THIS context's pending
        tags + sticky step, SNAPSHOTS the call into a persist job via the
        recorder's `build()` (on this thread, so the host's kwargs are captured
        before it can mutate them), then hands the job to the writer thread and
        returns immediately — the actual disk write happens off the call path.

        How the concurrency guarantees hold here:
        - `seq = next(self._seq)`: unique + monotonic across all threads/tasks,
          assigned in the calling context at enqueue time so ordering reflects
          call-completion order (the writer persists FIFO; reads are ORDER BY
          seq). `seq` stays a single counter across ALL agents — one global
          timeline, filtered per-agent by views.
        - tags/step come from the ContextVar (this task's/thread's own), and the
          tags are reset to empty IN THIS CONTEXT so they apply to exactly one
          call — with zero effect on any sibling context.

        Fail-open: never raises. `build()` and the writer's `submit()` are each
        internally guarded, and the whole body is ALSO wrapped so even an
        unexpected error (e.g. a monkeypatched `build` that throws) degrades
        capture silently instead of touching the host's own result/exception.
        `quiet` (set only by a stream proxy's best-effort `__del__` finalize)
        suppresses this method's own warning and is propagated into build/submit
        for the same GC/shutdown-time reasons described there."""
        seq = next(self._seq)
        try:
            # Read + reset THIS context's tags (one-call-only); read its step.
            tags = self._pending_tags.get()
            if tags:
                self._pending_tags.set(())
            step = self._step.get()
            if recorder is None or self._writer is None:
                return
            job = recorder.build(seq=seq, kwargs=kwargs, response=response,
                                 latency_ms=latency_ms, error=error,
                                 tagged=list(tags), agent=agent, step=step,
                                 provider=provider, quiet=quiet)
            if job is None:
                return
            # Bind THIS call's recorder + job into the writer-thread thunk, so
            # multi-provider runs persist through the correct adapter's recorder.
            self._writer.submit(lambda: recorder.persist(job, quiet=quiet), quiet=quiet)
        except Exception:  # noqa: BLE001 — fail-open guards the wiring, not just build()
            if not quiet:
                _log.warning("ctxdiff: failed to enqueue call seq=%s (tracing skipped)",
                             seq, exc_info=True)

    def close(self) -> None:
        """Flush and shut the run down cleanly. What: blocks until the writer
        thread has persisted every enqueued call, then stops it and closes the
        connection — no lost writes. How: delegates the flush/stop/close to the
        writer (which owns the connection and closes it on its own thread,
        honouring SQLite affinity); if no wrap ever happened there is no writer
        or store to close. Idempotent: clears `_writer`/`_ct` so a second call
        is a no-op."""
        if self._writer is not None:
            self._writer.close()   # flush queue, stop thread, close the connection
            self._writer = None
            self._ct = None
        elif self._ct is not None:
            self._ct.close()
            self._ct = None


class _ClientProxy:
    """A transparent proxy that forwards attribute access to the wrapped target,
    following the adapter's create path(s) down to the completion method(s) —
    which it replaces with an interceptor. Anything off those paths is returned
    as-is, so the wrapped client is behaviorally identical to the original.

    Multi-path: `create_paths` is a TUPLE of paths (usually length 1). An
    adapter like OpenAI's, which exposes two independent completion methods
    sharing one request/response shape (`chat.completions.create` and
    `responses.create`), resolves to a 2-tuple at wrap-time (see
    `Tracer.wrap`); every path-matching check below tests against ANY member
    of that tuple rather than a single path, so both methods get intercepted
    off the SAME proxy tree with no per-provider special-casing here."""

    def __init__(self, target: object, path: tuple[str, ...],
                 tracer: "Tracer", create_paths: tuple[tuple[str, ...], ...],
                 recorder: "Recorder", agent: str | None, provider: str | None,
                 adapter: object):
        """Store this proxy's bookkeeping: the wrapped object, how far along
        the path to the completion method this proxy sits, the owning tracer,
        the tuple of target paths, and the per-wrap recording context — the
        Recorder (built from THIS client's provider adapter), the agent name,
        the provider, and the adapter itself — which travel with the proxy so
        the interceptor records through the right adapter and attributes each
        call correctly. The adapter is threaded through SEPARATELY from the
        Recorder (which already holds its own reference internally) because
        the streaming path (see `_make_interceptor`) needs to call the
        adapter's `accumulate_stream_usage` directly, per-chunk, well before
        a Recorder.record() call happens at stream completion — reaching into
        `recorder._adapter` from here would work too, but couples this proxy
        to Recorder's private internals for no benefit over just carrying the
        adapter alongside it, the same way agent/provider already are.
        How: stored via `object.__setattr__` under `_ctx_`-prefixed names (a
        plain naming convention — NOT Python name-mangling, which only
        applies to `__dunder`-style names) so that `__getattr__` forwarding
        below can't accidentally shadow or recurse into real client
        attributes of the same name."""
        object.__setattr__(self, "_ctx_target", target)
        object.__setattr__(self, "_ctx_path", path)
        object.__setattr__(self, "_ctx_tracer", tracer)
        object.__setattr__(self, "_ctx_create_paths", create_paths)
        object.__setattr__(self, "_ctx_recorder", recorder)
        object.__setattr__(self, "_ctx_agent", agent)
        object.__setattr__(self, "_ctx_provider", provider)
        object.__setattr__(self, "_ctx_adapter", adapter)

    def __getattr__(self, name: str):
        """Resolve `name` on the wrapped target. If the new path IS one of the
        create paths, return the interceptor. If it is a prefix of ANY create
        path, wrap the intermediate object so traversal continues. If `name`
        is a known SDK response-wrapper hop (`with_raw_response`/
        `with_streaming_response`) encountered exactly one step before ANY
        create path, treat it as transparent — keep wrapping WITHOUT
        advancing the path, so the create step right after it still lands on
        a tracked path. If `name` is `aio` encountered at the client ROOT on a
        gemini-provider proxy, treat it as transparent the same way (see
        `_TRANSPARENT_ROOT_HOPS`) — google-genai's async surface hangs off
        `.aio` instead of a separate client class. Otherwise return the raw
        attribute — full pass-through.

        Multi-path: every check below tests `new_path`/`path` against ANY
        member of `create_paths` rather than a single path, since an adapter
        like OpenAI's resolves to more than one independent completion method
        (`chat.completions.create` and `responses.create`) sharing this same
        proxy tree from the client root. None of the existing single-path
        gating is weakened — a 1-tuple behaves identically to the old
        singular `create_path`."""
        target = object.__getattribute__(self, "_ctx_target")
        path = object.__getattribute__(self, "_ctx_path")
        tracer = object.__getattribute__(self, "_ctx_tracer")
        create_paths = object.__getattribute__(self, "_ctx_create_paths")
        recorder = object.__getattribute__(self, "_ctx_recorder")
        agent = object.__getattribute__(self, "_ctx_agent")
        provider = object.__getattribute__(self, "_ctx_provider")
        adapter = object.__getattribute__(self, "_ctx_adapter")

        attr = getattr(target, name)
        new_path = path + (name,)

        if new_path in create_paths:
            # Two DIFFERENT "this method always streams" shapes share the
            # generic "path segment says so" signal but need different
            # wrapping, distinguished by the LAST path segment's exact text:
            #
            # - Exactly "stream" (Anthropic's `messages.stream`, OpenAI's
            #   `chat.completions.stream`/`responses.stream`) is a `.stream()`
            #   CONVENIENCE-MANAGER helper — it returns a context-manager
            #   object (a `MessageStreamManager`/`ChatCompletionStreamManager`
            #   /...) whose `__enter__`/`__aenter__` is what actually fires
            #   the real HTTP request, not the `.stream()` call itself. Routed
            #   to `_StreamManagerProxy`/`_AsyncStreamManagerProxy`.
            # - Ends WITH "stream" but ISN'T literally "stream" (Gemini's
            #   `generate_content_stream` — confirmed via Phase 13 Step 0
            #   probe against real `google-genai`) is a distinctly-NAMED
            #   method (no `stream=True` kwarg exists for it) that returns a
            #   DIRECT iterator/async-iterator, not a manager — closer in
            #   shape to `create(stream=True)` than to `.stream()`. Routed
            #   through the EXISTING raw-stream path in `_make_interceptor`
            #   (`_StreamProxy`/`_AsyncStreamProxy`) by `is_named_stream_
            #   method`, which that path treats as equivalent to a truthy
            #   `stream` kwarg (there is no kwarg to read here at all).
            #
            # Both are STRUCTURAL signals read off the path the proxy itself
            # resolved (which method the caller is invoking) — never
            # duck-typed off the eventual return value.
            last = new_path[-1]
            is_manager = last == "stream"
            is_named_stream_method = last != "stream" and last.endswith("stream")
            return _make_interceptor(attr, tracer, recorder, agent, provider, adapter,
                                     is_manager=is_manager,
                                     is_named_stream_method=is_named_stream_method)
        # Is new_path a prefix of ANY create path? If so keep wrapping
        # (carrying this wrap's recording context down to the completion
        # method that new_path is heading towards).
        if any(new_path == cp[:len(new_path)] for cp in create_paths):
            return _ClientProxy(attr, new_path, tracer, create_paths,
                                recorder, agent, provider, adapter)
        # Transparent SDK hop: e.g. LangChain calls
        # `client.chat.completions.with_raw_response.create(...)` rather than
        # `client.chat.completions.create(...)`. That inserts a
        # `with_raw_response` attribute access exactly one step before the
        # tracked `create` step. Recognize it ONLY there (not anywhere else
        # in the tree, so unrelated same-named attributes never get
        # over-wrapped) and keep the SAME `path` — not `new_path` — so the
        # immediately-following `.create` access still computes
        # `path + ("create",) == <that create path>` and is intercepted
        # normally. "One step before" is checked against EVERY create path,
        # not just the first, so this generalizes cleanly to adapters with
        # more than one.
        if (name in _TRANSPARENT_HOPS
                and any(len(path) == len(cp) - 1 and path == cp[:len(path)]
                        for cp in create_paths)):
            return _ClientProxy(attr, path, tracer, create_paths,
                                recorder, agent, provider, adapter)
        # Gemini's `.aio` root hop: recognized ONLY at path == () (the client
        # root, before any traversal has happened) and ONLY when this proxy's
        # provider is gemini (whose create_path — ("models",
        # "generate_content") — is what `.aio` actually leads to). Keep the
        # SAME `path` (still empty) so the immediately-following `.models`
        # access advances toward create_path exactly as it would from the
        # sync client root.
        if (name in _TRANSPARENT_ROOT_HOPS
                and path == ()
                and getattr(adapter, "provider", None) == "gemini"):
            return _ClientProxy(attr, path, tracer, create_paths,
                                recorder, agent, provider, adapter)
        return attr


def _make_interceptor(real_create: Callable, tracer: "Tracer",
                      recorder: "Recorder", agent: str | None,
                      provider: str | None, adapter: object,
                      is_manager: bool = False,
                      is_named_stream_method: bool = False) -> Callable:
    """Wrap the provider's completion method: call the REAL method first
    (never delaying or altering the host's request/response), measure latency,
    then hand the (kwargs, response) to the tracer along with this wrap's
    recorder/agent/provider. On host error, record the failed call and re-raise
    the host's exception unchanged.

    `is_manager` (set by `_ClientProxy.__getattr__` whenever the matched
    create path's LAST segment is "stream"): the intercepted method is a
    `.stream()` convenience helper (Anthropic's `messages.stream`, OpenAI's
    `chat.completions.stream`/`responses.stream`), not a plain completion
    call. Confirmed empirically (Phase 13 Step 0 probe against real
    `anthropic`/`openai` SDKs) that these methods are ALWAYS plain, non-
    coroutine functions — even on an async client (`AsyncAnthropic().
    messages.stream(...)` returns its manager synchronously, with no
    `await` needed) — and that calling them makes NO HTTP request at all;
    the manager they return just closes over the request kwargs and defers
    the real request to `__enter__`/`__aenter__`. So the manager branch
    below is checked FIRST, before any of the awaitable/sync/raw-stream
    logic that follows (none of which applies to a `.stream()` call): the
    manager comes back from `real_create` directly, gets wrapped in
    `_StreamManagerProxy`/`_AsyncStreamManagerProxy` (chosen by duck-typing
    the manager's OWN `__aenter__`/`__enter__` — the only reliable signal
    available, since sync-vs-async can't be read off awaitability here the
    way the rest of this function does it), and returned immediately with
    NOTHING recorded yet — recording happens once the caller actually
    enters the `with`/`async with` block (see those classes' docstrings).

    Async-aware via CALL-TIME awaitable detection, not definition-time
    `inspect.iscoroutinefunction`: `AsyncOpenAI`/`AsyncAnthropic` (and any
    other async SDK client) wrap their `create` methods behind descriptors/
    bound-method machinery, so `iscoroutinefunction(client.chat.completions
    .create)` returns False even though CALLING it returns a coroutine —
    definition-time detection is unreliable for these SDKs. Instead,
    `real_create(*args, **kwargs)` is always invoked first exactly as in the
    sync path (for an async SDK this only constructs a coroutine object
    without running any of its body — safe even though it isn't awaited
    here); if the result `inspect.isawaitable(...)`, an async closure is
    returned that awaits the coroutine, records with latency measured to
    AWAIT COMPLETION (not coroutine creation, which is near-instant and would
    under-report the real call latency), and returns the awaited response —
    otherwise the original synchronous record-and-return path below runs
    unchanged, so sync clients pay zero extra cost or behavior change.

    Streaming (`kwargs.get("stream")` truthy OR `is_named_stream_method`):
    checked AFTER the real result is in hand (post-await, for an async SDK)
    rather than duck-typed off the result's shape. `kwargs.get("stream")` is
    the caller's own, unambiguous signal for what they asked the provider
    for on OpenAI/Anthropic's `create()`. `is_named_stream_method` (set by
    `_ClientProxy.__getattr__` whenever the matched path's last segment ends
    with, but isn't exactly, "stream") is the equivalent signal for a
    provider like Gemini whose streaming call is a DIFFERENTLY-NAMED method
    (`generate_content_stream`) rather than a kwarg — there is no `stream`
    kwarg to read at all, so the path itself is the only available signal,
    read once at proxy-resolution time rather than duck-typed off the
    result. Either signal routes to `_StreamProxy`/`_AsyncStreamProxy`
    instead of recording immediately; see the module docstring for why
    (usage isn't on the stream yet at this point — only later, as chunks
    are consumed). Confirmed (Phase 13 Step 0 probe) that Gemini's async
    `generate_content_stream` IS a coroutine function (unlike the `.stream()`
    manager helpers) — `await client.aio.models.generate_content_stream(...)`
    — so it flows through the EXISTING awaitable branch below unchanged,
    only needing the `is_named_stream_method` check added alongside
    `kwargs.get("stream")` post-await."""
    def interceptor(*args, **kwargs):
        """Stand in for the provider's completion method. What: times and
        forwards the call unchanged, reports it to the tracer, and returns
        (or re-raises) exactly what the real call produced — synchronously
        for sync SDKs, or as an awaitable for async ones (see the enclosing
        docstring). How: `real_create` is invoked first with the caller's own
        args/kwargs so the host request is never delayed, inspected, or
        altered; a plain exception here means either a sync SDK raised
        immediately, or an async SDK raised before even returning a
        coroutine — either way it's the host's own error, recorded and
        re-raised unchanged. Otherwise, if the result is awaitable, control
        hands off to `_record_after_await` (below) instead of recording now
        — recording an async call before it has actually run would report a
        call that hasn't happened yet, with no response/usage to extract."""
        if is_manager:
            # `.stream()` itself makes no HTTP request (see the enclosing
            # docstring) — any exception here is a pure construction-time
            # failure (e.g. bad kwargs) with no request ever attempted, so
            # there is nothing meaningful to record; just let it propagate
            # exactly as it would unwrapped.
            manager = real_create(*args, **kwargs)
            if hasattr(manager, "__aenter__"):
                return _AsyncStreamManagerProxy(manager, kwargs, tracer, recorder,
                                                agent, provider, adapter)
            return _StreamManagerProxy(manager, kwargs, tracer, recorder,
                                       agent, provider, adapter)

        start = time.perf_counter()
        try:
            result = real_create(*args, **kwargs)
        except Exception as exc:  # host's own LLM error
            latency_ms = int((time.perf_counter() - start) * 1000)
            tracer._on_create(kwargs, None, latency_ms, type(exc).__name__,
                              recorder, agent, provider)
            raise

        if inspect.isawaitable(result):
            async def _record_after_await():
                """Await the host's own coroutine/awaitable, then record
                success or failure with latency measured to THIS await's
                completion — the async mirror of the sync path below, one
                await later. Any exception here is the HOST's async error
                (e.g. the real HTTP call inside `AsyncOpenAI.create` failing);
                it is recorded then re-raised unchanged, same contract as the
                sync branch above. A truthy `stream` kwarg is checked here,
                post-await (an async client's create() returns the async
                stream object itself once awaited — the stream, not a
                response, so nothing is recorded yet; `_AsyncStreamProxy`
                takes over instead)."""
                try:
                    response = await result
                except Exception as exc:  # host's own async LLM error
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    tracer._on_create(kwargs, None, latency_ms, type(exc).__name__,
                                      recorder, agent, provider)
                    raise
                if kwargs.get("stream") or is_named_stream_method:
                    return _wrap_stream_result(
                        response, adapter,
                        lambda s: _AsyncStreamProxy(s, kwargs, tracer, recorder,
                                                    agent, provider, adapter, start))
                latency_ms = int((time.perf_counter() - start) * 1000)
                tracer._on_create(kwargs, response, latency_ms, None,
                                  recorder, agent, provider)
                return response
            return _record_after_await()

        if kwargs.get("stream") or is_named_stream_method:
            return _wrap_stream_result(
                result, adapter,
                lambda s: _StreamProxy(s, kwargs, tracer, recorder,
                                       agent, provider, adapter, start))

        latency_ms = int((time.perf_counter() - start) * 1000)
        tracer._on_create(kwargs, result, latency_ms, None,
                          recorder, agent, provider)
        return result
    return interceptor


def _wrap_stream_result(result: object, adapter: object,
                        make_proxy: Callable[[object], object]) -> object:
    """Return what the host should get back from a streaming call: either the
    stream proxy itself, or — for a provider whose streaming method returns
    an ENVELOPE around the stream — that same envelope with only its stream
    member proxied.

    Every provider but one hands back the iterator directly, so `make_proxy(
    result)` is the whole story. Bedrock's `converse_stream` does not: it
    returns a plain response dict `{"ResponseMetadata": {...}, "stream":
    <EventStream>}` (confirmed against real botocore, Step-0 probe), and
    wrapping THAT in a stream proxy would proxy a non-iterable — the host's
    `response["stream"]` would come back unwrapped and nothing would ever be
    recorded. An adapter declares the member by name via the optional
    `stream_envelope_key` attribute (see `BedrockAdapter`); when it is
    present and the result really is a mapping carrying that key, the
    envelope is SHALLOW-COPIED (never mutated — the host's object is not
    ours to alter) with the stream member replaced by its proxy, so
    `ResponseMetadata` and every other member reach the caller untouched.

    Fail-open: anything unexpected here (a non-copyable mapping, an adapter
    attribute that isn't a string) falls back to returning the host's own
    result UNWRAPPED — capture is lost for that call, the host's stream is
    not.

    That fallback is why the two "no proxy from an envelope" conditions below
    are SEPARATE checks rather than one. No declared key means this provider
    hands the stream back directly, so the result IS the stream and proxying
    it is correct. A declared key that the result does not carry means the
    opposite: an envelope was expected and the stream is not where it should
    be (an error-shaped response, a botocore that renamed the member), so
    there is nothing iterable to proxy — and wrapping the envelope anyway
    handed the host a `_StreamProxy` where its own next line does
    `response["ResponseMetadata"]`, raising TypeError inside the traced
    application. Capture is lost there; the host's call is not."""
    try:
        key = getattr(adapter, "stream_envelope_key", None)
        if key is None:
            return make_proxy(result)
        if not isinstance(result, dict) or key not in result:
            return result
        envelope = dict(result)
        envelope[key] = make_proxy(result[key])
        return envelope
    except Exception:  # noqa: BLE001 — the host's stream must survive regardless
        _log.warning("ctxdiff: failed to wrap a streamed response; this call "
                     "will not be recorded", exc_info=True)
        return result


def _accumulate_stream_usage(adapter: object, chunk: object, state: dict) -> None:
    """Fold one streamed chunk's usage into `state` via the adapter's
    OPTIONAL `accumulate_stream_usage` (see `capture/base.py`'s Adapter
    Protocol). Every shipped adapter — OpenAI, Anthropic, Gemini
    (`generate_content_stream`) and Bedrock (`converse_stream`) — now
    defines it, but it is still looked up with `getattr(..., None)` rather
    than assumed present, so a third-party or future adapter that omits it
    simply never accumulates usage: `state` stays empty and the eventual
    recorded call gets `usage=None`, the same as not capturing streaming
    usage at all, instead of an AttributeError mid-iteration. When the
    method IS present, the call is wrapped in its own try/except — on top of
    every provider adapter already being defensive internally — so a raise
    here can NEVER interrupt the caller's own chunk-by-chunk iteration; this
    is the hardest constraint on the whole streaming feature (see module
    docstring)."""
    accumulate = getattr(adapter, "accumulate_stream_usage", None)
    if accumulate is None:
        return
    try:
        accumulate(chunk, state)
    except Exception:  # noqa: BLE001 — fail-open: a chunk must reach the caller regardless
        _log.warning("ctxdiff: accumulate_stream_usage raised; usage for this "
                     "chunk not captured", exc_info=True)


def _finalize_stream_call(kwargs: dict, state: dict, start: float, tracer: "Tracer",
                          recorder: "Recorder", agent: str | None,
                          provider: str | None, error: str | None = None,
                          quiet: bool = False) -> None:
    """Record a streamed call ONCE, at stream completion — called by
    `_StreamProxy`/`_AsyncStreamProxy` from whichever completion path fires
    first (see their docstrings). Latency is measured from the ORIGINAL call
    start (captured before the host's `create()` even ran) to THIS moment,
    not to the last chunk received — the caller's wall-clock cost of the
    whole streamed exchange, mirroring what latency already means for a
    non-streaming call. `state` empty (no chunk ever carried usage — e.g. an
    OpenAI chat stream without the caller's own `stream_options={
    "include_usage": True}` opt-in, see capture/openai.py) means the call is
    still recorded, honestly, with `usage=None` — never a crash, never a
    fabricated zero. `error` is threaded straight into `_on_create` exactly
    like the non-streaming path's `type(exc).__name__` — a stream that raised
    mid-generation (see `_StreamProxy.__next__`) is recorded as a FAILED
    call, not a silently-successful one with merely-absent usage; the two
    must stay visually distinguishable in a trace. `quiet` is forwarded
    straight into `_on_create` (see its docstring) — set only by a stream
    proxy's best-effort `__del__` finalize. Goes through the tracer's
    existing fail-open `_on_create`, exactly like the non-streaming path, so
    a broken recorder can't break this either."""
    latency_ms = int((time.perf_counter() - start) * 1000)
    response = SyntheticUsageResponse(state) if state else None
    tracer._on_create(kwargs, response, latency_ms, error, recorder, agent, provider,
                      quiet=quiet)


class _StreamProxy:
    """Transparent wrapper around a SYNC provider stream (`openai.Stream`/
    `anthropic.Stream`, a Gemini `generate_content_stream` generator, or the
    `botocore.eventstream.EventStream` inside a Bedrock `converse_stream`
    response — see `_wrap_stream_result` for how that one is reached):
    yields every real chunk to the caller
    UNCHANGED and IMMEDIATELY — never buffered, dropped, reordered, or
    delayed, the one absolute constraint on this whole feature — while
    folding each chunk's usage (if any) into a running `state` dict via the
    adapter, then records the call ONCE the stream is done instead of at
    call-time (see module docstring for why recording is deferred at all).

    "Done" is whichever of these fires FIRST: the stream running out
    (`StopIteration` from `__next__`), the wrapped stream RAISING mid-
    generation (any other exception out of `__next__` — the provider's own
    stream failing partway through), an explicit `.close()`, exiting a
    `with ... as s:` block, or — best-effort, for a stream the caller never
    exhausts, closes, uses as a context manager, or gets an error from —
    garbage collection (`__del__`). Every one of those paths goes through the
    SAME `_finalize`, guarded by a `_finalized` flag, so no matter which path
    fires (or how many of them do — e.g. a mid-stream error immediately
    followed by an explicit `.close()` in the caller's own `finally`) the
    call is recorded EXACTLY once. A mid-stream error is recorded as a FAILED
    call (`error=type(exc).__name__`, mirroring the non-streaming path) with
    whatever usage was accumulated before the failure — never a silently
    "successful" call with merely-absent usage, and this happens
    DETERMINISTICALLY inside `__next__` at the moment of failure, not
    deferred to `__del__`.

    `__getattr__` forwards anything not defined on this class (`.response`,
    SDK-internal attributes, ...) straight to the wrapped stream, so any
    caller relying on those (LangChain included) keeps working unmodified —
    same transparency contract `_ClientProxy` already gives the client
    itself."""

    def __init__(self, stream: object, kwargs: dict, tracer: "Tracer",
                 recorder: "Recorder", agent: str | None, provider: str | None,
                 adapter: object, start: float):
        object.__setattr__(self, "_ctx_stream", stream)
        object.__setattr__(self, "_ctx_kwargs", kwargs)
        object.__setattr__(self, "_ctx_tracer", tracer)
        object.__setattr__(self, "_ctx_recorder", recorder)
        object.__setattr__(self, "_ctx_agent", agent)
        object.__setattr__(self, "_ctx_provider", provider)
        object.__setattr__(self, "_ctx_adapter", adapter)
        object.__setattr__(self, "_ctx_start", start)
        object.__setattr__(self, "_ctx_state", {})
        object.__setattr__(self, "_ctx_finalized", False)
        # The wrapped object's ITERATOR, materialized lazily on the first
        # `__next__` — see `_iterator()` for why this isn't just the stream.
        object.__setattr__(self, "_ctx_iter", None)

    def __getattr__(self, name: str):
        """Pass through to the wrapped stream. `object.__getattribute__` (not
        `self.`) avoids recursing back into `__getattr__` if `_ctx_stream`
        itself somehow isn't set yet — same defensive pattern as
        `_ClientProxy.__getattr__`."""
        return getattr(object.__getattribute__(self, "_ctx_stream"), name)

    def _iterator(self):
        """The wrapped object's iterator, created ONCE on first use.

        Why not just call `next()` on the stream itself: an ITERABLE is not
        necessarily an ITERATOR. `openai.Stream`/`anthropic.Stream` and
        Gemini's generator all define `__next__`, so `next(stream)` worked
        for them — but botocore's `EventStream` (what Bedrock's
        `converse_stream` yields) defines only `__iter__`, as a GENERATOR
        function, and no `__next__` at all; `next()` on it raises TypeError
        before a single event reaches the caller. `iter()` covers both
        cases — it returns an iterator's own self unchanged, so nothing
        changes for the providers that already worked — and it is called
        exactly once and cached, because for `EventStream` each `__iter__`
        would otherwise start a SECOND generator over the same underlying
        HTTP body and interleave two half-streams. Creating it is lazy and
        does no I/O, so this stays off the call path until the host actually
        starts consuming."""
        iterator = object.__getattribute__(self, "_ctx_iter")
        if iterator is None:
            iterator = iter(object.__getattribute__(self, "_ctx_stream"))
            object.__setattr__(self, "_ctx_iter", iterator)
        return iterator

    def _finalize(self, error: str | None = None, quiet: bool = False) -> None:
        """Record the call exactly once, with `error` (a type-name string, or
        None for a normal completion) threaded straight into
        `_finalize_stream_call` — see its docstring for why a mid-stream
        failure must be recorded AS a failure, not as a merely-usage-less
        success. Guarded twice over: the `_finalized` flag makes every caller
        of `_finalize` a no-op after the first (so an error-triggered
        finalize from `__next__` and a later `.close()`/`__exit__`/`__del__`
        on the SAME stream can't double-record), and the whole body is
        wrapped in try/except so a failure while recording can never
        propagate out of a completion path the caller is relying on
        (`.close()`, `__exit__`, `__del__`) to behave normally. `quiet`
        (set only by `__del__`) suppresses the trailing warning log
        entirely — logging a full `exc_info=True` traceback for the
        best-effort GC-time path is pure noise at best, and at interpreter
        shutdown (module globals possibly already torn down) can itself
        misbehave, so `__del__` opts out of it rather than risk it."""
        if self._ctx_finalized:
            return
        self._ctx_finalized = True
        try:
            _finalize_stream_call(self._ctx_kwargs, self._ctx_state, self._ctx_start,
                                  self._ctx_tracer, self._ctx_recorder,
                                  self._ctx_agent, self._ctx_provider,
                                  error=error, quiet=quiet)
        except Exception:  # noqa: BLE001 — fail-open: finalize must never raise
            if not quiet:
                _log.warning("ctxdiff: stream finalize raised; call not recorded",
                             exc_info=True)

    def __iter__(self):
        return self

    def __next__(self):
        """Pull the NEXT real chunk first. `StopIteration` (the stream is
        exhausted, the normal end) triggers a clean finalize and propagates
        unchanged. Any OTHER exception — the provider's own stream failing
        PARTWAY THROUGH generation — triggers finalize WITH that exception's
        type name as `error`, recording the call as failed (with whatever
        usage was accumulated up to this point) BEFORE re-raising the
        ORIGINAL exception unchanged; this happens deterministically, right
        here, rather than being left to `__del__` to maybe catch later (the
        caller's own stack frame — and the still-open store — are right
        here, right now). Usage accumulation for a chunk that DID arrive
        happens only after it's confirmed to have reached the caller, and
        never delays returning it."""
        try:
            chunk = next(self._iterator())
        except StopIteration:
            self._finalize()
            raise
        except Exception as exc:  # the wrapped stream's own mid-generation failure
            self._finalize(error=type(exc).__name__)
            raise
        _accumulate_stream_usage(self._ctx_adapter, chunk, self._ctx_state)
        return chunk

    def close(self) -> None:
        """Forward to the wrapped stream's own `close()` (if any) THEN
        finalize — `finally` guarantees finalize still runs even if the real
        `close()` itself raises, since an abandoned-but-explicitly-closed
        stream should still be recorded with whatever usage was accumulated
        so far."""
        try:
            real_close = getattr(self._ctx_stream, "close", None)
            if callable(real_close):
                real_close()
        finally:
            self._finalize()

    def __enter__(self):
        real_enter = getattr(self._ctx_stream, "__enter__", None)
        if callable(real_enter):
            real_enter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Forward to the wrapped stream's own `__exit__` (which typically
        closes its underlying HTTP connection) THEN finalize, so `with ... as
        s:` both behaves exactly as it did before wrapping AND triggers
        recording on block exit. The wrapped `__exit__`'s own return value
        (whether it wants to suppress a raised exception) is preserved."""
        real_exit = getattr(self._ctx_stream, "__exit__", None)
        suppress = real_exit(exc_type, exc, tb) if callable(real_exit) else False
        self._finalize()
        return bool(suppress)

    def __del__(self):
        """Best-effort finalize for a stream the caller never exhausted,
        closed, errored out of, or used as a context manager — pure silent
        abandonment (started iterating, stopped, dropped the reference) is
        still recorded (with whatever partial usage was accumulated before
        abandonment, possibly none) rather than silently vanishing from the
        trace. `quiet=True` so this GC-time best-effort path never logs a
        traceback — and the outer `except BaseException` (not just
        `Exception`) plus this whole method being wrapped at all is because
        `__del__` can run at an unpredictable time, including interpreter
        shutdown, when module-level names may already be torn down and
        even routine operations can raise unusual things; this must NEVER
        raise or print anything."""
        try:
            self._finalize(quiet=True)
        except BaseException:  # noqa: BLE001 — __del__ must never raise or spew
            pass


class _AsyncStreamProxy:
    """Async mirror of `_StreamProxy`, for `AsyncOpenAI`/`AsyncAnthropic`
    streams (`__aiter__`/`__anext__`, `async with ... as s:`). Shares the
    exact same completion/fail-open contract — see `_StreamProxy`'s
    docstring — only the iteration and context-manager protocols differ
    (async vs sync); `_finalize` itself is plain sync code shared in spirit
    (duplicated here rather than factored out, since the two classes'
    constructors/attribute storage are otherwise identical and a shared base
    would add a layer of indirection for ~15 lines of overlap)."""

    def __init__(self, stream: object, kwargs: dict, tracer: "Tracer",
                 recorder: "Recorder", agent: str | None, provider: str | None,
                 adapter: object, start: float):
        object.__setattr__(self, "_ctx_stream", stream)
        object.__setattr__(self, "_ctx_kwargs", kwargs)
        object.__setattr__(self, "_ctx_tracer", tracer)
        object.__setattr__(self, "_ctx_recorder", recorder)
        object.__setattr__(self, "_ctx_agent", agent)
        object.__setattr__(self, "_ctx_provider", provider)
        object.__setattr__(self, "_ctx_adapter", adapter)
        object.__setattr__(self, "_ctx_start", start)
        object.__setattr__(self, "_ctx_state", {})
        object.__setattr__(self, "_ctx_finalized", False)
        # The wrapped object's ASYNC ITERATOR, materialized lazily on the
        # first `__anext__` — see `_aiterator()`, the mirror of the sync
        # proxy's `_iterator()`.
        object.__setattr__(self, "_ctx_aiter", None)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_ctx_stream"), name)

    def _aiterator(self):
        """The wrapped object's async iterator, created ONCE on first use —
        the counterpart of `_StreamProxy._iterator()`, for the same reason.

        An async ITERABLE is not necessarily an async ITERATOR. Every async
        stream shipped by openai/anthropic/google-genai defines `__anext__`,
        so awaiting `stream.__anext__()` worked for them — but an object that
        defines only `__aiter__` (as a method returning a fresh async
        generator, which is exactly the shape botocore's `EventStream` has on
        the SYNC side) has no `__anext__` at all, and the direct call raised
        AttributeError before one chunk reached the caller. `aiter()` covers
        both cases: on a real async iterator it returns that same object
        unchanged, so nothing changes for the providers that already worked.
        It is called exactly once and cached, because a fresh `__aiter__`
        would start a SECOND generator over the same underlying body and
        interleave two half-streams. An object with `__anext__` but no
        `__aiter__` — which `aiter()` would reject — is still used directly,
        so this can only ever add a shape, never remove one. Creating the
        iterator is lazy and does no I/O, so it stays off the call path until
        the host actually starts consuming."""
        iterator = object.__getattribute__(self, "_ctx_aiter")
        if iterator is None:
            stream = object.__getattribute__(self, "_ctx_stream")
            iterator = aiter(stream) if hasattr(stream, "__aiter__") else stream
            object.__setattr__(self, "_ctx_aiter", iterator)
        return iterator

    def _finalize(self, error: str | None = None, quiet: bool = False) -> None:
        """See `_StreamProxy._finalize` — identical contract, `error` and
        `quiet` mean exactly the same thing here."""
        if self._ctx_finalized:
            return
        self._ctx_finalized = True
        try:
            _finalize_stream_call(self._ctx_kwargs, self._ctx_state, self._ctx_start,
                                  self._ctx_tracer, self._ctx_recorder,
                                  self._ctx_agent, self._ctx_provider,
                                  error=error, quiet=quiet)
        except Exception:  # noqa: BLE001 — fail-open: finalize must never raise
            if not quiet:
                _log.warning("ctxdiff: stream finalize raised; call not recorded",
                             exc_info=True)

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Async mirror of `_StreamProxy.__next__` — see its docstring for
        the full error-vs-exhaustion finalize contract, identical here. The
        iterator comes from `_aiterator()` rather than from the stream
        directly — see that method for why an async iterable may not be an
        async iterator."""
        try:
            chunk = await self._aiterator().__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as exc:  # the wrapped stream's own mid-generation failure
            self._finalize(error=type(exc).__name__)
            raise
        _accumulate_stream_usage(self._ctx_adapter, chunk, self._ctx_state)
        return chunk

    async def close(self) -> None:
        """Forward to the wrapped stream's own `close()` THEN finalize.
        `close()` is a coroutine function on the real async SDK streams
        (confirmed empirically, Phase 12 Step 0), but is awaited
        conditionally (`inspect.isawaitable`) rather than assumed, so a
        synchronous or absent `close()` on some other async-stream-shaped
        object is handled just as gracefully."""
        try:
            real_close = getattr(self._ctx_stream, "close", None)
            if callable(real_close):
                result = real_close()
                if inspect.isawaitable(result):
                    await result
        finally:
            self._finalize()

    async def __aenter__(self):
        real_enter = getattr(self._ctx_stream, "__aenter__", None)
        if callable(real_enter):
            await real_enter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        real_exit = getattr(self._ctx_stream, "__aexit__", None)
        suppress = await real_exit(exc_type, exc, tb) if callable(real_exit) else False
        self._finalize()
        return bool(suppress)

    def __del__(self):
        """Best-effort finalize on garbage collection — see `_StreamProxy.
        __del__` for the full `quiet=True`/`except BaseException` rationale.
        Deliberately calls the SYNC `_finalize` (not `close()`, which is
        async and can't be awaited from `__del__`): finalize itself is plain
        sync code — it's only the wrapped stream's OWN `close()` that's
        async — so this still records the call with whatever usage was
        accumulated before abandonment, it just doesn't also close the
        underlying HTTP connection (which `AsyncStream` itself already deals
        with a `__del__`/weakref finalizer for, independent of ctxdiff)."""
        try:
            self._finalize(quiet=True)
        except BaseException:  # noqa: BLE001 — __del__ must never raise or spew
            pass


class _StreamManagerProxy:
    """Wraps the StreamManager object returned by a `.stream()` convenience
    helper (Anthropic's `messages.stream`, OpenAI's `chat.completions.
    stream`/`responses.stream`) — used as `with client.messages.stream(...)
    as stream: ...`. Confirmed empirically (Phase 13 Step 0 probe) that
    `.stream()` itself never makes an HTTP request and is never awaitable,
    even on an async client: the manager it returns just closes over the
    request kwargs, deferring the ACTUAL request to `__enter__`. So
    `_make_interceptor` hands this proxy straight back from the `.stream()`
    call with nothing recorded yet — there's nothing to record until the
    caller enters the `with` block.

    `__enter__` calls the REAL manager's `__enter__` (this is where the
    request actually fires) with the latency clock starting right before
    that call, mirroring how the raw `create(stream=True)` path starts its
    own clock right before ITS real request. The real stream `__enter__`
    hands back is wrapped in the EXISTING `_StreamProxy` — reusing its
    usage-accumulation / finalize-once / fail-open machinery completely
    unchanged; this class adds nothing but the extra manager layer around
    it. A host error raised by the real `__enter__` (the request itself
    failing) is recorded as a failed call as usual, then re-raised
    unchanged — the caller's problem, not swallowed.

    `__exit__` does NOT close the real stream a second time via the wrapped
    stream (that would double-close the SAME underlying HTTP response the
    real manager's own `__exit__` already closes): it forwards to the real
    manager's `__exit__` first — closing exactly like unwrapped usage would
    — THEN calls the wrapped stream's `_finalize()` directly (not `.close()`
    /`.__exit__()`, which would try to close again). Reusing `_StreamProxy`'s
    own `_finalized` guard this way means the call is recorded exactly once
    regardless of whether the caller already exhausted the stream inside the
    `with` block (which finalizes from `_StreamProxy.__next__` on
    `StopIteration`/mid-error) or exited early instead."""

    def __init__(self, manager: object, kwargs: dict, tracer: "Tracer",
                 recorder: "Recorder", agent: str | None, provider: str | None,
                 adapter: object):
        object.__setattr__(self, "_ctx_manager", manager)
        object.__setattr__(self, "_ctx_kwargs", kwargs)
        object.__setattr__(self, "_ctx_tracer", tracer)
        object.__setattr__(self, "_ctx_recorder", recorder)
        object.__setattr__(self, "_ctx_agent", agent)
        object.__setattr__(self, "_ctx_provider", provider)
        object.__setattr__(self, "_ctx_adapter", adapter)
        object.__setattr__(self, "_ctx_wrapped_stream", None)

    def __getattr__(self, name: str):
        """Pass through to the wrapped manager — same transparency contract
        as `_StreamProxy.__getattr__`/`_ClientProxy.__getattr__`."""
        return getattr(object.__getattribute__(self, "_ctx_manager"), name)

    def __enter__(self):
        start = time.perf_counter()
        try:
            real_stream = self._ctx_manager.__enter__()
        except Exception as exc:  # the REAL request, fired here, failed
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._ctx_tracer._on_create(self._ctx_kwargs, None, latency_ms,
                                        type(exc).__name__, self._ctx_recorder,
                                        self._ctx_agent, self._ctx_provider)
            raise
        wrapped = _StreamProxy(real_stream, self._ctx_kwargs, self._ctx_tracer,
                              self._ctx_recorder, self._ctx_agent, self._ctx_provider,
                              self._ctx_adapter, start)
        object.__setattr__(self, "_ctx_wrapped_stream", wrapped)
        return wrapped

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Forward to the real manager's own `__exit__` (closes the real
        stream, same as unwrapped usage) inside a `finally` so a raise from
        it still lets the wrapped stream's finalize run — mirroring
        `_StreamProxy.close()`'s own `finally`-guarded forwarding. `suppress`
        is only read on the non-raising path (see that method's docstring
        for why an exception here skips straight past it)."""
        try:
            real_exit = getattr(self._ctx_manager, "__exit__", None)
            suppress = real_exit(exc_type, exc, tb) if callable(real_exit) else False
        finally:
            wrapped = self._ctx_wrapped_stream
            if wrapped is not None:
                wrapped._finalize()
        return bool(suppress)


class _AsyncStreamManagerProxy:
    """Async mirror of `_StreamManagerProxy` — see its docstring for the full
    contract (nothing recorded until `__aenter__`, real error fired there is
    recorded then re-raised, `__aexit__` forwards to the real manager then
    finalizes the wrapped stream directly rather than double-closing).
    Used for `async with client.messages.stream(...) as stream:` /
    `async with client.chat.completions.stream(...) as stream:` /
    `async with client.responses.stream(...) as stream:`."""

    def __init__(self, manager: object, kwargs: dict, tracer: "Tracer",
                 recorder: "Recorder", agent: str | None, provider: str | None,
                 adapter: object):
        object.__setattr__(self, "_ctx_manager", manager)
        object.__setattr__(self, "_ctx_kwargs", kwargs)
        object.__setattr__(self, "_ctx_tracer", tracer)
        object.__setattr__(self, "_ctx_recorder", recorder)
        object.__setattr__(self, "_ctx_agent", agent)
        object.__setattr__(self, "_ctx_provider", provider)
        object.__setattr__(self, "_ctx_adapter", adapter)
        object.__setattr__(self, "_ctx_wrapped_stream", None)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_ctx_manager"), name)

    async def __aenter__(self):
        start = time.perf_counter()
        try:
            real_stream = await self._ctx_manager.__aenter__()
        except Exception as exc:  # the REAL request, fired here, failed
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._ctx_tracer._on_create(self._ctx_kwargs, None, latency_ms,
                                        type(exc).__name__, self._ctx_recorder,
                                        self._ctx_agent, self._ctx_provider)
            raise
        wrapped = _AsyncStreamProxy(real_stream, self._ctx_kwargs, self._ctx_tracer,
                                    self._ctx_recorder, self._ctx_agent, self._ctx_provider,
                                    self._ctx_adapter, start)
        object.__setattr__(self, "_ctx_wrapped_stream", wrapped)
        return wrapped

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        try:
            real_aexit = getattr(self._ctx_manager, "__aexit__", None)
            suppress = await real_aexit(exc_type, exc, tb) if callable(real_aexit) else False
        finally:
            wrapped = self._ctx_wrapped_stream
            if wrapped is not None:
                wrapped._finalize()
        return bool(suppress)

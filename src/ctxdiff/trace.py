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

import inspect
import logging
import time
import types
import uuid
from datetime import datetime, timezone
from typing import Callable

from ctxdiff.capture.anthropic import AnthropicAdapter
from ctxdiff.capture.bedrock import BedrockAdapter
from ctxdiff.capture.gemini import GeminiAdapter
from ctxdiff.capture.openai import OpenAIAdapter
from ctxdiff.capture.recorder import Recorder
from ctxdiff.models import Block
from ctxdiff.store.ctrace import CTrace

_log = logging.getLogger("ctxdiff")

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


def init(project: str, redact: Callable[[Block], Block] | None = None,
         path: str | None = None) -> "Tracer":
    """Create a Tracer for one run. `project` names the run; `redact` is an
    optional per-block scrubber applied before storage; `path` is where the
    .ctrace is written (defaults to ./<project>-<uuid>.ctrace in the cwd)."""
    if path is None:
        path = f"{project}-{uuid.uuid4().hex[:8]}.ctrace"
    return Tracer(project=project, redact=redact, path=path)


class Tracer:
    """Owns the run's .ctrace and hands out recording proxies. The store's run
    row is created lazily on the first wrap(), when the provider becomes known."""

    def __init__(self, project: str, redact: Callable[[Block], Block] | None,
                 path: str):
        """Store the run's static config and initialize empty-run state. How:
        the store/recorder are NOT created here — they need the provider,
        which is only known once `wrap()` is called — so `_ct`/`_recorder`
        start as None and `_seq`/`_pending_tags` start at their zero values."""
        self.path = path
        self._project = project
        self._redact = redact
        self._ct: CTrace | None = None
        self._recorder: Recorder | None = None  # the FIRST wrap's recorder (kept
        # for backward compat: tests monkeypatch t._recorder.record to prove the
        # interceptor wiring is fail-open even when record() is broken)
        self._seq = 0                      # monotonically increasing turn index
        self._step: str | None = None      # sticky step label (see mark())
        self._pending_tags: list[tuple[str, str]] = []  # (label, needle) for next call

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
        if self._ct is None:
            # model is per-call, not known yet at run-creation time: pass ""
            # so CTrace.create() leaves run.models == [] rather than seeding
            # a bogus [""] — CTrace.record_call()/note_model() backfill the
            # real model(s) as calls come in (see store/ctrace.py).
            model = ""
            started = datetime.now(timezone.utc).isoformat()
            self._ct = CTrace.create(self.path, project=self._project,
                                     provider=provider, model=model,
                                     started_at=started)
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
                            recorder, agent, provider, adapter)

    def tag(self, label: str, items: list) -> None:
        """Buffer semantic tags for the NEXT recorded call only. Each item is
        reduced to its text (str as-is, else a 'text'/'content' field) and
        paired with `label`; the recorder marks any block containing that text
        as `label`. Contrast mark(): tag() is next-call-only (consumed and
        cleared after one call), whereas mark() is sticky across many calls."""
        for item in items:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if text:
                self._pending_tags.append((label, text))

    def mark(self, step: str | None) -> None:
        """Set the CURRENT step label stamped onto every subsequent recorded
        call — across ALL agents — until changed; `mark(None)` clears it. This
        is STICKY (persists until the next mark()), unlike tag(), which applies
        to the next call only and is then cleared. Use it to label phases of a
        run (e.g. 'plan', 'retrieve', 'answer') so per-step views can slice the
        timeline."""
        self._step = step

    def _on_create(self, kwargs: dict, response: object | None,
                   latency_ms: int | None, error: str | None,
                   recorder: Recorder | None, agent: str | None,
                   provider: str | None, quiet: bool = False) -> None:
        """Interceptor callback: advance the turn counter, hand everything to
        the wrapping proxy's own recorder (with the proxy's agent/provider and
        the tracer's current sticky step), then clear pending tags. `seq` stays
        a single monotonic counter across ALL agents — the global timeline is
        the source of truth and per-agent views filter it. Never raises:
        `Recorder.record` is internally fail-open, but this call is *also*
        wrapped so the wiring itself stays fail-open even if `record` is
        broken/replaced entirely (e.g. monkeypatched) and its own internal
        guard is bypassed. `quiet` (default False; propagated straight into
        `recorder.record`, see its docstring) ALSO suppresses this method's
        OWN `exc_info=True` warning log below — a stream proxy's best-effort
        `__del__` finalize (trace.py) is the only caller that ever sets it,
        so every other call site's logging behavior is unchanged."""
        self._seq += 1
        tags = self._pending_tags
        self._pending_tags = []
        step = self._step
        if recorder is not None:
            try:
                recorder.record(seq=self._seq, kwargs=kwargs, response=response,
                                latency_ms=latency_ms, error=error, tagged=tags,
                                agent=agent, step=step, provider=provider, quiet=quiet)
            except Exception:  # noqa: BLE001 — fail-open guards the wiring, not just record()
                if not quiet:
                    _log.warning("ctxdiff: recorder.record raised; tracing skipped for seq=%s",
                                 self._seq, exc_info=True)

    def close(self) -> None:
        """Close the underlying store, if one was opened."""
        if self._ct is not None:
            self._ct.close()


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
                and provider == "gemini"):
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
                    return _AsyncStreamProxy(response, kwargs, tracer, recorder,
                                             agent, provider, adapter, start)
                latency_ms = int((time.perf_counter() - start) * 1000)
                tracer._on_create(kwargs, response, latency_ms, None,
                                  recorder, agent, provider)
                return response
            return _record_after_await()

        if kwargs.get("stream") or is_named_stream_method:
            return _StreamProxy(result, kwargs, tracer, recorder,
                                agent, provider, adapter, start)

        latency_ms = int((time.perf_counter() - start) * 1000)
        tracer._on_create(kwargs, result, latency_ms, None,
                          recorder, agent, provider)
        return result
    return interceptor


def _accumulate_stream_usage(adapter: object, chunk: object, state: dict) -> None:
    """Fold one streamed chunk's usage into `state` via the adapter's
    OPTIONAL `accumulate_stream_usage` (see `capture/base.py`'s Adapter
    Protocol). Looked up with `getattr(..., None)` rather than assumed
    present: Bedrock's adapter doesn't define it (its streaming method —
    `converse_stream` — is a separate, still out-of-scope create path), so a
    stream wrapped for that provider simply never accumulates usage; `state`
    stays empty and the eventual recorded call gets `usage=None`, same as not
    capturing streaming at all. OpenAI, Anthropic, and Gemini all define it
    (Gemini's `generate_content_stream` is now covered too — Phase 13). When
    the method IS present, the call is wrapped in its own try/except — on
    top of every provider adapter already being defensive internally — so a
    raise here can NEVER interrupt the caller's own chunk-by-chunk
    iteration; this is the hardest constraint on the whole streaming feature
    (see module docstring)."""
    accumulate = getattr(adapter, "accumulate_stream_usage", None)
    if accumulate is None:
        return
    try:
        accumulate(chunk, state)
    except Exception:  # noqa: BLE001 — fail-open: a chunk must reach the caller regardless
        _log.warning("ctxdiff: accumulate_stream_usage raised; usage for this "
                     "chunk not captured", exc_info=True)


class _SyntheticStreamResponse:
    """Stands in for a completed `response` at RECORD time for a call that
    was actually a stream — it carries nothing but the usage accumulated
    across the stream's chunks, exposed as an ATTRIBUTE-based object (not the
    raw dict) because every adapter's `extract_usage` duck-types
    `response.<attr>.<field>` via `getattr`. Routing accumulated stream usage
    through THIS shape, into the SAME `extract_usage` code path a
    non-streaming call already uses, means there is no second, parallel
    usage-shaping path to keep in sync — a stream-derived call's stored
    `usage` dict is byte-for-byte what a non-streaming call's would have
    been, for the same accumulated numbers.

    Exposed under BOTH `.usage` (OpenAI's/Anthropic's `extract_usage` reads
    `response.usage`) AND `.usage_metadata` (Gemini's `extract_usage` reads
    `response.usage_metadata` instead — confirmed the mismatch empirically,
    Phase 13: without this, a Gemini stream's accumulated usage silently
    vanished at record time even though accumulation itself worked
    correctly) — both names point at the SAME namespace object, so whichever
    attribute a given adapter's `extract_usage` happens to duck-type off of,
    it finds the right data; neither adapter needs to know the other
    exists."""
    def __init__(self, state: dict):
        ns = types.SimpleNamespace(**state)
        self.usage = ns
        self.usage_metadata = ns


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
    response = _SyntheticStreamResponse(state) if state else None
    tracer._on_create(kwargs, response, latency_ms, error, recorder, agent, provider,
                      quiet=quiet)


class _StreamProxy:
    """Transparent wrapper around a SYNC provider stream (e.g. `openai.
    Stream`/`anthropic.Stream`): yields every real chunk to the caller
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

    def __getattr__(self, name: str):
        """Pass through to the wrapped stream. `object.__getattribute__` (not
        `self.`) avoids recursing back into `__getattr__` if `_ctx_stream`
        itself somehow isn't set yet — same defensive pattern as
        `_ClientProxy.__getattr__`."""
        return getattr(object.__getattribute__(self, "_ctx_stream"), name)

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
            chunk = next(self._ctx_stream)
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

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_ctx_stream"), name)

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
        the full error-vs-exhaustion finalize contract, identical here."""
        try:
            chunk = await self._ctx_stream.__anext__()
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

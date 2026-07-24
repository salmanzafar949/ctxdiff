"""The public entry point. `trace.init()` opens a .ctrace and returns a Tracer;
`tracer.wrap(client)` returns a transparent proxy that records every completion
call while behaving exactly like the original client."""
from __future__ import annotations

import inspect
import logging
import time
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
                            recorder, agent, provider)

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
                   provider: str | None) -> None:
        """Interceptor callback: advance the turn counter, hand everything to
        the wrapping proxy's own recorder (with the proxy's agent/provider and
        the tracer's current sticky step), then clear pending tags. `seq` stays
        a single monotonic counter across ALL agents — the global timeline is
        the source of truth and per-agent views filter it. Never raises:
        `Recorder.record` is internally fail-open, but this call is *also*
        wrapped so the wiring itself stays fail-open even if `record` is
        broken/replaced entirely (e.g. monkeypatched) and its own internal
        guard is bypassed."""
        self._seq += 1
        tags = self._pending_tags
        self._pending_tags = []
        step = self._step
        if recorder is not None:
            try:
                recorder.record(seq=self._seq, kwargs=kwargs, response=response,
                                latency_ms=latency_ms, error=error, tagged=tags,
                                agent=agent, step=step, provider=provider)
            except Exception:  # noqa: BLE001 — fail-open guards the wiring, not just record()
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
                 recorder: "Recorder", agent: str | None, provider: str | None):
        """Store this proxy's bookkeeping: the wrapped object, how far along
        the path to the completion method this proxy sits, the owning tracer,
        the tuple of target paths, and the per-wrap recording context — the
        Recorder (built from THIS client's provider adapter), the agent name,
        and the provider — which travel with the proxy so the interceptor
        records through the right adapter and attributes each call correctly.
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

        attr = getattr(target, name)
        new_path = path + (name,)

        if new_path in create_paths:
            return _make_interceptor(attr, tracer, recorder, agent, provider)
        # Is new_path a prefix of ANY create path? If so keep wrapping
        # (carrying this wrap's recording context down to the completion
        # method that new_path is heading towards).
        if any(new_path == cp[:len(new_path)] for cp in create_paths):
            return _ClientProxy(attr, new_path, tracer, create_paths,
                                recorder, agent, provider)
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
                                recorder, agent, provider)
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
                                recorder, agent, provider)
        return attr


def _make_interceptor(real_create: Callable, tracer: "Tracer",
                      recorder: "Recorder", agent: str | None,
                      provider: str | None) -> Callable:
    """Wrap the provider's completion method: call the REAL method first
    (never delaying or altering the host's request/response), measure latency,
    then hand the (kwargs, response) to the tracer along with this wrap's
    recorder/agent/provider. On host error, record the failed call and re-raise
    the host's exception unchanged.

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
    unchanged, so sync clients pay zero extra cost or behavior change."""
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
                sync branch above."""
                try:
                    response = await result
                except Exception as exc:  # host's own async LLM error
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    tracer._on_create(kwargs, None, latency_ms, type(exc).__name__,
                                      recorder, agent, provider)
                    raise
                latency_ms = int((time.perf_counter() - start) * 1000)
                tracer._on_create(kwargs, response, latency_ms, None,
                                  recorder, agent, provider)
                return response
            return _record_after_await()

        latency_ms = int((time.perf_counter() - start) * 1000)
        tracer._on_create(kwargs, result, latency_ms, None,
                          recorder, agent, provider)
        return result
    return interceptor

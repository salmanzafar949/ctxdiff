"""The public entry point. `trace.init()` opens a .ctrace and returns a Tracer;
`tracer.wrap(client)` returns a transparent proxy that records every completion
call while behaving exactly like the original client."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from ctxdiff.capture.anthropic import AnthropicAdapter
from ctxdiff.capture.gemini import GeminiAdapter
from ctxdiff.capture.openai import OpenAIAdapter
from ctxdiff.capture.recorder import Recorder
from ctxdiff.models import Block
from ctxdiff.store.ctrace import CTrace

_log = logging.getLogger("ctxdiff")

# Provider detection maps a client's top-level module to an adapter factory.
_ADAPTERS = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "gemini": GeminiAdapter}

# Some SDKs don't own their top-level module root — `google-genai`'s client
# lives under `google.genai...`, but the bare root `google` is shared with
# unrelated packages (google.cloud, google.protobuf, ...) and must NOT map to
# an adapter blindly. For those, detection falls back to matching a known
# DOTTED prefix of the full module path instead of just its root, so only
# `google.genai` (and its submodules) resolve to gemini while any other
# `google.*` package still falls through to the "unrecognized" error.
_DOTTED_PREFIXES = {"google.genai": "gemini"}

# SDK response-wrapper hops that sit BETWEEN a resource and its `.create`
# method without changing which HTTP call gets made — e.g. LangChain's
# `ChatOpenAI._generate()` calls `self.client.with_raw_response.create(...)`
# instead of `self.client.create(...)` to get at raw headers/status before
# parsing the body. `_ClientProxy` treats a name in this set as transparent
# (see `__getattr__`) when it is encountered exactly one step before the
# create path, so the call is still tracked and recorded.
_TRANSPARENT_HOPS = ("with_raw_response", "with_streaming_response")


def _detect_provider(client: object) -> str:
    """Infer the provider from the client's module path (e.g. an OpenAI client's
    class lives under the 'openai' package). Raises if unrecognized so wrap()
    fails loudly at setup time — not silently at record time.

    Two-stage lookup: first the module's top-level root (works for openai/
    anthropic, whose packages are dedicated to one provider). When that
    misses, fall back to matching a dotted prefix from `_DOTTED_PREFIXES` —
    needed for SDKs like google-genai whose module root ('google') is shared
    across unrelated packages, so root-only matching would be too broad."""
    module = type(client).__module__ or ""
    root = module.split(".", 1)[0]
    if root in _ADAPTERS:
        return root
    for prefix, provider in _DOTTED_PREFIXES.items():
        if module == prefix or module.startswith(prefix + "."):
            return provider
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
        self._recorder: Recorder | None = None
        self._seq = 0                      # monotonically increasing turn index
        self._pending_tags: list[tuple[str, str]] = []  # (label, needle) for next call

    def wrap(self, client: object) -> object:
        """Return a transparent proxy over `client` that records every call to
        the provider's completion method. Detects the provider, creates the
        run/store on first use, and wires a Recorder."""
        provider = _detect_provider(client)
        adapter = _ADAPTERS[provider]()
        if self._ct is None:
            model = ""  # model is per-call; the run stores the first seen later
            started = datetime.now(timezone.utc).isoformat()
            self._ct = CTrace.create(self.path, project=self._project,
                                     provider=provider, model=model,
                                     started_at=started)
            self._recorder = Recorder(self._ct, adapter, self._redact)
        return _ClientProxy(client, (), self, adapter.create_path)

    def tag(self, label: str, items: list) -> None:
        """Buffer semantic tags for the NEXT recorded call. Each item is reduced
        to its text (str as-is, else a 'text'/'content' field) and paired with
        `label`; the recorder marks any block containing that text as `label`."""
        for item in items:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if text:
                self._pending_tags.append((label, text))

    def _on_create(self, kwargs: dict, response: object | None,
                   latency_ms: int | None, error: str | None) -> None:
        """Interceptor callback: advance the turn counter, hand everything to
        the recorder, then clear pending tags. Never raises: `Recorder.record`
        is internally fail-open, but this call is *also* wrapped so the wiring
        itself stays fail-open even if `record` is broken/replaced entirely
        (e.g. monkeypatched) and its own internal guard is bypassed."""
        self._seq += 1
        tags = self._pending_tags
        self._pending_tags = []
        if self._recorder is not None:
            try:
                self._recorder.record(seq=self._seq, kwargs=kwargs, response=response,
                                      latency_ms=latency_ms, error=error, tagged=tags)
            except Exception:  # noqa: BLE001 — fail-open guards the wiring, not just record()
                _log.warning("ctxdiff: recorder.record raised; tracing skipped for seq=%s",
                             self._seq, exc_info=True)

    def close(self) -> None:
        """Close the underlying store, if one was opened."""
        if self._ct is not None:
            self._ct.close()


class _ClientProxy:
    """A transparent proxy that forwards attribute access to the wrapped target,
    following the adapter's `create_path` down to the completion method — which
    it replaces with an interceptor. Anything off that path is returned as-is,
    so the wrapped client is behaviorally identical to the original."""

    def __init__(self, target: object, path: tuple[str, ...],
                 tracer: "Tracer", create_path: tuple[str, ...]):
        """Store this proxy's bookkeeping: the wrapped object, how far along
        the path to the completion method this proxy sits, the owning
        tracer, and that target path. How: stored via `object.__setattr__`
        under `_ctx_`-prefixed names (a plain naming convention — NOT Python
        name-mangling, which only applies to `__dunder`-style names) so that
        `__getattr__` forwarding below can't accidentally shadow or recurse
        into real client attributes of the same name."""
        object.__setattr__(self, "_ctx_target", target)
        object.__setattr__(self, "_ctx_path", path)
        object.__setattr__(self, "_ctx_tracer", tracer)
        object.__setattr__(self, "_ctx_create_path", create_path)

    def __getattr__(self, name: str):
        """Resolve `name` on the wrapped target. If the new path IS the create
        path, return the interceptor. If it is a prefix of the create path, wrap
        the intermediate object so traversal continues. If `name` is a known
        SDK response-wrapper hop (`with_raw_response`/`with_streaming_response`)
        encountered exactly one step before create, treat it as transparent —
        keep wrapping WITHOUT advancing the path, so the create step right
        after it still lands on the tracked path. Otherwise return the raw
        attribute — full pass-through."""
        target = object.__getattribute__(self, "_ctx_target")
        path = object.__getattribute__(self, "_ctx_path")
        tracer = object.__getattribute__(self, "_ctx_tracer")
        create_path = object.__getattribute__(self, "_ctx_create_path")

        attr = getattr(target, name)
        new_path = path + (name,)

        if new_path == create_path:
            return _make_interceptor(attr, tracer)
        # Is new_path a prefix of create_path? If so keep wrapping.
        if new_path == create_path[:len(new_path)]:
            return _ClientProxy(attr, new_path, tracer, create_path)
        # Transparent SDK hop: e.g. LangChain calls
        # `client.chat.completions.with_raw_response.create(...)` rather than
        # `client.chat.completions.create(...)`. That inserts a
        # `with_raw_response` attribute access exactly one step before the
        # tracked `create` step. Recognize it ONLY there (not anywhere else
        # in the tree, so unrelated same-named attributes never get
        # over-wrapped) and keep the SAME `path` — not `new_path` — so the
        # immediately-following `.create` access still computes
        # `path + ("create",) == create_path` and is intercepted normally.
        if (name in _TRANSPARENT_HOPS
                and len(path) == len(create_path) - 1
                and path == create_path[:len(path)]):
            return _ClientProxy(attr, path, tracer, create_path)
        return attr


def _make_interceptor(real_create: Callable, tracer: "Tracer") -> Callable:
    """Wrap the provider's completion method: call the REAL method first
    (never delaying or altering the host's request/response), measure latency,
    then hand the (kwargs, response) to the tracer. On host error, record the
    failed call and re-raise the host's exception unchanged."""
    def interceptor(*args, **kwargs):
        """Stand in for the provider's completion method. What: times and
        forwards the call unchanged, reports it to the tracer, and returns
        (or re-raises) exactly what the real call produced. How: `real_create`
        is invoked first with the caller's own args/kwargs so the host request
        is never delayed, inspected, or altered; on success the response and
        latency are handed to `tracer._on_create` before returning it; on
        failure the call is still reported (with `error` set, no response)
        and the original exception is re-raised unchanged so the host sees
        its own error, not a ctxdiff one."""
        start = time.perf_counter()
        try:
            response = real_create(*args, **kwargs)
        except Exception as exc:  # host's own LLM error
            latency_ms = int((time.perf_counter() - start) * 1000)
            tracer._on_create(kwargs, None, latency_ms, error=type(exc).__name__)
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        tracer._on_create(kwargs, response, latency_ms, error=None)
        return response
    return interceptor

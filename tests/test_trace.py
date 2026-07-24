import asyncio
import time
import pytest
from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


class _FakeCompletions:
    """Stand-in for client.chat.completions with a recording create()."""
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()


class _FakeOpenAI:
    """Duck-typed OpenAI client: type module name drives provider detection."""
    __module__ = "openai"
    def __init__(self): self.chat = _FakeChat()


def test_wrap_passes_through_and_records(tmp_path):
    """A wrapped client behaves identically (same response, real create called)
    AND every create() is recorded to the .ctrace."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeOpenAI()
    wrapped = t.wrap(client)
    resp = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert isinstance(resp, _Resp)                    # real response passed through
    assert len(client.chat.completions.calls) == 1    # real create actually ran
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    blocks = ct.get_call_blocks(calls[0].id)
    assert blocks[0].block.text == "hi"
    ct.close()


def test_wrap_detects_provider_from_client(tmp_path):
    """Provider is inferred from the client's module so the right adapter and
    run.provider are used without the caller specifying it."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    t.wrap(_FakeOpenAI())
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_run().provider == "openai"
    ct.close(); t.close()


def test_wrap_backfills_run_models_from_the_call(tmp_path):
    """End-to-end regression for the `run.models == ['']` bug: wrap() no
    longer seeds a bogus blank model at run-creation time, and a single real
    call's `model` kwarg rolls up onto run.models — not `['']`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeOpenAI())
    wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_run().models == ["gpt-4o"]
    ct.close()


def test_tag_overrides_label_on_next_call(tmp_path):
    """tag() buffers a (label, text); the next recorded call labels any block
    containing that text with the tagged label and source 'tagged'."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeOpenAI())
    t.tag("rag", ["Enterprise pricing FAQ"])
    wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Context: Enterprise pricing FAQ 2026"}])
    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    blocks = ct.get_call_blocks(ct.get_calls()[0].id)
    assert blocks[0].label == "rag" and blocks[0].label_source == "tagged"
    ct.close()


class BoomError(Exception):
    """A distinctive host-side exception, distinguishable from any exception
    ctxdiff itself might raise, so the test can prove it is the host's own
    error being re-raised unchanged (not swallowed or replaced)."""


class _BoomingCompletions:
    """Stand-in for client.chat.completions whose create() always raises
    BoomError, simulating the underlying LLM call itself failing."""
    def create(self, **kwargs):
        raise BoomError("the host's own LLM error")


class _BoomingChat:
    def __init__(self): self.completions = _BoomingCompletions()


class _BoomingOpenAI:
    """Duck-typed OpenAI client whose completion call always fails."""
    __module__ = "openai"
    def __init__(self): self.chat = _BoomingChat()


def test_wrap_reraises_host_error_and_records_it(tmp_path):
    """On a host-side LLM error, the interceptor must (1) re-raise the host's
    own exception unchanged — never swallow or wrap it — and (2) still record
    the call, with `error` set to the exception's type name and `usage` None
    (there was no response to extract usage from)."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_BoomingOpenAI())
    with pytest.raises(BoomError):
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    assert calls[0].usage is None
    ct.close()


class _FakeRawResponseCompletions:
    """Stand-in for the SDK's `.with_raw_response` resource — same shape as
    `_FakeCompletions` (a recording create()) but reached via the
    with_raw_response hop, the way LangChain's `ChatOpenAI._generate()`
    calls `self.client.with_raw_response.create(**payload)` instead of
    `self.client.create(**payload)` directly."""
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _FakeCompletionsWithRawHop:
    """client.chat.completions augmented with a `.with_raw_response` hop,
    mirroring how the real openai SDK resource exposes both `.create()`
    directly and `.with_raw_response.create()` off the same resource."""
    def __init__(self): self.with_raw_response = _FakeRawResponseCompletions()


class _FakeChatWithRawHop:
    def __init__(self): self.completions = _FakeCompletionsWithRawHop()


class _FakeOpenAIWithRawHop:
    """Duck-typed OpenAI client exposing a with_raw_response hop nested
    exactly one step before create() (chat.completions.with_raw_response)
    AND an unrelated same-named attribute at the TOP level (off the create
    path), so one test can check both the transparent-hop success case and
    the no-over-wrapping guard case."""
    __module__ = "openai"
    def __init__(self):
        self.chat = _FakeChatWithRawHop()
        self.with_raw_response = object()  # off-path sentinel; must stay raw


def test_wrap_treats_with_raw_response_as_transparent_hop(tmp_path):
    """LangChain's ChatOpenAI calls `client.chat.completions.with_raw_response
    .create(...)` instead of `.create(...)` directly (spike §4). The proxy
    must keep tracking the create path THROUGH that hop — recording the
    call — while an unrelated `with_raw_response` attribute reached anywhere
    else (here: the top level) must stay a raw, unwrapped pass-through, i.e.
    NOT get swept up into transparent-hop wrapping it doesn't belong to."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeOpenAIWithRawHop()
    wrapped = t.wrap(client)

    resp = wrapped.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert isinstance(resp, _Resp)                                   # real response passed through
    assert len(client.chat.completions.with_raw_response.calls) == 1  # real create actually ran
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1  # the hop was transparent — the call WAS captured
    ct.close()

    # No-over-wrapping guard: with_raw_response accessed anywhere OTHER than
    # exactly one step before create() (here: the top level, not
    # chat.completions) must remain the raw, unwrapped attribute.
    assert wrapped.with_raw_response is client.with_raw_response


class _FakeModels:
    """Stand-in for client.models with a recording generate_content()."""
    def __init__(self): self.calls = []
    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _FakeGeminiClient:
    """Duck-typed google-genai client: its class lives under
    'google.genai.client', the same dotted-prefix path the real SDK uses, so
    detection must match via `_DOTTED_PREFIXES` (root 'google' alone is too
    broad to map to gemini)."""
    __module__ = "google.genai.client"
    def __init__(self): self.models = _FakeModels()


def test_wrap_detects_gemini_via_dotted_prefix_and_records(tmp_path):
    """A fake client whose module is 'google.genai.client' (root 'google', not
    in _ADAPTERS directly) is detected as gemini via the dotted-prefix
    fallback, and a call through `.models.generate_content(...)` is recorded
    to the .ctrace exactly like the other providers."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClient()
    wrapped = t.wrap(client)
    resp = wrapped.models.generate_content(model="gemini-2.0-flash", contents="hi")
    assert isinstance(resp, _Resp)
    assert len(client.models.calls) == 1
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_run().provider == "gemini"
    calls = ct.get_calls()
    assert len(calls) == 1
    blocks = ct.get_call_blocks(calls[0].id)
    assert blocks[0].block.text == "hi"
    ct.close()


class _FakeBedrockRuntime:
    """Duck-typed boto3 bedrock-runtime client: class name 'BedrockRuntime'
    living under module 'botocore.client' — the same module EVERY boto3
    client shares (s3, ec2, ...), so detection must key off the class name,
    not the module, once the botocore root is confirmed."""
    __module__ = "botocore.client"
    def __init__(self): self.calls = []
    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
                "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}}


_FakeBedrockRuntime.__name__ = "BedrockRuntime"


class _FakeS3:
    """Duck-typed boto3 S3 client: same shared module root ('botocore.client')
    as bedrock-runtime, but a class name NOT in `_BOTOCORE_CLASSES` — proves
    an unrecognized botocore client still raises rather than silently
    matching bedrock."""
    __module__ = "botocore.client"


_FakeS3.__name__ = "S3"


def test_wrap_detects_bedrock_via_botocore_class_name_and_records(tmp_path):
    """A fake client whose module is 'botocore.client' (root 'botocore', not
    in `_ADAPTERS`/`_DOTTED_PREFIXES` at all) but whose class name is
    'BedrockRuntime' is detected as bedrock via the class-name fallback, and
    a call through `.converse(...)` is recorded to the .ctrace exactly like
    the other providers."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeBedrockRuntime()
    wrapped = t.wrap(client)
    resp = wrapped.converse(modelId="anthropic.claude-3-haiku",
                            messages=[{"role": "user", "content": [{"text": "hi"}]}])
    assert resp["output"]["message"]["content"][0]["text"] == "hi"
    assert len(client.calls) == 1
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_run().provider == "bedrock"
    calls = ct.get_calls()
    assert len(calls) == 1
    blocks = ct.get_call_blocks(calls[0].id)
    assert blocks[0].block.text == "hi"
    ct.close()


def test_wrap_raises_for_unrecognized_botocore_client():
    """A botocore client with a class name NOT in `_BOTOCORE_CLASSES` (e.g.
    S3) must still raise ValueError — the botocore root alone is never
    sufficient to assume bedrock."""
    t = trace.init("agent", path="unused.ctrace")
    with pytest.raises(ValueError, match="unrecognized boto3 client class"):
        t.wrap(_FakeS3())


class _AnthUsage:
    input_tokens = 5; output_tokens = 2
class _AnthResp:
    usage = _AnthUsage()


class _FakeMessages:
    """Stand-in for client.messages with a recording create()."""
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _AnthResp()


class _FakeAnthropic:
    """Duck-typed Anthropic client: module 'anthropic' drives detection, and
    its completion path is `messages.create` — different from OpenAI's
    `chat.completions.create`."""
    __module__ = "anthropic"
    def __init__(self): self.messages = _FakeMessages()


def test_two_wraps_carry_their_own_agent(tmp_path):
    """Wrapping two clients with different agent names stamps each call with
    the agent of the proxy that made it; seq stays a single global counter."""
    t = trace.init("multi", path=str(tmp_path / "r.ctrace"))
    a = t.wrap(_FakeOpenAI(), agent="researcher")
    b = t.wrap(_FakeOpenAI(), agent="writer")
    a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "r"}])
    b.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "w"}])
    a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "r2"}])
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert [c.seq for c in calls] == [1, 2, 3]
    assert [c.agent for c in calls] == ["researcher", "writer", "researcher"]
    ct.close()


def test_multi_provider_wraps_record_through_own_adapter(tmp_path):
    """The multi-provider regression: an OpenAI-shaped and an Anthropic-shaped
    client wrapped in the SAME run must each record through their OWN adapter
    (the pre-v2 bug built one recorder from the first provider's adapter and
    mis-parsed the second). Proof: only the Anthropic adapter extracts a
    top-level `system` field into a block and reports usage as input_tokens —
    if the anthropic call had gone through the openai adapter, the system block
    would vanish and the usage shape would be wrong."""
    t = trace.init("multi", path=str(tmp_path / "r.ctrace"))
    oa = t.wrap(_FakeOpenAI(), agent="oa")
    an = t.wrap(_FakeAnthropic(), agent="an")

    oa.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi from openai"}])
    an.messages.create(
        model="claude-sonnet-4-5", system="anthropic system prompt",
        messages=[{"role": "user", "content": "hi from anthropic"}])
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    by_provider = {c.provider: c for c in ct.get_calls()}
    assert set(by_provider) == {"openai", "anthropic"}
    # Both providers' models roll up onto the single run, in call order —
    # the multi-agent/multi-provider case for the run.models backfill.
    assert ct.get_run().models == ["gpt-4o", "claude-sonnet-4-5"]

    # OpenAI call: openai usage shape, its user block present.
    oa_call = by_provider["openai"]
    assert "prompt_tokens" in (oa_call.usage or {})
    assert "input_tokens" not in (oa_call.usage or {})
    oa_texts = [cb.block.text for cb in ct.get_call_blocks(oa_call.id)]
    assert "hi from openai" in oa_texts

    # Anthropic call: anthropic usage shape, and the top-level system block
    # ONLY the anthropic adapter extracts — the smoking gun for correct routing.
    an_call = by_provider["anthropic"]
    assert "input_tokens" in (an_call.usage or {})
    an_texts = [cb.block.text for cb in ct.get_call_blocks(an_call.id)]
    assert "anthropic system prompt" in an_texts
    assert "hi from anthropic" in an_texts
    ct.close()


def test_mark_is_sticky_across_calls_and_clearable(tmp_path):
    """mark(step) applies to every subsequent call until changed (sticky,
    unlike tag which is next-call-only); mark(None) clears it."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    w = t.wrap(_FakeOpenAI())
    t.mark("plan")
    w.chat.completions.create(model="m", messages=[{"role": "user", "content": "a"}])
    w.chat.completions.create(model="m", messages=[{"role": "user", "content": "b"}])
    t.mark("answer")
    w.chat.completions.create(model="m", messages=[{"role": "user", "content": "c"}])
    t.mark(None)
    w.chat.completions.create(model="m", messages=[{"role": "user", "content": "d"}])
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert [c.step for c in ct.get_calls()] == ["plan", "plan", "answer", None]
    ct.close()


def test_wrap_is_fail_open_if_recording_breaks(tmp_path, monkeypatch):
    """If recording raises internally, the host call still returns normally."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeOpenAI())
    # Break the recorder; the wrapped call must still return the real response.
    monkeypatch.setattr(t._recorder, "record",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    # Should NOT raise despite record blowing up inside the interceptor's caller.
    # (record itself is fail-open, but this guards the interceptor wiring too.)
    resp = wrapped.chat.completions.create(model="gpt-4o", messages=[])
    assert resp is not None
    t.close()


# --- Async client support -------------------------------------------------
# All async tests below use `asyncio.run()` around plain sync `def test_...`
# functions rather than pytest-asyncio (no new test dependency needed) — the
# interceptor's own async closure is what's under test, not the test runner.


class _AsyncUsage:
    prompt_tokens = 7; completion_tokens = 2; total_tokens = 9
class _AsyncResp:
    usage = _AsyncUsage()


class _FakeAsyncCompletions:
    """Stand-in for an AsyncOpenAI-style client.chat.completions whose
    create() is a coroutine function — mirrors `_FakeCompletions` but awaits a
    small sleep first (the way a real network call would), so `latency_ms`
    has something non-trivial to measure at AWAIT-COMPLETION time rather than
    at coroutine-construction time."""
    def __init__(self): self.calls = []
    async def create(self, **kwargs):
        await asyncio.sleep(0.01)
        self.calls.append(kwargs)
        return _AsyncResp()


class _FakeAsyncChat:
    def __init__(self): self.completions = _FakeAsyncCompletions()


class _FakeAsyncOpenAI:
    """Duck-typed AsyncOpenAI client: same module/path shape as `_FakeOpenAI`,
    but `create()` is a coroutine function — this is what call-time awaitable
    detection must pick up, since `inspect.iscoroutinefunction` on the real
    SDKs' bound `create` methods returns False (they're wrapped descriptors),
    so definition-time detection would miss this."""
    __module__ = "openai"
    def __init__(self): self.chat = _FakeAsyncChat()


def test_wrap_async_create_passes_through_and_records(tmp_path):
    """An async wrapped client: the real (awaited) response passes through
    unchanged, the call lands on disk with usage + blocks exactly like the
    sync path, and `latency_ms` reflects time spent AWAITING the coroutine
    (>= the fake's own 0.01s sleep) — not the near-instant time to merely
    construct it."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncOpenAI()
    wrapped = t.wrap(client)

    resp = asyncio.run(wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi async"}]))
    assert isinstance(resp, _AsyncResp)                # real awaited response passed through
    assert len(client.chat.completions.calls) == 1     # real async create actually ran
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert calls[0].latency_ms >= 10  # >= the fake's 0.01s sleep, in whole ms
    blocks = ct.get_call_blocks(calls[0].id)
    assert blocks[0].block.text == "hi async"
    ct.close()


class _BoomingAsyncCompletions:
    """Stand-in for an async client.chat.completions whose create() always
    raises BoomError AFTER an await point, simulating the underlying async
    LLM call itself failing (not a sync setup error before any await)."""
    async def create(self, **kwargs):
        await asyncio.sleep(0)
        raise BoomError("the host's own async LLM error")


class _BoomingAsyncChat:
    def __init__(self): self.completions = _BoomingAsyncCompletions()


class _BoomingAsyncOpenAI:
    """Duck-typed AsyncOpenAI client whose completion call always fails."""
    __module__ = "openai"
    def __init__(self): self.chat = _BoomingAsyncChat()


def test_wrap_async_reraises_host_error_and_records_it(tmp_path):
    """On a host-side ASYNC LLM error (raised after the await), the
    interceptor must (1) re-raise the host's own exception unchanged and (2)
    still record the call, with `error` set to the exception's type name and
    `usage` None — the async mirror of
    `test_wrap_reraises_host_error_and_records_it`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_BoomingAsyncOpenAI())
    with pytest.raises(BoomError):
        asyncio.run(wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    assert calls[0].usage is None
    ct.close()


def test_wrap_async_is_fail_open_if_recording_breaks(tmp_path, monkeypatch):
    """If recording raises internally on the ASYNC path, the awaited host
    call must still return normally — the async mirror of
    `test_wrap_is_fail_open_if_recording_breaks`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeAsyncOpenAI())
    monkeypatch.setattr(t._recorder, "record",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    resp = asyncio.run(wrapped.chat.completions.create(model="gpt-4o", messages=[]))
    assert resp is not None
    t.close()


def test_wrap_async_client_carries_agent_attribution(tmp_path):
    """`wrap(async_client, agent=...)` composes with async interception: the
    recorded call carries the agent name exactly like the sync path."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeAsyncOpenAI(), agent="researcher")
    asyncio.run(wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert calls[0].agent == "researcher"
    ct.close()


class _FakeAsyncModels:
    """Stand-in for `client.aio.models`, with an async generate_content()."""
    def __init__(self): self.calls = []
    async def generate_content(self, **kwargs):
        await asyncio.sleep(0.01)
        self.calls.append(kwargs)
        return _Resp()


class _FakeAsyncGeminiNamespace:
    """Stand-in for `client.aio` itself — one level above `.models`."""
    def __init__(self): self.models = _FakeAsyncModels()


class _FakeGeminiClientWithAio:
    """Duck-typed google-genai client exposing BOTH the sync `.models`
    surface and the async `.aio.models` mirror, the way the real
    `genai.Client` does (one client, two namespaces, not two client
    classes)."""
    __module__ = "google.genai.client"
    def __init__(self):
        self.models = _FakeModels()
        self.aio = _FakeAsyncGeminiNamespace()


def test_wrap_gemini_aio_path_is_detected_and_captured(tmp_path):
    """`client.aio.models.generate_content(...)` is detected via the `.aio`
    root hop (recognized only at the client root, gated on provider ==
    'gemini') and recorded exactly like the sync `.models.generate_content`
    path, including the async latency/response handling."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithAio()
    wrapped = t.wrap(client)

    resp = asyncio.run(wrapped.aio.models.generate_content(
        model="gemini-2.0-flash", contents="hi via aio"))
    assert isinstance(resp, _Resp)
    assert len(client.aio.models.calls) == 1
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_run().provider == "gemini"
    calls = ct.get_calls()
    assert len(calls) == 1
    blocks = ct.get_call_blocks(calls[0].id)
    assert blocks[0].block.text == "hi via aio"
    ct.close()


class _FakeOpenAIWithRealAioAttr:
    """Duck-typed OpenAI-style client (provider != gemini) that ALSO happens
    to carry a top-level `.aio` attribute of its own — proves the `.aio`
    transparent-hop is gated on provider (not just structural root position),
    so an unrelated same-named attribute on a non-gemini client is never
    swept into wrapping it doesn't belong to."""
    __module__ = "openai"
    def __init__(self):
        self.chat = _FakeChat()
        self.aio = object()  # sentinel; must stay raw, unwrapped


def test_wrap_aio_on_non_gemini_client_passes_through_raw(tmp_path):
    """No-over-wrapping guard for the `.aio` root hop: on a non-gemini
    client, `.aio` reached at the root must remain the raw, unwrapped
    attribute — identical to the `with_raw_response` guard test above."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeOpenAIWithRealAioAttr()
    wrapped = t.wrap(client)
    assert wrapped.aio is client.aio
    t.close()


# --- Multi-path proxy: OpenAI's chat.completions.create + responses.create -
# These prove `_ClientProxy` tracks MORE THAN ONE create path off the same
# proxy tree (see trace.py's `create_paths`), not just a single path per wrap.


class _FakeResponses:
    """Stand-in for client.responses with a recording create()."""
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _FakeOpenAIWithBothPaths:
    """Duck-typed OpenAI client exposing BOTH `chat.completions.create` and
    `responses.create` off the same client root, the way the real openai SDK
    does — proof that a single wrap() intercepts both independently."""
    __module__ = "openai"
    def __init__(self):
        self.chat = _FakeChat()
        self.responses = _FakeResponses()


def test_wrap_multi_path_intercepts_both_chat_and_responses(tmp_path):
    """A single wrap() over a client exposing both completion methods records
    a call made through EITHER path, each landing on the .ctrace independently
    — the core proof that `create_paths` (plural) works, not just the first
    path an adapter declares."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeOpenAIWithBothPaths()
    wrapped = t.wrap(client)

    resp1 = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "via chat"}])
    resp2 = wrapped.responses.create(model="gpt-4o", input="via responses")
    assert isinstance(resp1, _Resp) and isinstance(resp2, _Resp)
    assert len(client.chat.completions.calls) == 1
    assert len(client.responses.calls) == 1
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 2
    texts = [cb.block.text for c in calls for cb in ct.get_call_blocks(c.id)]
    assert "via chat" in texts
    assert "via responses" in texts
    ct.close()


def test_wrap_with_raw_response_hop_still_works_for_chat_path(tmp_path):
    """Regression: the `with_raw_response` transparent hop (used by LangChain
    for the chat path) must still be intercepted correctly now that the proxy
    tracks MULTIPLE create paths instead of one — proves the multi-path
    change didn't weaken this existing gating."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeOpenAIWithRawHop()
    wrapped = t.wrap(client)

    resp = wrapped.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert isinstance(resp, _Resp)
    assert len(client.chat.completions.with_raw_response.calls) == 1
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert len(ct.get_calls()) == 1
    ct.close()


class _FakeAsyncResponses:
    """Stand-in for an AsyncOpenAI-style client.responses whose create() is a
    coroutine function — proves Responses async composes for free through
    the same call-time awaitable detection as chat completions."""
    def __init__(self): self.calls = []
    async def create(self, **kwargs):
        await asyncio.sleep(0.01)
        self.calls.append(kwargs)
        return _AsyncResp()


class _FakeAsyncOpenAIWithResponses:
    """Duck-typed AsyncOpenAI client exposing only the async responses.create
    path (the chat path isn't needed for this test)."""
    __module__ = "openai"
    def __init__(self): self.responses = _FakeAsyncResponses()


def test_wrap_async_responses_create_is_intercepted(tmp_path):
    """`AsyncOpenAI.responses.create(...)` — a coroutine function on the
    SECOND declared create path — is intercepted exactly like the async chat
    path: real response passes through, call lands on disk with usage."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncOpenAIWithResponses()
    wrapped = t.wrap(client)

    resp = asyncio.run(wrapped.responses.create(model="gpt-4o", input="hi async responses"))
    assert isinstance(resp, _AsyncResp)
    assert len(client.responses.calls) == 1
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    blocks = ct.get_call_blocks(calls[0].id)
    assert blocks[0].block.text == "hi async responses"
    ct.close()

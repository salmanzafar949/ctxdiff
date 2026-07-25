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
    run.provider are used without the caller specifying it.

    Read after `close()`, not straight after `wrap()`: the session is opened on
    the writer thread (so no store I/O ever sits on the host's call path — see
    `_DeferredStore`), which makes its creation asynchronous. `close()` is the
    point at which every write for the run, including that first one, is
    guaranteed flushed."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    t.wrap(_FakeOpenAI())
    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_run().provider == "openai"
    ct.close()


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


class _FakeLegacyVertexModel:
    """Duck-typed `vertexai.generative_models.GenerativeModel` — the LEGACY
    google-cloud-aiplatform SDK, which Google itself has deprecated in favour
    of google-genai. It is a MODEL object, not a client (the completion
    method hangs straight off it), and its `contents` are proto objects
    rather than the dicts `GeminiAdapter` reads, so it is deliberately NOT
    mapped to the gemini adapter."""
    __module__ = "vertexai.generative_models"

    def generate_content(self, *args, **kwargs):  # pragma: no cover — never called
        raise AssertionError("detection must fail before any call is made")


class _FakeBedrockStreamRuntime:
    """Duck-typed bedrock-runtime client whose `converse_stream` returns what
    the real one does: an ENVELOPE DICT carrying the event stream under
    "stream" — and, like botocore's own `EventStream`, one that is ITERABLE
    but NOT an ITERATOR (no `__next__`), which is the shape `_StreamProxy`
    has to cope with."""
    __module__ = "botocore.client"

    def __init__(self, events):
        self.events = events
        self.kwargs = None
        self.stream = None      # the real stream object handed back, for assertions

    def converse_stream(self, **kwargs):
        self.kwargs = kwargs
        self.stream = _IterableOnly(self.events)
        return {"ResponseMetadata": {"RequestId": "r-1"}, "stream": self.stream}


_FakeBedrockStreamRuntime.__name__ = "BedrockRuntime"


class _IterableOnly:
    """An iterable that is NOT an iterator — `__iter__` only, exactly like
    `botocore.eventstream.EventStream`. `closed` records whether the proxy's
    `close()` reached it."""

    def __init__(self, items):
        self.items = items
        self.closed = False

    def __iter__(self):
        return iter(self.items)

    def close(self):
        self.closed = True


def test_wrap_raises_for_the_legacy_vertexai_sdk():
    """`vertexai.generative_models.GenerativeModel` (the deprecated
    google-cloud-aiplatform SDK) is NOT supported: it must fail loudly at
    `wrap()` rather than be swept in with google-genai's Vertex mode, which
    IS supported and shares nothing with it but a product name. Silently
    accepting it would record a run with zero blocks — the proto `Content`
    objects it sends are not the dicts `GeminiAdapter` reads — which is worse
    than an error, because it looks like capture is working."""
    t = trace.init("agent", path="unused.ctrace")
    with pytest.raises(ValueError, match="unrecognized client module"):
        t.wrap(_FakeLegacyVertexModel())


def test_wrap_bedrock_converse_stream_records_usage_and_keeps_the_envelope(tmp_path):
    """Bedrock's `converse_stream` returns an envelope dict, not a stream.
    The host must get that SAME envelope back — `ResponseMetadata` intact —
    with only `["stream"]` proxied, every event passing through untouched,
    and the call recorded ONCE at completion with the usage carried by the
    trailing `metadata` event."""
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9}}},
    ]
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeBedrockStreamRuntime(events)
    wrapped = t.wrap(client)

    response = wrapped.converse_stream(
        modelId="anthropic.claude-3-haiku",
        messages=[{"role": "user", "content": [{"text": "hi"}]}])
    assert response["ResponseMetadata"] == {"RequestId": "r-1"}   # envelope preserved
    assert list(response["stream"]) == events                     # events untouched
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9}
    assert [b.block.text for b in ct.get_call_blocks(calls[0].id)] == ["hi"]
    ct.close()


def test_wrap_bedrock_converse_stream_close_forwards_and_records(tmp_path):
    """Closing the proxied stream forwards to the real object's `close()`
    (releasing the HTTP body) and still records the abandoned call — with
    only the usage seen so far, which here is none."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeBedrockStreamRuntime([{"messageStart": {"role": "assistant"}}])
    stream = t.wrap(client).converse_stream(
        modelId="m", messages=[{"role": "user", "content": [{"text": "hi"}]}])["stream"]
    stream.close()
    t.close()

    assert client.stream.closed is True          # close() reached the real object
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


class _FakeBedrockStreamRuntimeNoStream:
    """A bedrock-runtime client whose `converse_stream` returns an envelope
    that does NOT carry the declared `stream` member — an error-shaped
    response, a future botocore that renames it, a request the service
    answered without a body. The adapter still DECLARES
    `stream_envelope_key`, so this is the case where ctxdiff knows an
    envelope was expected and cannot find the stream inside it."""
    __module__ = "botocore.client"

    def __init__(self):
        self.response = {"ResponseMetadata": {"RequestId": "r-1", "HTTPStatusCode": 200}}

    def converse_stream(self, **kwargs):
        return self.response


_FakeBedrockStreamRuntimeNoStream.__name__ = "BedrockRuntime"


def test_stream_envelope_without_the_declared_member_reaches_the_host_unwrapped(tmp_path):
    """FAIL-OPEN, the hard case. When an adapter declares
    `stream_envelope_key` and the result does NOT carry it, ctxdiff has lost
    its capture — but the HOST must not lose its response. Wrapping the
    envelope itself in a stream proxy (which is what happened before) handed
    the caller an object that is not subscriptable, so the very next line of
    the host's own code — `response["ResponseMetadata"]` — raised TypeError.
    A debugger may drop its own data; it may never break the call it is
    watching."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeBedrockStreamRuntimeNoStream()
    wrapped = t.wrap(client)

    response = wrapped.converse_stream(
        modelId="anthropic.claude-3-haiku",
        messages=[{"role": "user", "content": [{"text": "hi"}]}])

    # The host's OWN object, untouched — not a proxy standing in for it.
    assert response is client.response
    assert response["ResponseMetadata"]["RequestId"] == "r-1"
    assert dict(response) == client.response
    t.close()

    # ...and capture is silently skipped, rather than recording a call whose
    # stream was never found.
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_calls() == []
    ct.close()


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


# --- Streaming usage capture -------------------------------------------------
# `kwargs.get("stream")` truthy routes the interceptor to `_StreamProxy`/
# `_AsyncStreamProxy` instead of recording at call-time (see trace.py's
# module docstring). These fakes mirror the REAL openai.Stream/
# AsyncStream/anthropic.Stream shape confirmed against the real SDKs
# (tests/eval/test_wrap_streaming.py, and the Phase 12 Step-0 probe): a
# generator-backed object with __iter__/__next__ (or __aiter__/__anext__),
# .close(), context-manager support, and a passthrough attribute.


class _StreamUsage:
    """Stand-in for a ChatCompletionChunk's `.usage` — only present, per
    real SDK behavior, on the FINAL chunk of a stream (see capture/openai.py)."""
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _StreamChunk:
    """Stand-in for one ChatCompletionChunk: content plus an optional usage."""
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


class _FakeSyncStream:
    """Stand-in for openai.Stream/anthropic.Stream: a plain iterator over
    pre-built chunks, exposing close(), context-manager support, and a
    passthrough attribute (`.response`) so `_StreamProxy`'s forwarding can
    be tested."""
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False
        self.close_calls = 0
        self.response = "raw-http-response-sentinel"

    def __iter__(self): return self
    def __next__(self): return next(self._chunks)

    def close(self):
        self.closed = True
        self.close_calls += 1

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _RaisingSyncStream:
    """Stand-in for a sync stream that yields `raise_after` normal chunks
    then RAISES mid-generation — the PROVIDER's own stream dying partway
    through (a dropped connection, a mid-response API error, ...), not
    anything ctxdiff does. Otherwise identical in shape to `_FakeSyncStream`
    (close(), context-manager support) so it exercises the exact same
    forwarding paths, just with a failure instead of a clean StopIteration."""
    def __init__(self, chunks, raise_after):
        self._chunks = list(chunks)
        self._raise_after = raise_after
        self._i = 0
        self.closed = False
        self.close_calls = 0

    def __iter__(self): return self

    def __next__(self):
        if self._i >= self._raise_after:
            raise BoomError("stream died mid-generation")
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    def close(self):
        self.closed = True
        self.close_calls += 1

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _FakeAsyncStream:
    """Async mirror of `_FakeSyncStream`."""
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._i = 0
        self.closed = False
        self.close_calls = 0
        self.response = "raw-http-response-sentinel"

    def __aiter__(self): return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    async def close(self):
        self.closed = True
        self.close_calls += 1

    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False


class _RaisingAsyncStream:
    """Async mirror of `_RaisingSyncStream`: yields `raise_after` normal
    chunks then RAISES mid-generation."""
    def __init__(self, chunks, raise_after):
        self._chunks = list(chunks)
        self._raise_after = raise_after
        self._i = 0
        self.closed = False
        self.close_calls = 0

    def __aiter__(self): return self

    async def __anext__(self):
        if self._i >= self._raise_after:
            raise BoomError("async stream died mid-generation")
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    async def close(self):
        self.closed = True
        self.close_calls += 1

    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False


class _FakeStreamingCompletions:
    """Stand-in for client.chat.completions whose create() returns a
    pre-built fake stream when stream=True (mirroring the real SDK) — the
    stream to return is set on the instance before the call, since each
    test builds its own exact chunk sequence."""
    def __init__(self):
        self.calls = []
        self.next_stream = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_stream


class _FakeStreamingChat:
    def __init__(self): self.completions = _FakeStreamingCompletions()


class _FakeStreamingOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeStreamingChat()


def test_stream_yields_chunks_unchanged_and_records_once_with_usage(tmp_path):
    """Full consumption of a sync stream: every chunk the caller receives is
    IDENTICAL and IN ORDER to what the fake stream produced, and after
    iteration completes, exactly one call is recorded, with usage folded
    from the final chunk."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [
        _StreamChunk("Hello"),
        _StreamChunk(" there"),
        _StreamChunk(None, usage=_StreamUsage(10, 2, 12)),
    ]
    client.chat.completions.next_stream = _FakeSyncStream(chunks)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        stream=True, stream_options={"include_usage": True})
    received = list(stream)
    assert received == chunks  # identical objects, in order — nothing dropped/reordered

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    ct.close()


def test_stream_double_iteration_does_not_double_record(tmp_path):
    """Exhausting the stream, then calling .close() on it afterward, does NOT
    record a second call — the _finalized guard covers finalize triggered
    from more than one completion path."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_stream = _FakeSyncStream([_StreamChunk("hi")])

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    list(stream)     # exhausts -> finalizes once
    stream.close()   # a second completion signal -> must be a no-op

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert len(ct.get_calls()) == 1
    ct.close()


def test_stream_partial_consumption_then_close_records_partial_usage(tmp_path):
    """Consuming only SOME chunks, then calling .close() before reaching the
    usage-bearing chunk, records the call once with usage=None (the honest
    reflection of what was actually accumulated before abandonment)."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [
        _StreamChunk("Hello"),
        _StreamChunk(" there"),
        _StreamChunk(None, usage=_StreamUsage(10, 2, 12)),
    ]
    client.chat.completions.next_stream = _FakeSyncStream(chunks)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    first = next(stream)
    assert first is chunks[0]
    stream.close()
    assert client.chat.completions.next_stream.closed  # real close() was forwarded to

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


def test_stream_context_manager_triggers_finalize_on_exit(tmp_path):
    """`with wrapped_stream as s:` both forwards to the wrapped stream's own
    __enter__/__exit__ (closing its real connection) and triggers finalize
    on block exit."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hello"), _StreamChunk(None, usage=_StreamUsage(5, 1, 6))]
    client.chat.completions.next_stream = _FakeSyncStream(chunks)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    with stream as s:
        received = list(s)
    assert received == chunks
    assert client.chat.completions.next_stream.closed

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    ct.close()


def test_stream_no_usage_records_usage_none_not_a_crash(tmp_path):
    """A stream with NO usage-bearing chunk at all (e.g. OpenAI chat without
    stream_options={"include_usage": True} — the honest documented gap) is
    still fully consumable and still recorded, with usage=None."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_stream = _FakeSyncStream(
        [_StreamChunk("Hello"), _StreamChunk(" there")])

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    received = list(stream)
    assert len(received) == 2

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


def test_stream_passthrough_attribute_forwards_to_wrapped_stream(tmp_path):
    """__getattr__ forwards attributes not defined on the proxy (e.g.
    `.response`) straight to the wrapped stream — same transparency contract
    `_ClientProxy` already gives the client itself."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_stream = _FakeSyncStream([_StreamChunk("hi")])

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    assert stream.response == "raw-http-response-sentinel"
    list(stream)
    t.close()


def test_stream_fail_open_accumulate_usage_raises_chunks_still_delivered(tmp_path, monkeypatch):
    """If the adapter's accumulate_stream_usage raises on every chunk, the
    caller must still receive every chunk, unchanged and in order — the
    hardest fail-open constraint. The call still ends up recorded (with
    usage=None, since nothing could be accumulated)."""
    from ctxdiff.capture.openai import OpenAIAdapter
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hello"), _StreamChunk(None, usage=_StreamUsage(1, 1, 2))]
    client.chat.completions.next_stream = _FakeSyncStream(chunks)

    monkeypatch.setattr(OpenAIAdapter, "accumulate_stream_usage",
                        lambda self, chunk, state: (_ for _ in ()).throw(RuntimeError("boom")))

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    received = list(stream)
    assert received == chunks  # every chunk still delivered despite accumulate() blowing up

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


def test_stream_fail_open_record_raises_iteration_still_completes(tmp_path, monkeypatch):
    """If recorder.record itself raises at finalize time, iteration must
    still complete normally for the caller — same fail-open contract the
    non-streaming path already has
    (test_wrap_is_fail_open_if_recording_breaks)."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_stream = _FakeSyncStream(
        [_StreamChunk("Hello"), _StreamChunk(None, usage=_StreamUsage(1, 1, 2))])

    monkeypatch.setattr(t._recorder, "record",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    received = list(stream)  # must not raise
    assert len(received) == 2


def test_stream_mid_error_records_error_and_reraises_unchanged(tmp_path):
    """A stream that yields K good chunks then RAISES mid-generation (the
    provider's own stream dying partway through): the K chunks are delivered
    to the caller unchanged before the raise, the ORIGINAL exception
    propagates unchanged (not swallowed, not wrapped), and the call is
    recorded EXACTLY ONCE with `error == 'BoomError'` — a mid-stream failure
    must never look like a silently-successful call that merely has no
    usage."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hello"), _StreamChunk(" there")]
    client.chat.completions.next_stream = _RaisingSyncStream(chunks, raise_after=2)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    received = []
    with pytest.raises(BoomError):
        for c in stream:
            received.append(c)
    assert received == chunks  # both good chunks delivered before the raise

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    assert calls[0].usage is None  # nothing accumulated before the failure
    ct.close()


def test_stream_mid_error_preserves_usage_accumulated_before_failure(tmp_path):
    """Usage accumulated from chunks BEFORE the failure is preserved on the
    error-recorded call — a stream that dies mid-generation AFTER already
    reporting some usage still reflects what was actually captured, not a
    blanket None."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hello", usage=_StreamUsage(10, 2, 12))]
    client.chat.completions.next_stream = _RaisingSyncStream(chunks, raise_after=1)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    with pytest.raises(BoomError):
        list(stream)

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert calls[0].error == "BoomError"
    assert calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    ct.close()


def test_stream_context_manager_mid_error_records_error_exactly_once(tmp_path):
    """`with wrapped_stream as s:` around a mid-stream error: __next__'s
    deterministic finalize records `error='BoomError'` at the moment of
    failure; __exit__'s own finalize call afterward (`with` always calls
    __exit__ when an exception propagates through the block) is a no-op
    under the `_finalized` guard — not a double record, and not an
    accidental overwrite back to error=None. The real stream's own
    __exit__/close() is still forwarded to."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    real_stream = _RaisingSyncStream([_StreamChunk("Hello")], raise_after=1)
    client.chat.completions.next_stream = real_stream

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    with pytest.raises(BoomError):
        with stream as s:
            for c in s:
                pass
    assert real_stream.closed  # __exit__ still forwarded to the real stream

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    ct.close()


def test_stream_abandoned_without_close_is_recorded_via_del(tmp_path, capsys):
    """A stream the caller never exhausts, closes, or uses as a context
    manager is still recorded on garbage collection (__del__), best-effort,
    with whatever partial usage was accumulated (here: none, since it was
    never consumed at all) — and no exception or traceback escapes/leaks to
    stderr from that best-effort GC-time path (the `quiet=True` +
    `except BaseException` contract on `__del__`)."""
    import gc
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamingOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_stream = _FakeSyncStream(
        [_StreamChunk("Hello"), _StreamChunk(None, usage=_StreamUsage(1, 1, 2))])

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    del stream
    gc.collect()  # must not raise or print a traceback

    t.close()
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Exception ignored" not in captured.err

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


# --- Async stream equivalents ------------------------------------------------


class _FakeAsyncStreamingCompletions:
    def __init__(self):
        self.calls = []
        self.next_stream = None

    async def create(self, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        return self.next_stream


class _FakeAsyncStreamingChat:
    def __init__(self): self.completions = _FakeAsyncStreamingCompletions()


class _FakeAsyncStreamingOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeAsyncStreamingChat()


def test_async_stream_yields_chunks_unchanged_and_records_once_with_usage(tmp_path):
    """Async mirror of
    test_stream_yields_chunks_unchanged_and_records_once_with_usage."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hi"), _StreamChunk(None, usage=_StreamUsage(4, 2, 6))]
    client.chat.completions.next_stream = _FakeAsyncStream(chunks)

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            stream=True, stream_options={"include_usage": True})
        return [c async for c in stream]

    received = asyncio.run(run())
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    ct.close()


def test_async_stream_context_manager_triggers_finalize_on_exit(tmp_path):
    """`async with wrapped_stream as s:` forwards to the wrapped stream and
    triggers finalize on block exit — async mirror of the sync context-
    manager test."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hi"), _StreamChunk(None, usage=_StreamUsage(3, 1, 4))]
    client.chat.completions.next_stream = _FakeAsyncStream(chunks)

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
        async with stream as s:
            return [c async for c in s]

    received = asyncio.run(run())
    assert received == chunks
    assert client.chat.completions.next_stream.closed

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    ct.close()


class _AsyncIterableOnly:
    """An ASYNC iterable that is NOT an async iterator: `__aiter__` only (a
    plain method returning a fresh async generator), no `__anext__` — the
    async mirror of `_IterableOnly`, i.e. of the shape
    `botocore.eventstream.EventStream` has on the sync side. `aiter_calls`
    counts how many times the iterator was materialized, so a test can prove
    it happens exactly ONCE (two would interleave two half-streams over one
    HTTP body)."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.aiter_calls = 0
        self.closed = False

    def __aiter__(self):
        self.aiter_calls += 1
        chunks = self.chunks

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    async def close(self):
        self.closed = True


def test_async_stream_that_is_iterable_but_not_an_iterator_is_consumed(tmp_path):
    """The async counterpart of `_StreamProxy._iterator()`. An async ITERABLE
    is not necessarily an async ITERATOR: an object may define `__aiter__`
    (returning a fresh async generator) and no `__anext__` at all — which is
    exactly the shape botocore's `EventStream` has on the sync side, and the
    reason the sync proxy materializes `iter(stream)` once. Calling
    `stream.__anext__()` directly on such an object raises AttributeError
    before a single chunk reaches the caller, so the async proxy resolves the
    iterator the same way, once and lazily."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hi"), _StreamChunk(None, usage=_StreamUsage(4, 2, 6))]
    stream_obj = _AsyncIterableOnly(chunks)
    client.chat.completions.next_stream = stream_obj

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            stream=True, stream_options={"include_usage": True})
        return [c async for c in stream]

    received = asyncio.run(run())
    assert received == chunks                     # every chunk reached the caller
    assert stream_obj.aiter_calls == 1            # materialized exactly once

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    ct.close()


def test_async_stream_fail_open_accumulate_raises_chunks_still_delivered(tmp_path, monkeypatch):
    """Async mirror of the sync fail-open accumulate test."""
    from ctxdiff.capture.openai import OpenAIAdapter
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hi"), _StreamChunk(None, usage=_StreamUsage(1, 1, 2))]
    client.chat.completions.next_stream = _FakeAsyncStream(chunks)

    monkeypatch.setattr(OpenAIAdapter, "accumulate_stream_usage",
                        lambda self, chunk, state: (_ for _ in ()).throw(RuntimeError("boom")))

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
        return [c async for c in stream]

    received = asyncio.run(run())
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_calls()[0].usage is None
    ct.close()


def test_async_stream_mid_error_records_error_and_reraises_unchanged(tmp_path):
    """Async mirror of test_stream_mid_error_records_error_and_reraises_unchanged."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamingOpenAI()
    wrapped = t.wrap(client)
    chunks = [_StreamChunk("Hi"), _StreamChunk(" there")]
    client.chat.completions.next_stream = _RaisingAsyncStream(chunks, raise_after=2)

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
        received = []
        with pytest.raises(BoomError):
            async for c in stream:
                received.append(c)
        return received

    received = asyncio.run(run())
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    assert calls[0].usage is None
    ct.close()


def test_async_stream_context_manager_mid_error_records_error_exactly_once(tmp_path):
    """Async mirror of test_stream_context_manager_mid_error_records_error_exactly_once:
    `async with wrapped_stream as s:` around a mid-stream error records
    error='BoomError' exactly once, and the real stream's own __aexit__ is
    still forwarded to."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamingOpenAI()
    wrapped = t.wrap(client)
    real_stream = _RaisingAsyncStream([_StreamChunk("Hi")], raise_after=1)
    client.chat.completions.next_stream = real_stream

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
        with pytest.raises(BoomError):
            async with stream as s:
                async for c in s:
                    pass

    asyncio.run(run())
    assert real_stream.closed

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    ct.close()


# --- `.stream()` convenience-manager helpers ---------------------------------
# Anthropic's `messages.stream(...)` / OpenAI's `chat.completions.stream(...)`
# and `responses.stream(...)` return a StreamManager, not a stream: nothing
# happens (no request, no recording) until the caller enters `with .../async
# with ... as stream:` — see trace.py's `_StreamManagerProxy`/
# `_AsyncStreamManagerProxy`. These fakes mirror that shape (confirmed against
# the real `anthropic`/`openai` SDKs, Phase 13 Step 0 probe): a manager whose
# `__enter__`/`__aenter__` returns a stream-shaped object and is where a host
# error would actually surface, plus `__exit__`/`__aexit__` that forward to
# the real stream's own close.


class _FakeStreamManager:
    """Stand-in for MessageStreamManager/ChatCompletionStreamManager. Makes no
    request itself — `enter_raises`, if set, simulates the REAL request
    (fired inside `__enter__`) failing."""
    def __init__(self, stream=None, enter_raises=None):
        self._stream = stream
        self._enter_raises = enter_raises
        self.entered = False
        self.exited = False

    def __enter__(self):
        if self._enter_raises is not None:
            raise self._enter_raises
        self.entered = True
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        real_exit = getattr(self._stream, "__exit__", None)
        return real_exit(exc_type, exc, tb) if callable(real_exit) else False


class _FakeAsyncStreamManager:
    """Async mirror of `_FakeStreamManager`."""
    def __init__(self, stream=None, enter_raises=None):
        self._stream = stream
        self._enter_raises = enter_raises
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        if self._enter_raises is not None:
            raise self._enter_raises
        self.entered = True
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        real_aexit = getattr(self._stream, "__aexit__", None)
        return await real_aexit(exc_type, exc, tb) if callable(real_aexit) else False


class _FakeStreamManagerCompletions:
    """Stand-in for client.chat.completions exposing `.stream()` (NOT
    `.create()`) — the manager-returning convenience helper, matching the
    real SDK shape 1:1 rather than reusing the `.create(stream=True)` fakes
    above."""
    def __init__(self):
        self.calls = []
        self.next_manager = None

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_manager


class _FakeStreamManagerChat:
    def __init__(self): self.completions = _FakeStreamManagerCompletions()


class _FakeStreamManagerOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeStreamManagerChat()


class _FakeAsyncStreamManagerCompletions:
    """Async mirror: `.stream()` is itself a PLAIN (non-async) method even on
    an async client — confirmed against real `AsyncOpenAI`/`AsyncAnthropic`
    (Phase 13 Step 0 probe) — it's only `__aenter__` that's async."""
    def __init__(self):
        self.calls = []
        self.next_manager = None

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_manager


class _FakeAsyncStreamManagerChat:
    def __init__(self): self.completions = _FakeAsyncStreamManagerCompletions()


class _FakeAsyncStreamManagerOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeAsyncStreamManagerChat()


def test_stream_manager_nothing_recorded_before_with_block_entered(tmp_path):
    """Calling `.stream()` itself makes no request and records nothing — the
    turn counter only advances once the caller actually enters the `with`
    block (mirroring the real SDKs, where `.stream()` just builds the
    manager and defers the request to `__enter__`)."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    real_stream = _FakeSyncStream([_StreamChunk("hi")])
    client.chat.completions.next_manager = _FakeStreamManager(real_stream)

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    # Nothing recorded yet: `.stream()` fires no request (the real manager's
    # __enter__ hasn't run) so no call has been enqueued.
    assert not client.chat.completions.next_manager.entered
    with manager as s:
        list(s)
    t.close()
    # Exactly one call recorded once the block ran (the flush on close()
    # guarantees the writer has persisted it before we read).
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert len(ct.get_calls()) == 1
    ct.close()


def test_stream_manager_sync_with_block_records_once_with_usage(tmp_path):
    """`with wrapped_manager as s: for ev in s: ...` delivers every event
    unchanged and in order, then records exactly once on block exit, with
    usage accumulated from the events — the manager-wrapped mirror of
    `test_stream_yields_chunks_unchanged_and_records_once_with_usage`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    events = [_StreamChunk("Hello"), _StreamChunk(" there"),
             _StreamChunk(None, usage=_StreamUsage(10, 2, 12))]
    real_stream = _FakeSyncStream(events)
    client.chat.completions.next_manager = _FakeStreamManager(real_stream)

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with manager as s:
        received = list(s)
    assert received == events  # identical objects, in order

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)  # request blocks still captured
    ct.close()


def test_stream_manager_exit_forwards_and_does_not_double_close(tmp_path):
    """`__exit__` forwards to the real manager's own `__exit__` (which closes
    the real stream once) and does NOT also close the wrapped stream a
    second time through `_StreamProxy.close()` — the real stream's `close()`
    is called exactly once even though ctxdiff sits in between."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    real_stream = _FakeSyncStream([_StreamChunk("hi")])
    client.chat.completions.next_manager = _FakeStreamManager(real_stream)

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with manager as s:
        list(s)
    assert real_stream.close_calls == 1
    assert client.chat.completions.next_manager.exited

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert len(ct.get_calls()) == 1
    ct.close()


def test_stream_manager_early_exit_before_exhaustion_still_finalizes(tmp_path):
    """Exiting the `with` block WITHOUT exhausting the stream still records
    the call once, via the manager's own `__exit__` triggering the wrapped
    stream's finalize directly."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    events = [_StreamChunk("Hello"), _StreamChunk(" there"),
             _StreamChunk(None, usage=_StreamUsage(10, 2, 12))]
    real_stream = _FakeSyncStream(events)
    client.chat.completions.next_manager = _FakeStreamManager(real_stream)

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with manager as s:
        first = next(s)
    assert first is events[0]

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None  # abandoned before the usage-bearing chunk
    ct.close()


def test_stream_manager_getattr_passthrough(tmp_path):
    """Attributes not defined on the manager proxy forward straight to the
    real manager (before __enter__) and, per _StreamProxy, straight to the
    real stream once entered — same transparency contract as everywhere
    else in this module."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    real_stream = _FakeSyncStream([_StreamChunk("hi")])
    fake_manager = _FakeStreamManager(real_stream)
    fake_manager.sentinel = "manager-sentinel"
    client.chat.completions.next_manager = fake_manager

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert manager.sentinel == "manager-sentinel"
    with manager as s:
        assert s.response == "raw-http-response-sentinel"
        list(s)
    t.close()


def test_stream_manager_enter_raises_records_error_and_reraises_unchanged(tmp_path):
    """A host error firing INSIDE `__enter__` (the real request) is the
    caller's problem: re-raised unchanged, and recorded as a failed call —
    the manager-wrapped mirror of the non-streaming host-error path."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_manager = _FakeStreamManager(
        enter_raises=BoomError("request failed"))

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(BoomError):
        with manager:
            pass

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    ct.close()


def test_stream_manager_mid_error_records_error_exactly_once(tmp_path):
    """A mid-stream error inside `with manager as s: for ev in s: ...`
    records error='BoomError' exactly once (from `_StreamProxy.__next__`),
    and the manager's own `__exit__` — which also fires afterward — is a
    clean no-op against the ALREADY-finalized wrapped stream, plus still
    forwards to the real manager's own `__exit__`/close."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    real_stream = _RaisingSyncStream([_StreamChunk("Hi")], raise_after=1)
    fake_manager = _FakeStreamManager(real_stream)
    client.chat.completions.next_manager = fake_manager

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(BoomError):
        with manager as s:
            for _ in s:
                pass
    assert fake_manager.exited
    assert real_stream.closed

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    ct.close()


def test_stream_manager_fail_open_accumulate_raises_events_still_delivered(tmp_path, monkeypatch):
    """A raising `accumulate_stream_usage` must never interrupt event
    delivery through the manager-wrapped path either — same hardest
    fail-open constraint as the raw stream path."""
    from ctxdiff.capture.openai import OpenAIAdapter
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    events = [_StreamChunk("Hello"), _StreamChunk(None, usage=_StreamUsage(1, 1, 2))]
    client.chat.completions.next_manager = _FakeStreamManager(_FakeSyncStream(events))

    monkeypatch.setattr(OpenAIAdapter, "accumulate_stream_usage",
                        lambda self, chunk, state: (_ for _ in ()).throw(RuntimeError("boom")))

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with manager as s:
        received = list(s)
    assert received == events

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_calls()[0].usage is None
    ct.close()


def test_stream_manager_fail_open_record_raises_with_block_still_completes(tmp_path, monkeypatch):
    """A broken recorder must never break the caller's `with` block — the
    manager-wrapped mirror of the raw stream's own fail-open record test."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeStreamManagerOpenAI()
    wrapped = t.wrap(client)
    only_chunk = _StreamChunk("hi")
    client.chat.completions.next_manager = _FakeStreamManager(_FakeSyncStream([only_chunk]))
    monkeypatch.setattr(t._recorder, "record",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("record boom")))

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    with manager as s:
        received = list(s)
    assert received == [only_chunk]  # delivered despite broken recorder
    t.close()


# --- Async `.stream()` manager equivalents -----------------------------------


def test_async_stream_manager_records_once_with_usage(tmp_path):
    """`async with wrapped_manager as s: async for ev in s: ...` — async
    mirror of `test_stream_manager_sync_with_block_records_once_with_usage`.
    `.stream()` is confirmed (Phase 13 Step 0 probe) to be a PLAIN method
    even on an async client — no `await` before it — only `__aenter__` is
    async, so `manager = wrapped.chat.completions.stream(...)` happens
    synchronously and only the `async with`/`async for` below run inside
    `asyncio.run`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamManagerOpenAI()
    wrapped = t.wrap(client)
    events = [_StreamChunk("Hi"), _StreamChunk(None, usage=_StreamUsage(4, 2, 6))]
    client.chat.completions.next_manager = _FakeAsyncStreamManager(_FakeAsyncStream(events))

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    async def run():
        async with manager as s:
            return [c async for c in s]

    received = asyncio.run(run())
    assert received == events

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    ct.close()


def test_async_stream_manager_enter_raises_records_error_and_reraises(tmp_path):
    """Async mirror of `test_stream_manager_enter_raises_records_error_and_reraises_unchanged`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamManagerOpenAI()
    wrapped = t.wrap(client)
    client.chat.completions.next_manager = _FakeAsyncStreamManager(
        enter_raises=BoomError("async request failed"))

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    async def run():
        async with manager:
            pass

    with pytest.raises(BoomError):
        asyncio.run(run())

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    ct.close()


def test_async_stream_manager_mid_error_records_error_exactly_once(tmp_path):
    """Async mirror of `test_stream_manager_mid_error_records_error_exactly_once`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncStreamManagerOpenAI()
    wrapped = t.wrap(client)
    real_stream = _RaisingAsyncStream([_StreamChunk("Hi")], raise_after=1)
    fake_manager = _FakeAsyncStreamManager(real_stream)
    client.chat.completions.next_manager = fake_manager

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    async def run():
        with pytest.raises(BoomError):
            async with manager as s:
                async for _ in s:
                    pass

    asyncio.run(run())
    assert fake_manager.exited
    assert real_stream.closed

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error == "BoomError"
    ct.close()


# --- Gemini `generate_content_stream` (a distinctly-NAMED streaming method,
# not a manager and not a `stream=True` kwarg) -------------------------------
# Confirmed (Phase 13 Step 0 probe against real `google-genai` 2.14.0):
# `client.models.generate_content_stream(...)` returns a DIRECT iterator
# (sync) — routed through the EXISTING raw-stream `_StreamProxy` via
# `is_named_stream_method` (see `_ClientProxy.__getattr__`), exactly like
# `create(stream=True)` on the other providers. `client.aio.models.
# generate_content_stream(...)` IS itself a coroutine function (unlike the
# `.stream()` manager helpers) that resolves, once awaited, to an async
# iterator — so it flows through the same await-then-check-stream branch
# `create(stream=True)` already uses on an async client. `usage_metadata` is
# CUMULATIVE (confirmed): each chunk already carries the running totals, so
# `GeminiAdapter.accumulate_stream_usage` OVERWRITES `state` each time —
# last chunk wins, never a sum.


class _GeminiUsage:
    def __init__(self, prompt_token_count, candidates_token_count, total_token_count):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


class _GeminiChunk:
    """Stand-in for a streamed `GenerateContentResponse` chunk."""
    def __init__(self, text, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class _FakeGeminiModelsWithStream(_FakeModels):
    """Adds `generate_content_stream` (a DIRECT iterator, not a manager) to
    the existing `_FakeModels` fake used for the non-streaming gemini
    tests."""
    def __init__(self):
        super().__init__()
        self.stream_calls = []
        self.next_stream = None

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return iter(self.next_stream)


class _FakeGeminiClientWithStream:
    __module__ = "google.genai.client"
    def __init__(self): self.models = _FakeGeminiModelsWithStream()


def test_gemini_stream_records_once_with_cumulative_usage_last_wins(tmp_path):
    """A real-shaped Gemini stream: every chunk reaches the caller unchanged
    and in order, and the recorded usage is the LAST chunk's cumulative
    totals — never a sum of the two chunks' counts."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithStream()
    wrapped = t.wrap(client)
    chunks = [
        _GeminiChunk("Hello", _GeminiUsage(10, 1, 11)),
        _GeminiChunk(" there!", _GeminiUsage(10, 5, 15)),
    ]
    client.models.next_stream = chunks

    stream = wrapped.models.generate_content_stream(
        model="gemini-2.0-flash", contents="hi")
    received = list(stream)
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15,
    }
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)
    ct.close()


def test_gemini_stream_no_usage_records_usage_none_not_a_crash(tmp_path):
    """A chunk with no `usage_metadata` at all still passes through fine;
    the call is recorded with usage=None, not a crash."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithStream()
    wrapped = t.wrap(client)
    client.models.next_stream = [_GeminiChunk("Hello"), _GeminiChunk(" there")]

    stream = wrapped.models.generate_content_stream(
        model="gemini-2.0-flash", contents="hi")
    assert len(list(stream)) == 2

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


def test_gemini_stream_fail_open_accumulate_raises_chunks_still_delivered(tmp_path, monkeypatch):
    """A raising `accumulate_stream_usage` must never interrupt delivery —
    same hardest fail-open constraint as the other providers' streams."""
    from ctxdiff.capture.gemini import GeminiAdapter
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithStream()
    wrapped = t.wrap(client)
    chunks = [_GeminiChunk("hi", _GeminiUsage(1, 1, 2))]
    client.models.next_stream = chunks
    monkeypatch.setattr(GeminiAdapter, "accumulate_stream_usage",
                        lambda self, chunk, state: (_ for _ in ()).throw(RuntimeError("boom")))

    stream = wrapped.models.generate_content_stream(model="gemini-2.0-flash", contents="hi")
    received = list(stream)
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_calls()[0].usage is None
    ct.close()


def test_gemini_stream_fail_open_record_raises_iteration_still_completes(tmp_path, monkeypatch):
    """A broken recorder must never break the caller's own iteration."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithStream()
    wrapped = t.wrap(client)
    chunks = [_GeminiChunk("hi", _GeminiUsage(1, 1, 2))]
    client.models.next_stream = chunks
    monkeypatch.setattr(t._recorder, "record",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("record boom")))

    stream = wrapped.models.generate_content_stream(model="gemini-2.0-flash", contents="hi")
    received = list(stream)
    assert received == chunks
    t.close()


class _AsyncGeminiChunkIterator:
    """Stand-in for the async iterator google-genai's async
    `generate_content_stream` resolves to once awaited (confirmed via Phase
    13 Step 0 probe: the method itself is a coroutine function, unlike the
    `.stream()` manager helpers)."""
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._i = 0

    def __aiter__(self): return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


class _FakeAsyncGeminiModelsWithStream(_FakeAsyncModels):
    """`generate_content_stream` here is itself `async def` — confirmed
    against the real SDK (`inspect.iscoroutinefunction` is True on
    `AsyncModels.generate_content_stream`) — resolving to an async iterator
    once awaited, not an async-generator function called directly."""
    def __init__(self):
        super().__init__()
        self.next_stream = None

    async def generate_content_stream(self, **kwargs):
        await asyncio.sleep(0)
        return _AsyncGeminiChunkIterator(self.next_stream)


class _FakeAsyncGeminiNamespaceWithStream:
    def __init__(self): self.models = _FakeAsyncGeminiModelsWithStream()


class _FakeGeminiClientWithAioStream:
    __module__ = "google.genai.client"
    def __init__(self):
        self.models = _FakeGeminiModelsWithStream()
        self.aio = _FakeAsyncGeminiNamespaceWithStream()


def test_gemini_async_stream_records_once_with_cumulative_usage(tmp_path):
    """`await client.aio.models.generate_content_stream(...)` then `async
    for` over the result — async mirror of
    `test_gemini_stream_records_once_with_cumulative_usage_last_wins`."""
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithAioStream()
    wrapped = t.wrap(client)
    chunks = [
        _GeminiChunk("Hi", _GeminiUsage(4, 1, 5)),
        _GeminiChunk("!", _GeminiUsage(4, 2, 6)),
    ]
    client.aio.models.next_stream = chunks

    async def run():
        stream = await wrapped.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hi async")
        return [c async for c in stream]

    received = asyncio.run(run())
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_token_count": 4, "candidates_token_count": 2, "total_token_count": 6,
    }
    ct.close()


def test_gemini_async_stream_fail_open_accumulate_raises(tmp_path, monkeypatch):
    """Async mirror of the sync fail-open accumulate test."""
    from ctxdiff.capture.gemini import GeminiAdapter
    t = trace.init("agent", path=str(tmp_path / "r.ctrace"))
    client = _FakeGeminiClientWithAioStream()
    wrapped = t.wrap(client)
    chunks = [_GeminiChunk("hi", _GeminiUsage(1, 1, 2))]
    client.aio.models.next_stream = chunks
    monkeypatch.setattr(GeminiAdapter, "accumulate_stream_usage",
                        lambda self, chunk, state: (_ for _ in ()).throw(RuntimeError("boom")))

    async def run():
        stream = await wrapped.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hi")
        return [c async for c in stream]

    received = asyncio.run(run())
    assert received == chunks

    t.close()
    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    assert ct.get_calls()[0].usage is None
    ct.close()

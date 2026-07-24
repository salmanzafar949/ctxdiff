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

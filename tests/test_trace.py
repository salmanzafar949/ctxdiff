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

from ctxdiff.capture.openai import OpenAIAdapter
from ctxdiff.capture.recorder import Recorder
from ctxdiff.models import Block
from ctxdiff.store.ctrace import CTrace


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


def _trace(tmp_path):
    return CTrace.create(str(tmp_path / "r.ctrace"), "agent", "openai", "gpt-4o")


def test_record_persists_blocks_with_tokens_and_labels(tmp_path):
    """record() runs the adapter, counts tokens, labels blocks, and stores a
    call readable via the store API. Token method for OpenAI is 'tiktoken'."""
    ct = _trace(tmp_path)
    rec = Recorder(ct, OpenAIAdapter(), redact=None)
    kwargs = {"model": "gpt-4o", "messages": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hi"}]}
    rec.record(seq=1, kwargs=kwargs, response=_Resp(), latency_ms=5, error=None, tagged=[])
    calls = ct.get_calls()
    assert len(calls) == 1 and calls[0].usage["total_tokens"] == 4
    blocks = ct.get_call_blocks(calls[0].id)
    assert [b.label for b in blocks] == ["system", "user"]
    assert all(b.block.token_method == "tiktoken" for b in blocks)
    assert all(b.block.token_count > 0 for b in blocks)
    ct.close()


def test_record_applies_redaction_before_store(tmp_path):
    """A redact hook rewrites block text before it is written — the stored text
    is the redacted text, never the original."""
    ct = _trace(tmp_path)
    def redact(block: Block) -> Block:
        return Block(block.content_hash, block.role, block.kind,
                     "[REDACTED]", block.token_count, block.token_method)
    rec = Recorder(ct, OpenAIAdapter(), redact=redact)
    rec.record(1, {"messages": [{"role": "user", "content": "secret"}]}, _Resp(), 1, None, [])
    blocks = ct.get_call_blocks(ct.get_calls()[0].id)
    assert blocks[0].block.text == "[REDACTED]"
    ct.close()


def test_record_is_fail_open_on_adapter_error(tmp_path):
    """If anything inside recording raises, record() swallows it (returns None,
    does not propagate) so the host app is never affected. Here a broken adapter
    raises, yet record() does not."""
    ct = _trace(tmp_path)
    class BrokenAdapter:
        provider = "openai"
        create_path = ("chat", "completions", "create")
        def extract_blocks(self, kwargs): raise RuntimeError("boom")
        def extract_params(self, kwargs): return {}
        def extract_usage(self, response): return None
    rec = Recorder(ct, BrokenAdapter(), redact=None)
    # Must not raise:
    rec.record(1, {"messages": []}, _Resp(), 1, None, [])
    ct.close()


def test_record_stores_error_calls(tmp_path):
    """When the host LLM call failed (response=None, error set), the call is
    still recorded with its error and whatever blocks were in the request."""
    ct = _trace(tmp_path)
    rec = Recorder(ct, OpenAIAdapter(), redact=None)
    rec.record(1, {"messages": [{"role": "user", "content": "hi"}]},
               response=None, latency_ms=9, error="RateLimitError", tagged=[])
    call = ct.get_calls()[0]
    assert call.error == "RateLimitError" and call.usage is None
    ct.close()

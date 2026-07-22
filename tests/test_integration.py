"""A realistic multi-turn run: a stable system prompt, growing history, and a
RAG chunk injected on turn 2, captured end-to-end through the public API."""
from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


class _Resp:
    class usage:  # noqa: N801
        prompt_tokens = 10; completion_tokens = 2; total_tokens = 12


class _Completions:
    def create(self, **kwargs): return _Resp()
class _Chat:
    def __init__(self): self.completions = _Completions()
class _OpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _Chat()


def test_multi_turn_run_captures_and_dedups(tmp_path):
    path = str(tmp_path / "run.ctrace")
    t = trace.init("support-agent", path=path)
    client = t.wrap(_OpenAI())

    system = {"role": "system", "content": "You are a support agent. Be precise."}

    # turn 1: system + first user question
    client.chat.completions.create(model="gpt-4o", messages=[
        system, {"role": "user", "content": "What's your refund window?"}])

    # turn 2: same system (should dedup), history grows, a RAG chunk is injected
    t.tag("rag", ["Refund policy: 30 days"])
    client.chat.completions.create(model="gpt-4o", messages=[
        system,
        {"role": "user", "content": "What's your refund window?"},
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "Context: Refund policy: 30 days. Answer now."}])
    t.close()

    ct = CTrace.open(path)
    calls = ct.get_calls()
    assert [c.seq for c in calls] == [1, 2]

    # The system prompt is identical across turns → stored once.
    n_system = ct._conn.execute(
        "SELECT COUNT(*) FROM block WHERE role='system'").fetchone()[0]
    assert n_system == 1

    # Turn 2's RAG-tagged block is labeled from the tag, not the role heuristic.
    t2_blocks = ct.get_call_blocks(calls[1].id)
    rag = [b for b in t2_blocks if b.label == "rag"]
    assert len(rag) == 1 and rag[0].label_source == "tagged"

    # Provider-reported usage is captured.
    assert calls[0].usage["total_tokens"] == 12
    ct.close()

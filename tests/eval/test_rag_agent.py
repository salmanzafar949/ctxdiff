"""THE agent evaluation: a realistic, self-contained multi-turn RAG support
agent, built on a real `openai.OpenAI` client wrapped by ctxdiff, HTTP fully
stubbed. This is ctxdiff's headline "evaluate a real agent" proof — it drives
the full capture lifecycle (growing history, stable system prompt, a tagged
RAG chunk) across three turns and then verifies everything the tracer claims
to do actually landed in the `.ctrace` file."""
from __future__ import annotations

import httpx
import openai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_openai_response

# A STABLE system prompt, reused VERBATIM on every turn — the exact scenario
# ctxdiff's content-addressed block store exists to dedup: the same system
# text should be written to the `block` table exactly once, no matter how
# many calls reference it.
SYSTEM_PROMPT = (
    "You are a helpful customer support agent for Acme Corp. "
    "Answer concisely and cite sources when provided."
)

# A tiny, deterministic "corpus" standing in for a real vector store. Keyed
# by the term the FOLLOW-UP question actually uses ("refund"), not the
# original question's term ("return policy"), since retrieval in this test
# fires on turn 2's query, not turn 1's.
_CORPUS = {
    "refund": "Returns are accepted within 30 days of purchase with a valid receipt.",
    "shipping": "Standard shipping takes 3-5 business days within the continental US.",
}


def retrieve(query: str) -> str:
    """Stand-in for a real retriever. What: returns the single corpus chunk
    whose key is a case-insensitive substring of `query`, or an empty string
    on no match. How: this is intentionally naive keyword matching, not
    semantic search — the point of this test is exercising ctxdiff's capture
    of a RAG-injected chunk, not retrieval quality, so determinism matters
    more than realism here."""
    q = query.lower()
    for key, chunk in _CORPUS.items():
        if key in q:
            return chunk
    return ""


def run_rag_conversation(wrapped_client, tracer) -> list[dict]:
    """Drive a 3-turn RAG support conversation through `wrapped_client` and
    return the final message history. What: turn 1 is a plain system+user
    question; turn 2 retrieves a doc chunk for the follow-up question, injects
    it as its own message ahead of the question, and tags it 'rag' via
    `tracer.tag()` before the call so the recorder labels that exact block;
    turn 3 is a further follow-up with no new retrieval, exercising a
    continuously growing history. How: `history` is mutated in place turn by
    turn — each call's `messages=history` snapshot is what actually gets sent
    (and therefore captured), mirroring how a real chat agent accumulates
    context."""
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Turn 1: plain system + user question, no retrieval yet.
    history.append({"role": "user", "content": "What is your return policy?"})
    resp1 = wrapped_client.chat.completions.create(model="gpt-4o", messages=history)
    history.append({"role": "assistant", "content": resp1.choices[0].message.content})

    # Turn 2: retrieve a chunk for the follow-up, inject it as its own
    # message immediately before the question, and tag it BEFORE the call so
    # the recorder attributes the tag to this turn only.
    chunk = retrieve("refund after 20 days")
    tracer.tag("rag", [chunk])
    history.append({"role": "user", "content": chunk})
    history.append({"role": "user", "content": "Can I get a refund after 20 days?"})
    resp2 = wrapped_client.chat.completions.create(model="gpt-4o", messages=history)
    history.append({"role": "assistant", "content": resp2.choices[0].message.content})

    # Turn 3: a further follow-up with no new retrieval — just growing
    # history, including the system prompt and the turn-2 chunk once more.
    history.append({"role": "user", "content": "And what about shipping?"})
    resp3 = wrapped_client.chat.completions.create(model="gpt-4o", messages=history)
    history.append({"role": "assistant", "content": resp3.choices[0].message.content})

    return history


def test_rag_agent_full_lifecycle(respx_mock, tmp_ctrace_path):
    """Run the 3-turn RAG conversation and verify the full capture lifecycle
    from a freshly reopened `.ctrace`: calls recorded in turn order with
    per-turn usage; the stable system prompt stored exactly ONCE in the
    content-addressed `block` table despite appearing in every turn's
    messages (the dedup mechanism); and the turn-2 RAG chunk labeled 'rag'/
    'tagged' specifically on turn 2's call_block row."""
    # Three distinct canned responses, returned in call order via side_effect
    # — respx supports a list of responses for successive requests to the
    # same route, which is exactly a multi-turn conversation's shape.
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=canned_openai_response(
                content="Sure — could you tell me your order date?",
                prompt_tokens=20, completion_tokens=8)),
            httpx.Response(200, json=canned_openai_response(
                content="Unfortunately 20 days is past our 30-day window... "
                        "actually no, 20 < 30, so yes, you're eligible.",
                prompt_tokens=45, completion_tokens=15)),
            httpx.Response(200, json=canned_openai_response(
                content="Standard shipping takes 3-5 business days.",
                prompt_tokens=60, completion_tokens=10)),
        ]
    )

    tracer = trace.init(project="rag-agent", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    run_rag_conversation(wrapped, tracer)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()

    # 1. Call sequence order.
    assert [c.seq for c in calls] == [1, 2, 3]

    # 2. Per-turn usage captured, distinct per call.
    assert calls[0].usage == {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    assert calls[1].usage == {"prompt_tokens": 45, "completion_tokens": 15, "total_tokens": 60}
    assert calls[2].usage == {"prompt_tokens": 60, "completion_tokens": 10, "total_tokens": 70}

    # 3. The stable system prompt is stored exactly ONCE in the block table,
    # even though every one of the 3 calls' messages included it verbatim.
    # This queries the store's raw connection directly (not get_call_blocks,
    # which reconstructs per-call views) to prove the dedup happened at the
    # storage layer itself.
    (system_block_count,) = ct._conn.execute(
        "SELECT COUNT(*) FROM block WHERE role = 'system'").fetchone()
    assert system_block_count == 1

    # 4. The turn-2 RAG chunk block is labeled 'rag'/'tagged' on turn 2's
    # call_block row specifically (not globally on the block itself — labels
    # are per-membership, so this must be checked via call 2's blocks).
    chunk_text = _CORPUS["refund"]
    turn2_blocks = ct.get_call_blocks(calls[1].id)
    rag_blocks = [cb for cb in turn2_blocks if cb.block.text == chunk_text]
    assert len(rag_blocks) == 1
    assert rag_blocks[0].label == "rag"
    assert rag_blocks[0].label_source == "tagged"

    ct.close()

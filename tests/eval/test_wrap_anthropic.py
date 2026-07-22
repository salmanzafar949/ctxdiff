"""Wrapping a REAL `anthropic.Anthropic` client end-to-end — the same shape as
`test_wrap_openai.py`, but proving the two adapters correctly diverge where
they must: `system` as a top-level kwarg (not a message) and `token_method`
routed to the character-based estimate, since Anthropic ships no public local
tokenizer (spike §3)."""
from __future__ import annotations

import anthropic
import httpx

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_anthropic_response


def test_wrap_anthropic_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """Wrap a real Anthropic client, make one stubbed system+user call, reopen
    the `.ctrace`, and assert: one call recorded, usage matches the canned
    response's input/output token fields exactly (Anthropic's usage shape,
    distinct from OpenAI's prompt/completion/total), blocks match the system
    kwarg and user message, and `token_method == 'estimate'` (the regression
    guard the spike calls out: this is the one place the two adapters' output
    shape *must* differ)."""
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=canned_anthropic_response(
            text="Hello there!", input_tokens=12, output_tokens=6)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = anthropic.Anthropic(api_key="x")
    wrapped = tracer.wrap(client)

    resp = wrapped.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        system="You are helpful.",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.content[0].text == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1

    call = calls[0]
    assert call.usage == {"input_tokens": 12, "output_tokens": 6}

    blocks = ct.get_call_blocks(call.id)
    assert len(blocks) == 2
    assert blocks[0].position == 0
    assert blocks[0].block.role == "system"
    assert blocks[0].block.text == "You are helpful."
    assert blocks[0].block.token_method == "estimate"
    assert blocks[1].position == 1
    assert blocks[1].block.role == "user"
    assert blocks[1].block.text == "hi"
    assert blocks[1].block.token_method == "estimate"
    ct.close()

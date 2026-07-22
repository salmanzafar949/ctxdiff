"""Wrapping a REAL `openai.OpenAI` client end-to-end: real SDK object, real
`_ClientProxy` interception, HTTP fully stubbed by respx, then reopen the
written `.ctrace` file to assert what actually landed on disk (spike §3)."""
from __future__ import annotations

import httpx
import openai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_openai_response


def test_wrap_openai_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """Wrap a real OpenAI client, make one stubbed system+user call through it,
    then reopen the `.ctrace` and assert the full capture shape: exactly one
    call recorded, usage matches the canned response exactly, blocks match
    the input messages' roles/texts/positions, and `token_method` is
    'tiktoken' (OpenAI's exact local tokenizer path, not the estimate)."""
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=canned_openai_response(
            content="Hello there!", prompt_tokens=10, completion_tokens=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    resp = wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    )
    # The real response object flows through untouched — the interceptor
    # never alters what the host application sees.
    assert resp.choices[0].message.content == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1

    call = calls[0]
    assert call.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    blocks = ct.get_call_blocks(call.id)
    assert len(blocks) == 2
    assert blocks[0].position == 0
    assert blocks[0].block.role == "system"
    assert blocks[0].block.text == "You are helpful."
    assert blocks[0].block.token_method == "tiktoken"
    assert blocks[1].position == 1
    assert blocks[1].block.role == "user"
    assert blocks[1].block.text == "hi"
    assert blocks[1].block.token_method == "tiktoken"
    ct.close()

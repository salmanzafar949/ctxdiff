"""Wrapping a REAL `openai.OpenAI` client pointed at an OSS/Ollama-style
OpenAI-compatible endpoint (`base_url="http://localhost:11434/v1"`). Same
shape as `test_wrap_openai.py` on purpose: this proves OSS models "just work"
via the OpenAI adapter — the client class/module (and therefore the detected
adapter) is identical to real OpenAI, only the `base_url` differs, and the
adapter's capture logic is entirely wire-protocol-based, not host-based."""
from __future__ import annotations

import httpx
import openai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_openai_response


def test_wrap_oss_openai_compatible_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """Wrap a real OpenAI client aimed at a local Ollama-style endpoint,
    make one stubbed call for an OSS model (`llama3`), reopen the `.ctrace`,
    and assert the same capture shape as real OpenAI: one call, matching
    usage, matching blocks, `token_method == 'tiktoken'` (the adapter has no
    way to know the model is non-OpenAI, so it applies its usual counting —
    an accepted approximation for OSS models, not a bug)."""
    respx_mock.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=canned_openai_response(
            content="Hello there!", prompt_tokens=10, completion_tokens=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x", base_url="http://localhost:11434/v1")
    wrapped = tracer.wrap(client)

    resp = wrapped.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    )
    assert resp.choices[0].message.content == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1

    call = calls[0]
    assert call.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    blocks = ct.get_call_blocks(call.id)
    assert len(blocks) == 2
    assert blocks[0].block.role == "system"
    assert blocks[0].block.text == "You are helpful."
    assert blocks[0].block.token_method == "tiktoken"
    assert blocks[1].block.role == "user"
    assert blocks[1].block.text == "hi"
    assert blocks[1].block.token_method == "tiktoken"
    ct.close()

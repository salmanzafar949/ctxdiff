"""Wrapping a REAL `google.genai.Client` end-to-end: real SDK object, real
`_ClientProxy` interception via the dotted-prefix ('google.genai') detection
path, HTTP fully stubbed by respx (google-genai's `_api_client.py` builds its
sync transport on `httpx.Client` — see `SyncHttpxClient` — so respx's httpx
transport mock intercepts it exactly like it does for openai/anthropic; no
mock.patch of the transport layer was needed), then reopen the written
`.ctrace` file to assert what actually landed on disk (Step-0 probe)."""
from __future__ import annotations

import httpx
from google import genai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_gemini_response


def test_wrap_gemini_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """Wrap a real `genai.Client`, make one stubbed system+user call through
    `models.generate_content`, then reopen the `.ctrace` and assert the full
    capture shape: exactly one call recorded, blocks ordered system-first
    then user, usage matches the canned response exactly (Gemini's
    prompt/candidates/total shape), and `token_method == 'estimate'` — gemini
    is not 'openai', so `count_tokens` routes to the character-based estimate,
    never tiktoken."""
    respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*generateContent.*"
    ).mock(return_value=httpx.Response(200, json=canned_gemini_response(
        text="Hello there!", prompt_token_count=10, candidates_token_count=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = genai.Client(api_key="x")
    wrapped = tracer.wrap(client)

    resp = wrapped.models.generate_content(
        model="gemini-2.0-flash",
        contents="hi",
        config={"system_instruction": "You are helpful."},
    )
    # The real response object flows through untouched — the interceptor
    # never alters what the host application sees.
    assert resp.text == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_run().provider == "gemini"
    calls = ct.get_calls()
    assert len(calls) == 1

    call = calls[0]
    assert call.usage == {
        "prompt_token_count": 10,
        "candidates_token_count": 5,
        "total_token_count": 15,
    }

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

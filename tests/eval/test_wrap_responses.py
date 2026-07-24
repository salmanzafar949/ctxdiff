"""Wrapping a REAL `openai.OpenAI` client's `.responses.create(...)` end-to-end:
real SDK object, real `_ClientProxy` multi-path interception, HTTP fully
stubbed by respx, then reopen the written `.ctrace` file to assert what
actually landed on disk (Step-0 probe). Mirrors `test_wrap_openai.py`'s shape
but drives the Responses API — the OpenAI Agents SDK's own completion
method — proving the second `create_paths` entry is captured for real, not
just against the fakes in `tests/test_trace.py`. Includes one async variant
(via `AsyncOpenAI` + `asyncio.run()`) proving the async/Responses combination
composes for free, per `_make_interceptor`'s call-time awaitable detection."""
from __future__ import annotations

import asyncio

import httpx
import openai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_responses_response


def test_wrap_openai_responses_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """Wrap a real OpenAI client, make one stubbed instructions+input call
    with a flat tool through `.responses.create(...)`, then reopen the
    `.ctrace` and assert the full capture shape: exactly one call recorded,
    blocks ordered instructions -> tool_schema -> input, usage matches the
    canned response's input_tokens/output_tokens/total_tokens exactly, and
    params carry `model` (never the content-bearing input/instructions/tools
    keys)."""
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=canned_responses_response(
            text="Hello there!", input_tokens=10, output_tokens=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    resp = wrapped.responses.create(
        model="gpt-4o",
        instructions="You are helpful.",
        input="hi",
        tools=[{"type": "function", "name": "lookup",
                "parameters": {"type": "object", "properties": {}}}],
    )
    # The real response object flows through untouched — the interceptor
    # never alters what the host application sees.
    assert resp.output[0].content[0].text == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1

    call = calls[0]
    assert call.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert call.params.get("model") == "gpt-4o"
    for leaked_key in ("input", "instructions", "tools"):
        assert leaked_key not in call.params

    blocks = ct.get_call_blocks(call.id)
    assert len(blocks) == 3
    assert blocks[0].position == 0
    assert blocks[0].block.role == "system"
    assert blocks[0].block.kind == "message"
    assert blocks[0].block.text == "You are helpful."
    assert blocks[1].position == 1
    assert blocks[1].block.kind == "tool_schema"
    assert "lookup" in blocks[1].block.text
    assert blocks[2].position == 2
    assert blocks[2].block.role == "user"
    assert blocks[2].block.kind == "message"
    assert blocks[2].block.text == "hi"
    ct.close()


def test_wrap_async_openai_responses_captures_call(respx_mock, tmp_ctrace_path):
    """Same call through a real `AsyncOpenAI` client via `asyncio.run()` —
    proves Responses-API capture composes with async client interception with
    no extra wiring (the async closure in `_make_interceptor` is path- and
    provider-agnostic)."""
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=canned_responses_response(
            text="Hi async!", input_tokens=8, output_tokens=4)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.AsyncOpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    resp = asyncio.run(wrapped.responses.create(model="gpt-4o", input="hi async"))
    assert resp.output[0].content[0].text == "Hi async!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
    blocks = ct.get_call_blocks(calls[0].id)
    assert len(blocks) == 1
    assert blocks[0].block.text == "hi async"
    ct.close()

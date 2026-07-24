"""Streaming-usage capture end-to-end against REAL SDK objects: real
`openai.OpenAI`/`AsyncOpenAI`/`anthropic.Anthropic` clients, HTTP fully
stubbed by respx as genuine Server-Sent-Events bodies (same technique as
`test_langchain.py`'s streaming test — the precedent for stubbing SSE with
respx), then reopen the written `.ctrace` to assert usage actually landed.

Exact chunk/event shapes below are empirically confirmed against real
`openai` 2.47.0 / `anthropic` 0.118.0 SSE parsing (Phase 12 Step 0 probe —
see .superpowers/sdd/phase12-streaming-report.md): OpenAI chat only reports
usage on a synthetic FINAL chunk (`choices: []`, `usage: {...}`), and ONLY
when the caller passes `stream_options={"include_usage": True}` — ctxdiff
never injects that itself, so a caller who doesn't opt in gets `usage=None`,
tested here too as the documented, honest gap. OpenAI Responses and
Anthropic streams report usage unconditionally, no caller opt-in needed."""
from __future__ import annotations

import asyncio

import httpx
import openai
import anthropic

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


def test_wrap_openai_chat_stream_with_include_usage_captures_usage(respx_mock, tmp_ctrace_path):
    """A real `OpenAI` client's `chat.completions.create(stream=True,
    stream_options={"include_usage": True})`: chunks stream through
    unaltered, and once the caller fully consumes the stream, exactly one
    call lands on disk with usage read off the synthetic final chunk."""
    sse_body = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there!"},'
        '"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,'
        '"completion_tokens":5,"total_tokens":15}}\n\n'
        'data: [DONE]\n\n'
    )
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        stream=True, stream_options={"include_usage": True})
    text = "".join(
        c.choices[0].delta.content
        for c in stream if c.choices and c.choices[0].delta.content)
    assert text == "Hello there!"  # real chunks, aggregated exactly as usual
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)  # request blocks still captured
    ct.close()


def test_wrap_openai_chat_stream_without_include_usage_records_usage_none(
        respx_mock, tmp_ctrace_path):
    """Same real stream, WITHOUT `stream_options={"include_usage": True}` —
    the documented, honest gap: ctxdiff never injects that opt-in itself
    (would alter the caller's own request), so no chunk ever carries usage
    and the call is recorded with usage=None, not a crash."""
    sse_body = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"Hi"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    )
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    stream = wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    list(stream)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


def test_wrap_async_openai_chat_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """Async mirror: a real `AsyncOpenAI` client streamed via `async for`,
    driven through `asyncio.run()` (no pytest-asyncio dependency, matching
    the rest of this suite's convention)."""
    sse_body = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"Hi"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":8,'
        '"completion_tokens":3,"total_tokens":11}}\n\n'
        'data: [DONE]\n\n'
    )
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.AsyncOpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    async def run():
        stream = await wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi async"}],
            stream=True, stream_options={"include_usage": True})
        return [c async for c in stream]

    chunks = asyncio.run(run())
    assert len(chunks) == 3
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11,
    }
    ct.close()


def test_wrap_openai_responses_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """A real `OpenAI` client's `responses.create(stream=True)`: the terminal
    `response.completed` event carries usage unconditionally (no caller
    opt-in needed, unlike Chat Completions)."""
    sse_body = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1",'
        '"object":"response","status":"in_progress","model":"gpt-4o","output":[]}}\n\n'
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1",'
        '"object":"response","status":"completed","model":"gpt-4o","output":'
        '[{"type":"message","id":"msg_1","role":"assistant","status":"completed",'
        '"content":[{"type":"output_text","text":"Hi","annotations":[]}]}],'
        '"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}\n\n'
    )
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    stream = wrapped.responses.create(model="gpt-4o", input="hi", stream=True)
    events = list(stream)
    assert any(getattr(e, "type", None) == "response.completed" for e in events)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)
    ct.close()


def test_wrap_anthropic_stream_captures_input_and_output_tokens(respx_mock, tmp_ctrace_path):
    """A real `Anthropic` client's `messages.create(stream=True)`: input
    tokens arrive on `message_start`, output tokens on `message_delta` — two
    separate events, both folded into ONE recorded call's usage."""
    sse_body = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message",'
        '"role":"assistant","model":"claude-opus-4-8","content":[],'
        '"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":12,"output_tokens":1}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        'event: content_block_stop\n'
        'data: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
        '"stop_sequence":null},"usage":{"output_tokens":6}}\n\n'
        'event: message_stop\n'
        'data: {"type":"message_stop"}\n\n'
    )
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = anthropic.Anthropic(api_key="x")
    wrapped = tracer.wrap(client)

    stream = wrapped.messages.create(
        model="claude-opus-4-8", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}], stream=True)
    events = list(stream)
    assert len(events) == 6
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    # Anthropic's usage shape carries no total_tokens (see AnthropicAdapter.extract_usage).
    assert calls[0].usage == {"input_tokens": 12, "output_tokens": 6}
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)
    ct.close()

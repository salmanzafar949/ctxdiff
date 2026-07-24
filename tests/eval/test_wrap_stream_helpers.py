"""`.stream()` convenience-helper capture, end-to-end against REAL SDK
objects: real `anthropic.Anthropic`/`AsyncAnthropic` and `openai.OpenAI`
clients, HTTP fully stubbed by respx as genuine Server-Sent-Events bodies
(same technique as `test_wrap_streaming.py`, its `create(stream=True)`
precedent), then reopen the written `.ctrace` to assert usage actually
landed.

Exact manager/stream shapes below are empirically confirmed against real
`anthropic` 0.118.0 / `openai` 2.47.0 (Phase 13 Step 0 probe — see
.superpowers/sdd/phase13-streamhelpers-report.md):

- `client.messages.stream(...)` / `client.chat.completions.stream(...)` /
  `client.responses.stream(...)` are ALL plain, non-coroutine methods — even
  on an async client — that make NO HTTP request themselves; they return a
  StreamManager whose `__enter__`/`__aenter__` is what actually fires the
  request. So `manager = wrapped...stream(...)` always happens outside
  `asyncio.run()` in the async tests below, exactly as a real caller would
  write it (no `await` before `.stream()`).
- Anthropic's `MessageStream` yields the SAME raw `message_start`/
  `message_delta`/... event objects a raw `create(stream=True)` stream
  yields, so `AnthropicAdapter.accumulate_stream_usage` needs no changes at
  all to also work here.
- OpenAI Responses' `ResponseStream` also re-fires the SAME
  `response.completed` event shape (with `.response.usage`) a raw
  `responses.create(stream=True)` stream carries — no changes needed either.
- OpenAI Chat's `ChatCompletionStream` wraps every real chunk one level
  deeper, in a `type == "chunk"` event whose `.chunk.usage` (not a top-level
  `.usage`) carries the caller-opted-in usage — `OpenAIAdapter.
  accumulate_stream_usage` gained a dedicated branch for this shape."""
from __future__ import annotations

import asyncio

import httpx
import openai
import anthropic

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


_ANTHROPIC_SSE = (
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


def test_wrap_anthropic_messages_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """`with client.messages.stream(...) as stream:` — the Anthropic-
    recommended convenience helper — records exactly once on block exit,
    with input tokens off `message_start` and output tokens off
    `message_delta`, same as the raw `create(stream=True)` path."""
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200, content=_ANTHROPIC_SSE, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = anthropic.Anthropic(api_key="x")
    wrapped = tracer.wrap(client)

    with wrapped.messages.stream(
            model="claude-opus-4-8", max_tokens=100,
            messages=[{"role": "user", "content": "hi"}]) as stream:
        events = list(stream)
    assert len(events) == 7  # every real event reaches the caller, including
    # the SDK-synthesized "text" convenience event alongside content_block_delta
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"input_tokens": 12, "output_tokens": 6}
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)  # request blocks still captured
    ct.close()


def test_wrap_async_anthropic_messages_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """`async with client.messages.stream(...) as stream:` on `AsyncAnthropic`
    — `.stream()` itself is called with NO `await` (confirmed non-coroutine),
    only the `async with`/`async for` below run inside `asyncio.run`."""
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200, content=_ANTHROPIC_SSE, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = anthropic.AsyncAnthropic(api_key="x")
    wrapped = tracer.wrap(client)

    manager = wrapped.messages.stream(
        model="claude-opus-4-8", max_tokens=100,
        messages=[{"role": "user", "content": "hi async"}])

    async def run():
        async with manager as stream:
            return [e async for e in stream]

    events = asyncio.run(run())
    assert len(events) == 7
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"input_tokens": 12, "output_tokens": 6}
    ct.close()


_OPENAI_CHAT_SSE_WITH_USAGE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant",'
    '"content":"Hello"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,'
    '"completion_tokens":5,"total_tokens":15}}\n\n'
    'data: [DONE]\n\n'
)


def test_wrap_openai_chat_completions_stream_with_include_usage_captures_usage(
        respx_mock, tmp_ctrace_path):
    """`with client.chat.completions.stream(...) as stream:` re-wraps every
    real chunk in a `ChunkEvent` (plus convenience `content.delta`/`content.
    done` events) — usage is confirmed to live at the FINAL chunk event's
    `.chunk.usage`, not a top-level `.usage` on the event itself, and ONLY
    when the caller opts in via `stream_options={"include_usage": True}` —
    the `.stream()` helper does NOT inject that itself (confirmed: it forwards
    kwargs verbatim, same fail-open contract as the raw path)."""
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=_OPENAI_CHAT_SSE_WITH_USAGE,
            headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    with wrapped.chat.completions.stream(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            stream_options={"include_usage": True}) as stream:
        events = list(stream)
    assert any(getattr(e, "type", None) == "chunk" for e in events)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)
    ct.close()


def test_wrap_openai_chat_completions_stream_without_include_usage_records_usage_none(
        respx_mock, tmp_ctrace_path):
    """Same helper, WITHOUT the caller's own opt-in — the documented, honest
    gap carries over unchanged to the `.stream()` manager path: no usage
    chunk is ever sent, so the call is recorded with usage=None, not a
    crash."""
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

    with wrapped.chat.completions.stream(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]) as stream:
        list(stream)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None
    ct.close()


def test_wrap_async_openai_chat_completions_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """Async mirror on `AsyncOpenAI` — `.stream()` itself is called with no
    `await` (confirmed non-coroutine even on the async client); only `async
    with`/`async for` run inside `asyncio.run`."""
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=_OPENAI_CHAT_SSE_WITH_USAGE,
            headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.AsyncOpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    manager = wrapped.chat.completions.stream(
        model="gpt-4o", messages=[{"role": "user", "content": "hi async"}],
        stream_options={"include_usage": True})

    async def run():
        async with manager as stream:
            return [e async for e in stream]

    events = asyncio.run(run())
    assert any(getattr(e, "type", None) == "chunk" for e in events)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }
    ct.close()


_OPENAI_RESPONSES_SSE = (
    'event: response.created\n'
    'data: {"type":"response.created","response":{"id":"resp_1",'
    '"object":"response","status":"in_progress","model":"gpt-4o","output":[]}}\n\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"resp_1",'
    '"object":"response","status":"completed","model":"gpt-4o","output":'
    '[{"type":"message","id":"msg_1","role":"assistant","status":"completed",'
    '"content":[{"type":"output_text","text":"Hi","annotations":[]}]}],'
    '"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}\n\n'
)


def test_wrap_openai_responses_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """`with client.responses.stream(...) as stream:` — the terminal
    `response.completed` event is re-fired with the SAME shape a raw
    `responses.create(stream=True)` call carries, so usage is captured
    unconditionally, no caller opt-in needed, unchanged from the raw path."""
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200, content=_OPENAI_RESPONSES_SSE, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(client)

    with wrapped.responses.stream(model="gpt-4o", input="hi") as stream:
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

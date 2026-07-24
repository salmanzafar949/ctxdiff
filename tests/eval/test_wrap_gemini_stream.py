"""`generate_content_stream` capture, end-to-end against a REAL
`google.genai.Client`, HTTP fully stubbed by respx as genuine
Server-Sent-Events (Phase 13 Step 0 probe — confirmed the exact stub below
against real `google-genai` 2.14.0), then reopen the written `.ctrace` to
assert usage actually landed.

Confirmed by the probe:
- `client.models.generate_content_stream(...)` returns a plain Python
  generator (sync) — no `__enter__`, has `.close()` — that lazily fires the
  real HTTP request on first iteration. Not a manager; routed through the
  EXISTING raw-stream `_StreamProxy` via `is_named_stream_method` (see
  `trace.py`'s `_ClientProxy.__getattr__`), same as
  `create(stream=True)` on the other providers.
- `client.aio.models.generate_content_stream(...)` IS itself a coroutine
  function (unlike Anthropic/OpenAI's `.stream()` manager helpers) that
  resolves, once awaited, to an async generator — `async for chunk in await
  client.aio.models.generate_content_stream(...)`.
- Each chunk's `usage_metadata` is CUMULATIVE — the last chunk already
  carries the full totals — so `GeminiAdapter.accumulate_stream_usage`
  overwrites `state` every chunk (last wins), never sums.
- The streaming endpoint is `...:streamGenerateContent` (with an
  `alt=sse` query param the SDK adds itself); stubbing via a URL regex that
  just requires the "streamGenerateContent" substring, same style as the
  existing `test_wrap_gemini.py`'s "generateContent" stub, works against the
  real SDK's sync `httpx.Client`-based transport with no extra mocking.
- Gemini is not 'openai', so block token counts land as `token_method ==
  'estimate'`, same as the non-streaming Gemini eval test."""
from __future__ import annotations

import asyncio

import httpx
from google import genai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


def _sse_body() -> str:
    """Two SSE chunks, one JSON object per `data:` line (no `event:` line —
    confirmed against the real SDK's SSE parsing), with CUMULATIVE
    `usageMetadata` — the second chunk's counts already include the first's."""
    chunk1 = (
        '{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},'
        '"index":0}],"usageMetadata":{"promptTokenCount":10,'
        '"candidatesTokenCount":1,"totalTokenCount":11},"modelVersion":"gemini-2.0-flash"}'
    )
    chunk2 = (
        '{"candidates":[{"content":{"parts":[{"text":" there!"}],"role":"model"},'
        '"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,'
        '"candidatesTokenCount":5,"totalTokenCount":15},"modelVersion":"gemini-2.0-flash"}'
    )
    return f"data: {chunk1}\n\ndata: {chunk2}\n\n"


def test_wrap_gemini_generate_content_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """`for chunk in client.models.generate_content_stream(...):` — chunks
    reach the caller unaltered, and the call is recorded once complete with
    the LAST chunk's cumulative usage (not a sum of the two chunks')."""
    respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*streamGenerateContent.*"
    ).mock(return_value=httpx.Response(
        200, content=_sse_body(), headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = genai.Client(api_key="x")
    wrapped = tracer.wrap(client)

    stream = wrapped.models.generate_content_stream(
        model="gemini-2.0-flash", contents="hi")
    chunks = list(stream)
    assert len(chunks) == 2
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_run().provider == "gemini"
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15,
    }
    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)  # request blocks still captured
    assert all(b.block.token_method == "estimate" for b in blocks)
    ct.close()


def test_wrap_async_gemini_generate_content_stream_captures_usage(respx_mock, tmp_ctrace_path):
    """`async for chunk in await client.aio.models.generate_content_stream(
    ...):` — the async mirror, driven through `asyncio.run()` (no
    pytest-asyncio dependency, matching the rest of this suite)."""
    respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*streamGenerateContent.*"
    ).mock(return_value=httpx.Response(
        200, content=_sse_body(), headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = genai.Client(api_key="x")
    wrapped = tracer.wrap(client)

    async def run():
        stream = await wrapped.aio.models.generate_content_stream(
            model="gemini-2.0-flash", contents="hi async")
        return [c async for c in stream]

    chunks = asyncio.run(run())
    assert len(chunks) == 2
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15,
    }
    ct.close()

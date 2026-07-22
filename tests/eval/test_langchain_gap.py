"""Encodes TODAY'S ground truth for LangChain's `ChatOpenAI` as regression
tests (spike §4). Unlike every other file in this suite, these tests assert
the DOCUMENTED LIMITATION, not a happy path — they exist to fail loudly if the
gap is ever silently fixed or silently worsened, so a future contributor who
changes `_ClientProxy` or the interceptor has to consciously update this file
rather than accidentally gaining/losing capability unnoticed."""
from __future__ import annotations

import httpx
import openai
import pytest

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

langchain_openai = pytest.importorskip("langchain_openai")

from .conftest import canned_openai_response  # noqa: E402 — after importorskip


def test_wrap_chatopenai_raises(tmp_ctrace_path):
    """`tracer.wrap()` on a `ChatOpenAI` instance must raise `ValueError` —
    `_detect_provider` doesn't recognize `langchain_openai` as a supported
    client module (spike §1/§4a). This is the fail-loud-at-setup-time
    behavior the module docstring promises; no HTTP is involved."""
    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(api_key="x", model="gpt-4o")
    with pytest.raises(ValueError, match="unrecognized client module"):
        tracer.wrap(llm)


def test_client_injection_default_invoke_not_captured(respx_mock, tmp_ctrace_path):
    """Client-injection wiring (`client=wrapped.chat.completions,
    root_client=wrapped`) is mechanically sound — pydantic accepts the proxy
    with no validation error, and `.invoke()` SUCCEEDS end-to-end against the
    stub — but the call is NOT captured: `len(ct.get_calls()) == 0`.

    ROOT CAUSE (spike §4, confirmed by reading langchain_openai 1.4.0's
    `ChatOpenAI._generate()`, base.py ~line 1733): the default/non-streaming
    invoke path calls `self.client.with_raw_response.create(**payload)`, NOT
    `self.client.create(**payload)`. `ctxdiff._ClientProxy.__getattr__` only
    keeps wrapping when the traversed attribute path stays an exact prefix of
    the adapter's fixed `create_path` tuple `("chat","completions","create")`.
    `with_raw_response` extends the path to `("chat","completions",
    "with_raw_response")`, which is NOT a prefix of `create_path`, so the
    proxy's prefix check fails and `__getattr__` falls through to returning
    the REAL, unwrapped `CompletionsWithRawResponse` object straight off the
    underlying SDK client — the interceptor is never installed on it, so
    `tracer._on_create` is never called for this path.

    THIS TEST MUST BE FLIPPED GREEN (its `== 0` assertion changed to `== 1`,
    deliberately, with a comment explaining why) by whoever fixes
    `_ClientProxy` to treat `with_raw_response`/`with_streaming_response` as
    transparent infixes rather than path-breaking hops. Until then, this
    assertion is the correct, intentional ground truth."""
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=canned_openai_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    oa = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(oa)
    llm = langchain_openai.ChatOpenAI(
        client=wrapped.chat.completions, root_client=wrapped,
        model="gpt-4o", api_key="x")

    result = llm.invoke("hi")
    assert result.content == "Hello there!"  # the wiring itself isn't broken
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert len(ct.get_calls()) == 0  # ...but ctxdiff genuinely misses it
    ct.close()


def test_client_injection_streaming_invoke_is_captured(respx_mock, tmp_ctrace_path):
    """With `streaming=True`, `ChatOpenAI._stream()` (the method backing
    `.invoke()` in this mode) calls `self.client.create(**payload)` directly —
    no `with_raw_response` hop — which DOES match the proxy's `create_path`
    exactly, so the interceptor IS installed and the call IS captured.

    This is the one LangChain invocation path client-injection captures
    today, but with a caveat this test also locks in: `real_create(...)`
    returns an `openai.Stream[ChatCompletionChunk]` iterator (not a completed
    response) at the moment the interceptor's `tracer._on_create` runs, so
    `OpenAIAdapter.extract_usage`'s `getattr(response, "usage", None)` finds
    nothing — `call.usage` is `None` even though the call itself succeeded
    and blocks were recorded. A future fix for the streaming-usage gap (e.g.
    wrapping the iterator to capture usage from its final chunk) should
    change this test's `is None` assertion deliberately, not by accident.

    How the canned response is built: the openai SDK's streaming client
    parses a Server-Sent-Events body (`data: <json>\\n\\n` per chunk,
    terminated by `data: [DONE]\\n\\n`) into `ChatCompletionChunk` objects.
    Each chunk here is the minimal valid shape for that pydantic model: an
    incremental `delta` (role+content split across chunks, as OpenAI's real
    API does) followed by a final chunk carrying `finish_reason`."""
    sse_body = (
        'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
        '"created":1700000000,"model":"gpt-4o","choices":[{"index":0,'
        '"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
        '"created":1700000000,"model":"gpt-4o","choices":[{"index":0,'
        '"delta":{"content":" there!"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
        '"created":1700000000,"model":"gpt-4o","choices":[{"index":0,'
        '"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    )
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    oa = openai.OpenAI(api_key="x")
    wrapped = tracer.wrap(oa)
    llm = langchain_openai.ChatOpenAI(
        client=wrapped.chat.completions, root_client=wrapped,
        model="gpt-4o", api_key="x", streaming=True)

    result = llm.invoke("hi")
    assert result.content == "Hello there!"  # chunks aggregated correctly
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None  # locks in the streaming-usage gap

    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)
    ct.close()

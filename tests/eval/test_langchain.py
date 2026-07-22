"""Encodes TODAY'S ground truth for LangChain's `ChatOpenAI` as regression
tests (spike §4). The default (non-streaming) `.invoke()` gap is now CLOSED:
`_ClientProxy` treats `with_raw_response`/`with_streaming_response` as
transparent hops (see `trace._TRANSPARENT_HOPS`), so the call IS captured,
WITH usage (via `extract_usage`'s `.parse()` fallback on the raw-response
wrapper). The streaming-usage gap remains OPEN and is intentionally still
asserted here as a DOCUMENTED LIMITATION (a v1 non-goal — the streaming path
hands the interceptor a `Stream` iterator with no `.parse()`, so usage stays
unrecoverable at call time). These tests exist to fail loudly if either
behavior is ever silently changed, so a future contributor who touches
`_ClientProxy`/the interceptor/`extract_usage` has to consciously update this
file rather than accidentally gaining/losing capability unnoticed."""
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


def test_client_injection_default_invoke_is_captured(respx_mock, tmp_ctrace_path):
    """Client-injection wiring (`client=wrapped.chat.completions,
    root_client=wrapped`) is mechanically sound — pydantic accepts the proxy
    with no validation error, and `.invoke()` SUCCEEDS end-to-end against the
    stub — and, as of the transparent-hop fix, the call IS captured, WITH
    usage: `len(ct.get_calls()) == 1` and `call.usage` reflects the canned
    token counts.

    PREVIOUS ROOT CAUSE, NOW FIXED (spike §4, confirmed by reading
    langchain_openai 1.4.0's `ChatOpenAI._generate()`, base.py ~line 1733):
    the default/non-streaming invoke path calls
    `self.client.with_raw_response.create(**payload)`, NOT
    `self.client.create(**payload)`. `ctxdiff._ClientProxy.__getattr__` used
    to only keep wrapping when the traversed attribute path stayed an exact
    prefix of the adapter's fixed `create_path` tuple
    `("chat","completions","create")` — `with_raw_response` extended the
    path to `("chat","completions","with_raw_response")`, not a prefix of
    `create_path`, so the proxy fell through to the REAL, unwrapped
    `CompletionsWithRawResponse` object and the interceptor never ran.

    THE FIX (trace.py `_TRANSPARENT_HOPS`): `__getattr__` now recognizes
    `with_raw_response`/`with_streaming_response` as transparent when
    encountered exactly one step before `create` — it keeps wrapping WITHOUT
    advancing the tracked path, so the following `.create` access still
    lands on `create_path` and IS intercepted. And because
    `with_raw_response.create()` returns a raw-response wrapper with no
    `.usage` of its own, `OpenAIAdapter.extract_usage` now falls back to
    calling the wrapper's (memoized) `.parse()` and reading `.usage` off the
    parsed body — which is why `call.usage` is populated here, not None."""
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=canned_openai_response(prompt_tokens=10, completion_tokens=5)))

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
    calls = ct.get_calls()
    assert len(calls) == 1  # the gap is closed: the hop is now captured
    assert calls[0].usage == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }  # usage recovered via the raw-response .parse() fallback

    blocks = ct.get_call_blocks(calls[0].id)
    assert any(b.block.text == "hi" for b in blocks)  # user content captured too
    ct.close()


def test_client_injection_streaming_invoke_is_captured(respx_mock, tmp_ctrace_path):
    """With `streaming=True`, `ChatOpenAI._stream()` (the method backing
    `.invoke()` in this mode) calls `self.client.create(**payload)` directly —
    no `with_raw_response` hop — which DOES match the proxy's `create_path`
    exactly, so the interceptor IS installed and the call IS captured.

    This LangChain invocation path is captured (as is, now, the default
    non-streaming path — see `test_client_injection_default_invoke_is_
    captured`), but with a caveat this test locks in as a SEPARATE, STILL
    OPEN gap (a v1 non-goal, not addressed by the with_raw_response
    transparent-hop fix): `real_create(...)` returns an
    `openai.Stream[ChatCompletionChunk]` iterator (not a completed response,
    and not a raw-response wrapper with a `.parse()`) at the moment the
    interceptor's `tracer._on_create` runs, so `OpenAIAdapter.extract_usage`
    finds neither a direct `.usage` nor a `.parse()` to fall back to —
    `call.usage` is `None` even though the call itself succeeded and blocks
    were recorded. A future fix for the streaming-usage gap (e.g. wrapping
    the iterator to capture usage from its final chunk) should change this
    test's `is None` assertion deliberately, not by accident.

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

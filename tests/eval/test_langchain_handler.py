"""The LangChain CALLBACK HANDLER, exercised against real LangChain,
langchain-openai, langchain-anthropic and LangGraph with HTTP stubbed by
respx — the idiomatic capture path that replaces reaching into
`ChatOpenAI`'s internals (the legacy client-injection path, still covered by
`test_langchain.py`).

The load-bearing assertion in this file is HASH IDENTITY: for the same
logical request, the blocks the handler records must be byte-identical — same
role, kind, text, and therefore same content hash — to the blocks the direct
SDK-wrapping path records. Several tests prove it two different ways:

  * end to end, by running the same prompt through a handler-instrumented
    `ChatOpenAI` and a directly-wrapped `openai.OpenAI` in one tracer and
    comparing the two calls' hashes;
  * against the WIRE, by capturing the exact JSON body LangChain sent and
    feeding that body to the provider's own adapter — so the handler is
    checked against what actually went to the provider, not against another
    copy of ctxdiff's own opinion.

All four provider branches of `to_wire` are covered here against a real
integration and a real request body: OpenAI, Anthropic, Gemini
(langchain-google-genai) and Bedrock Converse (langchain-aws). That coverage
is not decoration — the two branches that had NO wire-body test are precisely
the two that shipped dropping every non-text content part (an image recorded
as no block at all) and merging several text parts into one.

Cross-SDK identity is pinned here as shared literal hashes the JS suite
asserts too, INCLUDING the one case where the two SDKs are known to diverge —
a tool call, whose arguments LangChain re-serializes with each language's own
JSON serializer. Pinning the divergence is the point: it can then never move
without both suites noticing.
"""
from __future__ import annotations

import base64
import json
import struct
import zlib

import httpx
import openai
import pytest

from ctxdiff import trace
from ctxdiff.capture.recorder import build_block
from ctxdiff.models import RawBlock
from ctxdiff.store.ctrace import CTrace

langchain_openai = pytest.importorskip("langchain_openai")
langchain_anthropic = pytest.importorskip("langchain_anthropic")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from .conftest import (canned_anthropic_response, canned_gemini_response,  # noqa: E402
                       canned_openai_response)

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_GEMINI_URL_RE = r"https://generativelanguage\.googleapis\.com/.*"


def _png(width: int, height: int) -> bytes:
    """A structurally valid PNG declaring `width`x`height` in its IHDR — the
    same hand-built header the unit suites use, since ctxdiff's sniffer only
    reads the first few dozen bytes and no binary fixture belongs in the
    repo."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    ihdr = struct.pack(">II", width, height) + bytes([8, 2, 0, 0, 0])
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00")) + chunk(b"IEND", b""))


PNG_4x4 = _png(4, 4)
PNG_4x4_URI = "data:image/png;base64," + base64.b64encode(PNG_4x4).decode("ascii")

#: The message every multimodal test below sends: TWO text parts and an image
#: in one turn — the two things the Gemini/Bedrock branches used to lose.
VISION_MESSAGE = HumanMessage(content=[
    {"type": "text", "text": "look at this:"},
    {"type": "text", "text": " carefully"},
    {"type": "image_url", "image_url": {"url": PNG_4x4_URI}},
])


def _hashes_for(wire: dict, provider: str = "openai") -> list[str]:
    """The content hashes a DIRECT capture of `wire` would store: the real
    provider adapter's blocks, hashed by the real `build_block`. Used to
    check the handler against the request that actually went on the wire."""
    from ctxdiff.capture.anthropic import AnthropicAdapter
    from ctxdiff.capture.openai import OpenAIAdapter

    adapter = AnthropicAdapter() if provider == "anthropic" else OpenAIAdapter()
    return [build_block(raw, provider).content_hash
            for raw in adapter.extract_blocks(wire)]


def _gemini_hashes_for(body: dict) -> list[str]:
    """The hashes a direct capture of the Gemini REST body would store.

    One translation is needed and only one: google-genai's `generate_content`
    takes `contents=` plus a `config=` bag, while the REST body it builds
    puts the system instruction at the top level as
    `systemInstruction.parts[].text`. The CONTENT itself — `contents`, the
    part list this test exists to check — is passed through verbatim, so what
    is compared is the provider's own view of the request."""
    from ctxdiff.capture.gemini import GeminiAdapter

    texts = [part.get("text", "") for part
             in (body.get("systemInstruction") or {}).get("parts", [])]
    kwargs = {"contents": body.get("contents")}
    if texts:
        kwargs["config"] = {"system_instruction": "\n".join(texts)}
    return [build_block(raw, "gemini").content_hash
            for raw in GeminiAdapter().extract_blocks(kwargs)]


def _bedrock_hashes_for(body: dict) -> list[str]:
    """The hashes a direct capture of the Converse request body would store.
    No translation at all here: botocore's `converse(**kwargs)` sends its
    kwargs as the JSON body, so the captured body IS the shape
    `BedrockAdapter` reads (only `modelId` moves, into the URL path, and it
    carries no blocks)."""
    from ctxdiff.capture.bedrock import BedrockAdapter

    return [build_block(raw, "bedrock").content_hash
            for raw in BedrockAdapter().extract_blocks(body)]


def test_handler_captures_a_chat_call_with_usage(respx_mock, tmp_ctrace_path):
    """The basic path: a `ChatOpenAI` with `callbacks=[handler]` records one
    call carrying the system and user blocks in order, the model in params,
    and usage in OpenAI's own key names — the same dict a directly-wrapped
    client stores."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, json=canned_openai_response(prompt_tokens=10, completion_tokens=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                      callbacks=[tracer.langchain_handler()])
    assert llm.invoke([("system", "You are helpful."), ("human", "hi")]).content \
        == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_run().provider == "openai"
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].provider == "openai"
    assert calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15}
    assert calls[0].params["model"] == "gpt-4o"
    assert calls[0].error is None
    assert calls[0].latency_ms is not None

    blocks = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.kind, b.block.text) for b in blocks] == [
        ("system", "message", "You are helpful."), ("user", "message", "hi")]
    ct.close()


def test_handler_blocks_are_hash_identical_to_direct_capture(respx_mock, tmp_ctrace_path):
    """THE headline guarantee, end to end. The same prompt is sent twice into
    ONE tracer — once through a handler-instrumented `ChatOpenAI`, once
    through a directly-wrapped `openai.OpenAI` — and the two recorded calls
    must carry exactly the same block hashes in the same order. If they ever
    diverge, a team running both paths would see one prompt as two unrelated
    contexts and every diff between them would be noise."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, json=canned_openai_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                      callbacks=[tracer.langchain_handler()])
    llm.invoke([("system", "You are helpful."), ("human", "hi")])

    client = tracer.wrap(openai.OpenAI(api_key="x"))
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are helpful."},
                  {"role": "user", "content": "hi"}])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    via_handler, via_wrap = ct.get_calls()
    assert ([b.block.content_hash for b in ct.get_call_blocks(via_handler.id)]
            == [b.block.content_hash for b in ct.get_call_blocks(via_wrap.id)])
    # ...and the usage dicts match too, since both go through `extract_usage`.
    assert via_handler.usage == via_wrap.usage
    ct.close()


def test_cross_sdk_hashes_are_pinned(respx_mock, tmp_ctrace_path):
    """CROSS-SDK identity, pinned as literals. The JS SDK's LangChain handler
    asserts these EXACT hashes for the same prompt (`js/test/langchain.test.
    ts`), which is what makes "a `.ctrace` written by either SDK dedups
    against the other" a test rather than a claim. A change to either
    normalizer that moves these has to move both, deliberately."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, json=canned_openai_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                      callbacks=[tracer.langchain_handler()])
    llm.invoke([("system", "You are helpful."), ("human", "hi")])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert [b.block.content_hash for b in ct.get_call_blocks(calls[0].id)] == [
        "5a3c8882fb013e887693ebf8ce6c593b4f4f4131edbaffff2b5c163b412aca1e",
        "4e6c4093072114cd3ec3641653e12f750391cded3515bf460ccd07162c647685",
    ]
    ct.close()


#: The `arguments` string LangChain PYTHON puts on the wire when it
#: re-serializes a normalized tool call, and the one @langchain/core puts on
#: the wire for the identical logical call. `json.dumps` emits `", "` and
#: `": "` separators; `JSON.stringify` emits none. Neither SDK is wrong — each
#: reproduces its own framework's real request byte for byte — so the two are
#: pinned here as a KNOWN, deliberate divergence rather than papered over.
PY_TOOL_ARGUMENTS = '{"city": "Dubai"}'
JS_TOOL_ARGUMENTS = '{"city":"Dubai"}'

#: The content hash each SDK stores for the assistant tool-call block of the
#: SAME logical message (`get_weather({"city": "Dubai"})`, id `call_1`). Both
#: literals are asserted by BOTH suites.
PY_TOOL_CALL_HASH = "aed9a5ef806c38eb945801ff65af115866c127e282ce357e253ef0942f7a9cbe"
JS_TOOL_CALL_HASH = "65b07374b968257020127bfbf113d7fad62b16abad9fc957d195a87674b4f8c1"


def test_cross_sdk_tool_call_hashes_are_pinned_as_known_divergent(tmp_ctrace_path):
    """The limit of cross-SDK hash identity, pinned so it cannot drift
    unnoticed — and so nobody re-discovers it in production.

    A tool call is the one block whose text the two SDKs CANNOT agree on.
    LangChain re-serializes a normalized `.tool_calls` entry with the host
    language's own JSON serializer: `json.dumps` in Python (`{"city":
    "Dubai"}`), `JSON.stringify` in JS (`{"city":"Dubai"}`). The handler
    reproduces its own framework's real wire byte for byte — that is what
    `test_langgraph.py`'s wire-body test asserts — so the same logical
    message hashes differently in the two SDKs. Normalizing the handler's
    output to a common form would fix the cross-SDK hash by breaking the
    in-SDK one: the recorded block would no longer match the body Python's
    LangChain actually sent, nor a direct capture of it, which is the
    stronger guarantee and the one the whole design rests on. (And it would
    not even make the two agree in general — a DIRECT capture of the two
    frameworks' own requests still differs by the same bytes, with no ctxdiff
    in the picture.)

    The JS suite asserts the mirror of this test, so a change on either side
    fails on both."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import LLMResult

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    handler = tracer.langchain_handler()
    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
        [[HumanMessage(content="weather in Dubai?"),
          AIMessage(content="", tool_calls=[
              {"name": "get_weather", "args": {"city": "Dubai"}, "id": "call_1"}])]],
        run_id="run-1", invocation_params={"model": "gpt-4o"})
    handler.on_llm_end(LLMResult(generations=[]), run_id="run-1")
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    blocks = ct.get_call_blocks(ct.get_calls()[0].id)
    call_block = blocks[1].block
    assert json.loads(call_block.text)["function"]["arguments"] == PY_TOOL_ARGUMENTS

    # The Python hash, pinned. The JS suite pins the JS one for the same
    # message and asserts, as here, that the two are DIFFERENT — so a change
    # to either normalizer fails on both sides rather than silently making
    # one SDK's traces stop deduping against the other's.
    assert call_block.content_hash == PY_TOOL_CALL_HASH

    js_block = build_block(
        RawBlock(role=call_block.role, kind=call_block.kind,
                 text=json.dumps({"id": "call_1", "type": "function",
                                  "function": {"name": "get_weather",
                                               "arguments": JS_TOOL_ARGUMENTS}},
                                 sort_keys=True, ensure_ascii=False)),
        "openai")
    assert js_block.content_hash == JS_TOOL_CALL_HASH
    assert JS_TOOL_CALL_HASH != PY_TOOL_CALL_HASH
    ct.close()


def test_handler_blocks_match_the_wire_body_langchain_actually_sent(
        respx_mock, tmp_ctrace_path):
    """Checked against the PROVIDER's view rather than ctxdiff's: the exact
    JSON body LangChain put on the wire is captured from the stub and fed to
    the OpenAI adapter, and the handler's stored hashes must equal what a
    direct capture of that body would have produced. This is what catches a
    normalization that is self-consistent but wrong."""
    sent: list[dict] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=canned_openai_response())

    respx_mock.post(_OPENAI_URL).mock(side_effect=_respond)

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                      callbacks=[tracer.langchain_handler()])
    llm.invoke([("system", "Be terse."), ("human", "explain hashing")])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert [b.block.content_hash for b in ct.get_call_blocks(calls[0].id)] \
        == _hashes_for(sent[0])
    ct.close()


def test_handler_captures_streaming_with_usage(respx_mock, tmp_ctrace_path):
    """Streaming needs no special handling on this path — LangChain fires
    `on_llm_end` once the stream is consumed, with the aggregated result — so
    the call is recorded exactly once, with usage. Note this is STRICTLY
    better than the legacy client-injection path, where a streamed
    `.invoke()` records `usage=None` because LangChain never asks OpenAI for
    stream usage on the raw client (see `test_langchain.py`); through the
    callback, LangChain's own `stream_usage` handling supplies it."""
    sse = (
        'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there!"},'
        '"finish_reason":"stop"}]}\n\n'
        'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,'
        '"completion_tokens":5,"total_tokens":15}}\n\n'
        'data: [DONE]\n\n'
    )
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, content=sse, headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                      callbacks=[tracer.langchain_handler()])
    assert "".join(chunk.content for chunk in llm.stream("hi")) == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15}
    assert [b.block.text for b in ct.get_call_blocks(calls[0].id)] == ["hi"]
    ct.close()


def test_handler_captures_anthropic_and_derives_the_provider(respx_mock, tmp_ctrace_path):
    """The handler sees provider-agnostic messages, so the provider comes
    from the model description LangChain passes alongside them
    (`metadata["ls_provider"]`). A `ChatAnthropic` must therefore be recorded
    through the ANTHROPIC adapter: the system prompt lifted out of the
    message list into Anthropic's top-level `system` field, and usage stored
    under Anthropic's own input/output key names."""
    respx_mock.post(_ANTHROPIC_URL).mock(return_value=httpx.Response(
        200, json=canned_anthropic_response(input_tokens=12, output_tokens=6)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_anthropic.ChatAnthropic(
        model="claude-opus-4-8", api_key="x", max_tokens=100,
        callbacks=[tracer.langchain_handler()])
    llm.invoke([("system", "You are helpful."), ("human", "hi")])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].provider == "anthropic"
    assert calls[0].usage == {"input_tokens": 12, "output_tokens": 6}
    assert calls[0].params["model"] == "claude-opus-4-8"
    assert calls[0].params["max_tokens"] == 100

    blocks = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.kind, b.block.text) for b in blocks] == [
        ("system", "message", "You are helpful."), ("user", "message", "hi")]
    ct.close()


def test_handler_anthropic_blocks_match_the_wire_body(respx_mock, tmp_ctrace_path):
    """Same wire-level check as the OpenAI one, for the provider whose shape
    differs most: Anthropic's `system` is a top-level field, not a message,
    so a normalization that simply left it in the list would produce
    different blocks from a direct `client.messages.create(system=...)`."""
    sent: list[dict] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=canned_anthropic_response())

    respx_mock.post(_ANTHROPIC_URL).mock(side_effect=_respond)

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_anthropic.ChatAnthropic(
        model="claude-opus-4-8", api_key="x", max_tokens=100,
        callbacks=[tracer.langchain_handler()])
    llm.invoke([("system", "Be terse."), ("human", "explain hashing")])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert [b.block.content_hash for b in ct.get_call_blocks(calls[0].id)] \
        == _hashes_for(sent[0], provider="anthropic")
    ct.close()


# --------------------------------------------------------------------------
# The typed-part providers: Gemini and Bedrock Converse.
#
# These two branches build a list of TYPED PARTS instead of passing list
# content through the way the OpenAI/Anthropic branches do, and until these
# tests existed they were the only two with no wire-body verification at all
# — which is exactly why they shipped dropping every non-text part and
# merging multiple text parts into one. Both integrations are optional (the
# `eval` extra), so each test skips on its own rather than taking the file
# with it.


def test_handler_gemini_blocks_match_the_wire_body(respx_mock, tmp_ctrace_path):
    """The Gemini branch, checked against the body langchain-google-genai
    really sent — a VISION turn with two text parts and an image, which is
    where the branch was wrong. The real wire carries three parts
    (`{"text"}, {"text"}, {"inlineData"}`), so the trace must carry three
    blocks: two content parts and one IMAGE block whose text is a descriptor
    and whose identity is the picture's bytes."""
    google_genai = pytest.importorskip("langchain_google_genai")
    sent: list[dict] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=canned_gemini_response())

    respx_mock.post(url__regex=_GEMINI_URL_RE).mock(side_effect=_respond)

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = google_genai.ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key="x",
        callbacks=[tracer.langchain_handler()])
    llm.invoke([SystemMessage("Be terse."), VISION_MESSAGE])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].provider == "gemini"
    assert calls[0].usage == {"prompt_token_count": 10, "candidates_token_count": 5,
                              "total_token_count": 15}

    blocks = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.kind, b.block.text) for b in blocks] == [
        ("system", "message", "Be terse."),
        ("user", "content_part", "look at this:"),
        ("user", "content_part", " carefully"),
        ("user", "image", "[image 4×4 · ~258 tok]"),   # Gemini's own formula
    ]
    assert [b.block.content_hash for b in blocks] == _gemini_hashes_for(sent[0])
    ct.close()


def test_handler_gemini_vision_turn_is_hash_identical_to_direct_capture(
        respx_mock, tmp_ctrace_path):
    """The same picture, twice, in one tracer: once through
    `ChatGoogleGenerativeAI` + the handler (where LangChain carries it as an
    `image_url` data URI) and once through a directly-wrapped
    `google.genai.Client` (where it is `inline_data` bytes). Both must be the
    SAME blocks — image identity is the pixels, not the wrapper — or a
    vision agent's screenshot would look like a different image on every
    capture path."""
    google_genai = pytest.importorskip("langchain_google_genai")
    from google import genai

    respx_mock.post(url__regex=_GEMINI_URL_RE).mock(
        return_value=httpx.Response(200, json=canned_gemini_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = google_genai.ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key="x",
        callbacks=[tracer.langchain_handler()])
    llm.invoke([SystemMessage("Be terse."), VISION_MESSAGE])

    client = tracer.wrap(genai.Client(api_key="x"))
    client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [
            {"text": "look at this:"}, {"text": " carefully"},
            {"inline_data": {"mime_type": "image/png", "data": PNG_4x4}}]}],
        config={"system_instruction": "Be terse."})
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    via_handler, via_wrap = ct.get_calls()
    assert ([b.block.content_hash for b in ct.get_call_blocks(via_handler.id)]
            == [b.block.content_hash for b in ct.get_call_blocks(via_wrap.id)])
    ct.close()


def _stubbed_bedrock_client(sent: list, response: dict):
    """A real boto3 bedrock-runtime client whose `Converse` request never
    leaves the process: botocore's `before-send` hook records the JSON body
    and answers with `response`.

    Stubbed here rather than with `botocore.stub.Stubber` because the whole
    point is to read the REQUEST BODY langchain-aws produced — Stubber
    validates parameters but never exposes the serialized body, and it is the
    body that the recorded blocks have to match."""
    import boto3
    from botocore.awsrequest import AWSResponse

    class _RawBody:
        """The minimal urllib3-shaped body botocore's JSON parser needs."""

        def __init__(self, data: bytes):
            self._data = data

        def stream(self, *args, **kwargs):
            yield self._data

        def read(self, *args, **kwargs):
            return self._data

        def close(self) -> None:
            pass

    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")

    def _respond(request, **kwargs):
        sent.append(json.loads(request.body))
        return AWSResponse(request.url, 200, {"content-type": "application/json"},
                           _RawBody(json.dumps(response).encode()))

    client.meta.events.register("before-send.bedrock-runtime.Converse", _respond)
    return client


def _canned_converse_response(text: str = "Hello there!", input_tokens: int = 12,
                              output_tokens: int = 6) -> dict:
    """A Converse response in the shape botocore parses, including the
    required `metrics` member."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens,
                  "totalTokens": input_tokens + output_tokens},
        "metrics": {"latencyMs": 100},
    }


def test_handler_bedrock_blocks_match_the_wire_body(tmp_ctrace_path):
    """The Bedrock Converse branch, checked against the body langchain-aws
    really sent. Converse content is a list of typed blocks, so the same
    vision turn is three of them — and the image, which the branch used to
    drop entirely, is an image block hashed over its bytes."""
    pytest.importorskip("langchain_aws")
    from langchain_aws import ChatBedrockConverse

    sent: list[dict] = []
    client = _stubbed_bedrock_client(sent, _canned_converse_response())

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = ChatBedrockConverse(client=client,
                              model="anthropic.claude-3-haiku-20240307-v1:0",
                              callbacks=[tracer.langchain_handler()])
    llm.invoke([SystemMessage("Be terse."), VISION_MESSAGE])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].provider == "bedrock"
    assert calls[0].usage == {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18}

    blocks = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.kind, b.block.text) for b in blocks] == [
        ("system", "message", "Be terse."),
        ("user", "content_part", "look at this:"),
        ("user", "content_part", " carefully"),
        ("user", "image", "[image 4×4 · ~1 tok]"),   # Anthropic's formula, per images.py
    ]
    assert [b.block.content_hash for b in blocks] == _bedrock_hashes_for(sent[0])
    ct.close()


def test_handler_bedrock_vision_turn_is_hash_identical_to_direct_capture(tmp_ctrace_path):
    """The Bedrock twin of the Gemini identity test: the same picture through
    `ChatBedrockConverse` + the handler and through a directly-wrapped boto3
    client calling `converse(...)` with Converse's own `image` block must be
    the same blocks."""
    pytest.importorskip("langchain_aws")
    from langchain_aws import ChatBedrockConverse

    sent: list[dict] = []
    client = _stubbed_bedrock_client(sent, _canned_converse_response())

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    ChatBedrockConverse(client=client,
                        model="anthropic.claude-3-haiku-20240307-v1:0",
                        callbacks=[tracer.langchain_handler()]).invoke(
        [SystemMessage("Be terse."), VISION_MESSAGE])

    tracer.wrap(client).converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        system=[{"text": "Be terse."}],
        messages=[{"role": "user", "content": [
            {"text": "look at this:"}, {"text": " carefully"},
            {"image": {"format": "png", "source": {"bytes": PNG_4x4}}}]}])
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    via_handler, via_wrap = ct.get_calls()
    assert ([b.block.content_hash for b in ct.get_call_blocks(via_handler.id)]
            == [b.block.content_hash for b in ct.get_call_blocks(via_wrap.id)])
    ct.close()


def test_one_handler_serves_two_providers_in_one_run(respx_mock, tmp_ctrace_path):
    """A single handler passed to several models records each through the
    correct adapter — `_recorder_for` builds and caches one Recorder per
    provider — so a multi-provider agent lands in ONE session with each call
    normalized by its own provider's rules."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, json=canned_openai_response()))
    respx_mock.post(_ANTHROPIC_URL).mock(return_value=httpx.Response(
        200, json=canned_anthropic_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    handler = tracer.langchain_handler()
    langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                callbacks=[handler]).invoke("hi")
    langchain_anthropic.ChatAnthropic(model="claude-opus-4-8", api_key="x",
                                      max_tokens=100,
                                      callbacks=[handler]).invoke("hi")
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert [c.provider for c in calls] == ["openai", "anthropic"]
    assert calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15}
    assert calls[1].usage == {"input_tokens": 12, "output_tokens": 6}
    ct.close()


def test_handler_records_a_failed_call(respx_mock, tmp_ctrace_path):
    """A provider error must still leave a turn in the trace — the context
    that PRODUCED the failure is exactly what a debugger is wanted for — so
    `on_llm_error` records a call carrying the exception type name and the
    request's blocks, and the host's own exception is untouched."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        500, json={"error": {"message": "boom"}}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x", max_retries=0,
                                      callbacks=[tracer.langchain_handler()])
    with pytest.raises(Exception):
        llm.invoke("hi")
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error is not None
    assert calls[0].usage is None
    assert [b.block.text for b in ct.get_call_blocks(calls[0].id)] == ["hi"]
    ct.close()


def test_handler_stamps_the_agent_name(respx_mock, tmp_ctrace_path):
    """`langchain_handler(agent=...)` attributes every call it records to
    that agent, exactly as `wrap(client, agent=...)` does — which is how a
    multi-agent graph gets per-agent views."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, json=canned_openai_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    llm = langchain_openai.ChatOpenAI(
        model="gpt-4o", api_key="x",
        callbacks=[tracer.langchain_handler(agent="researcher")])
    llm.invoke("hi")
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_calls()[0].agent == "researcher"
    ct.close()


def test_handler_is_fail_open_when_recording_is_broken(respx_mock, tmp_ctrace_path,
                                                       monkeypatch):
    """Fail-open, on the framework path: with `Recorder.build` broken
    outright, the LangChain call must still succeed and return its answer,
    and the trace is simply empty. A debugger may lose its own data; it may
    never take down the agent."""
    respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(
        200, json=canned_openai_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    handler = tracer.langchain_handler()
    llm = langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x",
                                      callbacks=[handler])
    # The recorder only exists once a provider is known, so break the class.
    from ctxdiff.capture import recorder as recorder_module

    def _boom(*args, **kwargs):
        raise RuntimeError("recorder is broken")

    monkeypatch.setattr(recorder_module.Recorder, "build", _boom)
    assert llm.invoke("hi").content == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_calls() == []
    ct.close()


def test_handler_records_each_run_exactly_once(tmp_ctrace_path):
    """The callbacks are driven directly (no HTTP, real langchain-core
    types) to pin two properties of the run bookkeeping: an `on_llm_end` for
    a run that never started records nothing and raises nothing (a handler
    attached mid-run), and a REPEATED end for a run that did start records
    only once — the same record-exactly-once discipline the stream proxies
    get from their `_finalized` flag, here from claiming the pending run."""
    from langchain_core.messages import HumanMessage
    from langchain_core.outputs import LLMResult

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    handler = tracer.langchain_handler()

    handler.on_llm_end(LLMResult(generations=[]), run_id="never-started")

    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
        [[HumanMessage(content="hi")]], run_id="run-1",
        invocation_params={"model": "gpt-4o"})
    handler.on_llm_end(LLMResult(generations=[]), run_id="run-1")
    handler.on_llm_end(LLMResult(generations=[]), run_id="run-1")
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert [b.block.text for b in ct.get_call_blocks(calls[0].id)] == ["hi"]
    ct.close()

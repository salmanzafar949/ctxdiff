"""Google **Vertex AI** capture, end to end against a REAL
`google.genai.Client` constructed in Vertex mode — `Client(vertexai=True,
project=..., location=...)` — with HTTP fully stubbed by respx.

WHAT THE STEP-0 PROBE FOUND, and why this file is mostly tests rather than
new adapter code: a Vertex-mode client is the SAME CLASS as an API-key one
(`google.genai.client.Client`, module `google.genai.client`), reaches the
model through the SAME resource path (`client.models.generate_content` /
`generate_content_stream`, `client.aio.models...` for async), takes the SAME
request kwargs (`model`/`contents`/`config`) and parses the response into the
SAME `GenerateContentResponse` with the SAME `usage_metadata` field names.
Only the BASE URL differs (`https://<location>-aiplatform.googleapis.com/...`
instead of `https://generativelanguage.googleapis.com/...`), and ctxdiff
never looks at a URL: `_detect_provider` matches the dotted module prefix
`google.genai`, and `GeminiAdapter` reads kwargs, not endpoints.

So Vertex was ALREADY captured, and these tests are the evidence for that
claim rather than a coincidence nobody has run — the whole point of pinning
it: a future change to detection (say, narrowing it by base URL or by an
`api_key` attribute) would silently drop every Vertex user, and this file is
what would fail first.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from google import genai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_gemini_response

# A credentials object carrying an already-valid fake token, so the SDK never
# reaches for Application Default Credentials (which would look for a metadata
# server / gcloud config and make the test depend on the machine it runs on)
# and never tries to REFRESH either — `AnonymousCredentials` looks like the
# obvious choice but raises `InvalidOperation` the moment google-genai calls
# `refresh()` on it, since `.valid` is False. A bearer token with no expiry is
# valid on its face, so the SDK just signs the request and hands it to respx.
google_auth = pytest.importorskip("google.auth")
from google.oauth2.credentials import Credentials as OAuthCredentials  # noqa: E402

# The Vertex endpoint the SDK builds for a `us-central1` client. Matched as a
# regex substring the same way the API-key Gemini tests match theirs, so the
# project/model/method path segments don't have to be spelled out.
_VERTEX_GENERATE = r"https://us-central1-aiplatform\.googleapis\.com/.*:generateContent.*"
_VERTEX_STREAM = r"https://us-central1-aiplatform\.googleapis\.com/.*:streamGenerateContent.*"


def _vertex_client() -> genai.Client:
    """A real Vertex-mode `genai.Client` with no ambient credential lookup."""
    return genai.Client(vertexai=True, project="my-project", location="us-central1",
                        credentials=OAuthCredentials(token="fake-token"))


def test_vertex_client_is_detected_as_gemini():
    """Detection, stated as its own fact: a Vertex-mode client resolves to
    the `gemini` provider — it is the same class in the same module as an
    API-key client, so the existing dotted-prefix rule already covers it. No
    HTTP involved."""
    client = _vertex_client()
    assert type(client).__module__ == "google.genai.client"
    assert trace._detect_provider(client) == "gemini"


def test_wrap_vertex_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """One stubbed system+user call through a Vertex client's
    `models.generate_content`: exactly one call recorded, blocks ordered
    system-first then user, usage in Gemini's prompt/candidates/total shape,
    `provider == 'gemini'` on the session, and the response object reaching
    the host untouched."""
    respx_mock.post(url__regex=_VERTEX_GENERATE).mock(
        return_value=httpx.Response(200, json=canned_gemini_response(
            text="Hello there!", prompt_token_count=10, candidates_token_count=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(_vertex_client())

    resp = wrapped.models.generate_content(
        model="gemini-2.0-flash",
        contents="hi",
        config={"system_instruction": "You are helpful."},
    )
    assert resp.text == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_run().provider == "gemini"
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15,
    }
    assert calls[0].params["model"] == "gemini-2.0-flash"

    blocks = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.text) for b in blocks] == [
        ("system", "You are helpful."), ("user", "hi")]
    assert all(b.block.token_method == "estimate" for b in blocks)
    ct.close()


def test_wrap_vertex_blocks_match_the_api_key_path_exactly(respx_mock, tmp_ctrace_path):
    """The SAME prompt sent through Vertex and through the API-key endpoint
    must produce the SAME BLOCK HASHES — the two are one provider with two
    front doors, so a team that moves from AI Studio to Vertex mid-project
    must not see its whole context window light up as new. Both halves run
    through the real SDK; only the stubbed base URL differs."""
    respx_mock.post(url__regex=_VERTEX_GENERATE).mock(
        return_value=httpx.Response(200, json=canned_gemini_response()))
    respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*generateContent.*"
    ).mock(return_value=httpx.Response(200, json=canned_gemini_response()))

    request = {"model": "gemini-2.0-flash", "contents": "hi",
               "config": {"system_instruction": "You are helpful."}}

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    # Both clients are held in locals for the duration: google-genai closes
    # its httpx transport when the `Client` is collected, and a proxy over a
    # sub-resource (`.models`) does not keep the client itself alive — the
    # same lifetime rule as using the SDK unwrapped.
    vertex_client, api_key_client = _vertex_client(), genai.Client(api_key="x")
    tracer.wrap(vertex_client).models.generate_content(**request)
    tracer.wrap(api_key_client).models.generate_content(**request)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    vertex, api_key = ct.get_calls()
    assert ([b.block.content_hash for b in ct.get_call_blocks(vertex.id)]
            == [b.block.content_hash for b in ct.get_call_blocks(api_key.id)])
    ct.close()


def test_wrap_vertex_streaming_captures_usage(respx_mock, tmp_ctrace_path):
    """`generate_content_stream` on a Vertex client: chunks reach the caller
    unaltered and the call is recorded once, with the last chunk's cumulative
    usage — the same `_StreamProxy` path the API-key client takes, reached
    through the same `is_named_stream_method` signal."""
    chunk1 = ('{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},'
              '"index":0}],"usageMetadata":{"promptTokenCount":10,'
              '"candidatesTokenCount":1,"totalTokenCount":11}}')
    chunk2 = ('{"candidates":[{"content":{"parts":[{"text":" there!"}],"role":"model"},'
              '"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,'
              '"candidatesTokenCount":5,"totalTokenCount":15}}')
    respx_mock.post(url__regex=_VERTEX_STREAM).mock(return_value=httpx.Response(
        200, content=f"data: {chunk1}\n\ndata: {chunk2}\n\n",
        headers={"content-type": "text/event-stream"}))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(_vertex_client())

    text = "".join(chunk.text for chunk in wrapped.models.generate_content_stream(
        model="gemini-2.0-flash", contents="hi"))
    assert text == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage == {
        "prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15,
    }
    ct.close()


def test_wrap_vertex_async_captures_call(respx_mock, tmp_ctrace_path):
    """The async surface (`client.aio.models.generate_content`) works on a
    Vertex client too — it hangs off the same `.aio` transparent root-hop,
    which is gated on the gemini provider and so applies identically here."""
    respx_mock.post(url__regex=_VERTEX_GENERATE).mock(
        return_value=httpx.Response(200, json=canned_gemini_response()))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(_vertex_client())

    async def _run():
        return await wrapped.aio.models.generate_content(
            model="gemini-2.0-flash", contents="hi")

    assert asyncio.run(_run()).text == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert [b.block.text for b in ct.get_call_blocks(calls[0].id)] == ["hi"]
    ct.close()

"""Wrapping a REAL `openai.AzureOpenAI` client — structurally identical to
`test_wrap_openai.py` on purpose: the point of this file is proving Azure
needs ZERO ctxdiff-side special-casing (spike §3), so only the client
construction and the respx route pattern differ from the plain-OpenAI test."""
from __future__ import annotations

import httpx
import openai

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace

from .conftest import canned_openai_response

# Azure's actual request URL (confirmed in the spike) embeds the deployment
# name and api-version query string, e.g.:
#   https://ex.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-02-01
# A regex route is used instead of an exact string match because the
# deployment segment is caller-supplied.
_AZURE_DEPLOYMENTS_REGEX = r"https://ex\.openai\.azure\.com/openai/deployments/.*/chat/completions.*"


def test_wrap_azure_captures_call_and_blocks(respx_mock, tmp_ctrace_path):
    """Wrap a real AzureOpenAI client, make one stubbed system+user call
    (using a deployment name as `model`, as Azure requires), reopen the
    `.ctrace`, and assert the SAME capture shape as plain OpenAI: one call,
    matching usage, matching blocks, `token_method == 'tiktoken'`. Any
    divergence here would mean Azure needs special-casing ctxdiff doesn't
    have — this test exists to catch that regression."""
    respx_mock.post(url__regex=_AZURE_DEPLOYMENTS_REGEX).mock(
        return_value=httpx.Response(200, json=canned_openai_response(
            content="Hello there!", prompt_tokens=10, completion_tokens=5)))

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    client = openai.AzureOpenAI(api_key="x", azure_endpoint="https://ex.openai.azure.com",
                                api_version="2024-02-01")
    wrapped = tracer.wrap(client)

    resp = wrapped.chat.completions.create(
        model="my-gpt4-deployment",  # Azure deployment name, not a model id
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

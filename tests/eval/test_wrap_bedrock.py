"""Wrapping a REAL boto3 `bedrock-runtime` client end-to-end: real botocore
client object, real `_ClientProxy` interception via the botocore
class-name detection path, HTTP fully stubbed by `botocore.stub.Stubber`
(NOT respx — botocore's transport is urllib3-based, not httpx, so respx
can't intercept it; `Stubber` patches the client's low-level
`_make_api_call` instead, which the proxy's captured bound `converse`
method still ultimately calls through — Step-0 probe confirmed this holds
even when the client is wrapped), then reopen the written `.ctrace` file to
assert what actually landed on disk."""
from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import Stubber  # noqa: E402 — after the importorskip guard

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


def _canned_converse_response(text: str = "Hello there!", input_tokens: int = 12,
                              output_tokens: int = 6) -> dict:
    """Build a Converse response shape confirmed against the real botocore
    'Converse' output shape (Step-0 probe): 'metrics' is a required output
    member so it must be present for `Stubber` to validate the canned
    response, even though this adapter doesn't read it. `totalTokens` is
    always the sum of the two counts so callers never have to keep three
    numbers in sync by hand."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens,
                  "totalTokens": input_tokens + output_tokens},
        "metrics": {"latencyMs": 100},
    }


def test_wrap_bedrock_captures_call_and_blocks(tmp_ctrace_path):
    """Wrap a real bedrock-runtime client, make one stubbed system+user call
    (with an inferenceConfig) through `.converse(...)`, then reopen the
    `.ctrace` and assert the full capture shape: exactly one call recorded,
    blocks ordered system-first then user, usage matches the canned response
    exactly (Bedrock's inputTokens/outputTokens/totalTokens shape), params
    carry modelId plus the flattened maxTokens, and `token_method ==
    'estimate'` — bedrock is not 'openai', so `count_tokens` routes to the
    character-based estimate, never tiktoken."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")
    stubber = Stubber(client)
    stubber.add_response("converse", _canned_converse_response(
        text="Hello there!", input_tokens=12, output_tokens=6))
    stubber.activate()

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(client)

    resp = wrapped.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        system=[{"text": "You are helpful."}],
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 256},
    )
    # The real response dict flows through untouched — the interceptor never
    # alters what the host application sees.
    assert resp["output"]["message"]["content"][0]["text"] == "Hello there!"
    stubber.assert_no_pending_responses()
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_run().provider == "bedrock"
    calls = ct.get_calls()
    assert len(calls) == 1

    call = calls[0]
    assert call.usage == {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18}
    assert call.params["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert call.params["maxTokens"] == 256
    assert "system" not in call.params
    assert "messages" not in call.params
    assert "inferenceConfig" not in call.params

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

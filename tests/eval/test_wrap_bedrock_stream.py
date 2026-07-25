"""Wrapping a REAL boto3 `bedrock-runtime` client's `converse_stream` end to
end: real botocore client, real `_ClientProxy` interception via the botocore
class-name detection path, real `botocore.eventstream.EventStream` parsing of
a real binary `vnd.amazon.eventstream` body — then reopen the written
`.ctrace` and assert what landed on disk.

WHY THE HTTP IS STUBBED THE WAY IT IS. The non-streaming sibling
(`test_wrap_bedrock.py`) uses `botocore.stub.Stubber`, which patches
`_make_api_call` and validates a canned RESPONSE DICT against the operation's
output shape. That cannot express this operation: `ConverseStream`'s output
member `stream` is an `eventstream: true` shape, and Stubber's parameter
validator rejects any iterator/list handed in for it (confirmed — Step-0
probe: `Invalid type for parameter stream ... valid types: <class 'dict'>`).
So the stub is moved one layer DOWN, to botocore's `before-send` hook, which
returns a fully-formed `AWSResponse` whose raw body is genuine event-stream
FRAMES (`_frame` below). Everything above that — the signer, the endpoint,
`convert_to_response_dict`'s event-stream branch, `EventStream`'s parser — is
real botocore code, which is the point: the two facts this feature depends on
(the response is an envelope DICT containing the stream, and `EventStream` is
iterable but NOT an iterator) are properties of that real code, not of a
fixture.
"""
from __future__ import annotations

import binascii
import json
import struct

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.awsrequest import AWSResponse  # noqa: E402 — after the importorskip guard

from ctxdiff import trace  # noqa: E402
from ctxdiff.store.ctrace import CTrace  # noqa: E402


def _header(name: str, value: str) -> bytes:
    """Encode one event-stream header: 1-byte name length, the name, the
    value type (7 = UTF-8 string, the only type these headers use), a 2-byte
    big-endian value length, then the value."""
    n, v = name.encode(), value.encode()
    return bytes([len(n)]) + n + bytes([7]) + struct.pack(">H", len(v)) + v


def _frame(event_type: str, payload: dict) -> bytes:
    """Encode ONE `vnd.amazon.eventstream` message the way Bedrock puts it on
    the wire: a 12-byte prelude (total length, headers length, CRC32 of those
    eight bytes), the three headers botocore's parser dispatches on
    (`:event-type` — which member of `ConverseStreamOutput` this is,
    `:message-type`, `:content-type`), the JSON payload, and a trailing CRC32
    over everything before it. Hand-rolled rather than imported because
    botocore ships a DECODER (`EventStreamBuffer`) and no encoder."""
    headers = (_header(":event-type", event_type)
               + _header(":message-type", "event")
               + _header(":content-type", "application/json"))
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    prelude += struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + headers + body
    return message + struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)


class _RawBody:
    """The minimal urllib3-shaped raw body botocore's event-stream branch
    needs: `convert_to_response_dict` passes `http_response.raw` straight to
    `EventStream`, which only ever calls `.stream()` on it (and `.close()`
    when the caller closes the stream). `closed` records whether the proxy's
    `close()` really reached the underlying body."""

    def __init__(self, data: bytes):
        self._data = data
        self.closed = False

    def stream(self, *args, **kwargs):
        """Yield the whole canned body as one chunk — `EventStreamBuffer`
        reassembles frames across chunk boundaries either way."""
        yield self._data

    def close(self) -> None:
        self.closed = True


def _canned_stream_body(text: str = "Hello there!", input_tokens: int = 12,
                        output_tokens: int = 6) -> bytes:
    """The full event sequence a real Converse stream emits, in order:
    `messageStart`, two `contentBlockDelta`s, `contentBlockStop`,
    `messageStop`, and LAST the `metadata` event carrying usage — the only
    event with token counts on it (Step-0 probe). `totalTokens` is always the
    sum so callers never keep three numbers in sync by hand."""
    head, tail = text[:5], text[5:]
    return b"".join([
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {"delta": {"text": head}, "contentBlockIndex": 0}),
        _frame("contentBlockDelta", {"delta": {"text": tail}, "contentBlockIndex": 0}),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", {
            "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens,
                      "totalTokens": input_tokens + output_tokens},
            "metrics": {"latencyMs": 100},
        }),
    ])


def _stubbed_client(body: bytes) -> tuple[object, _RawBody]:
    """Build a real bedrock-runtime client whose ConverseStream request is
    answered from `body` without touching the network, and return it with the
    raw body object so a test can assert the stream was really closed."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="y")
    raw = _RawBody(body)

    def _respond(request, **kwargs):
        return AWSResponse(request.url, 200,
                           {"content-type": "application/vnd.amazon.eventstream"}, raw)

    client.meta.events.register("before-send.bedrock-runtime.ConverseStream", _respond)
    return client, raw


def test_converse_stream_records_once_with_usage(tmp_ctrace_path):
    """The headline path. Consuming a wrapped `converse_stream` must: hand
    the caller the SAME envelope shape it would get unwrapped (a dict with
    `ResponseMetadata` and `stream`), pass every event through untouched and
    in order, and record EXACTLY ONE call whose usage came out of the
    trailing `metadata` event in Bedrock's own inputTokens/outputTokens/
    totalTokens shape — identical to what a non-streaming `converse` call
    stores for the same numbers, because both go through the same
    `extract_usage`."""
    client, _raw = _stubbed_client(_canned_stream_body())

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(client)

    response = wrapped.converse_stream(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        system=[{"text": "You are helpful."}],
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 256},
    )
    # The envelope is still an envelope: the host's own `ResponseMetadata`
    # survives, and `stream` is what it iterates.
    assert isinstance(response, dict)
    assert "ResponseMetadata" in response
    events = list(response["stream"])

    # Every event reached the caller, unchanged and in wire order.
    assert [next(iter(e)) for e in events] == [
        "messageStart", "contentBlockDelta", "contentBlockDelta",
        "contentBlockStop", "messageStop", "metadata"]
    assert "".join(e["contentBlockDelta"]["delta"]["text"]
                   for e in events if "contentBlockDelta" in e) == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_run().provider == "bedrock"
    calls = ct.get_calls()
    assert len(calls) == 1                      # recorded exactly once, at completion
    assert calls[0].usage == {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18}
    assert calls[0].error is None
    assert calls[0].params["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert calls[0].params["maxTokens"] == 256  # inferenceConfig still flattened

    blocks = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.text) for b in blocks] == [
        ("system", "You are helpful."), ("user", "hi")]
    ct.close()


def test_converse_stream_blocks_match_converse_exactly(tmp_ctrace_path):
    """`converse` and `converse_stream` take the IDENTICAL request shape, so
    the same request must produce the same BLOCKS — same hashes, same order —
    whichever method sent it. This is what makes a streamed turn diff against
    a non-streamed one instead of looking like a whole new context.

    Driven through the streaming path twice rather than against a hand-built
    expectation: the first call streams, the second one streams the same
    kwargs, and the recorded hashes are compared to the hashes the adapter
    produces for a `converse` of the same kwargs via the real recorder."""
    from ctxdiff.capture.bedrock import BedrockAdapter
    from ctxdiff.capture.recorder import build_block

    kwargs = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "system": [{"text": "You are helpful."}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    client, _raw = _stubbed_client(_canned_stream_body())
    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(client)
    list(wrapped.converse_stream(**kwargs)["stream"])
    tracer.close()

    adapter = BedrockAdapter()
    expected = [build_block(rb, "bedrock").content_hash
                for rb in adapter.extract_blocks(kwargs)]

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert [b.block.content_hash for b in ct.get_call_blocks(calls[0].id)] == expected
    ct.close()


def test_converse_stream_abandoned_is_still_recorded(tmp_ctrace_path):
    """A caller that reads a couple of events and closes the stream still
    gets a recorded call — with whatever usage arrived before the close
    (none here: the `metadata` event is last and was never reached) — and the
    real `EventStream.close()` still reaches the underlying body, so the HTTP
    connection is released exactly as it would be unwrapped."""
    client, raw = _stubbed_client(_canned_stream_body())
    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(client)

    stream = wrapped.converse_stream(
        modelId="m", messages=[{"role": "user", "content": [{"text": "hi"}]}])["stream"]
    iterator = iter(stream)
    next(iterator)
    stream.close()
    tracer.close()

    assert raw.closed is True                   # close() forwarded to the real body
    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].usage is None               # honest: no metadata event was seen
    ct.close()


def test_converse_stream_mid_stream_error_is_recorded_as_failed(tmp_ctrace_path):
    """A stream that fails PARTWAY through is recorded as a FAILED call, with
    the exception type name as `error`, and the original exception still
    reaches the caller unchanged. A half-streamed turn must never look like a
    successful one that merely lacks usage.

    The failure is induced by CORRUPTING a byte inside the last frame's
    payload, which makes botocore's own decoder raise `ChecksumMismatch` when
    it validates that message. Simply TRUNCATING the body does not work
    (checked): `EventStreamBuffer` treats a partial trailing frame as "more
    data may yet arrive" and the generator ends cleanly instead of raising —
    which would have made this test pass for the wrong reason."""
    corrupt = bytearray(_canned_stream_body())
    corrupt[-40] ^= 0xFF                        # break the final frame's CRC32
    client, _raw = _stubbed_client(bytes(corrupt))
    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(client)

    stream = wrapped.converse_stream(
        modelId="m", messages=[{"role": "user", "content": [{"text": "hi"}]}])["stream"]
    with pytest.raises(Exception):              # botocore's own parse failure
        list(stream)
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].error is not None
    ct.close()


def test_converse_stream_survives_a_broken_recorder(tmp_ctrace_path, monkeypatch):
    """Fail-open, on the path most able to break a host: if recording itself
    is broken, the caller must STILL get every event of its stream. Breaks
    `Recorder.build` outright and asserts the full text still arrives and the
    trace is simply empty."""
    client, _raw = _stubbed_client(_canned_stream_body())
    tracer = trace.init(project="p", path=tmp_ctrace_path)
    wrapped = tracer.wrap(client)

    def _boom(*args, **kwargs):
        raise RuntimeError("recorder is broken")

    monkeypatch.setattr(tracer._recorder, "build", _boom)

    events = list(wrapped.converse_stream(
        modelId="m", messages=[{"role": "user", "content": [{"text": "hi"}]}])["stream"])
    assert "".join(e["contentBlockDelta"]["delta"]["text"]
                   for e in events if "contentBlockDelta" in e) == "Hello there!"
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    assert ct.get_calls() == []                 # nothing recorded, nothing raised
    ct.close()

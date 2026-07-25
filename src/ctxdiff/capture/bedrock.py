"""AWS Bedrock adapter (Converse API, via boto3's `bedrock-runtime` client).
Reads the request kwargs a caller passes to `client.converse(...)` — plain
dicts/lists, the shape boto3 callers always use since botocore has no typed
request objects of its own — and the response, with no dependency on boto3
itself.

Two shapes this adapter absorbs that OpenAI/Anthropic don't have: `system` is
a list of `{"text": ...}` blocks (never a bare string, unlike Anthropic's
`system`), and the response from `client.converse(...)` is a plain DICT, not
an SDK object — botocore parses the wire JSON straight into dicts/lists, so
`extract_usage` must read it with `.get`, not `getattr`."""
from __future__ import annotations

import json

from ctxdiff.images import image_raw_block
from ctxdiff.models import RawBlock

# Request keys excluded from params verbatim: "system"/"messages"/"toolConfig"
# carry block *content* (kept out so content is never stored twice);
# "inferenceConfig" is dropped in its raw dict form too because its scalar
# fields are flattened into params individually below instead.
_CONTENT_KEYS = {"system", "messages", "toolConfig", "inferenceConfig"}

# inferenceConfig fields that are sampling params, not content; flattened
# directly into params under their own (Converse-native) names.
_INFERENCE_CONFIG_FIELDS = ("maxTokens", "temperature", "topP", "stopSequences")


class BedrockAdapter:
    """Normalize Bedrock Converse requests into the block model."""

    provider = "bedrock"
    create_path = ("converse",)  # kept for backward compat
    # `converse_stream` is a SEPARATE method (not a `stream=True` kwarg on
    # `converse`, unlike OpenAI/Anthropic) taking the IDENTICAL request shape —
    # so every extractor above serves both unchanged — and returning an
    # event-stream instead of a completed response. trace.py's
    # `_ClientProxy.__getattr__` recognizes the shape by the path's LAST
    # segment ENDING WITH "stream" but not being exactly "stream" (which
    # instead means the Anthropic/OpenAI `.stream()` MANAGER helpers) and
    # routes it through the existing `_StreamProxy` machinery, the same as
    # Gemini's `generate_content_stream`.
    create_paths = (("converse",), ("converse_stream",))
    # WHERE the iterator lives in what `converse_stream(...)` returns.
    # Confirmed empirically against real botocore 1.43.55 (Step-0 probe):
    # unlike every other provider's streaming method — which returns the
    # stream itself — this one returns a plain RESPONSE DICT that merely
    # CONTAINS the stream: `{"ResponseMetadata": {...}, "stream":
    # <botocore.eventstream.EventStream>}`. Declaring the key here lets
    # trace.py's `_wrap_stream_result` proxy the inner iterator and hand the
    # host back its envelope otherwise untouched, rather than wrapping the
    # dict (which isn't iterable) and silently capturing nothing.
    stream_envelope_key = "stream"

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten a request into ordered RawBlocks: `system` first (one
        'system'-role 'message' block per entry — text from the entry's
        'text' key when present, else stable JSON for any other Converse
        system-block shape like `cachePoint`), then tool schemas from
        `toolConfig.tools` (one 'tool_schema' block per tool, JSON of the
        `toolSpec`), then `messages` (one 'content_part' block per content
        entry — text entries keep their text verbatim, anything else — image,
        toolUse, toolResult — is stable-JSON serialized). Order mirrors send
        order (system → tool schemas → messages) the way the tokens actually
        sit in the sent context. Roles are passed through as-is: Converse has
        no 'tool' role — toolResult content lives inside a user-role
        message's `content` list, not a separate role."""
        blocks: list[RawBlock] = []
        for entry in kwargs.get("system") or []:
            text = entry.get("text") if "text" in entry else json.dumps(
                entry, sort_keys=True, ensure_ascii=False)
            blocks.append(RawBlock(role="system", kind="message", text=text))

        tool_config = kwargs.get("toolConfig") or {}
        for tool in tool_config.get("tools") or []:
            spec = tool.get("toolSpec", tool)
            blocks.append(RawBlock(
                role="system", kind="tool_schema",
                text=json.dumps(spec, sort_keys=True, ensure_ascii=False)))

        for msg in kwargs.get("messages") or []:
            role = msg.get("role", "user")
            for part in msg.get("content") or []:
                # An `{"image": {"format": ..., "source": {"bytes": ...}}}`
                # part becomes an 'image' block whose text is a short
                # descriptor and whose identity is the image bytes; see
                # ctxdiff.images.
                image = image_raw_block(role, part, self.provider)
                if image is not None:
                    blocks.append(image)
                    continue
                if isinstance(part, dict) and "text" in part:
                    text = part["text"]
                else:
                    text = json.dumps(part, sort_keys=True, ensure_ascii=False)
                blocks.append(RawBlock(role=role, kind="content_part", text=text))
        return blocks

    def extract_params(self, kwargs: dict) -> dict:
        """Return every request kwarg except the content-bearing 'system'/
        'messages'/'toolConfig' keys (so 'modelId' survives untouched), plus
        any inferenceConfig scalars (maxTokens, temperature, topP,
        stopSequences) flattened in when present — `inferenceConfig` is a
        dict, read defensively via `.get` so a missing field never appears in
        params as a spurious None."""
        params = {k: v for k, v in kwargs.items() if k not in _CONTENT_KEYS}
        inference_config = kwargs.get("inferenceConfig") or {}
        for field in _INFERENCE_CONFIG_FIELDS:
            value = inference_config.get(field)
            if value is not None:
                params[field] = value
        return params

    def extract_usage(self, response: object) -> dict | None:
        """Pull provider-reported usage into a plain dict, or None when the
        response carries none. Dict-first: `client.converse(...)` returns a
        plain dict (botocore parses Converse's wire JSON directly into
        dicts/lists — no typed response object, unlike openai/anthropic SDKs)
        so `usage` is read with `.get`, not `getattr`. A getattr fallback is
        kept for an object-shaped response (e.g. a caller-supplied stand-in
        in tests, or a future botocore version that wraps responses) so this
        doesn't hard-depend on the dict shape."""
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if usage is None:
            return None
        get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
        return {
            "inputTokens": get("inputTokens"),
            "outputTokens": get("outputTokens"),
            "totalTokens": get("totalTokens"),
        }

    def accumulate_stream_usage(self, chunk: object, state: dict) -> None:
        """Fold usage from ONE `converse_stream` event into `state`.

        Confirmed empirically against real botocore 1.43.55 event-stream
        parsing (Step-0 probe): a Converse stream emits `messageStart` →
        `contentBlockDelta`* → `contentBlockStop` → `messageStop` → and,
        LAST, a single `metadata` event carrying the whole exchange's counts
        at `chunk["metadata"]["usage"]` — Bedrock reports input and output
        together, once, at the end (unlike Anthropic's split across two
        events, and with no caller opt-in to arrange, unlike OpenAI chat's
        `stream_options`). So this is a plain overwrite from that one event
        and a no-op for every other event type.

        DICT-first, not getattr: botocore parses each event straight into
        plain dicts/lists — there is no typed event object here, the same
        reason `extract_usage` reads a `converse` response with `.get`. The
        counts are written into `state` under the SAME key names
        `extract_usage` returns for a non-streaming call, so the synthetic
        response trace.py builds from `state` reads back identically (its
        `.usage` is an attribute namespace, which `extract_usage`'s getattr
        fallback handles). Wrapped in a catch-all exactly like the other
        adapters': a malformed/unexpected event must never interrupt the
        caller's own iteration."""
        try:
            if not isinstance(chunk, dict):
                return
            metadata = chunk.get("metadata")
            usage = metadata.get("usage") if isinstance(metadata, dict) else None
            if isinstance(usage, dict):
                state["inputTokens"] = usage.get("inputTokens")
                state["outputTokens"] = usage.get("outputTokens")
                state["totalTokens"] = usage.get("totalTokens")
        except Exception:  # noqa: BLE001 — never break the caller's iteration
            pass

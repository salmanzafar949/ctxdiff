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
    create_path = ("converse",)

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

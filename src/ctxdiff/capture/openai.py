"""OpenAI Chat Completions adapter. Reads the request kwargs a caller passes to
`client.chat.completions.create(...)` — plain dicts/lists — and the response
object, with no dependency on the openai SDK itself."""
from __future__ import annotations

import json

from ctxdiff.models import RawBlock

# Request keys that carry block *content* rather than sampling params; excluded
# from params so content is never stored twice.
_CONTENT_KEYS = {"messages", "tools"}


class OpenAIAdapter:
    """Normalize OpenAI chat requests into the block model."""

    provider = "openai"
    create_path = ("chat", "completions", "create")

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten a request into ordered RawBlocks: tool schemas first (they
        occupy the front of the context window), then each message. A message
        with list content becomes one 'content_part' block per part; otherwise
        one 'message' block. Tool schemas are serialized to JSON so their text
        is stable and diffable."""
        blocks: list[RawBlock] = []
        for tool in kwargs.get("tools") or []:
            blocks.append(RawBlock(
                role="system", kind="tool_schema",
                text=json.dumps(tool, sort_keys=True, ensure_ascii=False)))
        for msg in kwargs.get("messages") or []:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    blocks.append(RawBlock(
                        role=role, kind="content_part",
                        text=part if isinstance(part, str)
                        else json.dumps(part, sort_keys=True, ensure_ascii=False)))
            else:
                blocks.append(RawBlock(
                    role=role, kind="message", text=content or ""))
        return blocks

    def extract_params(self, kwargs: dict) -> dict:
        """Return every request kwarg except the content-bearing keys, so the
        stored params capture model/sampling settings without duplicating blocks."""
        return {k: v for k, v in kwargs.items() if k not in _CONTENT_KEYS}

    def extract_usage(self, response: object) -> dict | None:
        """Pull provider-reported usage off `response.usage` (duck-typed) into a
        plain dict. Returns None when the response carries no usage (e.g. an
        error path), so downstream code can treat usage as optional."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

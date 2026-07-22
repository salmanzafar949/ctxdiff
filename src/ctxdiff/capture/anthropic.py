"""Anthropic Messages adapter. Reads the kwargs passed to
`client.messages.create(...)`. Differs from OpenAI in two ways the adapter
absorbs: `system` is a top-level field (string or list of text blocks), and
usage is reported as input_tokens/output_tokens."""
from __future__ import annotations

import json

from ctxdiff.models import RawBlock

_CONTENT_KEYS = {"system", "messages", "tools"}


class AnthropicAdapter:
    """Normalize Anthropic messages requests into the block model."""

    provider = "anthropic"
    create_path = ("messages", "create")

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten a request into ordered RawBlocks: the top-level `system`
        first (one block for a string, or one per entry for a list of text
        blocks), then tool schemas, then messages. Order mirrors how the tokens
        actually sit in the sent context."""
        blocks: list[RawBlock] = []
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            blocks.append(RawBlock(role="system", kind="message", text=system))
        elif isinstance(system, list):
            for part in system:
                text = part if isinstance(part, str) else part.get("text", "")
                blocks.append(RawBlock(role="system", kind="message", text=text))
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
                blocks.append(RawBlock(role=role, kind="message", text=content or ""))
        return blocks

    def extract_params(self, kwargs: dict) -> dict:
        """Return request kwargs minus the content-bearing keys."""
        return {k: v for k, v in kwargs.items() if k not in _CONTENT_KEYS}

    def extract_usage(self, response: object) -> dict | None:
        """Map Anthropic's `response.usage.input_tokens/output_tokens` into a
        plain dict, or None when absent."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

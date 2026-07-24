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
        is stable and diffable.

        Order mirrors the wire payload: for a given message, its content
        block(s) come first, then one 'content_part' block per entry in
        `tool_calls` (and one for the legacy single-dict `function_call`, if
        present) — these carry the assistant's tool invocations, which ARE
        part of the context echoed back to the model on later turns and must
        not be silently dropped. When content is None/empty but tool_calls
        (or function_call) is present, the empty content 'message' block is
        skipped entirely: the tool_call part(s) ARE the message, there's
        nothing else to emit for it."""
        blocks: list[RawBlock] = []
        for tool in kwargs.get("tools") or []:
            blocks.append(RawBlock(
                role="system", kind="tool_schema",
                text=json.dumps(tool, sort_keys=True, ensure_ascii=False)))
        for msg in kwargs.get("messages") or []:
            role = msg.get("role", "user")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []
            function_call = msg.get("function_call")

            has_calls = bool(tool_calls) or bool(function_call)
            if isinstance(content, list):
                for part in content:
                    blocks.append(RawBlock(
                        role=role, kind="content_part",
                        text=part if isinstance(part, str)
                        else json.dumps(part, sort_keys=True, ensure_ascii=False)))
            elif content or not has_calls:
                # Preserve the pre-fix fallback (an empty 'message' block) for
                # ordinary messages with no content and no tool_calls; only
                # skip the block when tool_calls/function_call cover for it.
                blocks.append(RawBlock(role=role, kind="message", text=content or ""))

            for call in tool_calls:
                blocks.append(RawBlock(
                    role=role, kind="content_part",
                    text=call if isinstance(call, str)
                    else json.dumps(call, sort_keys=True, ensure_ascii=False)))
            if function_call:
                blocks.append(RawBlock(
                    role=role, kind="content_part",
                    text=function_call if isinstance(function_call, str)
                    else json.dumps(function_call, sort_keys=True, ensure_ascii=False)))
        return blocks

    def extract_params(self, kwargs: dict) -> dict:
        """Return every request kwarg except the content-bearing keys, so the
        stored params capture model/sampling settings without duplicating blocks."""
        return {k: v for k, v in kwargs.items() if k not in _CONTENT_KEYS}

    def extract_usage(self, response: object) -> dict | None:
        """Pull provider-reported usage off `response.usage` (duck-typed) into a
        plain dict. Returns None when the response carries no usage (e.g. an
        error path), so downstream code can treat usage as optional.

        Fallback: a raw-response wrapper (what `with_raw_response.create()`
        returns — the hop LangChain's `ChatOpenAI` takes, see trace.py's
        `_TRANSPARENT_HOPS`) has no `.usage` of its own; only its parsed body
        does. If `.usage` isn't directly present but a callable `.parse()`
        is, call it and read `.usage` off the parsed result instead. openai's
        raw-response `.parse()` is memoized, so calling it here does not
        consume the body or interfere with a caller (e.g. LangChain) parsing
        it again later. Any failure during this fallback is swallowed —
        extract_usage runs inside the fail-open recorder, but stays
        defensive here too rather than relying solely on that outer guard."""
        usage = getattr(response, "usage", None)
        if usage is None:
            parse = getattr(response, "parse", None)
            if callable(parse):
                try:
                    usage = getattr(parse(), "usage", None)
                except Exception:  # noqa: BLE001 — defensive; never raise from here
                    usage = None
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

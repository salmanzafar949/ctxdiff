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
    create_path = ("messages", "create")  # kept for backward compat
    # `messages.stream(**kwargs)` (the `with client.messages.stream(...) as
    # stream:` convenience helper) is a SECOND completion method sharing the
    # exact same request/response shape as `messages.create` — confirmed
    # empirically (Phase 13 Step 0 probe) that the events it yields
    # (message_start/message_delta/...) are the SAME raw event objects a
    # `create(stream=True)` call yields, so `accumulate_stream_usage` below
    # needs no changes at all to also work for the manager-wrapped path (see
    # trace.py's `_StreamManagerProxy`).
    create_paths = (("messages", "create"), ("messages", "stream"))

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
        plain dict, or None when absent.

        Fallback: a raw-response wrapper (what `with_raw_response.create()`
        returns) has no `.usage` of its own; only its parsed body does. If
        `.usage` isn't directly present but a callable `.parse()` is, call it
        and read `.usage` off the parsed result instead — the SDK's
        raw-response `.parse()` is memoized, so this doesn't consume the
        body or interfere with a later caller-side parse. Any failure during
        this fallback is swallowed (extract_usage runs inside the fail-open
        recorder, but stays defensive here too)."""
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
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

    def accumulate_stream_usage(self, chunk: object, state: dict) -> None:
        """Fold usage from ONE streamed Anthropic event into `state`.
        Confirmed empirically against real `anthropic` 0.118.0 SSE parsing
        (Phase 12 Step 0): unlike OpenAI, Anthropic splits input and output
        token counts across TWO different event types rather than reporting
        both together on one final chunk — `message_start` carries the
        (already-known) input token count at `chunk.message.usage.
        input_tokens`, and `message_delta` (emitted once, near the end of
        the stream, alongside the stop reason) carries the output token
        count at `chunk.usage.output_tokens`. Both are written into `state`
        under the SAME key names `extract_usage` uses for a non-streaming
        response, so the eventual synthetic response reads back
        identically. Duck-typed and defensive — see `OpenAIAdapter.
        accumulate_stream_usage` for why this stays self-guarding rather
        than relying solely on the caller's own try/except."""
        try:
            event_type = getattr(chunk, "type", None)
            if event_type == "message_start":
                usage = getattr(getattr(chunk, "message", None), "usage", None)
                input_tokens = getattr(usage, "input_tokens", None)
                if input_tokens is not None:
                    state["input_tokens"] = input_tokens
            elif event_type == "message_delta":
                usage = getattr(chunk, "usage", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if output_tokens is not None:
                    state["output_tokens"] = output_tokens
        except Exception:  # noqa: BLE001 — never break the caller's iteration
            pass

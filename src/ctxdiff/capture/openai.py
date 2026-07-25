"""OpenAI adapter, covering BOTH completion methods the SDK exposes: the
established Chat Completions API (`client.chat.completions.create(...)`) and
its successor, the Responses API (`client.responses.create(...)`, what the
Agents SDK builds on) — plain dicts/lists for kwargs, with no dependency on
the openai SDK itself. The two request shapes are disjoint (`messages` vs
`input`/`instructions`) so `extract_blocks`/`extract_params` dispatch on which
keys are present; `extract_usage` duck-types both response usage shapes."""
from __future__ import annotations

import json

from ctxdiff.images import image_raw_block
from ctxdiff.models import RawBlock

# Request keys that carry block *content* rather than sampling params for the
# Chat Completions shape; excluded from params so content is never stored
# twice.
_CHAT_CONTENT_KEYS = {"messages", "tools"}

# Same idea for the Responses shape: `input`/`instructions` carry the
# conversation content and `tools` the schemas, all extracted into blocks —
# `previous_response_id` is deliberately NOT here (see extract_params) since
# it's chain-linkage metadata, not content.
_RESPONSES_CONTENT_KEYS = {"input", "instructions", "tools"}

# Part types within a Responses `input` content list whose "text" field is the
# human-readable payload; anything else falls back to stable JSON of the part.
_RESPONSES_TEXT_PART_TYPES = {"input_text", "output_text"}


def _is_responses_shape(kwargs: dict) -> bool:
    """Distinguish the Responses request shape from Chat Completions. How:
    the two are disjoint on the wire — Chat Completions always sends
    `messages`; Responses never does, using `input`/`instructions` instead —
    so presence of either of the latter (and absence of `messages`) is
    sufficient. Kwargs with neither key (e.g. a nearly-empty test payload)
    default to the chat shape, preserving all pre-existing behavior for
    Chat-Completions-only callers."""
    return "messages" not in kwargs and ("input" in kwargs or "instructions" in kwargs)


class OpenAIAdapter:
    """Normalize OpenAI chat AND responses requests into the block model."""

    provider = "openai"
    create_path = ("chat", "completions", "create")  # kept for backward compat
    # The `.stream()` convenience-manager helpers (`with client.chat.
    # completions.stream(...) as stream:` / `with client.responses.stream(
    # ...) as stream:`) share the SAME request shape as their `.create()`
    # siblings, so they're just two more completion methods on this same
    # adapter — see trace.py's `_StreamManagerProxy`/`_AsyncStreamManagerProxy`
    # for how a path ending in "stream" gets manager-wrapped instead of
    # stream-wrapped directly.
    create_paths = (("chat", "completions", "create"), ("responses", "create"),
                    ("chat", "completions", "stream"), ("responses", "stream"))

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Dispatch to the Responses extractor when the kwargs shape says so;
        otherwise run the original Chat Completions extraction, byte-for-byte
        unchanged (see `_extract_chat_blocks`)."""
        if _is_responses_shape(kwargs):
            return self._extract_responses_blocks(kwargs)
        return self._extract_chat_blocks(kwargs)

    def _extract_responses_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten a Responses-API request into ordered RawBlocks:
        `instructions` FIRST (a system message occupying the front of the
        context, mirroring how Chat Completions' system message is
        conventionally first), then `tools` (already flat — Responses tool
        schemas are NOT nested under a "function" key the way Chat
        Completions' are — serialized to stable JSON same as the chat path),
        then `input` in wire order.

        `input` is either a plain string (one user message block) or a list
        of items, each handled defensively so a malformed/unexpected item
        shape never raises:
          - a dict with a "role" key is a message item: string content is one
            'message' block; list content is one 'content_part' block per
            part (using the part's own "text" for input_text/output_text
            parts, else stable JSON of the part) — role is passed through
            as-is from the wire.
          - {"type": "function_call", ...} is the model's own tool
            invocation, echoed back on later turns — recorded as an
            'assistant' 'content_part' block (stable JSON).
          - {"type": "function_call_output", ...} is the caller feeding a
            tool's result back in — recorded as a 'tool' 'content_part'
            block (stable JSON).
          - anything else (unrecognized item shape) falls back to a 'user'
            'content_part' block of stable JSON, so an unfamiliar/future item
            type is captured rather than dropped or raising."""
        blocks: list[RawBlock] = []
        instructions = kwargs.get("instructions")
        if instructions:
            blocks.append(RawBlock(role="system", kind="message", text=instructions))
        for tool in kwargs.get("tools") or []:
            blocks.append(RawBlock(
                role="system", kind="tool_schema",
                text=json.dumps(tool, sort_keys=True, ensure_ascii=False)))
        input_ = kwargs.get("input")
        if isinstance(input_, str):
            blocks.append(RawBlock(role="user", kind="message", text=input_))
        elif isinstance(input_, list):
            for item in input_:
                blocks.extend(self._extract_responses_input_item(item))
        return blocks

    def _extract_responses_input_item(self, item: object) -> list[RawBlock]:
        """Extract the block(s) for one entry of a Responses `input` list. See
        `_extract_responses_blocks` for the shape rules; kept as its own
        method purely for readability (one item's worth of dispatch logic)."""
        if isinstance(item, dict) and "role" in item:
            role = item.get("role", "user")
            content = item.get("content")
            if isinstance(content, list):
                out = []
                for part in content:
                    # An `input_image` part becomes an 'image' block whose text
                    # is a short descriptor and whose identity is the image
                    # bytes — never the base64 payload. See ctxdiff.images.
                    image = image_raw_block(role, part, self.provider)
                    if image is not None:
                        out.append(image)
                        continue
                    if isinstance(part, dict) and part.get("type") in _RESPONSES_TEXT_PART_TYPES:
                        text = part.get("text", "")
                    else:
                        text = json.dumps(part, sort_keys=True, ensure_ascii=False)
                    out.append(RawBlock(role=role, kind="content_part", text=text))
                return out
            return [RawBlock(role=role, kind="message", text=content or "")]
        if isinstance(item, dict) and item.get("type") == "function_call":
            return [RawBlock(role="assistant", kind="content_part",
                             text=json.dumps(item, sort_keys=True, ensure_ascii=False))]
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            return [RawBlock(role="tool", kind="content_part",
                             text=json.dumps(item, sort_keys=True, ensure_ascii=False))]
        # Defensive fallback: any other shape (unrecognized item type, or not
        # even a dict) is still captured rather than dropped or raising.
        return [RawBlock(role="user", kind="content_part",
                         text=json.dumps(item, sort_keys=True, ensure_ascii=False))]

    def _extract_chat_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten a Chat Completions request into ordered RawBlocks: tool
        schemas first (they occupy the front of the context window), then
        each message. A message with list content becomes one 'content_part'
        block per part; otherwise one 'message' block. Tool schemas are
        serialized to JSON so their text is stable and diffable.

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
                    # An `image_url` part becomes an 'image' block whose text
                    # is a short descriptor and whose identity is the image
                    # bytes — never the base64 data URI. See ctxdiff.images.
                    image = image_raw_block(role, part, self.provider)
                    if image is not None:
                        blocks.append(image)
                        continue
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
        stored params capture model/sampling settings without duplicating
        blocks. Which keys count as content-bearing depends on the request
        shape: Chat Completions drops `messages`/`tools`; Responses drops
        `input`/`instructions`/`tools` but DELIBERATELY KEEPS
        `previous_response_id` — it isn't content, it's meaningful chain
        linkage between calls (which prior response this one continues from),
        so it belongs in params alongside model/sampling settings."""
        content_keys = _RESPONSES_CONTENT_KEYS if _is_responses_shape(kwargs) else _CHAT_CONTENT_KEYS
        return {k: v for k, v in kwargs.items() if k not in content_keys}

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
        defensive here too rather than relying solely on that outer guard.

        Two naming families: Chat Completions reports
        prompt_tokens/completion_tokens/total_tokens; Responses reports
        input_tokens/output_tokens/total_tokens (confirmed: neither usage
        object carries the other family's attributes — `getattr(...,
        default=None)` cleanly distinguishes them). Both are duck-typed here
        so ONE adapter method serves both request shapes; if a usage object
        somehow carried both (never observed, but not assumed impossible),
        the prompt_tokens family wins, deterministically."""
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
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is not None or completion_tokens is not None:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is not None or output_tokens is not None:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        # Neither family present (e.g. a usage-less stub in a test): preserve
        # the pre-Responses default shape rather than returning an empty dict.
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def accumulate_stream_usage(self, chunk: object, state: dict) -> None:
        """Fold usage from ONE streamed chunk into `state`, covering both
        shapes a stream can carry (confirmed empirically against real
        `openai` 2.47.0 SSE parsing, Phase 12 Step 0):

        - Chat Completions (`ChatCompletionChunk`): every chunk has a
          `.usage` attribute, but it's `None` on all but the LAST chunk, and
          ONLY when the caller opted in via
          `stream_options={"include_usage": True}` — ctxdiff never injects
          that itself (see trace.py module docstring), so without caller
          opt-in no chunk ever carries usage and `state` stays empty. When
          present, `chunk.usage` duck-types identically to a non-streaming
          `ChatCompletion.usage` (prompt_tokens/completion_tokens/
          total_tokens), so the SAME key names are written into `state` —
          the eventual synthetic response `extract_usage` reads back off of
          this state is indistinguishable from a real one.
        - Responses (`ResponseStreamEvent`): the terminal `response.completed`
          event carries the completed `Response` at `.response`, whose
          `.usage` is the same `input_tokens`/`output_tokens`/`total_tokens`
          shape as a non-streaming Responses call — no caller opt-in needed,
          Responses streams emit usage unconditionally. Confirmed (Phase 13
          Step 0 probe) that `client.responses.stream(...)`'s manager-wrapped
          events include this SAME `response.completed` shape unchanged, so
          this branch already covers it with no changes.
        - Chat Completions via the `.stream()` CONVENIENCE MANAGER (`with
          client.chat.completions.stream(...) as stream:`), NOT `create(
          stream=True)`: confirmed (Phase 13 Step 0 probe) that this yields
          typed `ChatCompletionStreamEvent`s — `chunk`/`content.delta`/... —
          rather than raw `ChatCompletionChunk`s directly, so usage is NOT at
          `chunk.usage` (that attribute doesn't exist on these event objects
          at all) but one level deeper, at `chunk.chunk.usage`, ONLY on the
          `type == "chunk"` event wrapping the real final chunk. Same
          caller-opt-in requirement as the raw path (`stream_options={
          "include_usage": True}`) since it's the same underlying HTTP
          stream, just re-wrapped by the SDK.

        Duck-typed via `getattr(..., None)` throughout and wrapped in a
        catch-all: a malformed/unexpected chunk must never interrupt the
        caller's own iteration (trace.py's stream proxies also wrap this
        call in their own try/except, but this method stays defensive on
        its own rather than relying solely on that outer guard — same
        convention as `extract_usage`'s `.parse()` fallback above)."""
        try:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if prompt_tokens is not None or completion_tokens is not None:
                    state["prompt_tokens"] = prompt_tokens
                    state["completion_tokens"] = completion_tokens
                    state["total_tokens"] = getattr(usage, "total_tokens", None)
                    return
            if getattr(chunk, "type", None) == "response.completed":
                resp_usage = getattr(getattr(chunk, "response", None), "usage", None)
                if resp_usage is not None:
                    state["input_tokens"] = getattr(resp_usage, "input_tokens", None)
                    state["output_tokens"] = getattr(resp_usage, "output_tokens", None)
                    state["total_tokens"] = getattr(resp_usage, "total_tokens", None)
                    return
            if getattr(chunk, "type", None) == "chunk":
                inner_usage = getattr(getattr(chunk, "chunk", None), "usage", None)
                if inner_usage is not None:
                    prompt_tokens = getattr(inner_usage, "prompt_tokens", None)
                    completion_tokens = getattr(inner_usage, "completion_tokens", None)
                    if prompt_tokens is not None or completion_tokens is not None:
                        state["prompt_tokens"] = prompt_tokens
                        state["completion_tokens"] = completion_tokens
                        state["total_tokens"] = getattr(inner_usage, "total_tokens", None)
        except Exception:  # noqa: BLE001 — never break the caller's iteration
            pass

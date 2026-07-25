"""The LangChain callback handler — capture through LangChain's OWN extension
point instead of through its internals.

WHY THIS EXISTS. ctxdiff's original LangChain support was client injection:
`ChatOpenAI(client=wrapped.chat.completions, root_client=wrapped)`, i.e.
reaching past LangChain into the provider SDK object it happens to hold and
hoping it keeps holding it the same way. That works (and still does — it is
kept as the legacy path), but it is a bet on someone else's private
structure: a LangChain refactor of how `ChatOpenAI` calls the SDK breaks it
silently, capture just stops, and nothing fails loudly enough to notice. It
also only ever covered the providers whose SDK object ctxdiff could reach.

A callback handler is the supported, documented way in: LangChain hands every
chat model's request to `on_chat_model_start` and its result to
`on_llm_end`/`on_llm_error`, for EVERY provider integration, in streaming and
non-streaming mode alike, and LangGraph propagates the same callbacks through
a whole graph. So one handler covers ChatOpenAI, ChatAnthropic, ChatVertexAI,
ChatBedrockConverse and anything else that follows the interface.

THE HARD REQUIREMENT: HASH IDENTITY. A trace captured through LangChain and a
trace captured by wrapping the provider SDK directly must produce the SAME
BLOCKS for the same logical request — same role, same kind, same text, and
therefore the same content hash — or the two would never dedup against each
other, and a team using both would see phantom "everything changed" diffs.

The mechanism that guarantees it is deliberately NOT "write a second block
extractor for LangChain messages". It is:

    LangChain messages  ──►  the provider's OWN WIRE SHAPE  ──►  the SAME
                             (plain dicts)                       adapter

`to_wire()` below rebuilds the request dict that LangChain itself is about to
put on the wire (verified against real request bodies for ALL FOUR providers —
see `tests/eval/test_langchain_handler.py`), and that dict is handed to the
very same `OpenAIAdapter`/`AnthropicAdapter`/`GeminiAdapter`/`BedrockAdapter`
the direct path uses. There is exactly one block extractor per provider, so
identity is structural rather than something to keep in sync by hand. That
includes multimodal content: a message's parts are rebuilt ONE PER ENTRY (see
`_wire_parts`), so an image reaches the adapter as an image and is hashed over
its bytes exactly as a direct capture's would be.

The identity claim is "hash-identical to a direct capture IN THE SAME SDK",
and it holds without exception. Across SDKs it holds for everything except a
tool call, whose arguments LangChain re-serializes with the host language's
own JSON serializer — see `_tool_calls_of` for why that is inherent and why it
is pinned rather than normalized away.

Usage is folded the same way: LangChain normalizes every provider's token
counts into `usage_metadata` (input_tokens/output_tokens/total_tokens), which
is mapped back onto the key names THAT provider's `extract_usage` returns and
presented as a `SyntheticUsageResponse` — the same object the streaming path
already uses — so a LangChain-captured call's stored `usage` dict is
byte-identical to a directly-captured one's.
"""
from __future__ import annotations

import json
import logging
import threading
import time

# The synthetic usage response is SHARED with trace.py's streaming path (both
# live in capture/recorder.py), so accumulated counts reach every adapter's
# `extract_usage` through literally the same object either way.
from ctxdiff.capture.recorder import SyntheticUsageResponse

_log = logging.getLogger("ctxdiff")

# LangChain's own provider identifier (`metadata["ls_provider"]`, set by
# langchain-core for every chat model) mapped to ctxdiff's adapter name. This
# is the FIRST signal used because it is set by the framework rather than by
# the integration package, so it is stable across class renames.
_LS_PROVIDERS = {
    "openai": "openai",
    "azure_openai": "openai",
    "azure": "openai",
    "anthropic": "anthropic",
    "google_vertexai": "gemini",
    "google_genai": "gemini",
    "google_anthropic_vertex": "anthropic",
    "amazon_bedrock": "bedrock",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "ollama": "openai",          # OpenAI-compatible wire shape
    "together": "openai",
    "fireworks": "openai",
    "groq": "openai",
    "deepseek": "openai",
    "xai": "openai",
}

# Fallback signal: the model class name, from `serialized["id"][-1]` (or
# `serialized["name"]`). Used when `ls_provider` is absent — an older
# langchain-core, or a hand-rolled chat model that doesn't set metadata.
_CLASS_PROVIDERS = {
    "ChatOpenAI": "openai",
    "AzureChatOpenAI": "openai",
    "ChatAnthropic": "anthropic",
    "AnthropicChat": "anthropic",
    "ChatVertexAI": "gemini",
    "ChatGoogleGenerativeAI": "gemini",
    "ChatBedrock": "bedrock",
    "ChatBedrockConverse": "bedrock",
    "BedrockChat": "bedrock",
}

# What an unrecognized integration is treated as. OpenAI's chat shape is the
# de-facto lingua franca (every OSS-serving stack speaks it), and the choice
# is visible rather than silent: `provider` is stored on the call, so a trace
# says which adapter read it.
DEFAULT_PROVIDER = "openai"

# LangChain message `.type` -> the role that provider wire formats use. A
# `ChatMessage` carries its own arbitrary `.role`, which wins when present.
_ROLES = {"human": "user", "ai": "assistant", "system": "system",
          "tool": "tool", "function": "function", "developer": "developer"}

# Per provider, the key names `extract_usage` returns — so accumulated counts
# are written under the names that provider's adapter reads back. A `None`
# third entry means that provider reports no total (Anthropic doesn't).
_USAGE_KEYS = {
    "openai": ("prompt_tokens", "completion_tokens", "total_tokens"),
    "anthropic": ("input_tokens", "output_tokens", None),
    "gemini": ("prompt_token_count", "candidates_token_count", "total_token_count"),
    "bedrock": ("inputTokens", "outputTokens", "totalTokens"),
}

# Where each provider's request dict carries the model id.
_MODEL_KEYS = {"openai": "model", "anthropic": "model", "gemini": "model",
               "bedrock": "modelId"}


def provider_for(serialized: object, metadata: object) -> str:
    """Decide which ctxdiff adapter reads this LangChain request.

    The handler never sees a provider SDK client — only provider-AGNOSTIC
    messages — so the provider has to come from the model description
    LangChain passes alongside them. Two signals, in order: `metadata[
    "ls_provider"]` (set by langchain-core itself for every chat model, so it
    survives an integration renaming its class), then the class name from
    `serialized["id"][-1]`/`serialized["name"]` (covers a model that predates
    or omits the metadata). Anything unrecognized falls back to
    `DEFAULT_PROVIDER` rather than raising — a debugger that refuses to
    record an unfamiliar integration is worse than one that records it in the
    most widely-compatible shape and says which shape it used."""
    if isinstance(metadata, dict):
        ls = metadata.get("ls_provider")
        if isinstance(ls, str):
            mapped = _LS_PROVIDERS.get(ls.lower())
            if mapped:
                return mapped
    name = None
    if isinstance(serialized, dict):
        identifier = serialized.get("id")
        if isinstance(identifier, list) and identifier:
            name = identifier[-1]
        if not isinstance(name, str):
            name = serialized.get("name")
    if isinstance(name, str) and name in _CLASS_PROVIDERS:
        return _CLASS_PROVIDERS[name]
    return DEFAULT_PROVIDER


def _role_of(message: object) -> str:
    """The wire role for one LangChain message. Duck-typed via `getattr` so
    this module never imports langchain types: `.role` (a `ChatMessage`'s own
    free-form role) wins, then `.type` mapped through `_ROLES`, then "user"
    for anything unrecognized — a message is always attributed to somebody
    rather than dropped."""
    role = getattr(message, "role", None)
    if isinstance(role, str) and role:
        return role
    kind = getattr(message, "type", None)
    if not isinstance(kind, str):
        getter = getattr(message, "_getType", None) or getattr(message, "get_type", None)
        kind = getter() if callable(getter) else None
    return _ROLES.get(kind or "", "user")


def _tool_calls_of(message: object) -> list:
    """The provider-shaped tool calls on an assistant message, in OpenAI's
    wire form.

    Two sources, and the order matters for hash identity. The NORMALIZED
    `.tool_calls` comes first, rebuilt with `json.dumps(args)` — because that
    is exactly what LangChain's own converter does when it sends the message
    back, so the resulting `arguments` string matches the wire character for
    character (verified against captured request bodies in
    `tests/eval/test_langgraph.py`). Using `additional_kwargs["tool_calls"]`
    verbatim looks more faithful and is not: it holds the PROVIDER's original
    JSON text, whose whitespace LangChain does not preserve when it
    re-serializes, so the block text would differ from what actually went out
    by exactly the bytes the model happened to use. It stays as the fallback
    for an integration that keeps only the raw form and no normalized one.

    THE ONE PLACE CROSS-SDK IDENTITY STOPS. `json.dumps` writes `{"city":
    "Dubai"}` and JS's `JSON.stringify` writes `{"city":"Dubai"}`, so this
    block — and only this block — hashes differently in the two SDKs. That is
    inherent, not a bug to normalize away: each handler is reproducing its own
    framework's real request, and two DIRECT captures of those same two
    requests diverge by exactly the same bytes with no ctxdiff involved.
    Emitting a common form here would trade a guarantee that is verified
    against the wire (`tests/eval/test_langgraph.py`) for one that is not.
    The divergence is pinned in both suites — see
    `test_cross_sdk_tool_call_hashes_are_pinned_as_known_divergent` — and
    stated in both READMEs."""
    calls = getattr(message, "tool_calls", None)
    if isinstance(calls, list) and calls:
        out = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            out.append({
                "id": call.get("id"),
                "type": "function",
                "function": {"name": call.get("name"),
                             "arguments": json.dumps(call.get("args") or {})},
            })
        if out:
            return out
    raw = getattr(message, "additional_kwargs", None)
    if isinstance(raw, dict):
        existing = raw.get("tool_calls")
        if isinstance(existing, list) and existing:
            return existing
    return []


def _text_of(content: object) -> str:
    """Flatten a message's content down to plain text, for the places a wire
    format takes ONE string: a system prompt, a tool result's text. A string
    passes through; a list of content blocks contributes each block's "text"
    (LangChain's own multimodal shape) joined in order; anything else is
    stringified rather than dropped.

    NOT for message content on the typed-part providers — see `_wire_parts`,
    which is what a message's content list must go through."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _wire_parts(content: object) -> list:
    """A message's content as ONE PART PER ENTRY, for the wire formats whose
    content is a list of typed parts — Gemini's `parts` and Bedrock
    Converse's `content`.

    Text entries become that format's text part, `{"text": ...}` (the shape
    both `GeminiAdapter` and `BedrockAdapter` read, and the shape verified
    against the real bodies langchain-google-genai and langchain-aws put on
    the wire). EVERY OTHER ENTRY IS PASSED THROUGH UNCHANGED, so the
    adapter's own per-part handling gets to see it — above all
    `ctxdiff.images.image_raw_block`, which recognizes an image part in any
    provider's shape and turns it into an image block whose identity is the
    picture's BYTES. That is what makes the same screenshot ONE block whether
    it arrived as LangChain's `image_url` data URI here or as Gemini
    `inline_data` / Bedrock `image` bytes in a direct capture.

    The alternative — flattening the whole list to one string through
    `_text_of` — is what these two branches used to do, and it lost data
    twice over: every non-text part was dropped outright (a vision turn
    recorded 1 block instead of 2, and the image's entire token cost
    disappeared from `ctxdiff tokens`), and several text parts collapsed into
    a single block where the real wire — and ctxdiff's own OpenAI branch —
    keep them separate. Both made the same turn hash differently through
    LangChain than through a direct capture, which is the one thing this
    module exists to prevent.

    Empty text contributes NO part rather than an empty one: LangChain does
    not put an empty text part on the wire, so emitting one would itself be a
    divergence (a phantom empty block)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        text = str(content)
        return [{"text": text}] if text else []
    parts: list = []
    for entry in content:
        if isinstance(entry, str):
            if entry:
                parts.append({"text": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            if entry["text"]:
                parts.append({"text": entry["text"]})
        else:
            parts.append(entry)
    return parts


def _openai_messages(messages: list) -> list[dict]:
    """LangChain messages -> OpenAI Chat Completions `messages`.

    Verified against the real request bodies langchain-openai sends (see the
    eval tests): a plain message is `{"role", "content"}`; an assistant
    message carrying tool calls sends `content: None` alongside `tool_calls`;
    a `ToolMessage` becomes `{"role": "tool", "content", "tool_call_id"}`.
    List content (multimodal parts) is passed through untouched — the OpenAI
    adapter already turns each part into its own block, images included."""
    out: list[dict] = []
    for message in messages:
        role = _role_of(message)
        content = getattr(message, "content", None)
        entry: dict = {"role": role, "content": content}
        tool_calls = _tool_calls_of(message) if role == "assistant" else []
        if tool_calls:
            entry["tool_calls"] = tool_calls
            if not content:
                entry["content"] = None
        if role == "tool":
            entry["tool_call_id"] = getattr(message, "tool_call_id", None)
        out.append(entry)
    return out


def _anthropic_wire(messages: list) -> dict:
    """LangChain messages -> Anthropic Messages `system` + `messages`.

    Anthropic takes the system prompt OUT of the message list, as a
    top-level field, which is exactly what `AnthropicAdapter` expects. A
    single string system message becomes the bare string form (identical
    blocks to a direct `client.messages.create(system="...")` call); several
    become the list-of-text-blocks form. Tool calls and tool results become
    Anthropic's own `tool_use`/`tool_result` content blocks when the message
    doesn't already carry them verbatim — ChatAnthropic keeps the provider's
    raw content blocks on the message, in which case they pass straight
    through and the wire is reproduced exactly."""
    systems: list[str] = []
    out: list[dict] = []
    for message in messages:
        role = _role_of(message)
        content = getattr(message, "content", None)
        if role == "system":
            systems.append(_text_of(content))
            continue
        if role == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": getattr(message, "tool_call_id", None),
                "content": content,
            }]})
            continue
        calls = getattr(message, "tool_calls", None) if role == "assistant" else None
        if calls and not isinstance(content, list):
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": _text_of(content)})
            for call in calls:
                blocks.append({"type": "tool_use", "id": call.get("id"),
                               "name": call.get("name"), "input": call.get("args") or {}})
            out.append({"role": role, "content": blocks})
            continue
        out.append({"role": role, "content": content})
    wire: dict = {"messages": out}
    if len(systems) == 1:
        wire["system"] = systems[0]
    elif systems:
        wire["system"] = [{"type": "text", "text": text} for text in systems]
    return wire


def _gemini_wire(messages: list) -> dict:
    """LangChain messages -> google-genai `contents` + `config.
    system_instruction`.

    Gemini names the assistant role "model" and wraps every message in a
    `parts` list, which is what `GeminiAdapter` reads; a tool call becomes a
    `function_call` part and a tool result a `function_response` part, the
    same shapes ChatVertexAI/ChatGoogleGenerativeAI put on the wire.

    Message content becomes one part per content entry (`_wire_parts`) — the
    real wire carries `[{"text": "a"}, {"text": "b"}, {"inlineData": {...}}]`
    for a two-text-plus-image turn, so the blocks must too."""
    system: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = _role_of(message)
        content = getattr(message, "content", None)
        if role == "system":
            system.append(_text_of(content))
            continue
        if role == "tool":
            contents.append({"role": "user", "parts": [{"function_response": {
                "name": getattr(message, "name", None),
                "response": {"result": _text_of(content)},
            }}]})
            continue
        parts: list = _wire_parts(content)
        for call in (getattr(message, "tool_calls", None) or []) if role == "assistant" else []:
            parts.append({"function_call": {"name": call.get("name"),
                                            "args": call.get("args") or {}}})
        contents.append({"role": "model" if role == "assistant" else "user",
                         "parts": parts})
    wire: dict = {"contents": contents}
    if system:
        wire["config"] = {"system_instruction": "\n".join(system)}
    return wire


def _bedrock_wire(messages: list) -> dict:
    """LangChain messages -> Bedrock Converse `system` + `messages`.

    Converse takes system prompts as a list of `{"text": ...}` blocks and
    every message's content as a list of typed blocks, with tool calls as
    `toolUse` and results as `toolResult` inside a USER-role message (the
    Converse API has no tool role) — the shapes `BedrockAdapter` reads.

    Message content becomes one block per content entry (`_wire_parts`) — the
    real wire carries `[{"text": "a"}, {"text": "b"}, {"image": {...}}]` for a
    two-text-plus-image turn, so the recorded blocks must too."""
    system: list[dict] = []
    out: list[dict] = []
    for message in messages:
        role = _role_of(message)
        content = getattr(message, "content", None)
        if role == "system":
            system.append({"text": _text_of(content)})
            continue
        if role == "tool":
            out.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": getattr(message, "tool_call_id", None),
                "content": [{"text": _text_of(content)}],
            }}]})
            continue
        blocks: list = _wire_parts(content)
        for call in (getattr(message, "tool_calls", None) or []) if role == "assistant" else []:
            blocks.append({"toolUse": {"toolUseId": call.get("id"),
                                       "name": call.get("name"),
                                       "input": call.get("args") or {}}})
        out.append({"role": role, "content": blocks})
    wire: dict = {"messages": out}
    if system:
        wire["system"] = system
    return wire


def to_wire(provider: str, messages: list, invocation_params: object,
            model_name: object = None) -> dict:
    """Rebuild the request dict LangChain is about to send, in `provider`'s
    own wire shape — the single function the whole hash-identity promise
    rests on (see the module docstring).

    Content comes from the messages, normalized per provider. TOOL SCHEMAS
    and sampling params come from `invocation_params`, which LangChain has
    ALREADY converted into the target provider's format (`ChatOpenAI` hands
    over OpenAI-shaped tool dicts, `ChatAnthropic` Anthropic-shaped ones), so
    they are carried across verbatim into whichever key that provider's
    request uses — `tools` for OpenAI/Anthropic, `config.tools` for Gemini,
    `toolConfig.tools` for Bedrock — and are therefore byte-identical to what
    a direct call would have recorded. Everything else in
    `invocation_params` (temperature, max_tokens, ...) is merged in as
    ordinary request kwargs, which is exactly where each adapter's
    `extract_params` picks up params from, and the model id is written under
    the key that provider uses so the session's model roll-up still works."""
    params = dict(invocation_params) if isinstance(invocation_params, dict) else {}
    tools = params.pop("tools", None)
    params.pop("functions", None)  # legacy alias; schemas are carried via `tools`

    if provider == "anthropic":
        wire = _anthropic_wire(messages)
        if tools:
            wire["tools"] = tools
    elif provider == "gemini":
        wire = _gemini_wire(messages)
        if tools:
            wire.setdefault("config", {})["tools"] = tools
    elif provider == "bedrock":
        wire = _bedrock_wire(messages)
        if tools:
            wire["toolConfig"] = {"tools": tools}
    else:
        wire = {"messages": _openai_messages(messages)}
        if tools:
            wire["tools"] = tools

    model_key = _MODEL_KEYS.get(provider, "model")
    # Pop EVERY alias unconditionally, then pick the first one that had a
    # value — not `pop(a) or pop(b) or ...`, which short-circuits: `ChatOpenAI`
    # reports both `model` and `model_name`, so the loser stayed in `params`
    # and was merged in below as a spurious extra request key that no direct
    # capture would ever carry. The alias ORDER matches the JS twin's.
    aliases = [params.pop(key, None)
               for key in ("model_id", "modelId", "model", "model_name")]
    model = next((alias for alias in aliases if alias is not None), model_name)
    for key, value in params.items():
        # Never let a param overwrite content the messages already produced.
        if key not in wire:
            wire[key] = value
    if model is not None:
        wire[model_key] = model
    return wire


def usage_state(provider: str, result: object) -> dict:
    """Map LangChain's normalized token counts onto the key names
    `provider`'s own `extract_usage` returns, so a LangChain-captured call
    stores the same `usage` dict a directly-captured one would.

    Two sources, in order. `generations[0][0].message.usage_metadata` is
    langchain-core's PROVIDER-INDEPENDENT shape (input_tokens/output_tokens/
    total_tokens) and is populated for streaming runs too, where
    `llm_output` is None — which is why it is tried first. `llm_output` is
    the fallback, under either of the two names integrations use for it
    (`token_usage` for OpenAI-family, `usage` for Anthropic-family). An empty
    dict means no counts were reported at all, and the call is recorded
    honestly with `usage=None` rather than fabricated zeros."""
    counts = _usage_metadata(result)
    if not counts:
        return {}
    input_key, output_key, total_key = _USAGE_KEYS.get(provider, _USAGE_KEYS["openai"])
    state = {input_key: counts.get("input_tokens"),
             output_key: counts.get("output_tokens")}
    if total_key is not None:
        state[total_key] = counts.get("total_tokens")
    return state


def _usage_metadata(result: object) -> dict:
    """Pull `{input_tokens, output_tokens, total_tokens}` out of an
    `LLMResult`, whichever place this integration put them. Duck-typed and
    defensive throughout — a result shape nobody anticipated must yield no
    usage, never an exception, since this runs inside the recording path of
    a host's live agent."""
    try:
        generations = getattr(result, "generations", None) or []
        for batch in generations:
            for generation in batch or []:
                usage = getattr(getattr(generation, "message", None),
                                "usage_metadata", None)
                if isinstance(usage, dict) and usage:
                    return usage
        output = getattr(result, "llm_output", None)
        if isinstance(output, dict):
            for key in ("token_usage", "usage"):
                usage = output.get(key)
                if isinstance(usage, dict) and usage:
                    return {
                        "input_tokens": usage.get("prompt_tokens",
                                                  usage.get("input_tokens")),
                        "output_tokens": usage.get("completion_tokens",
                                                   usage.get("output_tokens")),
                        "total_tokens": usage.get("total_tokens"),
                    }
    except Exception:  # noqa: BLE001 — usage is best-effort, never a failure
        pass
    return {}


class _Pending:
    """One in-flight LangChain run: everything `on_llm_end` needs that only
    `on_chat_model_start` knew. Kept per `run_id` because LangChain
    interleaves runs freely (a LangGraph fan-out, a `RunnableParallel`), so
    "the last request seen" is never a safe assumption."""

    __slots__ = ("start", "kwargs", "provider", "recorder")

    def __init__(self, start: float, kwargs: dict, provider: str, recorder: object):
        self.start = start
        self.kwargs = kwargs
        self.provider = provider
        self.recorder = recorder


# How many concurrently-open runs are tracked before the oldest are dropped.
# A run whose end callback never fires (a cancelled task, a handler removed
# mid-flight) would otherwise pin its request payload forever; a debugging
# tool must not be able to leak a host's memory. Far above any real fan-out.
_MAX_PENDING = 2048


class _HandlerCore:
    """The whole handler, minus its base class.

    Written as a standalone mixin so the logic is readable at module level
    and importable without langchain: `build_handler` combines it with
    `langchain_core`'s `BaseCallbackHandler` at call time (LangChain's
    pydantic-validated `callbacks=` field rejects anything that isn't a real
    subclass — duck typing does NOT work here, confirmed against
    langchain-core 1.5). ctxdiff's own dependencies stay tiktoken-only.

    EVERY callback is fail-open. LangChain does catch handler exceptions by
    default, but that is not a guarantee this code is entitled to lean on
    (`raise_error=True` exists, and other frameworks re-dispatch these
    callbacks), so each method swallows its own failures: a broken tracer
    must never break the agent it is watching."""

    def __init__(self, tracer: object, agent: str | None = None):
        """Hold the tracer to record into and the agent name to stamp on
        every call this handler records, plus the per-run bookkeeping (a
        plain dict under a lock — LangChain dispatches callbacks from
        whatever thread or task the run is on)."""
        super().__init__()
        self._tracer = tracer
        self._agent = agent
        self._pending: dict = {}
        self._lock = threading.Lock()

    # LangChain reads these to decide whether to dispatch at all; ctxdiff
    # wants every LLM/chat-model event and nothing else, but the base class
    # defaults already say so, so they are left alone.

    def on_chat_model_start(self, serialized, messages, *, run_id=None,
                            **kwargs) -> None:
        """A chat model is about to be called: derive the provider, rebuild
        the provider's wire request from the messages + invocation params,
        and park it under this run id until the matching end/error callback.
        `messages` is a list of message LISTS (one per prompt in a batch);
        chat models are effectively always called with one, and only the
        first is recorded — a genuinely batched `generate([...])` reports one
        run id for the whole batch, so recording several calls against it
        would invent turns that never happened."""
        try:
            batch = messages[0] if messages else []
            self._start(serialized, list(batch), run_id, kwargs)
        except Exception:  # noqa: BLE001 — fail-open: never break the host's run
            _log.warning("ctxdiff: langchain on_chat_model_start failed "
                         "(this call will not be recorded)", exc_info=True)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        """The non-chat (completion-style) entry point. Each prompt string is
        treated as a user message so the same normalization, the same
        adapter and the same block shapes apply; as with a chat batch, only
        the first prompt of a batched call is recorded."""
        try:
            prompt = prompts[0] if prompts else ""
            self._start(serialized, [_PlainPrompt(prompt)], run_id, kwargs)
        except Exception:  # noqa: BLE001 — fail-open
            _log.warning("ctxdiff: langchain on_llm_start failed "
                         "(this call will not be recorded)", exc_info=True)

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        """The call finished: pair the result with the parked request, map
        its token counts into the provider's own usage shape, and record —
        through the tracer's ordinary fail-open `_on_create`, so a
        LangChain-captured call goes through exactly the same seq
        assignment, tag/step attribution, snapshotting and writer path as a
        directly-captured one."""
        try:
            pending = self._take(run_id)
            if pending is None:
                return
            state = usage_state(pending.provider, response)
            self._record(pending, SyntheticUsageResponse(state) if state else None, None)
        except Exception:  # noqa: BLE001 — fail-open
            _log.warning("ctxdiff: langchain on_llm_end failed "
                         "(this call will not be recorded)", exc_info=True)

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        """The call failed: record it as a FAILED call carrying the
        exception's type name, exactly as the direct path records a provider
        error, so a failed turn stays visible in the trace instead of
        vanishing from it."""
        try:
            pending = self._take(run_id)
            if pending is None:
                return
            self._record(pending, None, type(error).__name__)
        except Exception:  # noqa: BLE001 — fail-open
            _log.warning("ctxdiff: langchain on_llm_error failed "
                         "(this call will not be recorded)", exc_info=True)

    def _start(self, serialized, messages: list, run_id, kwargs: dict) -> None:
        """Shared body of the two start callbacks: resolve provider ->
        recorder -> wire request, and park it. The latency clock starts here,
        so a recorded LangChain call measures the same span the direct path
        does (request issued -> result in hand)."""
        metadata = kwargs.get("metadata")
        provider = provider_for(serialized, metadata)
        recorder = self._tracer._recorder_for(provider)
        if recorder is None:
            return
        model_name = metadata.get("ls_model_name") if isinstance(metadata, dict) else None
        wire = to_wire(provider, messages, kwargs.get("invocation_params"), model_name)
        self._park(run_id, _Pending(time.perf_counter(), wire, provider, recorder))

    def _park(self, run_id, pending: _Pending) -> None:
        """Remember an in-flight run, bounded. If `_MAX_PENDING` runs are
        already open — which means end callbacks are going missing, not that
        anyone really has that much in flight — the OLDEST is dropped so
        memory stays bounded; dicts preserve insertion order, so "oldest" is
        just the first key."""
        with self._lock:
            if len(self._pending) >= _MAX_PENDING:
                self._pending.pop(next(iter(self._pending)), None)
            self._pending[run_id] = pending

    def _take(self, run_id) -> _Pending | None:
        """Claim an in-flight run, exactly once — a duplicate end callback
        (or an error following an end) then finds nothing and records
        nothing, which is the same "record exactly once" guarantee the stream
        proxies get from their `_finalized` flag."""
        with self._lock:
            return self._pending.pop(run_id, None)

    def _record(self, pending: _Pending, response: object, error: str | None) -> None:
        """Hand one finished call to the tracer. Nothing here is
        LangChain-specific any more: from this line on the call is
        indistinguishable from one captured by wrapping the SDK directly."""
        latency_ms = int((time.perf_counter() - pending.start) * 1000)
        self._tracer._on_create(pending.kwargs, response, latency_ms, error,
                                pending.recorder, self._agent, pending.provider)


class _PlainPrompt:
    """A completion-API prompt string dressed as a message, so `on_llm_start`
    can reuse the chat normalizers instead of having its own path."""

    type = "human"

    def __init__(self, content: str):
        self.content = content


# Built once, on first use, and cached: the class object must be stable so
# repeated `langchain_handler()` calls produce handlers of the same type.
_HANDLER_CLASS = None
_HANDLER_CLASS_LOCK = threading.Lock()


def _handler_class():
    """The `CtxdiffCallbackHandler` type — `_HandlerCore` combined with
    langchain-core's `BaseCallbackHandler`.

    Built lazily because ctxdiff does not depend on langchain: importing it
    at module scope would make `import ctxdiff` fail for everyone who
    doesn't use LangChain. A missing langchain-core raises here, loudly, with
    the install line — this is a setup-time mistake the caller made
    explicitly by asking for a LangChain handler, exactly like `wrap()`
    raising on an unrecognized client, and silently returning a handler that
    records nothing would be far worse."""
    global _HANDLER_CLASS
    with _HANDLER_CLASS_LOCK:
        if _HANDLER_CLASS is None:
            try:
                from langchain_core.callbacks.base import BaseCallbackHandler
            except ImportError as exc:  # pragma: no cover — needs langchain absent
                raise ImportError(
                    "ctxdiff: tracer.langchain_handler() needs langchain-core; "
                    "install it with `pip install langchain-core`") from exc
            _HANDLER_CLASS = type("CtxdiffCallbackHandler",
                                  (_HandlerCore, BaseCallbackHandler), {})
        return _HANDLER_CLASS


def build_handler(tracer: object, agent: str | None = None):
    """Return a LangChain callback handler that records every chat-model call
    into `tracer`, stamped with `agent`. Used by `Tracer.langchain_handler()`
    — see that method for the user-facing docs."""
    return _handler_class()(tracer, agent)

"""Unit tests for the LangChain normalization layer (`ctxdiff.capture.
langchain`) — provider derivation, message -> wire-shape rebuilding, and
usage mapping.

These run WITHOUT langchain installed: the module never imports langchain
types (it duck-types messages via `getattr`), so the stand-ins below are
enough to exercise every branch. The end-to-end proof that these shapes match
what LangChain really sends lives in `tests/eval/test_langchain_handler.py`
and `tests/eval/test_langgraph.py`, which compare the resulting hashes
against captured wire bodies; this file covers the branches those two can't
reach without every provider integration installed."""
from __future__ import annotations

import base64
import json
import struct
import zlib

from ctxdiff.capture.anthropic import AnthropicAdapter
from ctxdiff.capture.bedrock import BedrockAdapter
from ctxdiff.capture.gemini import GeminiAdapter
from ctxdiff.capture.langchain import (DEFAULT_PROVIDER, provider_for, to_wire,
                                       usage_state)
from ctxdiff.capture.openai import OpenAIAdapter
from ctxdiff.images import ImageRef, image_hash_input


def _png(width: int, height: int) -> bytes:
    """A structurally valid PNG declaring `width`x`height` in its IHDR — the
    same hand-built header `tests/test_images.py` uses, since the sniffer only
    ever reads the first few dozen bytes and no binary fixture belongs in the
    repo for that."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    ihdr = struct.pack(">II", width, height) + bytes([8, 2, 0, 0, 0])
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00")) + chunk(b"IEND", b""))


#: One tiny picture, and its base64, reused by every multimodal test below.
PNG_4x4 = _png(4, 4)
_B64_PNG = base64.b64encode(PNG_4x4).decode("ascii")


class Msg:
    """A stand-in LangChain message: `.type`, `.content`, and whichever of
    `.tool_calls`/`.tool_call_id`/`.additional_kwargs` the real class would
    carry. Exactly the attributes the normalizer reads."""

    def __init__(self, type_, content, tool_calls=None, tool_call_id=None,
                 additional_kwargs=None, name=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.additional_kwargs = additional_kwargs or {}
        self.name = name


def _serialized(class_name: str) -> dict:
    """The `serialized` dict LangChain passes to `on_chat_model_start`."""
    return {"id": ["langchain", "chat_models", "x", class_name], "name": class_name}


# --------------------------------------------------------------------------
# provider derivation


def test_provider_comes_from_langchains_own_ls_provider_first():
    """`metadata["ls_provider"]` is set by langchain-core itself, so it is
    the primary signal — it survives an integration renaming its class."""
    assert provider_for(_serialized("SomethingNew"),
                        {"ls_provider": "anthropic"}) == "anthropic"
    assert provider_for(None, {"ls_provider": "google_vertexai"}) == "gemini"
    assert provider_for(None, {"ls_provider": "amazon_bedrock"}) == "bedrock"
    assert provider_for(None, {"ls_provider": "azure_openai"}) == "openai"


def test_provider_falls_back_to_the_model_class_name():
    """Without metadata (an older langchain-core, a hand-rolled model), the
    class name from `serialized["id"][-1]` decides."""
    assert provider_for(_serialized("ChatAnthropic"), None) == "anthropic"
    assert provider_for(_serialized("ChatVertexAI"), None) == "gemini"
    assert provider_for(_serialized("ChatBedrockConverse"), None) == "bedrock"
    assert provider_for(_serialized("ChatOpenAI"), None) == "openai"


def test_unknown_integration_falls_back_to_the_openai_wire_shape():
    """An unrecognized integration is recorded in the most widely-compatible
    shape rather than dropped — and the choice is visible, because the
    provider is stored on the call."""
    assert provider_for(_serialized("ChatSomethingBrandNew"), {}) == DEFAULT_PROVIDER
    assert provider_for(None, None) == DEFAULT_PROVIDER


# --------------------------------------------------------------------------
# OpenAI wire shape


def test_openai_wire_rebuilds_messages_tools_and_params():
    """The default shape: roles mapped from LangChain's own type names,
    tool schemas carried through verbatim under `tools`, and everything else
    from `invocation_params` merged in as ordinary request kwargs."""
    tools = [{"type": "function", "function": {"name": "f"}}]
    wire = to_wire("openai",
                   [Msg("system", "sys"), Msg("human", "hi")],
                   {"model": "gpt-4o", "temperature": 0.2, "tools": tools})
    assert wire["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "hi"}]
    assert wire["tools"] == tools
    assert wire["model"] == "gpt-4o"
    assert wire["temperature"] == 0.2


def test_openai_wire_rebuilds_tool_calls_the_way_langchain_reserializes_them():
    """The normalized `.tool_calls` is rebuilt with `json.dumps(args)` —
    the identical call LangChain's own converter makes when it sends the
    message back — so the `arguments` string matches the wire character for
    character, whitespace included. (Using the provider's raw JSON text
    instead would differ by exactly the bytes the model happened to use, and
    LangChain doesn't preserve those.)"""
    wire = to_wire("openai", [Msg("ai", "", tool_calls=[
        {"name": "f", "args": {"city": "Dubai"}, "id": "call_1"}])], {})
    assert wire["messages"][0]["tool_calls"] == [
        {"id": "call_1", "type": "function",
         "function": {"name": "f", "arguments": '{"city": "Dubai"}'}}]
    assert wire["messages"][0]["content"] is None      # matches the real wire


def test_openai_wire_falls_back_to_the_raw_tool_calls_when_thats_all_there_is():
    """An integration that keeps only `additional_kwargs["tool_calls"]` (no
    normalized form) still gets its tool calls captured, verbatim."""
    raw = [{"id": "call_1", "type": "function",
            "function": {"name": "f", "arguments": '{"x": 1}'}}]
    wire = to_wire("openai", [Msg("ai", "", additional_kwargs={"tool_calls": raw})], {})
    assert wire["messages"][0]["tool_calls"] == raw


def test_openai_wire_keeps_tool_results_and_multimodal_parts():
    """A tool result becomes a `tool`-role message with its call id; list
    content (multimodal parts) passes through untouched, so the adapter's
    per-part block extraction — images included — applies unchanged."""
    parts = [{"type": "text", "text": "look"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    wire = to_wire("openai", [Msg("tool", "sunny", tool_call_id="call_1"),
                              Msg("human", parts)], {})
    assert wire["messages"][0] == {"role": "tool", "content": "sunny",
                                   "tool_call_id": "call_1"}
    assert wire["messages"][1]["content"] == parts


def test_openai_wire_blocks_match_a_hand_written_direct_request():
    """The identity claim, stated at the block level and checked against the
    adapter directly: the rebuilt wire produces the same blocks a caller's
    own hand-written request would."""
    wire = to_wire("openai", [Msg("system", "sys"), Msg("human", "hi")],
                   {"model": "gpt-4o"})
    direct = {"model": "gpt-4o",
              "messages": [{"role": "system", "content": "sys"},
                           {"role": "user", "content": "hi"}]}
    adapter = OpenAIAdapter()
    assert ([(b.role, b.kind, b.text) for b in adapter.extract_blocks(wire)]
            == [(b.role, b.kind, b.text) for b in adapter.extract_blocks(direct)])


# --------------------------------------------------------------------------
# the other three providers


def test_anthropic_wire_lifts_the_system_prompt_out_of_the_messages():
    """Anthropic takes the system prompt as a TOP-LEVEL field. A single
    string system message becomes the bare string form — identical blocks to
    a direct `messages.create(system="...")` — and several become the
    list-of-text-blocks form."""
    wire = to_wire("anthropic", [Msg("system", "sys"), Msg("human", "hi")],
                   {"model": "claude", "max_tokens": 100})
    assert wire["system"] == "sys"
    assert wire["messages"] == [{"role": "user", "content": "hi"}]
    assert wire["max_tokens"] == 100

    two = to_wire("anthropic", [Msg("system", "a"), Msg("system", "b")], {})
    assert two["system"] == [{"type": "text", "text": "a"},
                             {"type": "text", "text": "b"}]
    assert [(b.role, b.kind, b.text) for b in AnthropicAdapter().extract_blocks(two)] \
        == [("system", "message", "a"), ("system", "message", "b")]


def test_anthropic_wire_uses_tool_use_and_tool_result_blocks():
    """Anthropic expresses tool calls as `tool_use` content blocks and their
    results as `tool_result` blocks inside a USER message — the shapes
    `AnthropicAdapter` reads."""
    wire = to_wire("anthropic", [
        Msg("ai", "", tool_calls=[{"name": "f", "args": {"x": 1}, "id": "t1"}]),
        Msg("tool", "sunny", tool_call_id="t1"),
    ], {})
    assert wire["messages"][0]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}}]
    assert wire["messages"][1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}]}


def test_gemini_wire_uses_contents_parts_and_the_model_role():
    """Gemini has no `messages`: content goes in `contents`, each entry
    wrapping `parts`, with the assistant role named "model" and the system
    prompt moved into `config.system_instruction`."""
    wire = to_wire("gemini", [Msg("system", "sys"), Msg("human", "hi"),
                              Msg("ai", "there")], {"model": "gemini-2.0-flash"})
    assert wire["config"]["system_instruction"] == "sys"
    assert wire["contents"] == [{"role": "user", "parts": [{"text": "hi"}]},
                                {"role": "model", "parts": [{"text": "there"}]}]
    assert wire["model"] == "gemini-2.0-flash"
    assert [(b.role, b.kind, b.text) for b in GeminiAdapter().extract_blocks(wire)] \
        == [("system", "message", "sys"), ("user", "content_part", "hi"),
            ("assistant", "content_part", "there")]


def test_bedrock_wire_uses_converse_shapes_and_the_model_id_key():
    """Converse takes `system` as a list of text blocks, every message's
    content as a list of typed blocks, tool schemas under `toolConfig`, and
    names the model `modelId` — which is also the key the session's model
    roll-up reads for this provider."""
    tools = [{"toolSpec": {"name": "f"}}]
    wire = to_wire("bedrock", [Msg("system", "sys"), Msg("human", "hi")],
                   {"model_id": "anthropic.claude-3-haiku", "tools": tools})
    assert wire["system"] == [{"text": "sys"}]
    assert wire["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert wire["toolConfig"] == {"tools": tools}
    assert wire["modelId"] == "anthropic.claude-3-haiku"
    assert [(b.role, b.kind, b.text) for b in BedrockAdapter().extract_blocks(wire)] \
        == [("system", "message", "sys"), ("system", "tool_schema",
                                           json.dumps({"name": "f"}, sort_keys=True)),
            ("user", "content_part", "hi")]


# --------------------------------------------------------------------------
# one part per content entry (the typed-parts providers)
#
# Gemini and Bedrock are the two branches that build a list of TYPED PARTS
# rather than passing list content through the way OpenAI/Anthropic do, so
# they are the two that can silently lose a part. Both are checked the same
# way: through the real adapter, because "how many blocks does this turn
# record" is the question that actually matters.


def _content_blocks(provider: str, wire: dict) -> list:
    """The blocks the provider's own adapter extracts from a rebuilt wire —
    the same adapter, and therefore the same block extraction, a direct
    capture of that request would go through."""
    adapter = GeminiAdapter() if provider == "gemini" else BedrockAdapter()
    return adapter.extract_blocks(wire)


def test_gemini_wire_emits_one_part_per_content_entry():
    """A vision turn is TWO parts, not one. LangChain's multimodal content is
    a list of entries — text entries plus an image entry — and each one is a
    separate part on the real Gemini wire, so each must become its own block.
    Flattening the list to a single string dropped the image entirely (its
    whole token cost vanished) and merged the two text entries into one
    block, so the same turn recorded fewer blocks through LangChain than
    through a direct capture and stopped deduping against it."""
    image_part = {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{_B64_PNG}"}}
    wire = to_wire("gemini", [Msg("human", [{"type": "text", "text": "a"},
                                            {"type": "text", "text": "b"},
                                            image_part])], {})
    assert wire["contents"] == [{"role": "user", "parts": [
        {"text": "a"}, {"text": "b"}, image_part]}]

    blocks = _content_blocks("gemini", wire)
    assert [(b.role, b.kind, b.text) for b in blocks[:2]] == [
        ("user", "content_part", "a"), ("user", "content_part", "b")]
    # The image is an IMAGE block — descriptor text, bytes-based identity, a
    # real vision-token estimate — never a JSON blob and never absent.
    assert blocks[2].kind == "image"
    assert blocks[2].text == "[image 4×4 · ~258 tok]"
    assert blocks[2].hash_input == image_hash_input(ImageRef(data=PNG_4x4))
    assert len(blocks) == 3


def test_bedrock_wire_emits_one_content_block_per_content_entry():
    """The Bedrock Converse twin of the Gemini case: Converse content is a
    list of typed blocks, so two text entries and an image are three blocks —
    with the image hashed over its BYTES, identical to the same picture
    captured directly."""
    image_part = {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{_B64_PNG}"}}
    wire = to_wire("bedrock", [Msg("human", [{"type": "text", "text": "a"},
                                             {"type": "text", "text": "b"},
                                             image_part])], {})
    assert wire["messages"] == [{"role": "user", "content": [
        {"text": "a"}, {"text": "b"}, image_part]}]

    blocks = _content_blocks("bedrock", wire)
    assert [(b.role, b.kind, b.text) for b in blocks[:2]] == [
        ("user", "content_part", "a"), ("user", "content_part", "b")]
    assert blocks[2].kind == "image"
    assert blocks[2].hash_input == image_hash_input(ImageRef(data=PNG_4x4))
    assert len(blocks) == 3


def test_typed_part_providers_hash_a_vision_turn_like_a_direct_capture():
    """The identity claim for images, stated where it broke. The blocks the
    handler's rebuilt wire produces must be hash-identical to the blocks a
    DIRECT capture of the provider's own request shape produces — the same
    picture arriving as an OpenAI-style data URI (what LangChain hands the
    handler) and as Gemini `inline_data` / Bedrock `image` bytes (what goes
    on the wire) is ONE block, because identity is the image bytes."""
    image_part = {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{_B64_PNG}"}}
    messages = [Msg("human", [{"type": "text", "text": "what is this?"}, image_part])]

    gemini_direct = {"contents": [{"role": "user", "parts": [
        {"text": "what is this?"},
        {"inline_data": {"mime_type": "image/png", "data": _B64_PNG}}]}]}
    assert ([b.hash_input for b in _content_blocks("gemini", to_wire("gemini", messages, {}))]
            == [b.hash_input for b in GeminiAdapter().extract_blocks(gemini_direct)])

    bedrock_direct = {"messages": [{"role": "user", "content": [
        {"text": "what is this?"},
        {"image": {"format": "png", "source": {"bytes": PNG_4x4}}}]}]}
    assert ([b.hash_input for b in _content_blocks("bedrock", to_wire("bedrock", messages, {}))]
            == [b.hash_input for b in BedrockAdapter().extract_blocks(bedrock_direct)])


def test_typed_part_providers_keep_plain_string_content_as_one_text_part():
    """The common case is unchanged: a plain string message is still exactly
    one text part, and an empty one contributes no part at all rather than a
    phantom empty block."""
    gemini = to_wire("gemini", [Msg("human", "hi"), Msg("ai", "")], {})
    assert gemini["contents"] == [{"role": "user", "parts": [{"text": "hi"}]},
                                  {"role": "model", "parts": []}]
    bedrock = to_wire("bedrock", [Msg("human", "hi"), Msg("ai", "")], {})
    assert bedrock["messages"] == [{"role": "user", "content": [{"text": "hi"}]},
                                   {"role": "assistant", "content": []}]


def test_model_key_aliases_are_all_consumed_from_the_params():
    """Every alias a model id can arrive under is popped, not just the first
    one that matched. `ChatOpenAI`'s invocation params carry BOTH `model` and
    `model_name`, so a short-circuiting `pop(...) or pop(...)` left the
    loser behind and it was merged in as a spurious extra request param —
    which then showed up in the stored params of a LangChain-captured call
    and nowhere else."""
    wire = to_wire("openai", [Msg("human", "hi")],
                   {"model": "gpt-4o", "model_name": "gpt-4o", "temperature": 0.2})
    assert wire["model"] == "gpt-4o"
    assert "model_name" not in wire
    assert wire["temperature"] == 0.2

    # ...and the aliases are tried in the same order the JS twin tries them,
    # so a Bedrock integration naming it `modelId` lands under `modelId` too.
    bedrock = to_wire("bedrock", [Msg("human", "hi")],
                      {"modelId": "anthropic.claude-3-haiku", "model_name": "claude"})
    assert bedrock["modelId"] == "anthropic.claude-3-haiku"
    assert "model_name" not in bedrock
    assert "model" not in bedrock


def test_model_name_falls_back_to_the_metadata_when_params_have_none():
    """`ls_model_name` backfills the model when `invocation_params` carries
    none, so the session's model roll-up is never blank for a provider whose
    integration doesn't echo the model in its params."""
    wire = to_wire("openai", [Msg("human", "hi")], {}, model_name="gpt-4o-mini")
    assert wire["model"] == "gpt-4o-mini"


def test_content_keys_are_never_overwritten_by_invocation_params():
    """A stray `messages`/`contents` key in `invocation_params` must not
    clobber the content rebuilt from the actual messages — the params are
    merged around the content, never over it."""
    wire = to_wire("openai", [Msg("human", "hi")],
                   {"messages": [{"role": "user", "content": "WRONG"}]})
    assert wire["messages"] == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# usage mapping


class _Gen:
    def __init__(self, usage):
        self.message = type("M", (), {"usage_metadata": usage})()


class _Result:
    def __init__(self, usage=None, llm_output=None):
        self.generations = [[_Gen(usage)]] if usage is not None else []
        self.llm_output = llm_output


def test_usage_is_mapped_onto_each_providers_own_key_names():
    """LangChain normalizes every provider's counts into one shape; ctxdiff
    maps them BACK onto the names that provider's `extract_usage` returns, so
    a LangChain-captured call's usage dict is identical to a directly
    captured one's."""
    counts = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    result = _Result(usage=counts)
    assert usage_state("openai", result) == {"prompt_tokens": 10,
                                             "completion_tokens": 5,
                                             "total_tokens": 15}
    assert usage_state("anthropic", result) == {"input_tokens": 10,
                                                "output_tokens": 5}
    assert usage_state("gemini", result) == {"prompt_token_count": 10,
                                             "candidates_token_count": 5,
                                             "total_token_count": 15}
    assert usage_state("bedrock", result) == {"inputTokens": 10, "outputTokens": 5,
                                              "totalTokens": 15}


def test_usage_falls_back_to_llm_output_under_either_name():
    """Older/other integrations report only `llm_output`, under
    `token_usage` (OpenAI family) or `usage` (Anthropic family)."""
    assert usage_state("openai", _Result(llm_output={"token_usage": {
        "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})) == {
        "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert usage_state("anthropic", _Result(llm_output={"usage": {
        "input_tokens": 3, "output_tokens": 1}})) == {"input_tokens": 3,
                                                      "output_tokens": 1}


def test_no_usage_reported_yields_an_empty_state_not_zeros():
    """No counts anywhere means the call is recorded honestly with
    `usage=None` — never fabricated zeros, and never an exception, whatever
    shape the result turns out to be."""
    assert usage_state("openai", _Result()) == {}
    assert usage_state("openai", _Result(llm_output={})) == {}
    assert usage_state("openai", object()) == {}

from ctxdiff.capture.bedrock import BedrockAdapter


def test_extract_blocks_orders_system_then_tools_then_messages():
    """`system` (list of {"text":...} dicts) is emitted first, one 'system'-role
    'message' block per entry; then `toolConfig.tools` as 'tool_schema' blocks;
    then `messages` content parts. Order mirrors send order (system → tool
    schemas → messages)."""
    kwargs = {
        "modelId": "anthropic.claude-3-haiku",
        "system": [{"text": "Be helpful."}],
        "toolConfig": {"tools": [{"toolSpec": {"name": "lookup", "inputSchema": {}}}]},
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    blocks = BedrockAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["message", "tool_schema", "content_part"]
    assert [b.role for b in blocks] == ["system", "system", "user"]
    assert blocks[0].text == "Be helpful."
    assert "lookup" in blocks[1].text
    assert blocks[2].text == "hi"


def test_extract_blocks_system_non_text_block_serializes_stable_json():
    """A `system` entry without a 'text' key (some other Converse system block
    shape) falls back to stable (sort_keys) JSON so it stays diffable."""
    kwargs = {"system": [{"cachePoint": {"type": "default"}}], "messages": []}
    blocks = BedrockAdapter().extract_blocks(kwargs)
    assert blocks[0].kind == "message"
    assert blocks[0].role == "system"
    assert "cachePoint" in blocks[0].text


def test_extract_blocks_tool_config_multiple_tools():
    """Each tool in `toolConfig.tools` yields its own 'tool_schema' block, JSON
    of the toolSpec (not the outer {"toolSpec": ...} wrapper is asserted here,
    just that both schemas appear as separate blocks)."""
    kwargs = {
        "messages": [],
        "toolConfig": {"tools": [
            {"toolSpec": {"name": "a", "inputSchema": {}}},
            {"toolSpec": {"name": "b", "inputSchema": {}}},
        ]},
    }
    blocks = BedrockAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["tool_schema", "tool_schema"]
    assert "\"a\"" in blocks[0].text
    assert "\"b\"" in blocks[1].text


def test_extract_blocks_message_content_parts_text_and_non_text():
    """Each entry in a message's `content` list becomes its own 'content_part'
    block: a {"text": ...} entry keeps the text verbatim; any other shape
    (image, toolUse, toolResult) is stable-JSON serialized. Role is passed
    through as-is (Converse only has 'user'/'assistant' at the message level;
    toolResult content lives inside a user-role message, not a 'tool' role)."""
    kwargs = {
        "messages": [
            {"role": "user", "content": [
                {"text": "look this up"},
                {"toolResult": {"toolUseId": "1", "content": [{"text": "42"}]}},
            ]},
            {"role": "assistant", "content": [{"text": "the answer is 42"}]},
        ],
    }
    blocks = BedrockAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["content_part"] * 3
    assert [b.role for b in blocks] == ["user", "user", "assistant"]
    assert blocks[0].text == "look this up"
    assert "toolResult" in blocks[1].text
    assert "toolUseId" in blocks[1].text
    assert blocks[2].text == "the answer is 42"


def test_extract_blocks_no_system_or_tool_config():
    """Absent `system`/`toolConfig` keys don't raise; only message blocks come
    out."""
    kwargs = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
    blocks = BedrockAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["content_part"]
    assert blocks[0].text == "hi"


def test_extract_params_drops_content_keys_keeps_model_id_and_flattens_inference_config():
    """params exclude system/messages/toolConfig, keep modelId, and flatten
    inferenceConfig scalars (maxTokens, temperature, topP, stopSequences)."""
    kwargs = {
        "modelId": "anthropic.claude-3-haiku",
        "system": [{"text": "x"}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "toolConfig": {"tools": []},
        "inferenceConfig": {
            "maxTokens": 256, "temperature": 0.2, "topP": 0.9,
            "stopSequences": ["END"],
        },
    }
    params = BedrockAdapter().extract_params(kwargs)
    assert params == {
        "modelId": "anthropic.claude-3-haiku",
        "maxTokens": 256, "temperature": 0.2, "topP": 0.9,
        "stopSequences": ["END"],
    }


def test_extract_params_with_no_inference_config():
    """params work fine when inferenceConfig is absent entirely."""
    kwargs = {"modelId": "anthropic.claude-3-haiku", "messages": []}
    params = BedrockAdapter().extract_params(kwargs)
    assert params == {"modelId": "anthropic.claude-3-haiku"}


def test_extract_params_with_partial_inference_config():
    """Only the inferenceConfig fields actually present are flattened in —
    missing ones (accessed via .get) don't appear as None."""
    kwargs = {"modelId": "m", "messages": [], "inferenceConfig": {"maxTokens": 100}}
    params = BedrockAdapter().extract_params(kwargs)
    assert params == {"modelId": "m", "maxTokens": 100}


def test_extract_usage_from_dict_response():
    """The Converse API returns a plain dict; usage is read via .get, not
    getattr, and returned as a plain dict with the three Bedrock usage keys."""
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
    }
    usage = BedrockAdapter().extract_usage(response)
    assert usage == {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18}


def test_extract_usage_none_when_absent():
    """A dict response with no 'usage' key yields None."""
    assert BedrockAdapter().extract_usage({"output": {}}) is None


def test_extract_usage_object_shaped_response_fallback():
    """Duck-type fallback: if `response` isn't a dict (or has no usable
    'usage' key) but exposes `.usage` as an attribute, that's used instead —
    so an object-shaped response doesn't break usage extraction."""
    class Usage:
        inputTokens = 3
        outputTokens = 1
        totalTokens = 4

    class Resp:
        usage = Usage()

    usage = BedrockAdapter().extract_usage(Resp())
    assert usage == {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}


def test_extract_usage_none_for_object_without_usage():
    """An object response with no usage attribute at all yields None."""
    assert BedrockAdapter().extract_usage(object()) is None

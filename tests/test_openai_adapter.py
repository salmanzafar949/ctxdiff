from ctxdiff.capture.openai import OpenAIAdapter


def test_extract_blocks_orders_tools_then_messages():
    """Tool schemas are emitted first (they sit at the front of the context
    budget), then messages in order. Each tool → one 'tool_schema' block; each
    message → one block with its role."""
    kwargs = {
        "model": "gpt-4o",
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ],
    }
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["tool_schema", "message", "message"]
    assert [b.role for b in blocks] == ["system", "system", "user"]
    assert "lookup" in blocks[0].text
    assert blocks[2].text == "hi"


def test_extract_blocks_handles_multipart_content():
    """A message whose content is a list of parts yields one block per part,
    kind 'content_part', preserving order."""
    kwargs = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look at this"},
        {"type": "text", "text": "and this"},
    ]}]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["content_part", "content_part"]
    assert "look at this" in blocks[0].text


def test_extract_blocks_tool_calls_only_emits_content_part_no_empty_message():
    """An assistant message with content=None and tool_calls present must
    NOT emit an empty-text 'message' block — the tool_call part(s) ARE the
    message. Exactly one content_part block per tool call, role mirrors the
    message's role, text is stable JSON of the tool_call dict containing the
    function name."""
    kwargs = {"messages": [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Dubai"}'},
        }],
    }]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 1
    assert blocks[0].kind == "content_part"
    assert blocks[0].role == "assistant"
    assert "get_weather" in blocks[0].text


def test_extract_blocks_content_and_tool_calls_both_present():
    """When a message has both non-empty content and tool_calls, content is
    emitted first (mirrors wire payload order), then one block per tool call."""
    kwargs = {"messages": [{
        "role": "assistant",
        "content": "Let me check that for you.",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }],
    }]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 2
    assert blocks[0].kind == "message"
    assert blocks[0].text == "Let me check that for you."
    assert blocks[1].kind == "content_part"
    assert blocks[1].role == "assistant"
    assert "get_weather" in blocks[1].text


def test_extract_blocks_legacy_function_call_handled():
    """The legacy single-dict `function_call` field is handled the same way
    as `tool_calls`: emitted as a content_part block, no empty message block
    when content is None."""
    kwargs = {"messages": [{
        "role": "assistant",
        "content": None,
        "function_call": {"name": "get_weather", "arguments": '{"city":"Dubai"}'},
    }]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 1
    assert blocks[0].kind == "content_part"
    assert blocks[0].role == "assistant"
    assert "get_weather" in blocks[0].text


def test_extract_blocks_tool_calls_entry_not_a_dict_never_raises():
    """Defensive duck-typing: a tool_calls entry that isn't a dict is still
    serialized (stable JSON of whatever it is) rather than raising."""
    kwargs = {"messages": [{
        "role": "assistant",
        "content": None,
        "tool_calls": ["not-a-dict"],
    }]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 1
    assert blocks[0].kind == "content_part"
    assert "not-a-dict" in blocks[0].text


def test_extract_params_drops_content_keys():
    """params keep model/temperature but never messages/tools (block content
    is stored as blocks, not duplicated into params)."""
    kwargs = {"model": "gpt-4o", "temperature": 0.2, "messages": [], "tools": []}
    params = OpenAIAdapter().extract_params(kwargs)
    assert params == {"model": "gpt-4o", "temperature": 0.2}


def test_extract_usage_reads_object_or_none():
    """usage is pulled off the response's `.usage` (duck-typed) as a dict;
    a response without usage yields None."""
    class Usage:
        def __init__(self): self.prompt_tokens = 10; self.completion_tokens = 3; self.total_tokens = 13
    class Resp:
        usage = Usage()
    u = OpenAIAdapter().extract_usage(Resp())
    assert u["total_tokens"] == 13
    assert OpenAIAdapter().extract_usage(object()) is None


def test_extract_usage_falls_back_to_parse_for_raw_response():
    """`with_raw_response.create()` (the hop LangChain's ChatOpenAI uses)
    returns a raw-response wrapper with NO `.usage` directly — only its
    parsed body does. When `.usage` is absent but a callable `.parse()` is
    present, extract_usage calls it and reads `.usage` off the parsed
    result."""
    class Usage:
        def __init__(self): self.prompt_tokens = 7; self.completion_tokens = 2; self.total_tokens = 9
    class Parsed:
        usage = Usage()
    class RawResponse:
        """No `.usage` attribute — only `.parse()`, mirroring openai's
        `LegacyAPIResponse`/`APIResponse` raw-response wrapper shape."""
        def parse(self): return Parsed()

    u = OpenAIAdapter().extract_usage(RawResponse())
    assert u == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


def test_extract_usage_parse_fallback_failure_degrades_to_none():
    """If `.parse()` exists but raises, extract_usage must degrade to None
    rather than propagate — it runs inside the fail-open recorder, but stays
    defensive here too."""
    class BoomRawResponse:
        def parse(self): raise RuntimeError("boom")
    assert OpenAIAdapter().extract_usage(BoomRawResponse()) is None

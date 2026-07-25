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


# --- Responses API ----------------------------------------------------------
# The Responses shape (`input`/`instructions`/flat `tools`) is disjoint from
# Chat Completions' (`messages`/nested `tools`) so these tests use kwargs
# shaped like `client.responses.create(...)` calls; the chat-path tests above
# are untouched, proving the two shapes don't interfere with each other.


def test_extract_blocks_responses_instructions_first_then_tools_then_input():
    """Ordering contract: instructions (system message) FIRST, then flat tool
    schemas, then the input — regardless of kwarg dict insertion order."""
    kwargs = {
        "model": "gpt-4o",
        "input": "hi",
        "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
        "instructions": "be terse",
    }
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert [(b.role, b.kind) for b in blocks] == [
        ("system", "message"), ("system", "tool_schema"), ("user", "message"),
    ]
    assert blocks[0].text == "be terse"
    assert "lookup" in blocks[1].text
    assert blocks[2].text == "hi"


def test_extract_blocks_responses_input_as_plain_string():
    """A plain-string `input` becomes exactly one 'user'/'message' block."""
    blocks = OpenAIAdapter().extract_blocks({"model": "gpt-4o", "input": "hello there"})
    assert len(blocks) == 1
    assert blocks[0].role == "user" and blocks[0].kind == "message"
    assert blocks[0].text == "hello there"


def test_extract_blocks_responses_input_list_message_item_plain_content():
    """A message item in the `input` list with plain string content becomes
    one 'message' block, role passed through as-is."""
    kwargs = {"model": "gpt-4o", "input": [{"role": "assistant", "content": "sure"}]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 1
    assert blocks[0].role == "assistant" and blocks[0].kind == "message"
    assert blocks[0].text == "sure"


def test_extract_blocks_responses_input_list_message_item_content_parts():
    """A message item whose content is a list of parts yields one
    'content_part' block per part; input_text/output_text parts use their own
    "text" field, other part shapes fall back to stable JSON."""
    kwargs = {"model": "gpt-4o", "input": [
        {"role": "user", "content": [
            {"type": "input_text", "text": "look at this"},
            {"type": "input_file", "file_id": "file-1"},
        ]},
    ]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 2
    assert blocks[0].kind == "content_part" and blocks[0].role == "user"
    assert blocks[0].text == "look at this"
    # A non-image, non-text part keeps the stable-JSON fallback. `input_image`
    # is the one part type that leaves this path — it becomes an 'image' block
    # instead; see tests/test_images.py.
    assert "input_file" in blocks[1].text


def test_extract_blocks_responses_function_call_and_output_items():
    """Typed items in `input`: `function_call` maps to an 'assistant'
    'content_part' block, `function_call_output` maps to a 'tool'
    'content_part' block — both stable JSON of the whole item."""
    kwargs = {"model": "gpt-4o", "input": [
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
    ]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 2
    assert blocks[0].role == "assistant" and blocks[0].kind == "content_part"
    assert "get_weather" in blocks[0].text
    assert blocks[1].role == "tool" and blocks[1].kind == "content_part"
    assert "sunny" in blocks[1].text


def test_extract_blocks_responses_unrecognized_item_never_raises():
    """Defensive fallback: an `input` list item that is neither a message dict
    nor a recognized typed item is still captured (stable JSON, role 'user',
    kind 'content_part') rather than dropped or raising — including a
    non-dict entry."""
    kwargs = {"model": "gpt-4o", "input": [
        {"type": "reasoning", "summary": []},
        "a bare string item",
    ]}
    blocks = OpenAIAdapter().extract_blocks(kwargs)
    assert len(blocks) == 2
    assert all(b.role == "user" and b.kind == "content_part" for b in blocks)
    assert "reasoning" in blocks[0].text
    assert "a bare string item" in blocks[1].text


def test_extract_params_responses_keeps_model_and_previous_response_id_drops_content():
    """Responses params keep model/sampling settings AND `previous_response_id`
    (chain linkage, not content) but never leak input/instructions/tools."""
    kwargs = {
        "model": "gpt-4o", "temperature": 0.2,
        "input": "hi", "instructions": "be terse",
        "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
        "previous_response_id": "resp_prev",
    }
    params = OpenAIAdapter().extract_params(kwargs)
    assert params == {"model": "gpt-4o", "temperature": 0.2, "previous_response_id": "resp_prev"}
    assert "input" not in params and "instructions" not in params and "tools" not in params


def test_extract_usage_reads_input_output_tokens_family():
    """Responses-shaped usage (input_tokens/output_tokens/total_tokens, no
    prompt_tokens/completion_tokens attributes at all) is duck-typed into the
    input_tokens family, not silently coerced into the prompt_tokens shape."""
    class Usage:
        def __init__(self): self.input_tokens = 10; self.output_tokens = 5; self.total_tokens = 15
    class Resp:
        usage = Usage()
    u = OpenAIAdapter().extract_usage(Resp())
    assert u == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert "prompt_tokens" not in u


def test_extract_usage_prefers_prompt_tokens_family_when_both_present():
    """If a usage object somehow carried BOTH families, prompt_tokens wins —
    deterministic, documented tie-break."""
    class Usage:
        def __init__(self):
            self.prompt_tokens = 10; self.completion_tokens = 5
            self.input_tokens = 999; self.output_tokens = 999
            self.total_tokens = 15
    class Resp:
        usage = Usage()
    u = OpenAIAdapter().extract_usage(Resp())
    assert u == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

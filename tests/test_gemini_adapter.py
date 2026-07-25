from ctxdiff.capture.gemini import GeminiAdapter


def test_extract_blocks_orders_system_then_tools_then_contents_string():
    """system_instruction is emitted first (one 'system'-role 'message' block),
    then tool schemas from config.tools, then a plain string `contents` becomes
    a single 'user' 'message' block. Order mirrors how tokens actually sit in
    the sent context."""
    kwargs = {
        "model": "gemini-2.0-flash",
        "contents": "hi",
        "config": {
            "system_instruction": "Be helpful.",
            "tools": [{"function_declarations": [{"name": "lookup", "parameters": {}}]}],
        },
    }
    blocks = GeminiAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["message", "tool_schema", "message"]
    assert [b.role for b in blocks] == ["system", "system", "user"]
    assert blocks[0].text == "Be helpful."
    assert "lookup" in blocks[1].text
    assert blocks[2].text == "hi"


def test_extract_blocks_system_instruction_as_list():
    """A list system_instruction yields one 'system' block per entry, string
    entries kept verbatim."""
    kwargs = {"contents": "hi", "config": {"system_instruction": ["Be helpful.", "Be concise."]}}
    blocks = GeminiAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks[:2]] == ["message", "message"]
    assert blocks[0].text == "Be helpful."
    assert blocks[1].text == "Be concise."


def test_extract_blocks_handles_contents_list_with_dict_parts_and_role_mapping():
    """A list `contents` entry that is a dict with 'parts' yields one
    'content_part' block per part (part text extracted if str/dict-with-text,
    else stable JSON); role 'model' maps to 'assistant', anything else passes
    through, missing role defaults to 'user'. A plain string entry in the list
    becomes its own 'user' message block."""
    kwargs = {
        "contents": [
            "first user turn",
            {"role": "model", "parts": [{"text": "reply part 1"}, {"text": "reply part 2"}]},
            {"role": "user", "parts": [{"text": "follow up"}]},
            {"parts": [{"text": "no role given"}]},
        ],
    }
    blocks = GeminiAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == [
        "message", "content_part", "content_part", "content_part", "content_part",
    ]
    assert [b.role for b in blocks] == ["user", "assistant", "assistant", "user", "user"]
    assert blocks[0].text == "first user turn"
    assert blocks[1].text == "reply part 1"
    assert blocks[2].text == "reply part 2"
    assert blocks[3].text == "follow up"
    assert blocks[4].text == "no role given"


def test_extract_blocks_content_part_non_text_part_serializes_stable_json():
    """A part that is neither a plain string, nor a dict with a 'text' key, nor
    an image is serialized to stable (sort_keys) JSON so it stays diffable.
    Non-image `inline_data` (audio here) deliberately keeps this path: only an
    image MIME type is rerouted to an image block (see test_images.py)."""
    kwargs = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "audio/wav", "data": "xyz"}}]}]}
    blocks = GeminiAdapter().extract_blocks(kwargs)
    assert blocks[0].kind == "content_part"
    assert "inline_data" in blocks[0].text
    assert "mime_type" in blocks[0].text


def test_extract_blocks_config_as_object_duck_typed():
    """config may be an SDK-style object instead of a dict; system_instruction
    and tools are read via getattr in that case."""
    class FakeConfig:
        system_instruction = "Be helpful."
        tools = None

    kwargs = {"contents": "hi", "config": FakeConfig()}
    blocks = GeminiAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["message", "message"]
    assert blocks[0].text == "Be helpful."
    assert blocks[1].text == "hi"


def test_extract_params_drops_contents_and_config_but_keeps_model_and_sampling():
    """params exclude the content-bearing 'contents'/'config' keys, but pull
    non-content sampling fields (temperature, max_output_tokens, top_p, top_k)
    out of config (when present) and keep the top-level 'model' kwarg."""
    kwargs = {
        "model": "gemini-2.0-flash",
        "contents": "hi",
        "config": {"system_instruction": "x", "temperature": 0.2, "max_output_tokens": 256,
                   "top_p": 0.9, "top_k": 40},
    }
    params = GeminiAdapter().extract_params(kwargs)
    assert params == {
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.9,
        "top_k": 40,
    }


def test_extract_params_with_no_config():
    """params work fine when config is absent entirely."""
    kwargs = {"model": "gemini-2.0-flash", "contents": "hi"}
    params = GeminiAdapter().extract_params(kwargs)
    assert params == {"model": "gemini-2.0-flash"}


def test_extract_params_with_object_config():
    """Sampling fields are pulled off an object-style config via getattr too."""
    class FakeConfig:
        temperature = 0.5
        top_p = None  # absent value should not appear in params

    kwargs = {"model": "gemini-2.0-flash", "contents": "hi", "config": FakeConfig()}
    params = GeminiAdapter().extract_params(kwargs)
    assert params == {"model": "gemini-2.0-flash", "temperature": 0.5}


def test_extract_usage_reads_object_or_none():
    """usage is pulled off `response.usage_metadata` (duck-typed) into a plain
    dict; a response without usage_metadata yields None."""
    class UsageMetadata:
        prompt_token_count = 10
        candidates_token_count = 5
        total_token_count = 15

    class Resp:
        usage_metadata = UsageMetadata()

    u = GeminiAdapter().extract_usage(Resp())
    assert u == {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}
    assert GeminiAdapter().extract_usage(object()) is None

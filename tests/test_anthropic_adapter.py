from ctxdiff.capture.anthropic import AnthropicAdapter


def test_system_is_first_then_tools_then_messages():
    """Anthropic puts `system` at the top level; the adapter emits it as a
    leading 'system' block, then tool schemas, then messages in order."""
    kwargs = {
        "model": "claude-sonnet-4",
        "system": "Be precise.",
        "tools": [{"name": "lookup", "input_schema": {}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    blocks = AnthropicAdapter().extract_blocks(kwargs)
    assert [b.kind for b in blocks] == ["message", "tool_schema", "message"]
    assert blocks[0].role == "system" and blocks[0].text == "Be precise."
    assert "lookup" in blocks[1].text
    assert blocks[2].role == "user"


def test_structured_system_blocks_each_become_a_block():
    """Anthropic also accepts `system` as a list of text blocks; each becomes
    its own 'system' block so a changed one diffs in isolation."""
    kwargs = {"system": [{"type": "text", "text": "rule A"},
                         {"type": "text", "text": "rule B"}],
              "messages": []}
    blocks = AnthropicAdapter().extract_blocks(kwargs)
    assert [b.text for b in blocks] == ["rule A", "rule B"]
    assert all(b.role == "system" for b in blocks)


def test_extract_params_drops_content_keys():
    """params exclude system/messages/tools content keys."""
    kwargs = {"model": "claude-sonnet-4", "max_tokens": 512,
              "system": "x", "messages": [], "tools": []}
    params = AnthropicAdapter().extract_params(kwargs)
    assert params == {"model": "claude-sonnet-4", "max_tokens": 512}


def test_extract_usage_maps_anthropic_fields():
    """Anthropic reports input_tokens/output_tokens; map them to a plain dict."""
    class Usage:
        input_tokens = 20; output_tokens = 5
    class Resp:
        usage = Usage()
    u = AnthropicAdapter().extract_usage(Resp())
    assert u["input_tokens"] == 20 and u["output_tokens"] == 5


def test_extract_usage_falls_back_to_parse_for_raw_response():
    """A raw-response wrapper (e.g. from `with_raw_response.create()`) has no
    `.usage` directly — only its parsed body does. extract_usage falls back
    to calling `.parse()` and reading `.usage` off the parsed result."""
    class Usage:
        input_tokens = 20; output_tokens = 5
    class Parsed:
        usage = Usage()
    class RawResponse:
        def parse(self): return Parsed()

    u = AnthropicAdapter().extract_usage(RawResponse())
    assert u == {"input_tokens": 20, "output_tokens": 5}


def test_extract_usage_parse_fallback_failure_degrades_to_none():
    """If `.parse()` exists but raises, extract_usage must degrade to None
    rather than propagate."""
    class BoomRawResponse:
        def parse(self): raise RuntimeError("boom")
    assert AnthropicAdapter().extract_usage(BoomRawResponse()) is None

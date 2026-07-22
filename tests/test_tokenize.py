from ctxdiff.tokenize.counter import count_tokens


def test_openai_uses_tiktoken_and_is_positive():
    """OpenAI counts are exact via tiktoken; non-empty text yields >0 tokens."""
    count, method = count_tokens("hello world, this is a test", "openai")
    assert method == "tiktoken"
    assert count > 0


def test_anthropic_uses_estimate_and_is_positive():
    """Anthropic has no public local tokenizer, so we estimate and label it
    honestly as 'estimate' — never presented as exact."""
    count, method = count_tokens("hello world, this is a test", "anthropic")
    assert method == "estimate"
    assert count > 0


def test_empty_text_is_zero_tokens():
    """Empty text costs zero tokens under either provider."""
    assert count_tokens("", "openai")[0] == 0
    assert count_tokens("", "anthropic")[0] == 0


def test_estimate_scales_with_length():
    """The estimator is monotonic: longer text never counts fewer tokens."""
    short, _ = count_tokens("one two three", "anthropic")
    long, _ = count_tokens("one two three " * 50, "anthropic")
    assert long > short

import ctxdiff.tokenize.counter as counter
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


def test_openai_falls_back_to_estimate_when_tiktoken_unavailable(monkeypatch):
    """Spec §8: 'tokenizer unavailable/throws → fall back to estimate, mark
    token_method=estimate'. Forcing the encoder builder to raise (simulating
    a missing package or a blocked/failed network download) must NOT crash
    count_tokens — it must fall back and still return a positive count."""
    monkeypatch.setattr(counter, "_ENCODER", None)
    monkeypatch.setattr(counter, "_get_encoder",
                        lambda: (_ for _ in ()).throw(RuntimeError("no network")))
    count, method = count_tokens("hello world", "openai")
    assert method == "estimate"
    assert count > 0


def test_openai_tiktoken_failure_is_cached_and_not_retried(monkeypatch):
    """After a forced failure, the encoder is marked unavailable so subsequent
    calls fall back immediately without re-attempting the (possibly network)
    load — the failing builder must be invoked at most once."""
    monkeypatch.setattr(counter, "_ENCODER", None)
    call_count = {"n": 0}

    def failing_builder():
        call_count["n"] += 1
        raise RuntimeError("no network")

    monkeypatch.setattr(counter, "_get_encoder", failing_builder)

    first_count, first_method = count_tokens("hello world", "openai")
    second_count, second_method = count_tokens("hello world again", "openai")

    assert first_method == "estimate" and second_method == "estimate"
    assert first_count > 0 and second_count > 0
    assert call_count["n"] == 1  # builder attempted once, then cached as unavailable

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


def test_openai_encoder_construction_failure_is_cached_and_not_retried(monkeypatch):
    """A CONSTRUCTION failure latches. When `_get_encoder` itself raises, no
    later call can succeed either (the package is missing, the encoding is
    unknown, the download is blocked), so the sentinel is set and subsequent
    calls fall back immediately without re-attempting the possibly-network load
    — the failing builder must be invoked at most once."""
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


def test_a_literal_special_token_encodes_as_text_rather_than_raising():
    """`<|endoftext|>` in ordinary content is COUNTED, not estimated.

    tiktoken refuses by default to encode text that spells a control token, but
    ctxdiff is measuring a payload that was already sent — and the OpenAI API
    escapes those spellings, so the model saw plain characters too. Counting
    them as plain characters (`disallowed_special=()`) is therefore both the
    truthful number and the one the JS SDK produces. Nine tokens for these
    seventeen characters; the character estimate would have said five."""
    assert count_tokens("a <|endoftext|> b", "openai") == (9, "tiktoken")


def test_a_special_token_block_does_not_degrade_its_neighbours(monkeypatch):
    """THE REGRESSION TEST for the process-wide latch.

    One block quoting a control token must not change how any OTHER block in
    the same process is counted. This used to fail: the encode refusal was
    latched into `_ENCODER_UNAVAILABLE`, so every openai count after it — for
    the rest of the process — silently became an estimate while the report went
    on presenting the numbers as exact. Asserted per block, method included,
    because a count that is merely 'close' is exactly the failure mode."""
    monkeypatch.setattr(counter, "_ENCODER", None)  # a clean process
    before = count_tokens("hello world", "openai")
    offending = count_tokens("a <|endoftext|> b", "openai")
    after = count_tokens("hello world", "openai")

    assert before == (2, "tiktoken")
    assert offending == (9, "tiktoken")
    assert after == (2, "tiktoken")
    assert counter._ENCODER is not counter._ENCODER_UNAVAILABLE


def test_a_per_text_encode_failure_estimates_that_text_alone(monkeypatch):
    """An encode error that is NOT about special tokens still fails open — for
    that text only.

    `disallowed_special=()` removes the one refusal ctxdiff actually meets in
    the wild, but the defensive fallback has to stay: a corrupt input or a
    future tokenizer bug must degrade one block, not the process. This drives a
    tokenizer that throws for exactly one string and asserts the neighbours
    still come back exact — the scope the old code got wrong."""
    real = counter._get_encoder()

    class OneBadString:
        """A stand-in encoder that raises for one specific text and otherwise
        delegates to the real tiktoken encoder."""

        def encode(self, text, **kwargs):
            if text == "boom":
                raise ValueError("synthetic encode failure")
            return real.encode(text, **kwargs)

    monkeypatch.setattr(counter, "_ENCODER", OneBadString())

    assert count_tokens("hello world", "openai") == (2, "tiktoken")
    count, method = count_tokens("boom", "openai")
    assert (count, method) == (1, "estimate")  # ceil(4/4), marked honestly
    assert count_tokens("hello world", "openai") == (2, "tiktoken")
    assert counter._ENCODER is not counter._ENCODER_UNAVAILABLE


def test_estimate_counts_code_points_not_utf16_units():
    """The estimate's unit is the CODE POINT, and this pins it as a cross-SDK
    contract rather than an implementation detail of `len()`.

    Python gets this for free — `len("🚀")` is 1 — but JS strings are UTF-16 and
    `"🚀".length` is 2, so the JS twin counted every astral character twice. Since
    openai is the ONLY provider counted exactly, that made bedrock, anthropic and
    gemini render different token numbers in the two SDKs for byte-identical
    content (a Converse system block of `Répondez en français 🇫🇷` was 6 here and
    7 there). The numbers below are the shared expectation; `js/test/
    tokenize.test.ts` asserts the same ones, and `js/test/conformance.test.ts`
    compares the two implementations directly across a spread of astral inputs."""
    assert count_tokens("🚀" * 8, "anthropic") == (2, "estimate")   # 8 code points
    assert count_tokens("Répondez en français 🇫🇷", "bedrock") == (6, "estimate")
    assert count_tokens("👨‍👩‍👧‍👦", "gemini") == (2, "estimate")        # 4 emoji + 3 ZWJ
    assert count_tokens("𝕌𝕟𝕚𝕔𝕠𝕕𝕖", "anthropic") == (2, "estimate")  # math alphanumerics
    assert count_tokens("𠜎𤭢𰻞", "bedrock") == (1, "estimate")      # CJK ext B/G
    # BMP text is untouched by the distinction — a control on both sides.
    assert count_tokens("a" * 400, "gemini") == (100, "estimate")
    assert count_tokens("日本語のトークン化テスト", "anthropic") == (3, "estimate")

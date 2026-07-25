"""Token counting. Exact for OpenAI via tiktoken; a documented heuristic
estimate for Anthropic, which ships no public local tokenizer. The returned
`method` string lets every view mark estimates as approximate."""
from __future__ import annotations

import logging
import math

_log = logging.getLogger("ctxdiff")

# Lazily-built tiktoken encoder, cached across calls (construction is not free).
# Three states: None = not attempted yet; _ENCODER_UNAVAILABLE = CONSTRUCTION
# was attempted and failed (so we stop retrying); anything else = a live encoder
# ready to use. The sentinel is set ONLY for construction failure — see
# `count_tokens` for why an encode failure must never reach it.
_ENCODER_UNAVAILABLE = object()
_ENCODER = None


def _get_encoder():
    """Build (or return the already-cached) tiktoken encoder. How: calls
    `tiktoken.get_encoding`, which can perform a network download on first use
    and can raise for import errors, download failures, or load errors. This
    function does not swallow those — it is the single point `count_tokens`
    calls (and can substitute in tests) to attempt the load; failure handling
    and unavailable-caching live in the caller so they still apply even if
    this whole function is replaced."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("o200k_base")
    return _ENCODER


def _tiktoken_count(encoder, text: str) -> int:
    """Exact OpenAI token count for `text` under an already-built `encoder`.
    Uses the `o200k_base` encoding (GPT-4o family); it is a close, stable proxy
    across current OpenAI chat models and avoids a per-model lookup.

    `disallowed_special=()` is the load-bearing argument. By default tiktoken
    RAISES on any text that literally spells a special token (`<|endoftext|>`,
    `<|fim_prefix|>`, …) so a caller cannot smuggle a control token into a
    prompt by accident. ctxdiff is not building a prompt — it is measuring one
    that was already sent — and the wire payload it measures is ordinary text
    to the model too, because the OpenAI API escapes those spellings rather
    than honouring them. Passing an empty disallowed set makes the literal
    encode as the plain characters it is (nine tokens for `a <|endoftext|> b`),
    which is both the truthful count and exactly what the JS SDK's
    `disallowedSpecial: new Set()` produces. Without it, a user quoting
    `<|endoftext|>` — a prompt-injection writeup, a tokenizer tutorial, a
    pasted model card — would silently downgrade their trace to estimates.

    The encoder is passed in rather than fetched here so the caller can tell a
    CONSTRUCTION failure (permanent, latch it) from an ENCODE failure (specific
    to this one text, fall back for it alone)."""
    return len(encoder.encode(text, disallowed_special=()))


def _estimate_count(text: str) -> int:
    """Estimate tokens when no exact tokenizer exists (Anthropic). Uses the
    well-known ~4-characters-per-token rule of thumb, rounded up so any
    non-empty text is at least 1 token. Deliberately simple and monotonic;
    reconciled against provider-reported usage at the call level elsewhere."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def count_tokens(text: str, provider: str) -> tuple[int, str]:
    """Count tokens for `text` under `provider`, returning (count, method).
    OpenAI → exact tiktoken ('tiktoken') when available; anything else, or an
    OpenAI tokenizer that is unavailable/throws → heuristic estimate
    ('estimate'). Empty text is always zero tokens.

    How the OpenAI path stays network-free/fail-safe (spec §8: "tokenizer
    unavailable/throws → fall back to estimate, mark token_method='estimate'"):
    `tiktoken.get_encoding` can perform a network download on first use, and
    can also fail for mundane reasons (missing package, corrupt cache). This
    function never lets that escape — it always falls back to the
    character-based estimate instead.

    TWO DIFFERENT FAILURES, TWO DIFFERENT SCOPES — the distinction is the whole
    reason this function is shaped like this:

      * CONSTRUCTION failure (`_get_encoder` raised: tiktoken not installed,
        unknown encoding, blocked download). No future call can succeed either,
        so it latches into `_ENCODER_UNAVAILABLE` and every later count skips
        the retry — which is what the sentinel was always for.
      * ENCODE failure (the encoder exists but choked on THIS text). That says
        nothing about the next block, so it falls back for this text ALONE and
        leaves the encoder live. Latching here is what used to make one message
        quoting `<|endoftext|>` turn an entire trace's numbers into estimates
        while the JS SDK kept counting exactly — a silent, process-wide loss of
        the "exact counts for OpenAI" promise from a single user message.

    Either way the fallback is LABELLED: the affected block carries
    `token_method='estimate'` so no estimate is ever rendered as exact, and its
    neighbours keep `'tiktoken'`."""
    global _ENCODER
    if not text:
        return (0, "tiktoken" if provider == "openai" else "estimate")
    if provider != "openai":
        return (_estimate_count(text), "estimate")

    if _ENCODER is not _ENCODER_UNAVAILABLE:
        try:
            encoder = _get_encoder()
        except Exception:  # noqa: BLE001 — a permanently broken tokenizer
            _log.warning("ctxdiff: tiktoken encoder unavailable; falling back "
                         "to estimate token counts for the rest of this "
                         "process", exc_info=True)
            _ENCODER = _ENCODER_UNAVAILABLE
        else:
            try:
                return (_tiktoken_count(encoder, text), "tiktoken")
            except Exception:  # noqa: BLE001 — this ONE text, not the encoder
                # Deliberately does NOT set _ENCODER_UNAVAILABLE: the encoder
                # is demonstrably alive, so the very next block gets an exact
                # count. Only this block degrades, and it is marked as such.
                _log.warning("ctxdiff: tiktoken could not encode one block; "
                             "estimating that block only", exc_info=True)
    return (_estimate_count(text), "estimate")

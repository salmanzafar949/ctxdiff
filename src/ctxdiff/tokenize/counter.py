"""Token counting. Exact for OpenAI via tiktoken; a documented heuristic
estimate for Anthropic, which ships no public local tokenizer. The returned
`method` string lets every view mark estimates as approximate."""
from __future__ import annotations

import logging
import math

_log = logging.getLogger("ctxdiff")

# Lazily-built tiktoken encoder, cached across calls (construction is not free).
# Three states: None = not attempted yet; _ENCODER_UNAVAILABLE = attempted and
# failed (so we stop retrying); anything else = a live encoder ready to use.
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


def _tiktoken_count(text: str) -> int:
    """Exact OpenAI token count. Uses the `o200k_base` encoding (GPT-4o family);
    it is a close, stable proxy across current OpenAI chat models and avoids a
    per-model lookup. The encoder is built once and reused."""
    return len(_get_encoder().encode(text))


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
    can also fail for mundane reasons (missing package, corrupt cache, encode
    error). This function never lets that escape — it always falls back to
    the character-based estimate instead, and remembers a prior failure via
    the `_ENCODER_UNAVAILABLE` sentinel so it does not retry the (possibly
    network) load on every subsequent call in the process."""
    global _ENCODER
    if not text:
        return (0, "tiktoken" if provider == "openai" else "estimate")
    if provider == "openai":
        if _ENCODER is not _ENCODER_UNAVAILABLE:
            try:
                return (_tiktoken_count(text), "tiktoken")
            except Exception:  # noqa: BLE001 — any tokenizer failure falls back
                _log.warning("ctxdiff: tiktoken unavailable; falling back to "
                             "estimate token counts", exc_info=True)
                _ENCODER = _ENCODER_UNAVAILABLE
        return (_estimate_count(text), "estimate")
    return (_estimate_count(text), "estimate")

"""Token counting. Exact for OpenAI via tiktoken; a documented heuristic
estimate for Anthropic, which ships no public local tokenizer. The returned
`method` string lets every view mark estimates as approximate."""
from __future__ import annotations

import math

# Lazily-built tiktoken encoder, cached across calls (construction is not free).
_ENCODER = None


def _tiktoken_count(text: str) -> int:
    """Exact OpenAI token count. Uses the `o200k_base` encoding (GPT-4o family);
    it is a close, stable proxy across current OpenAI chat models and avoids a
    per-model lookup. The encoder is built once and reused."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("o200k_base")
    return len(_ENCODER.encode(text))


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
    OpenAI → exact tiktoken ('tiktoken'); anything else → heuristic estimate
    ('estimate'). Empty text is always zero tokens."""
    if not text:
        return (0, "tiktoken" if provider == "openai" else "estimate")
    if provider == "openai":
        return (_tiktoken_count(text), "tiktoken")
    return (_estimate_count(text), "estimate")

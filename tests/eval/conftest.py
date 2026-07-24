"""Shared fixtures for the real-SDK eval suite. What: stubs HTTP with respx so
every test in this directory exercises genuine `openai`/`anthropic`/
`google-genai` SDK code paths with zero network calls. How: the
`pytest.importorskip` calls below make the whole `tests/eval/` directory SKIP
(not error) when the `eval` extra isn't installed, so the default `dev` test
run (`pytest -q` against just `openai`-free installs) stays green regardless
of whether these optional deps exist."""
from __future__ import annotations

import pytest

# Module-level guards: if any of these imports fail, every test collected from
# this directory is skipped rather than erroring. This must live at import
# time (not inside a fixture) so collection itself never raises on a bare
# `dev`-only install.
respx = pytest.importorskip("respx")
openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")
genai = pytest.importorskip("google.genai")


@pytest.fixture
def respx_mock():
    """Yield a respx router with `assert_all_called=True`. What: wraps
    `respx.mock(...)` as a context-manager fixture so every test gets a fresh,
    fully-isolated HTTP mock. How: `assert_all_called=True` makes the router
    itself raise on teardown if any registered route was never hit — this is
    the mechanism that catches a silently-uncaptured call (e.g. the LangChain
    `with_raw_response` gap) as a *route* left uncalled, not just a missing
    ctxdiff assertion."""
    with respx.mock(assert_all_called=True) as router:
        yield router


def canned_openai_response(content: str = "Hello there!", prompt_tokens: int = 10,
                           completion_tokens: int = 5) -> dict:
    """Build the exact OpenAI chat-completion JSON shape confirmed against real
    `openai` 2.47.0 `ChatCompletion` parsing (spike §2). How: `total_tokens` is
    always `prompt_tokens + completion_tokens` so callers never have to keep
    two numbers in sync by hand."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"},
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def canned_anthropic_response(text: str = "Hello there!", input_tokens: int = 12,
                              output_tokens: int = 6) -> dict:
    """Build the exact Anthropic Messages JSON shape confirmed against real
    `anthropic` 0.118.0 `Message` parsing (spike §2)."""
    return {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def canned_gemini_response(text: str = "Hello there!", prompt_token_count: int = 10,
                          candidates_token_count: int = 5) -> dict:
    """Build the exact Gemini `generate_content` JSON shape confirmed against
    real `google-genai` 2.14.0 `GenerateContentResponse` parsing (Step-0
    probe). `totalTokenCount` is always the sum of the two counts so callers
    never have to keep three numbers in sync by hand."""
    return {
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": prompt_token_count,
            "candidatesTokenCount": candidates_token_count,
            "totalTokenCount": prompt_token_count + candidates_token_count,
        },
        "modelVersion": "gemini-2.0-flash",
    }


@pytest.fixture
def tmp_ctrace_path(tmp_path, request) -> str:
    """Return a unique `.ctrace` path for this test. How: derives the filename
    from pytest's own `tmp_path` (already unique per test) plus the test's own
    node name, so paths are deterministic and human-readable in failures
    without needing a random/uuid/date component."""
    return str(tmp_path / f"{request.node.name}.ctrace")

"""The adapter contract. An adapter is the ONLY provider-aware code: it turns a
provider's request kwargs into role-tagged RawBlocks and pulls usage/params off
the request and response. Everything downstream is provider-agnostic."""
from __future__ import annotations

from typing import Protocol

from ctxdiff.models import RawBlock


class Adapter(Protocol):
    """Structural contract every provider adapter satisfies."""

    provider: str
    # The attribute path from the client to the completion method, e.g.
    # ("chat","completions","create"). The proxy uses this to know what to wrap.
    # Kept for backward compat with single-path adapters and as the fallback
    # when `create_paths` (below) isn't defined.
    create_path: tuple[str, ...]
    # Optional: a TUPLE of attribute paths, for adapters whose SDK exposes more
    # than one completion method sharing the same request/response shape —
    # e.g. OpenAI's `chat.completions.create` AND `responses.create`. When
    # present, `Tracer.wrap` resolves `paths = adapter.create_paths` instead of
    # the singular `create_path`; when absent (most adapters), callers fall
    # back to `(adapter.create_path,)` so single-path adapters need no change
    # at all. Declared `| None` here (rather than required) since Protocol
    # attributes aren't given defaults, and most adapters simply omit it.
    create_paths: tuple[tuple[str, ...], ...] | None

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten the request payload into ordered context blocks."""
        ...

    def extract_params(self, kwargs: dict) -> dict:
        """Return sampling/model params (everything except block content)."""
        ...

    def extract_usage(self, response: object) -> dict | None:
        """Return provider-reported token usage as a plain dict, or None."""
        ...

    def accumulate_stream_usage(self, chunk: object, state: dict) -> None:
        """OPTIONAL — fold any usage carried by one streamed chunk into
        `state`, using this provider's own usage-dict key names (e.g.
        prompt_tokens/completion_tokens for OpenAI chat, input_tokens/
        output_tokens for OpenAI Responses and Anthropic) so `state` ends up
        shaped exactly like what `extract_usage` would have returned from a
        non-streaming response. Called once per chunk from inside the
        caller's own iteration (see trace.py's `_StreamProxy`/
        `_AsyncStreamProxy`), so it must be duck-typed and never raise.

        This is genuinely optional: `Adapter` is a structural Protocol with
        no runtime enforcement, and trace.py looks this method up with
        `getattr(adapter, "accumulate_stream_usage", None)` before calling
        it — an adapter that omits it (Gemini's `generate_content_stream`,
        Bedrock's `converse_stream` — both separate methods, out of scope
        for this pass) simply never accumulates stream usage; `state` stays
        empty and the recorded call's usage is None, same as today."""
        ...

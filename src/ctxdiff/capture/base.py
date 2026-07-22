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
    create_path: tuple[str, ...]

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten the request payload into ordered context blocks."""
        ...

    def extract_params(self, kwargs: dict) -> dict:
        """Return sampling/model params (everything except block content)."""
        ...

    def extract_usage(self, response: object) -> dict | None:
        """Return provider-reported token usage as a plain dict, or None."""
        ...

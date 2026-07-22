"""The fail-open recording path. Turns one (request, response) pair into a
stored call: adapter → token counts → hashes → labels → redaction → store.
Its single public method never raises — a debugging tool must not be able to
crash the program it is debugging."""
from __future__ import annotations

import logging
from typing import Callable

from ctxdiff.capture.base import Adapter
from ctxdiff.models import Block, CallBlock, basic_label, content_hash
from ctxdiff.store.ctrace import CTrace
from ctxdiff.tokenize.counter import count_tokens

_log = logging.getLogger("ctxdiff")


class Recorder:
    """Records calls into a CTrace using a provider adapter. Holds an optional
    redaction hook applied to every block just before it is stored."""

    def __init__(self, ctrace: CTrace, adapter: Adapter,
                 redact: Callable[[Block], Block] | None):
        """Wire the three collaborators `record()` needs: the store to write
        to, the provider-specific adapter that knows how to pull blocks/
        params/usage out of raw request/response objects, and an optional
        redaction hook. How: just stores the references; no I/O happens
        until `record()` is called."""
        self._ct = ctrace
        self._adapter = adapter
        self._redact = redact

    def record(self, seq: int, kwargs: dict, response: object | None,
               latency_ms: int | None, error: str | None,
               tagged: list[tuple[str, str]]) -> None:
        """Build and store one call from its request kwargs and response. Every
        step runs inside a catch-all: any failure is logged once and swallowed,
        leaving the host application's own call path untouched (fail-open).
        `tagged` is a list of (label, needle) pairs used to override labels."""
        try:
            raw = self._adapter.extract_blocks(kwargs)
            params = self._adapter.extract_params(kwargs)
            usage = self._adapter.extract_usage(response) if response is not None else None
            provider = self._adapter.provider

            call_blocks: list[CallBlock] = []
            for position, rb in enumerate(raw):
                # Count tokens for this text under the provider, then build the
                # content-addressed Block.
                token_count, token_method = count_tokens(rb.text, provider)
                block = Block(
                    content_hash=content_hash(rb.role, rb.kind, rb.text),
                    role=rb.role, kind=rb.kind, text=rb.text,
                    token_count=token_count, token_method=token_method,
                )
                # Redact after hashing/counting but before storage. The hash is
                # kept from the original text so identity/dedup is stable even if
                # redaction is nondeterministic; only the stored text changes.
                if self._redact is not None:
                    block = self._safe_redact(block)
                label, label_source = basic_label(rb.role, rb.kind, rb.text, tagged)
                call_blocks.append(CallBlock(
                    block=block, position=position,
                    label=label, label_source=label_source))

            self._ct.record_call(seq=seq, params=params, usage=usage,
                                 latency_ms=latency_ms, error=error,
                                 call_blocks=call_blocks)
        except Exception:  # noqa: BLE001 — fail-open is the whole point
            _log.warning("ctxdiff: failed to record call seq=%s (tracing skipped)",
                         seq, exc_info=True)

    def _safe_redact(self, block: Block) -> Block:
        """Apply the redaction hook, but never let a throwing redactor break
        recording: on error, replace the text with a sentinel so nothing
        sensitive leaks and the run continues."""
        try:
            return self._redact(block)
        except Exception:  # noqa: BLE001
            _log.warning("ctxdiff: redact hook raised; using sentinel", exc_info=True)
            return Block(block.content_hash, block.role, block.kind,
                         "[redaction-error]", block.token_count, block.token_method)

"""The fail-open recording path. Turns one (request, response) pair into a
stored call: adapter → token counts → hashes → labels → redaction → store.
Its single public method never raises — a debugging tool must not be able to
crash the program it is debugging."""
from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass
from typing import Callable

from ctxdiff.capture.base import Adapter
from ctxdiff.models import Block, CallBlock, basic_label, content_hash
from ctxdiff.store.ctrace import CTrace
from ctxdiff.tokenize.counter import count_tokens

_log = logging.getLogger("ctxdiff")


@dataclass(frozen=True)
class PersistJob:
    """One fully-built, fully-SNAPSHOTTED call ready to be written to the store.

    This is the hand-off between the two halves of recording (see `Recorder`):
    `build()` produces it on the CALLING thread — so every field is derived from
    the request/response AT CALL-COMPLETION TIME and owns its own immutable data
    (frozen dataclass, plain dicts, CallBlocks whose text is already copied out),
    with no lingering reference to the host's mutable `kwargs` — and `persist()`
    consumes it on the single WRITER thread. Because the snapshot is taken before
    the host can mutate its `messages`/`contents` list for the next turn (the
    normal agent-loop pattern of appending to one list and calling again), a
    deferred write can never record a later turn's contents against this seq."""
    seq: int
    params: dict
    usage: dict | None
    latency_ms: int | None
    error: str | None
    call_blocks: list[CallBlock]
    agent: str | None
    step: str | None
    provider: str | None


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
        # One-time-warning state for persist failures (mirrors _Writer._degrade):
        # a persistently-broken store must warn ONCE for the run, not once per
        # call. Guarded by a lock because persist() runs on the writer thread.
        self._persist_warn_lock = threading.Lock()
        self._persist_warned = False

    def build(self, seq: int, kwargs: dict, response: object | None,
              latency_ms: int | None, error: str | None,
              tagged: list[tuple[str, str]],
              agent: str | None = None, step: str | None = None,
              provider: str | None = None, quiet: bool = False) -> PersistJob | None:
        """Phase 1 of recording — runs on the CALLING thread. What: turns a
        (request, response) pair into a fully self-contained `PersistJob`
        (blocks + params + usage + attribution), doing all the CPU work —
        adapter extraction, token counting, hashing, labelling, redaction — and,
        crucially, SNAPSHOTTING the request contents out of `kwargs` into owned,
        immutable data right here, before returning to the host. That snapshot
        is why recording can then be deferred to another thread without risk:
        the host may reuse/mutate its `messages` list for the next turn the
        instant this returns, but the block text was already copied out.

        How: mirrors the old inline pipeline exactly (so a sequentially-driven
        call produces a byte-identical stored row), but STOPS before touching
        the store — persistence is `persist()`'s job. Fail-open: any failure is
        logged once (unless `quiet`) and swallowed, returning None so the caller
        simply skips this call instead of the host ever seeing an error.
        `quiet` (set only by a stream proxy's best-effort `__del__` finalize —
        see trace.py) suppresses the warning at GC/interpreter-shutdown time,
        when logging is pure noise at best and can itself misbehave."""
        try:
            raw = self._adapter.extract_blocks(kwargs)
            # DEEP-COPY the params here, on the calling thread, before returning.
            # `extract_params` is a shallow dict comprehension whose values still
            # alias the host's kwargs objects; params is only json.dumps'd later,
            # on the writer thread. A host that passes a mutable param (metadata,
            # extra_body, stop, logit_bias, response_format) and mutates it
            # before that deferred write would otherwise corrupt this stored row.
            # deepcopy (not a json round-trip) so a non-JSON-serializable value
            # can't raise here and defeat the snapshot — and if deepcopy itself
            # ever raised, the surrounding fail-open guard still swallows it.
            params = copy.deepcopy(self._adapter.extract_params(kwargs))
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

            return PersistJob(seq=seq, params=params, usage=usage,
                              latency_ms=latency_ms, error=error,
                              call_blocks=call_blocks,
                              agent=agent, step=step, provider=provider)
        except Exception:  # noqa: BLE001 — fail-open is the whole point
            if not quiet:
                _log.warning("ctxdiff: failed to build call seq=%s (tracing skipped)",
                             seq, exc_info=True)
            return None

    def persist(self, job: PersistJob, quiet: bool = False) -> None:
        """Phase 2 of recording — runs on the single WRITER thread. What: writes
        one already-built `PersistJob` to the store in a single transaction.
        How: a thin pass-through to `CTrace.record_call`; because every write
        for a run funnels through one writer thread, this is the ONLY place the
        connection is used for writing, so SQLite's thread-affinity holds even
        though the connection was opened on a different thread. Fail-open: a
        failed write is swallowed — a broken store must never take down the
        writer loop or, by extension, the host — and warned AT MOST ONCE for the
        run (see `_warn_persist_once`), so a persistently-broken store (disk
        full, read-only mount) can't flood the host's logs one line per call."""
        try:
            self._ct.record_call(seq=job.seq, params=job.params, usage=job.usage,
                                 latency_ms=job.latency_ms, error=job.error,
                                 call_blocks=job.call_blocks,
                                 agent=job.agent, step=job.step, provider=job.provider)
        except Exception:  # noqa: BLE001 — fail-open is the whole point
            if not quiet:
                self._warn_persist_once(job.seq)

    def _warn_persist_once(self, seq: int) -> None:
        """Emit the persist-failure warning AT MOST ONCE for this recorder, then
        stay silent — mirrors `_Writer._degrade`'s one-time mechanism. Why: a
        store that fails every write (disk full, read-only mount) would, with a
        per-call log, spam one warning line per recorded call; a lock-guarded
        `_persist_warned` flag makes only the FIRST failure log (with traceback)
        and every later one a no-op. Never host-facing regardless — this only
        ever reaches the `ctxdiff` logger."""
        with self._persist_warn_lock:
            if self._persist_warned:
                return
            self._persist_warned = True
        _log.warning("ctxdiff: failed to persist call seq=%s; further persist "
                     "failures in this run will be silent (tracing degraded)",
                     seq, exc_info=True)

    def record(self, seq: int, kwargs: dict, response: object | None,
               latency_ms: int | None, error: str | None,
               tagged: list[tuple[str, str]],
               agent: str | None = None, step: str | None = None,
               provider: str | None = None, quiet: bool = False) -> None:
        """Build AND persist one call inline, on the calling thread — the
        combined, synchronous path (`build()` then `persist()`). Retained for
        callers/tests that drive the whole pipeline in one step and don't route
        through the writer thread; the live capture path in trace.py instead
        calls `build()` on the host thread and hands `persist()` to the writer.
        Fail-open throughout via the two halves' own guards."""
        job = self.build(seq, kwargs, response, latency_ms, error, tagged,
                         agent, step, provider, quiet)
        if job is not None:
            self.persist(job, quiet)

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

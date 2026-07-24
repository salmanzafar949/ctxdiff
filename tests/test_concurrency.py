"""Concurrency-correctness + single-writer-thread tests for the capture path.

These are the regression tests for the v0.5 concurrency model: under an
`asyncio.gather` fan-out or a `ThreadPoolExecutor` fan-out (the normal shape of
a modern agent), each call must record ITS OWN tag/step with no cross-context
contamination, every enqueued write must survive `close()` in seq order, and
every degradation path must stay fail-open (the host call always returns its
real value / raises its real error unchanged).

They exercise the contextvars isolation (per asyncio-task AND per-thread) and
the dedicated writer thread behind the queue — see `ctxdiff/trace.py` and
`ctxdiff/capture/recorder.py`.
"""
import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


class _FakeSyncCompletions:
    """Sync chat.completions.create that records nothing itself but simulates a
    tiny bit of work so concurrent worker threads genuinely overlap."""
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        # A hair of work so ThreadPoolExecutor workers actually interleave.
        sum(range(1000))
        self.calls.append(kwargs)
        return _Resp()


class _FakeSyncChat:
    def __init__(self): self.completions = _FakeSyncCompletions()


class _FakeSyncOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeSyncChat()


class _FakeAsyncCompletions:
    """Async chat.completions.create that awaits a real sleep so gathered tasks
    are forced to interleave BETWEEN each task's tag()/mark() and its record —
    exactly the window the old global-state bug corrupted."""
    def __init__(self): self.calls = []
    async def create(self, **kwargs):
        await asyncio.sleep(0.01)
        self.calls.append(kwargs)
        return _Resp()


class _FakeAsyncChat:
    def __init__(self): self.completions = _FakeAsyncCompletions()


class _FakeAsyncOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeAsyncChat()


def _needle(i: int) -> str:
    """The unique text carried in call i's user message; also the tag needle so
    the recorded block for call i is labelled with call i's own tag."""
    return f"needle-{i:03d}"


def _index_of(text: str) -> int | None:
    """Recover the call index i from a stored block's text (see _needle)."""
    for tok in text.split():
        if tok.startswith("needle-"):
            return int(tok.split("-")[1])
    return None


# --- (a) asyncio.gather: per-task tag/step isolation -------------------------

def test_asyncio_gather_each_call_records_its_own_tag_and_step(tmp_path):
    """N coroutines run concurrently under asyncio.gather; each sets its OWN
    mark()/tag() then makes one call. With contextvars isolation, every
    recorded call must carry its own step and its own tag — never a sibling
    task's. Under the old global-state model, tasks interleaving at the await
    point cross-contaminate, which this asserts against."""
    n = 25
    t = trace.init("async-fan", path=str(tmp_path / "r.ctrace"))
    client = _FakeAsyncOpenAI()
    wrapped = t.wrap(client)

    async def one(i: int):
        t.mark(f"step-{i:03d}")
        t.tag(f"tag-{i:03d}", [_needle(i)])
        await wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"hello {_needle(i)} world"}])

    async def run():
        await asyncio.gather(*(one(i) for i in range(n)))

    asyncio.run(run())
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == n
    for c in calls:
        blocks = ct.get_call_blocks(c.id)
        i = _index_of(blocks[0].block.text)
        assert i is not None
        # Each call's own step, and its own tag applied to its own block.
        assert c.step == f"step-{i:03d}", f"call {i} got step {c.step!r}"
        assert blocks[0].label == f"tag-{i:03d}", f"call {i} got label {blocks[0].label!r}"
        assert blocks[0].label_source == "tagged"
    ct.close()


# --- (b) ThreadPoolExecutor: per-thread tag/step isolation -------------------

def test_threadpool_fanout_each_call_records_its_own_tag_and_step(tmp_path):
    """Same isolation guarantee across REAL OS threads: a ThreadPoolExecutor
    fan-out where each worker sets its own mark()/tag() then calls. Each thread
    gets its own contextvars context, so no attribution bleeds across threads."""
    n = 40
    t = trace.init("thread-fan", path=str(tmp_path / "r.ctrace"))
    client = _FakeSyncOpenAI()
    wrapped = t.wrap(client)

    def one(i: int):
        t.mark(f"step-{i:03d}")
        t.tag(f"tag-{i:03d}", [_needle(i)])
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"hello {_needle(i)} world"}])

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, range(n)))
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == n
    for c in calls:
        blocks = ct.get_call_blocks(c.id)
        i = _index_of(blocks[0].block.text)
        assert i is not None
        assert c.step == f"step-{i:03d}", f"call {i} got step {c.step!r}"
        assert blocks[0].label == f"tag-{i:03d}", f"call {i} got label {blocks[0].label!r}"
    ct.close()


def _step_by_needle(ct):
    """Map each recorded call's needle index -> its stored step, for the tests
    below that assert per-task step attribution."""
    out = {}
    for c in ct.get_calls():
        blocks = ct.get_call_blocks(c.id)
        i = _index_of(blocks[0].block.text)
        out[i] = c.step
    return out


def test_mark_step_leaks_across_reused_worker_without_scoped_step(tmp_path):
    """Documents mark()'s CAVEAT (see Tracer.mark): sticky-within-context means
    a raw ThreadPoolExecutor — which reuses workers without resetting their
    context — lets a mark() linger on a worker. A LATER task on that same worker
    that does NOT call mark() inherits the previous task's step. max_workers=1
    forces a single reused worker so the leak is deterministic: call 0 marks
    'phase-A', call 1 marks nothing yet records 'phase-A' too (inherited). This
    is exactly why tracer.step() (next test) is the recommended concurrent form."""
    t = trace.init("mark-leak", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())

    def first():
        t.mark("phase-A")
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(0)}"}])

    def second():  # deliberately does NOT call mark()
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(1)}"}])

    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(first).result()
        ex.submit(second).result()
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    steps = _step_by_needle(ct)
    assert steps[0] == "phase-A"           # the task that set it
    assert steps[1] == "phase-A"           # LEAKED onto the reused worker
    ct.close()


def test_tracer_step_contextmanager_no_leak_on_reused_worker(tmp_path):
    """The scoped tracer.step() context manager resets the step on block exit,
    so even on a SINGLE reused worker (max_workers=1) a later task that opens no
    step() block records step=None — never the previous task's label. Same
    worker-reuse setup as the mark() leak test above, opposite (correct) result."""
    t = trace.init("step-noleak-1", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())

    def first():
        with t.step("phase-A"):
            wrapped.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(0)}"}])

    def second():  # no step() block at all
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(1)}"}])

    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(first).result()
        ex.submit(second).result()
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    steps = _step_by_needle(ct)
    assert steps[0] == "phase-A"           # scoped to its block
    assert steps[1] is None                # NO leak — reset on exit cleared it
    ct.close()


def test_tracer_step_scoped_no_leak_across_pool_when_only_some_tasks_use_it(tmp_path):
    """tracer.step() is leak-proof across a real reused pool (max_workers=2)
    where ONLY SOME tasks open a step() block. Even-index tasks scope their own
    phase label; odd-index tasks use no step at all. Because step() resets on
    exit, every odd task records step=None regardless of which worker it lands
    on — proving no even task's label can ever leak onto an odd one."""
    n = 40
    t = trace.init("step-noleak-pool", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())

    def one(i: int):
        if i % 2 == 0:
            with t.step(f"phase-{i:03d}"):
                wrapped.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"m {_needle(i)}"}])
        else:
            wrapped.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(i)}"}])

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(one, range(n)))
    t.close()

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    steps = _step_by_needle(ct)
    assert len(steps) == n
    for i in range(n):
        if i % 2 == 0:
            assert steps[i] == f"phase-{i:03d}", f"call {i} got step {steps[i]!r}"
        else:
            assert steps[i] is None, f"call {i} leaked step {steps[i]!r}"
    ct.close()


# --- (c) close() flushes the queue: no lost writes, seq order ----------------

def test_close_flushes_all_enqueued_writes_in_seq_order(tmp_path):
    """Every call enqueued before close() is persisted (none lost) and the
    stored seqs are the contiguous 1..N in order — proof the writer drains the
    whole queue before the connection is closed."""
    n = 60
    t = trace.init("flush", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())
    for i in range(n):
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(i)}"}])
    t.close()  # must block until the writer has drained every enqueued job

    ct = CTrace.open(str(tmp_path / "r.ctrace"))
    calls = ct.get_calls()
    assert len(calls) == n                       # nothing lost
    assert [c.seq for c in calls] == list(range(1, n + 1))  # contiguous, in order
    ct.close()


# --- (d) fail-open under forced writer/queue/recorder failures ---------------

def test_fail_open_when_write_queue_overflows(tmp_path, caplog):
    """A jammed/overflowing write queue must NOT break the host call: the
    create() still returns its real response, the overflowed record is dropped,
    and exactly one capture-degradation warning is emitted (not one per call)."""
    t = trace.init("overflow", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())

    # Force every enqueue to look like an overflow.
    def _boom(_job):
        raise queue.Full()
    t._writer._queue.put_nowait = _boom  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="ctxdiff"):
        r1 = wrapped.chat.completions.create(model="gpt-4o", messages=[])
        r2 = wrapped.chat.completions.create(model="gpt-4o", messages=[])
    assert isinstance(r1, _Resp) and isinstance(r2, _Resp)  # host unaffected
    degraded = [r for r in caplog.records if "degraded" in r.getMessage()]
    assert len(degraded) == 1  # one-time warning, not per-call
    t.close()


def test_fail_open_when_build_raises_on_call_thread(tmp_path):
    """If the calling-thread build phase raises, the host call still returns its
    real response — the enqueue path is wrapped, not just the writer."""
    t = trace.init("buildboom", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())
    t._recorder.build = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("build boom"))
    resp = wrapped.chat.completions.create(model="gpt-4o", messages=[])
    assert isinstance(resp, _Resp)
    t.close()


def test_fail_open_when_persist_raises_on_writer_thread(tmp_path):
    """If the writer-thread persist phase raises for every job, the host calls
    still return, close() still completes (no hang), and nothing propagates."""
    t = trace.init("persistboom", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())
    t._recorder.persist = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("persist boom"))
    for _ in range(5):
        resp = wrapped.chat.completions.create(model="gpt-4o", messages=[])
        assert isinstance(resp, _Resp)
    t.close()  # must not hang or raise even though every persist blew up


def test_normal_concurrency_emits_no_degradation_warning(tmp_path, caplog):
    """Ordinary concurrent capture must NOT emit the capture-degradation
    warning — that warning is reserved for genuine degradation (writer dead /
    queue overflow), never for normal parallelism."""
    t = trace.init("quiet", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())

    def one(i: int):
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"m {_needle(i)}"}])

    with caplog.at_level(logging.WARNING, logger="ctxdiff"):
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(one, range(40)))
        t.close()
    assert not [r for r in caplog.records if "degraded" in r.getMessage()]


def test_close_drains_straggler_enqueued_behind_sentinel(tmp_path, caplog):
    """A submit() that read _closed==False can land its put_nowait AFTER close()
    enqueued the sentinel — that job then sits BEHIND the sentinel and would be
    silently abandoned. The writer must drain such stragglers before exiting.

    Deterministic setup: hold the writer thread inside a gate job, then enqueue
    the sentinel followed by a straggler DIRECTLY (simulating the race). When the
    gate releases, the writer finishes the gate job, sees the sentinel, and must
    still run the straggler before closing."""
    t = trace.init("straggler", path=str(tmp_path / "r.ctrace"))
    t.wrap(_FakeSyncOpenAI())     # creates the writer + store
    w = t._writer

    started = threading.Event()
    gate = threading.Event()
    processed = []

    def gate_job():
        started.set()
        gate.wait(5)              # pin the writer thread here

    w.submit(gate_job)
    assert started.wait(5)        # writer is now busy in gate_job

    # Enqueue the sentinel, then a straggler behind it — the exact lost-write
    # ordering close() can produce under the submit/close race.
    def straggler():
        processed.append("straggler")

    w._queue.put(w._SENTINEL)
    w._queue.put(straggler)

    with caplog.at_level(logging.WARNING, logger="ctxdiff"):
        gate.set()                # release the writer
        w._thread.join(5)
    assert not w._thread.is_alive()
    assert processed == ["straggler"]     # drained, not lost
    assert any("drained" in r.getMessage() for r in caplog.records)
    t.close()                     # idempotent cleanup (thread already exited)


def test_persist_failure_warns_once_not_per_call(tmp_path, caplog):
    """A persistently-failing store (disk full) must warn ONCE for the run, not
    once per call — mirroring the one-time _degrade mechanism. Every store write
    is forced to raise; across many calls only a single persist warning fires,
    and the host calls all still return their real response (fail-open)."""
    t = trace.init("persistwarn", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())

    def _boom(*a, **k):
        raise OSError("disk full")
    t._recorder._ct.record_call = _boom   # every write fails, via recorder.persist

    with caplog.at_level(logging.WARNING, logger="ctxdiff"):
        for _ in range(8):
            resp = wrapped.chat.completions.create(model="gpt-4o", messages=[])
            assert isinstance(resp, _Resp)   # host unaffected
        t.close()                            # flush: all 8 jobs run on the writer
    persist_warnings = [r for r in caplog.records
                        if "persist" in r.getMessage()]
    assert len(persist_warnings) == 1        # one-time, not per-call


def test_close_is_idempotent(tmp_path):
    """Calling close() twice must be safe (second is a no-op) — the writer is
    stopped and the connection closed exactly once."""
    t = trace.init("twice", path=str(tmp_path / "r.ctrace"))
    wrapped = t.wrap(_FakeSyncOpenAI())
    wrapped.chat.completions.create(model="gpt-4o", messages=[])
    t.close()
    t.close()  # no raise

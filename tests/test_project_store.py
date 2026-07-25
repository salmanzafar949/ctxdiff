"""Project-scoped storage: one `.ctrace` per PROJECT holding MULTIPLE sessions.

These are the regression tests for the project-scoped DB model: each
`trace.init(project)` opens (or creates) a stable `./<project>.ctrace` and
APPENDS a new session (a fresh `run` row) to it, instead of writing one file
per run. Multiple Tracers in one process and multiple processes may write the
same project DB concurrently; the reader must list 1..N sessions and read each
one's calls in isolation, while old single-session files still open unchanged.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace, Session, parse_started_at


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


class _FakeSyncCompletions:
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        sum(range(1000))  # a hair of real work so concurrent workers interleave
        self.calls.append(kwargs)
        return _Resp()


class _FakeSyncChat:
    def __init__(self): self.completions = _FakeSyncCompletions()


class _FakeSyncOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeSyncChat()


def _call(wrapped, text):
    wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": text}])


# --- (1) stable project path + append semantics ------------------------------

def test_init_defaults_to_stable_project_path(tmp_path, monkeypatch):
    """init(project) with no path defaults to a STABLE ./<project>.ctrace (not a
    per-run <project>-<uuid>.ctrace), so successive runs share one project DB."""
    monkeypatch.chdir(tmp_path)
    t = trace.init("myproj")
    assert os.path.basename(t.path) == "myproj.ctrace"
    t.wrap(_FakeSyncOpenAI())  # force store creation
    t.close()
    assert (tmp_path / "myproj.ctrace").exists()


def test_two_inits_append_two_sessions_to_same_file(tmp_path):
    """(a) Two trace.init("proj") in a row land TWO sessions in the SAME
    proj.ctrace, each with its own calls, correctly isolated by session id."""
    path = str(tmp_path / "proj.ctrace")

    t1 = trace.init("proj", path=path)
    w1 = t1.wrap(_FakeSyncOpenAI())
    _call(w1, "session-one-a"); _call(w1, "session-one-b")
    t1.close()

    t2 = trace.init("proj", path=path)
    w2 = t2.wrap(_FakeSyncOpenAI())
    _call(w2, "session-two-a")
    t2.close()

    # Only ONE file exists — the sessions were appended, not written separately.
    assert [p.name for p in tmp_path.glob("*.ctrace")] == ["proj.ctrace"]

    ct = CTrace.open(path)
    sessions = ct.list_sessions()
    assert len(sessions) == 2
    # Two calls in the first session, one in the second — isolated by run id.
    by_turns = sorted(s.turn_count for s in sessions)
    assert by_turns == [1, 2]
    # Each session's calls contain only its own needles.
    for s in sessions:
        texts = []
        for c in ct.get_calls(session_id=s.id):
            for b in ct.get_call_blocks(c.id):
                texts.append(b.block.text)
        if s.turn_count == 2:
            assert set(texts) == {"session-one-a", "session-one-b"}
        else:
            assert set(texts) == {"session-two-a"}
    ct.close()


def test_explicit_path_appends_when_file_exists(tmp_path):
    """An explicit path= still appends a new session when the file already
    exists (same behavior as the default stable path)."""
    path = str(tmp_path / "custom.ctrace")
    t1 = trace.init("p", path=path); t1.wrap(_FakeSyncOpenAI()); t1.close()
    t2 = trace.init("p", path=path); t2.wrap(_FakeSyncOpenAI()); t2.close()
    ct = CTrace.open(path)
    assert len(ct.list_sessions()) == 2
    ct.close()


# --- (b) concurrent multi-writer to one project file -------------------------

def test_concurrent_two_tracers_write_one_project_file(tmp_path):
    """(b) Two Tracers writing one project file concurrently (ThreadPoolExecutor)
    both land their session with no corruption and no lost writes — leans on
    busy_timeout + retry-on-locked in the writer's persist path."""
    path = str(tmp_path / "concurrent.ctrace")
    n_per = 30

    def run_session(label):
        t = trace.init("concurrent", path=path)
        w = t.wrap(_FakeSyncOpenAI())
        for i in range(n_per):
            _call(w, f"{label}-{i:03d}")
        t.close()

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run_session, ("A", "B")))

    ct = CTrace.open(path)
    sessions = ct.list_sessions()
    assert len(sessions) == 2                       # both sessions present
    assert sum(s.turn_count for s in sessions) == 2 * n_per  # nothing lost
    # No corruption: integrity check passes and every session reads back cleanly.
    assert ct._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    for s in sessions:
        assert s.turn_count == n_per
    ct.close()


# --- (b') barrier-released concurrent session CREATION: fail-open + lossless --

def _barrier_worker(barrier, path, err_q):
    """One worker PROCESS of the barrier race: block on the shared barrier so
    every worker is released at the SAME instant (maximizing the create-a-new-
    file lock-collision window — a real multi-PROCESS collision, not just
    threads sharing the GIL), then run a full session against the SAME project
    file: init(path) + wrap() + a few calls + close(). ANY exception escaping
    init/wrap/calls/close is the fail-open breach under test — its repr is put
    on the shared error queue AND re-raised so the child also exits non-zero,
    giving the test two independent breach signals."""
    try:
        barrier.wait(30)
        t = trace.init("barrier", path=path)
        w = t.wrap(_FakeSyncOpenAI())
        for i in range(3):
            _call(w, f"call-{i:03d}")
        t.close()
    except Exception as exc:  # noqa: BLE001 — the fail-open breach we assert against
        err_q.put(repr(exc))
        raise


def test_barrier_concurrent_session_creation_is_fail_open_and_lossless(tmp_path):
    """Many sessions created CONCURRENTLY on the SAME project file by many
    PROCESSES, all released from a common barrier so they collide inside the
    create-a-new-file window — the exact race the reviewer reproduced with 6-12
    processes, which two non-barrier'd tracers miss. Uses real processes (spawn)
    because the crash is an OS-level SQLite write-lock collision that the GIL
    hides when the workers are mere threads.

    Asserts the two fail-open guarantees, per round on a fresh file:
      (i) NO worker breaches: none raises out of init/wrap/calls/close (empty
          error queue) AND every child exits 0 — `wrap()` must never let a
          `database is locked` escape into the host.
      (ii) EVERY session survives: exactly one run row per worker (none lost),
          all calls present, and the file passes integrity_check. This proves
          the busy_timeout-first + locked-retry setup actually SUCCEEDS under
          contention, not merely that the last-resort fail-open guard swallowed
          the crash.

    Fails against the pre-fix code: WAL/DDL ran before busy_timeout and outside
    the retry, and `wrap()` had no guard, so under this barrier collision some
    workers raise OperationalError straight out (breaching (i) — non-zero exit
    and a populated error queue)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    n = 12
    rounds = 3
    for r in range(rounds):
        path = str(tmp_path / f"barrier-{r}.ctrace")
        barrier = ctx.Barrier(n)
        err_q = ctx.Queue()
        procs = [ctx.Process(target=_barrier_worker, args=(barrier, path, err_q))
                 for _ in range(n)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(60)

        errors = []
        while not err_q.empty():
            errors.append(err_q.get())
        exit_codes = [p.exitcode for p in procs]

        assert all(c == 0 for c in exit_codes), (
            f"round {r}: worker exit codes {exit_codes}; errors {errors[:3]}")
        assert not errors, f"round {r}: {len(errors)} worker(s) raised: {errors[:3]}"

        ct = CTrace.open(path)
        try:
            assert len(ct.list_sessions()) == n, f"round {r}: lost run rows"
            assert ct._conn.execute(
                "PRAGMA integrity_check").fetchone()[0] == "ok", f"round {r}: corrupt"
            assert sum(s.turn_count for s in ct.list_sessions()) == n * 3
        finally:
            ct.close()


# --- (c) started_at: canonical UTC-with-offset + legacy tolerance ------------

def test_started_at_roundtrips_as_utc_with_offset(tmp_path):
    """(c) A session's started_at is stored as an unambiguous UTC-with-offset
    ISO string and parses back to a tz-aware UTC datetime."""
    path = str(tmp_path / "ts.ctrace")
    t = trace.init("ts", path=path); t.wrap(_FakeSyncOpenAI()); t.close()
    ct = CTrace.open(path)
    started = ct.get_run().started_at
    dt = parse_started_at(started)
    assert dt.tzinfo is not None                    # unambiguous — carries offset
    assert dt.utcoffset() == timezone.utc.utcoffset(None)
    ct.close()


def test_parse_started_at_tolerates_naive_legacy_row():
    """(c) A legacy naive/UTC-without-offset timestamp still parses — assumed
    UTC — so old rows render correctly alongside new UTC-with-offset ones."""
    naive = parse_started_at("2026-01-01T12:00:00")       # legacy, no offset
    aware = parse_started_at("2026-01-01T12:00:00+00:00")  # new canonical
    assert naive.tzinfo is not None                        # coerced to aware UTC
    assert naive == aware                                  # same instant


# --- (d) back-compat: an existing single-session file reads identically ------

def test_existing_single_session_file_opens_identically(tmp_path):
    """(d) A single-session .ctrace (one run row) opens and reads exactly as
    before — it is just a project DB with one session. list_sessions() returns
    exactly one, and get_run()/get_calls() (no session_id) still work."""
    path = str(tmp_path / "single.ctrace")
    t = trace.init("single", path=path)
    w = t.wrap(_FakeSyncOpenAI())
    _call(w, "only-call")
    t.close()

    ct = CTrace.open(path)
    assert len(ct.list_sessions()) == 1
    # Default (session-less) reader API is unchanged for single-session files.
    assert ct.get_run().project == "single"
    calls = ct.get_calls()
    assert len(calls) == 1
    assert ct.get_call_blocks(calls[0].id)[0].block.text == "only-call"
    ct.close()


# --- (d') default binding reads the NEWEST session ---------------------------

def test_default_binding_reads_newest_session(tmp_path):
    """With no selector, a multi-session project file binds to the NEWEST
    session (the run the user just made), not the oldest. Since runs APPEND to a
    stable <project>.ctrace, the old oldest-default made the CLI forever analyze
    session #1; this asserts the fix. Three sessions are appended in order; a
    fresh open() with no session_id must read the THIRD.

    list_sessions() stays OLDEST-first for display — asserted here too so the
    two orderings don't get conflated."""
    path = str(tmp_path / "newest.ctrace")
    for label in ("first", "second", "third"):
        t = trace.init("newest", path=path)
        _call(t.wrap(_FakeSyncOpenAI()), label)
        t.close()

    ct = CTrace.open(path)
    try:
        # Default (session-less) read binds to the NEWEST session.
        calls = ct.get_calls()
        assert len(calls) == 1
        assert ct.get_call_blocks(calls[0].id)[0].block.text == "third"
        # list_sessions() remains oldest-first for display.
        assert [
            ct.get_call_blocks(ct.get_calls(session_id=s.id)[0].id)[0].block.text
            for s in ct.list_sessions()
        ] == ["first", "second", "third"]
    finally:
        ct.close()


# --- (e) session listing: agents present + turn counts -----------------------

def test_list_sessions_reports_agents_and_turn_counts(tmp_path):
    """(e) list_sessions returns each session with its agent set and turn count.
    One session drives two agents across three calls; the reader reports both
    agents (first-appearance order) and turn_count == 3."""
    path = str(tmp_path / "agents.ctrace")
    t = trace.init("agents", path=path)
    researcher = t.wrap(_FakeSyncOpenAI(), agent="researcher")
    writer = t.wrap(_FakeSyncOpenAI(), agent="writer")
    _call(researcher, "r1"); _call(writer, "w1"); _call(researcher, "r2")
    t.close()

    ct = CTrace.open(path)
    sessions = ct.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert isinstance(s, Session)
    assert s.turn_count == 3
    assert set(s.agents) == {"researcher", "writer"}
    assert s.project == "agents"
    ct.close()


def test_list_sessions_multi_session_agents_isolated(tmp_path):
    """Each session reports only ITS OWN agents — no bleed across sessions in
    the same project file."""
    path = str(tmp_path / "multi.ctrace")
    t1 = trace.init("multi", path=path)
    _call(t1.wrap(_FakeSyncOpenAI(), agent="alpha"), "a")
    t1.close()
    t2 = trace.init("multi", path=path)
    _call(t2.wrap(_FakeSyncOpenAI(), agent="beta"), "b")
    t2.close()

    ct = CTrace.open(path)
    sessions = ct.list_sessions()
    assert len(sessions) == 2
    # Each session's agent set is exactly its own — no cross-session bleed.
    agent_sets = sorted(tuple(sorted(s.agents)) for s in sessions)
    assert agent_sets == [("alpha",), ("beta",)]
    ct.close()

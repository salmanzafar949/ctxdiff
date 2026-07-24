from ctxdiff.analyze.cache import analyze_cache
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace


def _cb(text, position, role="user", kind="message", label="user", token_count=None):
    """Build a CallBlock whose hash is derived from (role, kind, text) —
    mirrors real content-addressing without pulling in a tokenizer."""
    if token_count is None:
        token_count = len(text)
    block = Block(content_hash=f"h:{role}:{kind}:{text}", role=role, kind=kind,
                  text=text, token_count=token_count, token_method="tiktoken")
    return CallBlock(block=block, position=position, label=label, label_source="heuristic")


def _record(ct, seq, blocks, agent=None):
    ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=blocks, agent=agent)


SYSTEM = _cb("system prompt", 0, role="system", label="system")


# --- append-only happy path -----------------------------------------------------


def test_append_only_run_has_zero_breaks_and_full_stable_prefix(tmp_path):
    """A run that only ever appends to history (system+user, then
    system+user+assistant+user, ...) never breaks the cache prefix: zero
    breaks, zero rebilled tokens, and the stable prefix equals the full
    length of the shorter call in every pair — the cache-friendly happy
    path must never warn."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    turn1 = [SYSTEM, _cb("hello", 1, label="user")]
    turn2 = turn1 + [_cb("hi there", 2, role="assistant", label="history"),
                      _cb("how are you", 3, label="user")]
    turn3 = turn2 + [_cb("good, thanks", 4, role="assistant", label="history"),
                      _cb("great", 5, label="user")]
    _record(ct, 1, turn1)
    _record(ct, 2, turn2)
    _record(ct, 3, turn3)

    report = analyze_cache(ct)
    ct.close()

    assert report.pairs_analyzed == 2
    assert report.breaks == []
    assert report.rebilled_tokens_total == 0
    # smallest stable prefix across pairs == the full shorter call each time
    assert report.stable_prefix_tokens_min == sum(cb.block.token_count for cb in turn1)


# --- dynamic system-block timestamp -----------------------------------------------


def _turn_with_timestamp(ts):
    return [
        _cb(f"you are an assistant. current time: {ts}", 0, role="system", label="system"),
        _cb("static rule", 1, role="system", label="system"),
        _cb("hello", 2, label="user"),
    ]


def test_changing_system_timestamp_breaks_every_pair_with_fix_hint(tmp_path):
    """A system block whose only change is an embedded timestamp breaks the
    prefix on every consecutive pair, with the same culprit position/kind
    each time, an inline first-difference in the detail text, and the
    dynamic-early-system fix hint."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    _record(ct, 1, _turn_with_timestamp("10:00:00"))
    _record(ct, 2, _turn_with_timestamp("10:00:05"))
    _record(ct, 3, _turn_with_timestamp("10:00:10"))

    report = analyze_cache(ct)
    ct.close()

    assert report.pairs_analyzed == 2
    assert len(report.breaks) == 2
    positions = {b.divergent_position for b in report.breaks}
    kinds = {b.culprit_kind for b in report.breaks}
    labels = {b.culprit_label for b in report.breaks}
    assert positions == {0}
    assert kinds == {"modified"}
    assert labels == {"system"}
    for b in report.breaks:
        assert "first difference at char" in b.detail
        assert "modified system block" in b.detail
    assert report.fix_hint is not None
    assert "dynamic value" in report.fix_hint


# --- evicted early block --------------------------------------------------------


def test_evicted_early_block_reports_evicted_kind_no_dynamic_hint(tmp_path):
    """Call N drops a rule block call N-1 had at position 1: the break is
    attributed as an eviction (not a modification), and the dynamic-field
    fix hint must NOT fire for an insert/remove pattern."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    turn1 = [SYSTEM, _cb("rule: be terse", 1, role="system", label="system"),
             _cb("hello", 2, label="user")]
    turn2 = [SYSTEM, _cb("hello", 1, label="user")]  # "rule: be terse" evicted
    _record(ct, 1, turn1)
    _record(ct, 2, turn2)

    report = analyze_cache(ct)
    ct.close()

    assert len(report.breaks) == 1
    b = report.breaks[0]
    assert b.divergent_position == 1
    assert b.culprit_kind == "evicted"
    assert "evict" in b.detail or "remov" in b.detail
    assert report.fix_hint is None


# --- reordered blocks (the reachable 'changed'-fallback shape) -------------------


def test_reordered_blocks_break_prefix_with_reordered_culprit_kind(tmp_path):
    """Two identical-content blocks simply swap position between calls
    (old=[A,B], new=[B,A], same hashes). diff_calls' move-reconciliation
    folds BOTH into 'unchanged' entries (same content, different position),
    so no modified/added/evicted entry aligns with the divergence index —
    this is the reachable shape that must NOT fall through to a generic,
    mis-attributed 'changed' label: it must classify as 'reordered', with
    correct rebilled math."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    a = _cb("block A text", 0, role="system", label="system", token_count=7)
    b = _cb("block B text", 1, role="system", label="system", token_count=9)
    turn1 = [a, b]
    # Same two blocks (identical content_hash each), swapped positions.
    b_swapped = _cb("block B text", 0, role="system", label="system", token_count=9)
    a_swapped = _cb("block A text", 1, role="system", label="system", token_count=7)
    turn2 = [b_swapped, a_swapped]
    _record(ct, 1, turn1)
    _record(ct, 2, turn2)

    report = analyze_cache(ct)
    ct.close()

    assert len(report.breaks) == 1
    b = report.breaks[0]
    assert b.divergent_position == 0
    assert b.stable_blocks == 0
    assert b.stable_tokens == 0
    assert b.culprit_kind == "reordered"
    assert b.culprit_label == "system"
    assert "reorder" in b.detail
    # rebilled = every token in turn2 from the divergence point (0) onward.
    assert report.rebilled_tokens_total == 9 + 7
    assert report.stable_prefix_tokens_min == 0


# --- rebilled math: hand-computable fixture -------------------------------------


def test_rebilled_and_stable_token_math_is_exact(tmp_path):
    """A hand-computable fixture: turn 1 has three blocks (10, 20, 30
    tokens); turn 2 keeps block 1 unchanged, changes block 2's text (10
    tokens), and keeps a 30-token block 3. Stable prefix = 10 tokens (only
    the first block matches); rebilled = every token in turn 2 from the
    divergence point onward = 10 + 30 = 40."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    turn1 = [
        _cb("A" * 10, 0, role="system", label="system", token_count=10),
        _cb("B" * 20, 1, role="system", label="system", token_count=20),
        _cb("C" * 30, 2, role="user", label="user", token_count=30),
    ]
    turn2 = [
        _cb("A" * 10, 0, role="system", label="system", token_count=10),
        _cb("B-changed", 1, role="system", label="system", token_count=10),
        _cb("C" * 30, 2, role="user", label="user", token_count=30),
    ]
    _record(ct, 1, turn1)
    _record(ct, 2, turn2)

    report = analyze_cache(ct)
    ct.close()

    assert len(report.breaks) == 1
    b = report.breaks[0]
    assert b.stable_blocks == 1
    assert b.stable_tokens == 10
    assert report.stable_prefix_tokens_min == 10
    assert report.rebilled_tokens_total == 10 + 30  # changed block + everything after


# --- <2 calls ---------------------------------------------------------------------


def test_fewer_than_two_calls_returns_empty_report_no_crash(tmp_path):
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record(ct, 1, [SYSTEM, _cb("hello", 1, label="user")])

    report = analyze_cache(ct)
    ct.close()

    assert report.pairs_analyzed == 0
    assert report.breaks == []
    assert report.rebilled_tokens_total == 0
    assert report.stable_prefix_tokens_min == 0
    assert report.fix_hint is None


def test_zero_calls_returns_empty_report_no_crash(tmp_path):
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    report = analyze_cache(ct)
    ct.close()

    assert report.pairs_analyzed == 0
    assert report.breaks == []


# --- multi-agent grouping (the correctness fix) ---------------------------------


def test_interleaved_agents_each_internally_stable_have_zero_breaks(tmp_path):
    """Two agents interleaved on the global timeline, each only ever appending
    to its OWN history, must produce ZERO breaks: a cross-agent adjacent pair
    (which shares no prefix) is never a cache break. Before per-agent grouping
    this run would have flagged every hand-off — nearly all-breaks."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    a1 = [_cb("sys A", 0, role="system", label="system"), _cb("a-q1", 1, label="user")]
    b1 = [_cb("sys B", 0, role="system", label="system"), _cb("b-q1", 1, label="user")]
    a2 = a1 + [_cb("a-ans", 2, role="assistant", label="history"),
               _cb("a-q2", 3, label="user")]
    b2 = b1 + [_cb("b-ans", 2, role="assistant", label="history"),
               _cb("b-q2", 3, label="user")]
    _record(ct, 1, a1, agent="researcher")
    _record(ct, 2, b1, agent="writer")
    _record(ct, 3, a2, agent="researcher")
    _record(ct, 4, b2, agent="writer")

    report = analyze_cache(ct)
    ct.close()

    assert report.breaks == []
    assert report.rebilled_tokens_total == 0
    assert report.pairs_analyzed == 2   # one within-agent pair each, no cross pairs
    assert report.agents_analyzed == 2


def test_break_is_attributed_to_the_single_offending_agent(tmp_path):
    """One agent has a changing-timestamp system block (a break); the other is
    stable append growth. The single break is attributed to the offending
    agent only, and the stable agent contributes none."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    def _a(ts):
        return [_cb(f"sys A time {ts}", 0, role="system", label="system"),
                _cb("a-q", 1, label="user")]
    b1 = [_cb("sys B", 0, role="system", label="system"), _cb("b-q1", 1, label="user")]
    b2 = b1 + [_cb("b-ans", 2, role="assistant", label="history"),
               _cb("b-q2", 3, label="user")]
    _record(ct, 1, _a("10:00:00"), agent="researcher")
    _record(ct, 2, b1, agent="writer")
    _record(ct, 3, _a("10:00:05"), agent="researcher")  # researcher break here
    _record(ct, 4, b2, agent="writer")                  # writer stays stable

    report = analyze_cache(ct)
    ct.close()

    assert len(report.breaks) == 1
    b = report.breaks[0]
    assert b.agent == "researcher"
    assert b.culprit_kind == "modified"
    assert b.divergent_position == 0
    assert report.agents_analyzed == 2


def test_agent_filter_analyzes_only_that_agents_calls(tmp_path):
    """analyze_cache(ct, agent='writer') restricts to the writer's own calls;
    the researcher's break is not counted, and the result is a single-timeline
    report (agents_analyzed None)."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    def _a(ts):
        return [_cb(f"sys A time {ts}", 0, role="system", label="system"),
                _cb("a-q", 1, label="user")]
    b1 = [_cb("sys B", 0, role="system", label="system"), _cb("b-q1", 1, label="user")]
    b2 = b1 + [_cb("b-ans", 2, role="assistant", label="history"),
               _cb("b-q2", 3, label="user")]
    _record(ct, 1, _a("10:00:00"), agent="researcher")
    _record(ct, 2, b1, agent="writer")
    _record(ct, 3, _a("10:00:05"), agent="researcher")
    _record(ct, 4, b2, agent="writer")

    report = analyze_cache(ct, agent="writer")
    ct.close()

    assert report.breaks == []            # writer never breaks
    assert report.pairs_analyzed == 1     # writer's single within-agent pair
    assert report.agents_analyzed is None

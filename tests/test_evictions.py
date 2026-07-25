"""The tagged-eviction detector.

Almost every test here is a COUNTER-example, and deliberately so. "The block you
tagged 'rag' at turn 3 was evicted at turn 6" is a strong claim — it tells a
developer their agent lost something it was told to keep — and a report that
cries wolf once is a report nobody reads again. So the detector is narrowed
three ways, and each narrowing gets a test that would fail loudly if it were
dropped:

- only blocks the developer TAGGED (heuristic history churns every turn by
  design);
- only within ONE agent's timeline (a hand-off is not an eviction);
- only when the block never comes back (absent-for-one-turn is not forgotten).

The fourth property is not a narrowing but a consequence of REUSING the differ:
a tagged block whose text was edited in place is `modified`, not `evicted`, and
a naive hash-presence scan would get that wrong.
"""
from __future__ import annotations

import pytest

from ctxdiff import trace
from ctxdiff.analyze.evictions import analyze_evictions
from ctxdiff.cli import main
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace


def _cb(text, position, *, role="user", label=None, tagged=False, tokens=10):
    """A CallBlock whose label source is the thing under test. `tagged=True`
    stands in for what `tracer.tag()` produces during capture."""
    block = Block(content_hash=f"h:{role}:{text}", role=role, kind="message",
                  text=text, token_count=tokens, token_method="tiktoken")
    return CallBlock(block=block, position=position,
                     label=label or ("rag" if tagged else role),
                     label_source="tagged" if tagged else "heuristic")


def _write(path, turns):
    """Write a trace from a list of (agent, [CallBlock]) turns, numbered from 1."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq, (agent, blocks) in enumerate(turns, start=1):
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None,
                       latency_ms=1, error=None, call_blocks=blocks, agent=agent)
    ct.close()
    return path


def _reopen(path):
    return CTrace.open(path)


# --- the positive case ------------------------------------------------------------


def test_a_tagged_block_that_leaves_for_good_is_named_with_both_turns(tmp_path):
    """The headline claim: where it entered, and where it disappeared. Both, not
    just the second — the turn the developer remembers is the one they wrote the
    retrieval in."""
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]),
        (None, [sys, rag, _cb("reply", 2, role="assistant")]),
        (None, [sys, _cb("reply", 1, role="assistant")]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert len(report.evictions) == 1
    e = report.evictions[0]
    assert (e.label, e.entered_seq, e.last_seen_seq, e.evicted_seq) == ("rag", 1, 2, 3)
    assert e.tokens == 42
    assert report.tagged_blocks == 1
    assert report.pairs_analyzed == 2


# --- taggedness is a property of the CONTENT, not of one call ------------------------


def test_one_tag_call_marks_the_content_for_the_whole_run(tmp_path):
    """The defect this module shipped with, and the exact shape of its own
    headline example.

    `tracer.tag()` is next-call-only BY DESIGN — it buffers for the NEXT recorded
    call and clears itself — so a block tagged once is `label_source='tagged'` on
    exactly ONE call and `heuristic` on every later one that still carries the
    same text. An eviction is detected on the pair (turn 4, turn 5), and asking
    "was the block tagged on turn 4?" answers no for a block tagged on turn 1.
    The whole feature was therefore silent on the one story it advertises: tag at
    turn 1, evicted at turn 5.

    Taggedness is decided per CONTENT HASH over the agent's whole timeline: if
    the developer ever said "this text is load-bearing", it stays load-bearing
    until it leaves."""
    sys = _cb("system", 0, role="system", label="system")
    tagged_rag = _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42)
    # The SAME text, one turn later, carrying only the role heuristic — which is
    # precisely what the recorder writes once tag()'s one-call buffer is spent.
    plain_rag = _cb("Enterprise pricing FAQ", 1, tokens=42)
    assert plain_rag.block.content_hash == tagged_rag.block.content_hash
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, tagged_rag]),
        (None, [sys, plain_rag, _cb("a", 2, role="assistant")]),
        (None, [sys, plain_rag, _cb("a", 2, role="assistant")]),
        (None, [sys, _cb("a", 1, role="assistant")]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert len(report.evictions) == 1
    e = report.evictions[0]
    assert (e.label, e.tagged_seq, e.entered_seq, e.last_seen_seq,
            e.evicted_seq) == ("rag", 1, 1, 3, 4)


def test_the_headline_names_the_turn_the_tag_was_actually_applied(tmp_path):
    """The turn the block ENTERED and the turn it was TAGGED are two different
    facts, and the sentence must not mix them: a block present from turn 1 but
    tagged from turn 2 was not tagged at turn 1, and saying so describes a tag
    that never existed."""
    sys = _cb("system", 0, role="system", label="system")
    plain = _cb("Enterprise pricing FAQ", 1)
    tagged = _cb("Enterprise pricing FAQ", 1, tagged=True)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, plain, _cb("ask", 2)]),
        (None, [sys, tagged, _cb("ask", 2)]),
        (None, [sys, _cb("ask", 1)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    e = report.evictions[0]
    assert e.entered_seq == 1      # the content was there from the start
    assert e.tagged_seq == 2       # ...but nobody vouched for it until turn 2


def test_the_label_comes_from_the_first_turn_that_tagged_the_content(tmp_path):
    """Re-tagging the same text under a second name must not make the report
    quote the later name against the earlier turn. One block, one label, one
    turn — all three taken from the same place."""
    sys = _cb("system", 0, role="system", label="system")
    first = _cb("Enterprise pricing FAQ", 1, tagged=True, label="rag")
    later = _cb("Enterprise pricing FAQ", 1, tagged=True, label="notes")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, first, _cb("ask", 2)]),
        (None, [sys, later, _cb("ask", 2)]),
        (None, [sys, _cb("ask", 1)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert (report.evictions[0].label, report.evictions[0].tagged_seq) == ("rag", 1)


def test_one_lost_block_is_one_event_however_many_memberships(tmp_path):
    """A call may legitimately carry the same text twice. The differ reports two
    `evicted` entries for it — two memberships really did disappear — but the
    developer lost ONE block, and counting the memberships is what made `check`
    say '2 tagged blocks evicted of 1' and `tokens` print the same stanza
    twice."""
    sys = _cb("system", 0, role="system", label="system")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys,
                _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42),
                _cb("Enterprise pricing FAQ", 2, tagged=True, tokens=42),
                _cb("ask", 3)]),
        (None, [sys, _cb("ask", 1)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert len(report.evictions) == 1
    assert report.tagged_blocks == 1


# --- the three narrowings ----------------------------------------------------------


def test_heuristically_labeled_blocks_are_never_reported(tmp_path):
    """Every multi-turn agent evicts ordinary history — that is what a context
    window IS. Reporting it would bury the one line worth reading."""
    sys = _cb("system", 0, role="system", label="system")
    old = _cb("ancient history", 1, role="assistant")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, old]),
        (None, [sys]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert report.evictions == []
    assert report.tagged_blocks == 0


def test_an_agent_hand_off_is_not_an_eviction(tmp_path):
    """The researcher's tagged block is 'missing' from the writer's very next
    call because it was never in the writer's context. Pairing is per agent, the
    rule `analyze_cache` already established."""
    r_sys = _cb("researcher", 0, role="system", label="system")
    w_sys = _cb("writer", 0, role="system", label="system")
    rag = _cb("Retrieved passage", 1, tagged=True)
    path = _write(str(tmp_path / "t.ctrace"), [
        ("researcher", [r_sys, rag]),
        ("writer", [w_sys, _cb("compose", 1)]),
        ("researcher", [r_sys, rag, _cb("more", 2)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert report.evictions == []
    assert report.agents_analyzed == 2


def test_an_eviction_on_one_agents_own_timeline_is_still_found(tmp_path):
    """...and the per-agent scoping must not become a way to miss a real loss:
    the same interleaved run, with the researcher genuinely dropping it later."""
    r_sys = _cb("researcher", 0, role="system", label="system")
    w_sys = _cb("writer", 0, role="system", label="system")
    rag = _cb("Retrieved passage", 1, tagged=True)
    path = _write(str(tmp_path / "t.ctrace"), [
        ("researcher", [r_sys, rag]),
        ("writer", [w_sys, _cb("compose", 1)]),
        ("researcher", [r_sys, _cb("later", 1, role="assistant")]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert len(report.evictions) == 1
    e = report.evictions[0]
    assert (e.agent, e.entered_seq, e.evicted_seq) == ("researcher", 1, 3)


def test_a_block_that_comes_back_was_never_forgotten(tmp_path):
    """Absent for one turn and back the next is a re-rank or a momentary crowd-
    out, not a loss — and claiming otherwise is exactly the false alarm that
    would make this report ignorable."""
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]),
        (None, [sys, _cb("filler", 1)]),
        (None, [sys, rag, _cb("filler", 2)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert report.evictions == []


def test_a_block_that_leaves_twice_is_reported_once_for_the_final_departure(tmp_path):
    """Out, back, then out for good: one event, naming the departure it did not
    return from, and still crediting the turn the content first entered."""
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]),
        (None, [sys, _cb("filler", 1)]),
        (None, [sys, rag, _cb("filler", 2)]),
        (None, [sys, _cb("filler", 1), _cb("more", 2)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert len(report.evictions) == 1
    e = report.evictions[0]
    assert (e.entered_seq, e.last_seen_seq, e.evicted_seq) == (1, 3, 4)


# --- the differ-reuse consequence ----------------------------------------------------


def test_an_edited_tagged_block_is_modified_not_evicted(tmp_path):
    """The reason this module calls `diff_calls` instead of comparing hash sets:
    a same-slot content change is `modified`, and reporting an edit as a loss
    would fire on every prompt someone tunes.

    This is also the detector's one documented BLIND SPOT, and it is a blind spot
    on purpose: a tagged block replaced in the same slot by different text of the
    same role and kind reads to the differ as an edit, so no eviction is
    reported. Teaching this module to second-guess that classification would mean
    a similarity heuristic of its own — a second opinion about what the differ
    already decided, which is precisely the reimplementation this design avoids.
    Whatever `ctxdiff diff` calls it, this report calls it."""
    sys = _cb("system", 0, role="system", label="system")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, _cb("Brief revision A", 1, tagged=True, label="brief")]),
        (None, [sys, _cb("Brief revision B", 1, tagged=True, label="brief")]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert report.evictions == []
    assert report.tagged_blocks == 2  # two distinct texts were tagged


# --- reporting ---------------------------------------------------------------------


def test_a_run_with_nothing_to_pair_says_so_rather_than_reassuring(tmp_path):
    """One turn: nothing could have been evicted because nothing was compared.
    The count is what lets `check` tell that apart from a genuine all-clear."""
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [_cb("Enterprise pricing FAQ", 0, tagged=True)]),
    ])
    ct = _reopen(path)
    try:
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert report.pairs_analyzed == 0
    assert report.tagged_blocks == 1


def test_tokens_prints_the_eviction_in_the_words_the_bug_is_described_in(
        tmp_path, capsys):
    """The whole feature, end to end through the CLI."""
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]),
        (None, [sys, rag, _cb("reply", 2, role="assistant")]),
        (None, [sys, _cb("reply", 1, role="assistant")]),
    ])
    main(["tokens", "--project", path])
    out = capsys.readouterr().out
    assert "⚠ the block you tagged 'rag' at turn 1 was evicted at turn 3" in out
    assert "'Enterprise pricing FAQ'" in out
    assert ("[rag·user] 42 tok · entered at turn 1 · last present at turn 2 · "
            "never returned") in out


def test_tokens_turn_n_reports_only_that_turns_eviction(tmp_path, capsys):
    """`--turn 1` selects ONE turn, and the eviction stanza has to obey the same
    selector: printing "…was evicted at turn 3" under a turn-1 report describes
    something the reader did not ask about and cannot see."""
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]),
        (None, [sys, rag, _cb("reply", 2, role="assistant")]),
        (None, [sys, _cb("reply", 1, role="assistant")]),
    ])
    main(["tokens", "--project", path, "--turn", "1"])
    assert "evicted" not in capsys.readouterr().out
    main(["tokens", "--project", path, "--turn", "3"])
    assert "was evicted at turn 3" in capsys.readouterr().out


def test_tokens_says_nothing_when_there_is_nothing_to_say(tmp_path, capsys):
    """No stanza, no reassuring line: a section that prints on every clean run
    trains people to stop reading it."""
    sys = _cb("system", 0, role="system", label="system")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys]), (None, [sys, _cb("hi", 1)]),
    ])
    main(["tokens", "--project", path])
    assert "evicted" not in capsys.readouterr().out


# --- the check assertion --------------------------------------------------------------


def _evicting_trace(tmp_path):
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42)
    return _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]),
        (None, [sys, rag, _cb("reply", 2, role="assistant")]),
        (None, [sys, _cb("reply", 1, role="assistant")]),
    ])


def test_check_fails_the_build_on_a_tagged_eviction(tmp_path, capsys):
    """Exit 1 and a line naming the block — the same sentence `tokens` prints,
    because both come from the same analyzer."""
    path = _evicting_trace(tmp_path)
    assert main(["check", "--project", path, "--no-tagged-eviction"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  no-tagged-eviction" in out
    assert "the block you tagged 'rag' at turn 1 was evicted at turn 3 · 42 tok" in out


def test_check_never_reports_more_blocks_lost_than_it_counted(tmp_path, capsys):
    """The summary's numerator and its denominator have to be the same KIND of
    thing. Counting eviction EVENTS against distinct tagged BLOCKS produced the
    self-contradiction '2 tagged blocks evicted of 1' the moment one call carried
    the same tagged text twice."""
    sys = _cb("system", 0, role="system", label="system")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys,
                _cb("Enterprise pricing FAQ", 1, tagged=True, tokens=42),
                _cb("Enterprise pricing FAQ", 2, tagged=True, tokens=42),
                _cb("ask", 3)]),
        (None, [sys, _cb("ask", 1)]),
    ])
    assert main(["check", "--project", path, "--no-tagged-eviction"]) == 1
    out = capsys.readouterr().out
    assert "1 tagged block evicted of 1 across 1 turn pair" in out
    assert out.count("was evicted at turn 2") == 1


def test_check_distinguishes_an_untagged_run_from_a_clean_one(tmp_path, capsys):
    """A run with no tags passes, and SAYS the assertion was vacuous — a tick
    that measured nothing is the failure mode this whole command exists to
    avoid."""
    sys = _cb("system", 0, role="system", label="system")
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys]), (None, [sys, _cb("hi", 1)]),
    ])
    assert main(["check", "--project", path, "--no-tagged-eviction"]) == 0
    out = capsys.readouterr().out
    assert "no tagged blocks in this run" in out


def test_check_reports_a_genuine_all_clear_with_its_denominators(tmp_path, capsys):
    """Tagged blocks existed, pairs existed, nothing was lost — the only pass
    that is actually reassuring, and it quotes both counts."""
    sys = _cb("system", 0, role="system", label="system")
    rag = _cb("Enterprise pricing FAQ", 1, tagged=True)
    path = _write(str(tmp_path / "t.ctrace"), [
        (None, [sys, rag]), (None, [sys, rag, _cb("hi", 2)]),
    ])
    assert main(["check", "--project", path, "--no-tagged-eviction"]) == 0
    assert "all 1 tagged block survived 1 turn pair" in capsys.readouterr().out


def test_the_new_assertion_is_offered_when_nothing_was_asked_for(tmp_path, capsys):
    """A `check` with no thresholds lists what it can do; a flag missing from
    that menu is a flag nobody discovers."""
    path = _evicting_trace(tmp_path)
    assert main(["check", "--project", path]) == 2
    assert "--no-tagged-eviction" in capsys.readouterr().err


# --- through the REAL capture path -----------------------------------------------------


class _Resp:
    class usage:  # noqa: N801
        prompt_tokens = 10; completion_tokens = 2; total_tokens = 12


class _Completions:
    def create(self, **kwargs): return _Resp()
class _Chat:
    def __init__(self): self.completions = _Completions()
class _OpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _Chat()


def test_a_single_tag_call_survives_capture_and_is_still_reported(tmp_path, capsys):
    """The headline example, driven through `tracer.tag()` and a wrapped client
    rather than through hand-built CallBlocks — because the hand-built fixtures
    are exactly what hid this: they can set `label_source='tagged'` on every
    call, and the recorder never does.

    `tag()` is next-call-only by design (`trace.py` reads and clears the pending
    buffer inside `_finalize`), so this is the one and only call where the RAG
    block is stored as `tagged`; on turns 2-4 the same text comes back labeled by
    the role heuristic. Turn 5 drops it for good."""
    path = str(tmp_path / "run.ctrace")
    t = trace.init("support-agent", path=path)
    client = t.wrap(_OpenAI())
    system = {"role": "system", "content": "You are a support agent. Be precise."}
    rag = {"role": "user",
           "content": "Refund policy: 30 days from delivery, no restocking fee."}
    question = {"role": "user", "content": "What's your refund window?"}

    t.tag("rag", ["Refund policy: 30 days from delivery, no restocking fee."])
    client.chat.completions.create(model="gpt-4o",
                                   messages=[system, rag, question])
    for i in range(2, 5):  # nobody re-tags; the block is simply still there
        client.chat.completions.create(model="gpt-4o", messages=[
            system, rag, question,
            {"role": "assistant", "content": f"Answer {i}."}])
    # turn 5: a sliding-window trimmer drops it, and it never comes back.
    client.chat.completions.create(model="gpt-4o", messages=[
        system, question, {"role": "assistant", "content": "Answer 4."},
        {"role": "user", "content": "And below that?"}])
    t.close()

    ct = CTrace.open(path)
    try:
        sources = [cb.label_source
                   for c in ct.get_calls()
                   for cb in ct.get_call_blocks(c.id)
                   if cb.block.text.startswith("Refund policy")]
        # The premise: tagged on exactly ONE call, heuristic on the other three.
        assert sources == ["tagged", "heuristic", "heuristic", "heuristic"]
        report = analyze_evictions(ct)
    finally:
        ct.close()
    assert len(report.evictions) == 1
    e = report.evictions[0]
    assert (e.label, e.tagged_seq, e.entered_seq, e.evicted_seq) == ("rag", 1, 1, 5)

    main(["tokens", "--project", path])
    assert ("⚠ the block you tagged 'rag' at turn 1 was evicted at turn 5"
            in capsys.readouterr().out)
    assert main(["check", "--project", path, "--no-tagged-eviction"]) == 1
    assert "FAIL  no-tagged-eviction" in capsys.readouterr().out

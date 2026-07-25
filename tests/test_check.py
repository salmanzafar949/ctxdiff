"""`ctxdiff check` — the CI gate.

Two properties are worth more than any individual assertion here, and most of
this file exists to defend them:

1. **A check must never pass by not looking.** No assertions, an empty session,
   an `--agent` matching nobody — every one of those is a non-zero exit, never
   a green tick over an unexamined trace. A CI gate whose failure mode is
   "silently verified nothing" is worse than no gate.
2. **`check` and the analysis commands can never disagree.** The thresholds are
   compared against the very numbers `ctxdiff tokens` and `ctxdiff cache`
   print, so a red build and a hand-run report always tell one story. The tests
   that pin this compare `check`'s output against the other commands' output on
   the same trace rather than against a re-derived constant.
"""
from __future__ import annotations

import os
import re

import pytest

from ctxdiff.analyze.check import Thresholds, analyze_check
from ctxdiff.cli import main
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace
from ctxdiff.store.mysql import MySQLStore
from ctxdiff.store.postgres import PostgresStore

from tests import fakedb

# A tool schema referenced by name in an assistant block (so bloat detection
# sees it used) and one that is never mentioned anywhere (so it is dead).
USED_SCHEMA = '{"name": "get_weather", "parameters": {}}'
UNUSED_SCHEMA = '{"name": "delete_account", "parameters": {}}'


def _cb(text, position, role="user", kind="message", label="user",
        token_count=None, token_method="tiktoken"):
    """Build a CallBlock whose hash is derived from (role, kind, text,
    token_method), so the same visible text can be stored once as exact and
    once as an estimate without colliding."""
    if token_count is None:
        token_count = len(text)
    block = Block(content_hash=f"h:{role}:{kind}:{text}:{token_method}", role=role,
                  kind=kind, text=text, token_count=token_count,
                  token_method=token_method)
    return CallBlock(block=block, position=position, label=label,
                     label_source="heuristic")


def _record(ct, seq, blocks, agent=None):
    ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=blocks, agent=agent)


def _make_clean_trace(path):
    """A 3-turn append-only run with exactly one registered tool schema, and
    that schema invoked. Everything a check can assert about it PASSES: the
    prefix never breaks (pure history growth), no schema is dead, and the turns
    are small. The baseline every failure case below deviates from by exactly
    one thing."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    base = [
        _cb("system prompt", 0, role="system", label="system", token_count=50),
        _cb(USED_SCHEMA, 1, role="system", kind="tool_schema",
            label="tool_schema", token_count=30),
        _cb("hello", 2, label="user", token_count=20),
    ]
    turn2 = base + [_cb("calling get_weather", 3, role="assistant",
                        label="history", token_count=10),
                    _cb("and now?", 4, label="user", token_count=20)]
    turn3 = turn2 + [_cb("sunny", 5, role="assistant", label="history",
                         token_count=10),
                     _cb("thanks", 6, label="user", token_count=20)]
    _record(ct, 1, base)
    _record(ct, 2, turn2)
    _record(ct, 3, turn3)
    ct.close()
    # turn totals: 100, 130, 160
    return path


def _make_dead_schema_trace(path):
    """The clean trace plus a second registered schema nobody ever invokes —
    the only difference, so a `--no-dead-schemas` failure can only be about
    that."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    blocks = [
        _cb("system prompt", 0, role="system", label="system", token_count=50),
        _cb(USED_SCHEMA, 1, role="system", kind="tool_schema",
            label="tool_schema", token_count=30),
        _cb(UNUSED_SCHEMA, 2, role="system", kind="tool_schema",
            label="tool_schema", token_count=40),
        _cb("calling get_weather", 3, role="assistant", label="history",
            token_count=10),
        _cb("hello", 4, label="user", token_count=20),
    ]
    for seq in (1, 2):
        _record(ct, seq, blocks)
    ct.close()
    return path


def _make_broken_prefix_trace(path):
    """A run whose first system block carries a changing timestamp, so the
    cache prefix breaks at position 0 on every pair."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq, ts in ((1, "10:00:00"), (2, "10:00:05"), (3, "10:00:10")):
        _record(ct, seq, [
            _cb(f"you are an assistant. current time: {ts}", 0, role="system",
                label="system", token_count=40),
            _cb("static rule", 1, role="system", label="system", token_count=10),
            _cb("hello", 2, label="user", token_count=20),
        ])
    ct.close()
    return path


def _make_unmeasured_trace(path):
    """THE false-pass shape, from the reviewer's report.

    Two turns, each carrying an image whose cost ctxdiff cannot know — a remote
    URL it must never fetch, stored as a zero-token `estimate` block. The
    provider bills 800 then 1,600 prompt tokens (recorded verbatim in `usage`),
    while the stored block totals are 4 and 8. Comparing those totals to a
    budget is comparing a FLOOR to a budget, and the only wrong answer it can
    give is a silent PASS."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    img1 = _cb("[image]", 1, kind="image", token_count=0, token_method="estimate")
    turn1 = [_cb("look", 0, token_count=4), img1]
    turn2 = turn1 + [
        _cb("ok", 2, role="assistant", label="history", token_count=4),
        _cb("[image 2]", 3, kind="image", token_count=0, token_method="estimate"),
    ]
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage={"prompt_tokens": 800},
                   latency_ms=10, error=None, call_blocks=turn1)
    ct.record_call(seq=2, params={"model": "gpt-4o"}, usage={"prompt_tokens": 1600},
                   latency_ms=10, error=None, call_blocks=turn2)
    ct.close()
    return path


def _make_fanout_trace(path):
    """A FAN-OUT run: four turns, four agents, one turn each — a supervisor
    dispatching four workers, which is an ordinary topology and not an edge
    case. Nothing here can be paired (pairing is per-agent by design), so every
    pair-based assertion has nothing to measure and must say which nothing."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq, agent in enumerate(("scout", "planner", "coder", "critic"), start=1):
        _record(ct, seq, [_cb(f"sys {agent}", 0, role="system", label="system",
                              token_count=20 + seq)], agent=agent)
    ct.close()
    return path


def _make_multi_agent_trace(path):
    """Two agents interleaved on one timeline. `researcher` grows sharply
    (100 → 400 tokens) while `writer` stays flat, so an agent-scoped check can
    be shown to gate ONE agent and not the other. Both are internally
    append-only, so neither breaks its own prefix — which also means the
    adjacent cross-agent pairs must not be mistaken for breaks OR for growth."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    r1 = [_cb("sys R", 0, role="system", label="system", token_count=100)]
    r2 = r1 + [_cb("big research dump", 1, role="assistant", label="history",
                   token_count=300)]
    w1 = [_cb("sys W", 0, role="system", label="system", token_count=50)]
    w2 = w1 + [_cb("draft", 1, role="assistant", label="history", token_count=10)]
    _record(ct, 1, r1, agent="researcher")
    _record(ct, 2, w1, agent="writer")
    _record(ct, 3, r2, agent="researcher")
    _record(ct, 4, w2, agent="writer")
    ct.close()
    return path


# --- the analyzer: each assertion, passing and failing ---------------------------


def test_max_context_passes_and_reports_the_peak_turn(tmp_path):
    """A run under budget passes AND states its high-water mark: a PASS that
    only says "PASS" tells nobody how close the run came, which is the number
    worth watching across successive pull requests."""
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(max_context=1000))
    ct.close()

    assert report.passed
    [a] = report.assertions
    assert a.name == "max-context"
    assert a.summary == "peak 160 tok at turn 3 · limit 1,000"
    assert a.details == []


def test_max_context_failure_names_every_offending_turn_and_the_overage(tmp_path):
    """Each turn over the limit gets its own violation line naming the turn,
    its total and by how much it blew the budget — "the build is red" is not
    actionable; "turn 3 is 30 tokens over" is."""
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(max_context=120))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary == "2 turns over limit · peak 160 tok at turn 3 · limit 120"
    assert a.details == [
        "turn 2 · 130 tok · 10 over limit",
        "turn 3 · 160 tok · 40 over limit",
    ]


def test_max_context_marks_an_approximate_total_as_approximate(tmp_path):
    """A turn whose total mixes in estimated blocks is flagged `(~approx)` in
    its violation, exactly as `ctxdiff tokens` flags it. A threshold verdict
    computed from a partly-estimated number must never read as an exact
    overage."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record(ct, 1, [_cb("guessed", 0, token_count=500, token_method="estimate")])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(max_context=100))
    ct.close()

    assert report.assertions[0].details == ["turn 1 · 500 tok (~approx) · 400 over limit"]


def test_max_context_pct_uses_the_supplied_window_as_the_denominator(tmp_path):
    """The percentage assertion is expressed against a window the USER states —
    ctxdiff ships no model→window table — and reports both the percentage and
    the token budget it works out to, so no reader has to recompute it."""
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(context_window=1000, max_context_pct=15.0))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.name == "max-context-pct"
    assert a.summary == ("1 turn over limit · peak 16.0% of 1,000 tok window "
                         "at turn 3 · limit 15.0% (150 tok)")
    assert a.details == ["turn 3 · 160 tok · 16.0% of 1,000 tok window · "
                         "limit 15.0% (150 tok)"]


def test_max_context_pct_passes_under_the_window(tmp_path):
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(context_window=1000, max_context_pct=50.0))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == ("peak 16.0% of 1,000 tok window at "
                                            "turn 3 · limit 50.0% (500 tok)")


def test_require_stable_prefix_passes_on_an_append_only_run(tmp_path):
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(require_stable_prefix=True))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == ("prefix stable across all 2 turn "
                                            "pairs · min stable prefix 100 tok")


def test_require_stable_prefix_failure_names_where_and_why(tmp_path):
    """The violation carries the turn pair, the culprit block's label and kind,
    how often it breaks, and the analyzer's own first-difference explanation —
    all of it lifted verbatim from the cache profiler rather than re-derived."""
    ct = CTrace.open(_make_broken_prefix_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(require_stable_prefix=True))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary == "2 breaks across 2 turn pairs · 140 tok re-billed"
    assert len(a.details) == 1  # one culprit, collapsed across both pairs
    assert a.details[0].startswith("turn 1 → turn 2 [system·modified] breaks 2/2 pairs — ")
    assert "first difference at char" in a.details[0]


def test_require_stable_prefix_on_a_single_turn_says_so_rather_than_claiming_stability(tmp_path):
    """One turn has no pair to be stable against. The assertion passes (there
    is nothing to violate) but says WHY, instead of asserting a stability it
    never measured."""
    path = str(tmp_path / "one.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record(ct, 1, [_cb("only turn", 0)])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(require_stable_prefix=True))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == "fewer than 2 turns — no pairs to check"


def test_no_dead_schemas_passes_when_every_schema_is_invoked(tmp_path):
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(no_dead_schemas=True))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == "all 1 registered tool schema invoked"


def test_no_dead_schemas_failure_names_the_offending_schema(tmp_path):
    """The violation names the tool by name — the only identity a dead schema
    has — and the summary carries its recurring per-call cost."""
    ct = CTrace.open(_make_dead_schema_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(no_dead_schemas=True))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary.startswith("1 of 2 registered tools never used · 40 tok/call (")
    assert a.details == ["tool schema 'delete_account' registered but never invoked"]


def test_no_dead_schemas_on_a_run_with_no_schemas_says_there_are_none(tmp_path):
    """A run registering no tools at all passes and says so. Reporting `0 of 0`
    would read like a measurement that was never taken."""
    path = str(tmp_path / "noschema.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record(ct, 1, [_cb("just a message", 0)])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(no_dead_schemas=True))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == "no tool schemas registered"


def test_max_growth_passes_and_reports_the_largest_jump(tmp_path):
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(max_growth=100))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == "peak growth 30 tok at turn 2 · limit 100"


def test_max_growth_failure_shows_the_before_and_after_totals(tmp_path):
    """A growth violation names both turns and both totals: "grew by 30" is
    only interpretable next to what it grew from."""
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(max_growth=20))
    ct.close()

    assert not report.passed
    assert report.assertions[0].details == [
        "turn 1 → turn 2 · +30 tok (100 → 130) · limit 20",
        "turn 2 → turn 3 · +30 tok (130 → 160) · limit 20",
    ]


def test_max_growth_pct_reports_percentages_against_the_previous_turn(tmp_path):
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(max_growth_pct=25.0))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary == "1 turn over limit · peak growth 30.0% at turn 2 · limit 25.0%"
    assert a.details == ["turn 1 → turn 2 · +30.0% (100 → 130 tok) · limit 25.0%"]


def test_growth_is_measured_within_an_agent_never_across_a_hand_off(tmp_path):
    """THE multi-agent correctness property. On an interleaved timeline the
    adjacent pairs are researcher→writer and writer→researcher; measuring
    growth across those would report a 100→50 "shrink" and a 50→400 "explosion"
    that describe nothing but the hand-offs. Pairing within each agent — the
    same rule the cache profiler uses — reports the two REAL growths."""
    ct = CTrace.open(_make_multi_agent_trace(str(tmp_path / "multi.ctrace")))
    report = analyze_check(ct, Thresholds(max_growth=0))
    ct.close()

    [a] = report.assertions
    assert a.details == [
        "turn 1 → turn 3 [agent:researcher] · +300 tok (100 → 400) · limit 0",
        "turn 2 → turn 4 [agent:writer] · +10 tok (50 → 60) · limit 0",
    ]


def test_agent_scoping_gates_one_agent_and_not_the_other(tmp_path):
    """`--agent` narrows what is asserted, so one pipeline can hold each of its
    agents to its own budget. The researcher blows a 200-token limit; the
    writer, checked with the very same limit, does not."""
    path = _make_multi_agent_trace(str(tmp_path / "multi.ctrace"))

    ct = CTrace.open(path)
    researcher = analyze_check(ct, Thresholds(max_context=200), agent="researcher")
    writer = analyze_check(ct, Thresholds(max_context=200), agent="writer")
    ct.close()

    assert not researcher.passed
    assert researcher.turns_analyzed == 2
    assert researcher.agent == "researcher"
    assert researcher.assertions[0].details == [
        "turn 3 [agent:researcher] · 400 tok · 200 over limit"]
    assert writer.passed
    assert writer.turns_analyzed == 2


def test_assertions_are_reported_in_a_fixed_order_regardless_of_request(tmp_path):
    """The report's row order is the analyzer's, not the caller's — so two
    workflows asking for the same assertions get the same output, and a golden
    or a diff of two CI logs is meaningful."""
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(
        max_growth_pct=90.0, no_dead_schemas=True, max_context=1000,
        require_stable_prefix=True, max_growth=1000,
        context_window=10000, max_context_pct=90.0))
    ct.close()

    assert [a.name for a in report.assertions] == [
        "max-context", "max-context-pct", "require-stable-prefix",
        "no-dead-schemas", "max-growth", "max-growth-pct"]
    assert report.passed


def test_only_requested_assertions_appear(tmp_path):
    """An assertion nobody asked for is absent, not a vacuous PASS. A report
    listing checks that were never requested would let a reader believe a
    budget is being enforced when nothing is enforcing it."""
    ct = CTrace.open(_make_clean_trace(str(tmp_path / "run.ctrace")))
    report = analyze_check(ct, Thresholds(no_dead_schemas=True))
    ct.close()

    assert [a.name for a in report.assertions] == ["no-dead-schemas"]


# --- unmeasured turns: a floor is never certified ---------------------------------


def test_max_context_refuses_to_certify_a_turn_whose_total_is_a_floor(tmp_path):
    """THE false-pass regression. A turn holding an image of unknowable cost has
    a total that is a lower bound, not a measurement — 8 stored tokens against
    1,600 the provider actually billed. Comparing that to a 500-token budget can
    only ever produce a PASS, and it would be silent and permanent.

    So the turn is not compared at all: it is reported as UNMEASURED and the
    assertion fails. A gate must not go green on a number it knows is a floor."""
    ct = CTrace.open(_make_unmeasured_trace(str(tmp_path / "img.ctrace")))
    report = analyze_check(ct, Thresholds(max_context=500))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary == ("2 turns unmeasured · peak 8 tok (~approx) at turn 2 · "
                         "limit 500")
    assert a.details == [
        "turn 1 · 4 tok (~approx) · 1 block of unknown token cost — a floor, "
        "not a measurement · limit 500",
        "turn 2 · 8 tok (~approx) · 2 blocks of unknown token cost — a floor, "
        "not a measurement · limit 500",
    ]


def test_the_check_cli_exits_1_on_the_unmeasured_repro(tmp_path, capsys):
    """The same trace through the CLI: `ctxdiff check --max-context 500` over a
    run whose turns really cost 800 and 1,600 tokens must fail the build, not
    print `PASS max-context peak 8 tok` and exit 0."""
    path = _make_unmeasured_trace(str(tmp_path / "img.ctrace"))

    code = main(["check", "--project", path, "--max-context", "500"])

    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL  max-context" in out
    assert "PASS" not in out
    assert "a floor, not a measurement" in out


def test_a_turn_both_over_the_limit_and_unmeasured_is_reported_once(tmp_path):
    """The overage is already proved and the floor cannot un-prove it, so such a
    turn is an over-limit violation and not also an unmeasured one — two lines
    for one turn would read as two problems."""
    ct = CTrace.open(_make_unmeasured_trace(str(tmp_path / "img.ctrace")))
    report = analyze_check(ct, Thresholds(max_context=5))
    ct.close()

    [a] = report.assertions
    assert a.summary == ("1 turn over limit · 1 turn unmeasured · peak 8 tok "
                         "(~approx) at turn 2 · limit 5")
    assert a.details == [
        "turn 2 · 8 tok (~approx) · 3 over limit",
        "turn 1 · 4 tok (~approx) · 1 block of unknown token cost — a floor, "
        "not a measurement · limit 5",
    ]


def test_max_context_pct_refuses_an_unmeasured_turn_too(tmp_path):
    """A percentage computed from a floor is a floor. The line still quotes the
    percentage — it is a real lower bound — beside the reason it cannot settle
    the question."""
    ct = CTrace.open(_make_unmeasured_trace(str(tmp_path / "img.ctrace")))
    report = analyze_check(ct, Thresholds(context_window=1000, max_context_pct=50.0))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary == ("2 turns unmeasured · peak 0.8% (~approx) of 1,000 tok "
                         "window at turn 2 · limit 50.0% (500 tok)")
    assert a.details[0] == (
        "turn 1 · 4 tok (~approx) · 0.4% of 1,000 tok window · 1 block of "
        "unknown token cost — a floor, not a measurement · limit 50.0% (500 tok)")


def test_max_growth_refuses_a_pair_with_an_unmeasured_turn(tmp_path):
    """A difference between two totals is only knowable when both are, and the
    error runs both ways — an unmeasured EARLIER turn overstates the growth, an
    unmeasured LATER one understates it. Neither a pass nor a numeric violation
    would be defensible, so the pair is reported as unmeasured."""
    ct = CTrace.open(_make_unmeasured_trace(str(tmp_path / "img.ctrace")))
    report = analyze_check(ct, Thresholds(max_growth=1000))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.summary == ("1 turn unmeasured · peak growth 4 tok (~approx) at "
                         "turn 2 · limit 1,000")
    assert a.details == [
        "turn 1 → turn 2 · +4 tok (~approx) (4 → 8) · 3 blocks of unknown token "
        "cost — a floor, not a measurement · limit 1,000",
    ]


def test_max_growth_pct_reports_an_unmeasured_pair_without_a_percentage(tmp_path):
    """The percentage would be derived from a floor, and quoting it beside a
    limit is exactly the confusion the refusal exists to avoid — so the line
    states the two totals and stops."""
    ct = CTrace.open(_make_unmeasured_trace(str(tmp_path / "img.ctrace")))
    report = analyze_check(ct, Thresholds(max_growth_pct=500.0))
    ct.close()

    assert not report.passed
    [a] = report.assertions
    assert a.details == [
        "turn 1 → turn 2 · 4 → 8 tok (~approx) · 3 blocks of unknown token cost "
        "— a floor, not a measurement · limit 500.0%",
    ]


def test_a_zero_token_estimate_over_EMPTY_text_is_not_called_unmeasured(tmp_path):
    """The distinction that keeps this from crying wolf. The estimator rounds any
    non-empty text up to at least one token, so a zero there means "cost
    unknown" — but empty content legitimately costs zero and is measured
    perfectly well. A run of empty estimate blocks must still pass."""
    path = str(tmp_path / "empty.ctrace")
    ct = CTrace.create(path, project="demo", provider="anthropic", model="claude")
    for seq in (1, 2):
        _record(ct, seq, [
            _cb("real content", 0, token_count=30, token_method="estimate"),
            _cb("", 1, token_count=0, token_method="estimate"),
        ])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(max_context=100, max_growth=10))
    ct.close()

    assert report.passed
    # Ties on the peak go to the EARLIEST turn — both turns total 30.
    assert report.assertions[0].summary == "peak 30 tok (~approx) at turn 1 · limit 100"


# --- an assertion with nothing to measure says WHICH nothing ----------------------


def test_a_fan_out_run_says_no_pairs_rather_than_fewer_than_2_turns(tmp_path):
    """Four turns, four agents, one turn each. Pairing is per-agent by design, so
    every adjacent pair is a hand-off and none is analyzable — but reporting
    "fewer than 2 turns" over a four-turn run reads as a typo and hides the fact
    that three assertions measured nothing at all. Each says which nothing it
    found."""
    ct = CTrace.open(_make_fanout_trace(str(tmp_path / "fan.ctrace")))
    report = analyze_check(ct, Thresholds(require_stable_prefix=True, max_growth=0,
                                          max_growth_pct=0.0))
    ct.close()

    assert report.passed
    prefix, growth, growth_pct = report.assertions
    assert prefix.summary == ("no consecutive same-agent pairs (4 agents, 1 turn "
                              "each) — no pairs to check")
    assert growth.summary == ("no consecutive same-agent pairs (4 agents, 1 turn "
                              "each) — no growth to measure")
    assert growth_pct.summary == growth.summary


def test_a_genuinely_single_turn_run_still_says_fewer_than_2_turns(tmp_path):
    """The other state, unchanged: one turn really is fewer than two."""
    path = str(tmp_path / "one.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record(ct, 1, [_cb("only turn", 0)])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(require_stable_prefix=True, max_growth=0,
                                          max_growth_pct=0.0))
    ct.close()

    assert [a.summary for a in report.assertions] == [
        "fewer than 2 turns — no pairs to check",
        "fewer than 2 turns — no growth to measure",
        "fewer than 2 turns — no growth to measure",
    ]


def test_max_growth_pct_says_when_every_pair_was_skipped_for_a_zero_token_turn(tmp_path):
    """The third state, which `_check_max_growth_pct`'s docstring has always
    promised and which used to be reported with the "fewer than 2 turns" wording
    of a completely different situation. A pair whose EARLIER turn totalled zero
    has no denominator; when that is every pair, the assertion names the skip and
    its reason."""
    path = str(tmp_path / "zero.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record(ct, 1, [_cb("", 0, token_count=0)])
    _record(ct, 2, [_cb("", 0, token_count=0), _cb("something", 1, token_count=9)])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(max_growth_pct=10.0))
    ct.close()

    assert report.passed
    assert report.assertions[0].summary == ("all 1 pair skipped — the earlier "
                                            "turn had 0 tokens")


# --- the report is a single line per violation, whatever the payload says ---------


def test_a_tool_schema_named_across_two_lines_stays_one_violation_line(tmp_path):
    """A dead-schema violation quotes a name that came out of a captured payload.
    The report is pasted into markdown — the GitHub Action fences it into the job
    summary — so a name carrying a newline and a run of backticks would close the
    fence and render contributor-controlled markup. Flattening at the source
    makes every violation exactly one line."""
    path = str(tmp_path / "hostile.ctrace")
    hostile = '{"name": "wipe\\n```\\n### \\u2705 all checks passed", "parameters": {}}'
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq in (1, 2):
        _record(ct, seq, [
            _cb(USED_SCHEMA, 0, role="system", kind="tool_schema",
                label="tool_schema", token_count=30),
            _cb(hostile, 1, role="system", kind="tool_schema",
                label="tool_schema", token_count=40),
            _cb("calling get_weather", 2, role="assistant", label="history",
                token_count=10),
        ])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(no_dead_schemas=True))
    ct.close()

    [detail] = report.assertions[0].details
    assert "\n" not in detail
    assert detail == ("tool schema 'wipe ``` ### ✅ all checks passed' registered "
                      "but never invoked")


def test_a_prefix_break_over_multi_line_text_stays_one_violation_line(tmp_path):
    """The same guarantee for the OTHER captured fragment a report quotes: the
    differing substring in a cache-break explanation, which is prompt text and
    therefore whatever a user (or an outside contributor's fixture) wrote."""
    path = str(tmp_path / "breaks.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq, tail in ((1, "one"), (2, "```\n### ✅ ctxdiff check passed\n")):
        _record(ct, seq, [_cb(f"system rules: {tail}", 0, role="system",
                              label="system", token_count=40),
                          _cb("hello", 1, token_count=20)])
    ct.close()

    ct = CTrace.open(path)
    report = analyze_check(ct, Thresholds(require_stable_prefix=True))
    ct.close()

    [detail] = report.assertions[0].details
    assert "\n" not in detail
    # The newline that would have closed a markdown fence is now a space, and
    # the quoted fragment stays inside the one line the report allotted it.
    assert "'``` ### ✅ ctxdiff ch" in detail


# --- the CLI: exit codes and messages ----------------------------------------------


def test_check_exits_0_and_prints_a_pass_verdict(tmp_path, capsys):
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    code = main(["check", "--project", path, "--max-context", "1000",
                 "--require-stable-prefix", "--no-dead-schemas"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("ctxdiff check · 3 turns · session ")
    assert "check passed · 3 assertions" in out
    assert out.count("PASS") == 3
    assert "FAIL" not in out


def test_check_exits_1_on_a_violation_so_ci_fails_the_build(tmp_path, capsys):
    """The headline contract: a blown budget is a non-zero exit. Without this
    the whole feature is a pretty report nobody's CI reacts to."""
    path = _make_dead_schema_trace(str(tmp_path / "run.ctrace"))

    code = main(["check", "--project", path, "--max-context", "1000",
                 "--no-dead-schemas"])

    out = capsys.readouterr().out
    assert code == 1
    assert "check FAILED · 1 of 2 assertions failed" in out
    assert "FAIL  no-dead-schemas" in out
    assert "PASS  max-context" in out


def test_check_with_no_assertions_exits_2_rather_than_passing_vacuously(tmp_path, capsys):
    """Refusing to run is the only safe answer: exiting 0 for "you asked me to
    verify nothing" is a green tick that means nothing, in the one command
    whose entire job is to mean something."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    code = main(["check", "--project", path])

    err = capsys.readouterr().err
    assert code == 2
    assert "nothing to assert" in err
    for flag in ("--max-context", "--require-stable-prefix", "--no-dead-schemas",
                 "--max-growth", "--max-growth-pct", "--max-context-pct"):
        assert flag in err


def test_max_context_pct_without_a_window_exits_2(tmp_path, capsys):
    """A percentage with no denominator cannot be evaluated, and ctxdiff will
    not invent one from a model name."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    code = main(["check", "--project", path, "--max-context-pct", "80"])

    err = capsys.readouterr().err
    assert code == 2
    assert "--max-context-pct needs a denominator" in err
    assert "--context-window" in err


def test_context_window_without_the_percentage_exits_2(tmp_path, capsys):
    """Nothing would consume it. Silently ignoring a flag someone typed is how
    a CI gate ends up asserting less than its author believes it does — and the
    plausible mistake here (`--max-context 8000 --context-window 8000`, from
    someone who read the window AS the budget) would otherwise look like it
    worked."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    code = main(["check", "--project", path, "--max-context", "8000",
                 "--context-window", "8000"])

    err = capsys.readouterr().err
    assert code == 2
    assert "--context-window is only used by --max-context-pct" in err


def test_non_positive_and_negative_limits_exit_2(tmp_path, capsys):
    """Range errors are usage errors. A zero context window is a division by
    zero and a negative growth budget is not a thing to assert — but a growth
    budget of exactly ZERO is ("this context must not grow at all"), so it is
    allowed."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    assert main(["check", "--project", path, "--max-context", "0"]) == 2
    assert "--max-context must be greater than 0 (got 0)" in capsys.readouterr().err

    assert main(["check", "--project", path, "--max-context-pct", "0",
                 "--context-window", "100"]) == 2
    assert "--max-context-pct must be greater than 0 (got 0.0)" in capsys.readouterr().err

    assert main(["check", "--project", path, "--max-growth", "-5"]) == 2
    assert "--max-growth cannot be negative (got -5)" in capsys.readouterr().err

    capsys.readouterr()
    assert main(["check", "--project", path, "--max-growth", "0"]) == 1


def test_check_on_an_unknown_agent_exits_2_and_lists_the_real_ones(tmp_path, capsys):
    """A typo'd agent must not filter every call away and report a table of
    passes — that is a check that would stay green forever."""
    path = _make_multi_agent_trace(str(tmp_path / "multi.ctrace"))

    code = main(["check", "--project", path, "--agent", "resercher",
                 "--max-context", "10"])

    err = capsys.readouterr().err
    assert code == 2
    assert "researcher" in err


def test_check_agent_scoping_from_the_cli(tmp_path, capsys):
    """The same limit gates the researcher and clears the writer, and the
    header states which agent was checked."""
    path = _make_multi_agent_trace(str(tmp_path / "multi.ctrace"))

    assert main(["check", "--project", path, "--agent", "researcher",
                 "--max-context", "200"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("ctxdiff check · 2 turns · agent researcher · session ")

    assert main(["check", "--project", path, "--agent", "writer",
                 "--max-context", "200"]) == 0


def test_the_header_names_the_trace_the_verdict_was_computed_from(tmp_path, monkeypatch, capsys):
    """With no `--project`, the CLI reads the most recently modified `*.ctrace`
    in the working directory — which is the GitHub Action's default. So an
    unrelated newer trace can be checked, pass, and leave a report
    indistinguishable from one over the intended run. The header names the file
    and the session, so the verdict says what it was computed from."""
    monkeypatch.chdir(tmp_path)
    _make_clean_trace(str(tmp_path / "intended.ctrace"))
    # A newer, unrelated trace — the one the newest-file default will pick.
    _make_clean_trace(str(tmp_path / "stray.ctrace"))
    os.utime(tmp_path / "stray.ctrace", (2_000_000_000, 2_000_000_000))

    assert main(["check", "--max-context", "1000"]) == 0

    header = capsys.readouterr().out.splitlines()[0]
    assert header.startswith("ctxdiff check · 3 turns · stray.ctrace (session ")


def test_a_named_project_is_reported_as_its_session_id(tmp_path, capsys):
    """When the user named the project themselves the filename adds nothing —
    it is in the command they just ran — so the header carries the session id
    alone, exactly as every selector error in the CLI already spells it."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    main(["check", "--project", path, "--max-context", "1000"])

    header = capsys.readouterr().out.splitlines()[0]
    assert re.fullmatch(r"ctxdiff check · 3 turns · session [0-9a-f]{12}", header)


def test_a_stray_positional_after_check_is_a_usage_error(tmp_path, capsys):
    """`--require-stable-prefix` is a boolean flag, so `--require-stable-prefix
    false` leaves `false` as a positional nothing claims. `check` registers no
    positional, so argparse refuses it — and the JS CLI must refuse it the same
    way rather than adopting it as `--project` and reporting on some other
    trace."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    # argparse reports it from the top-level parser and calls `parser.error()`,
    # which raises SystemExit(2) — hence the `raises` rather than a return code.
    with pytest.raises(SystemExit) as excinfo:
        main(["check", "--project", path, "--require-stable-prefix", "false"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments: false" in capsys.readouterr().err


def test_check_no_color_output_has_no_ansi_escapes(tmp_path, monkeypatch, capsys):
    """The report is pasted into CI logs and job summaries, which render escape
    codes as literal garbage."""
    monkeypatch.setenv("NO_COLOR", "1")
    path = _make_dead_schema_trace(str(tmp_path / "run.ctrace"))

    main(["check", "--project", path, "--no-dead-schemas"])

    assert "\x1b[" not in capsys.readouterr().out


def test_check_with_no_trace_exits_1_not_0(tmp_path, monkeypatch, capsys):
    """No trace is a FAILURE, not a pass. The day capture silently breaks, a
    check that greened on an absent trace would keep the build green forever —
    the single most expensive way this feature could be wrong."""
    monkeypatch.chdir(tmp_path)

    code = main(["check", "--max-context", "100"])

    assert code == 1
    assert "no .ctrace here" in capsys.readouterr().err


# --- the same check against a DATABASE-backed project --------------------------------

PG_DSN = "postgresql://u:p@localhost:5432/ctxdiff"
MY_DSN = "mysql://u:p@localhost:3306/ctxdiff"


def _fill(store):
    """Write the dead-schema scenario through a Store's public write path, so
    the trace is materialized by the very code a real capture uses rather than
    by hand-written SQL."""
    blocks = [
        _cb("system prompt", 0, role="system", label="system", token_count=50),
        _cb(USED_SCHEMA, 1, role="system", kind="tool_schema",
            label="tool_schema", token_count=30),
        _cb(UNUSED_SCHEMA, 2, role="system", kind="tool_schema",
            label="tool_schema", token_count=40),
        _cb("calling get_weather", 3, role="assistant", label="history",
            token_count=10),
        _cb("hello", 4, label="user", token_count=20),
    ]
    for seq in (1, 2):
        store.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None,
                          latency_ms=10, error=None, call_blocks=blocks)


@pytest.mark.parametrize("driver,backend_cls,dsn", [
    ("psycopg", PostgresStore, PG_DSN),
    ("pymysql", MySQLStore, MY_DSN),
])
def test_check_reads_a_configured_database_exactly_like_a_ctrace(
        monkeypatch, tmp_path, capsys, driver, backend_cls, dsn):
    """`ctxdiff check --project <DSN>` gates a database-backed project with the
    same verdicts, the same violation text and the same exit codes as a local
    `.ctrace`.

    This is not a formality: the team most likely to WANT a CI gate is the one
    already pointing several containers at a shared store, and a check that
    only worked on a file would be unavailable to exactly them. The store is
    driven through `tests/fakedb`, so the adapters' real SQL runs (see that
    module) — the read path is genuinely exercised, not mocked."""
    fakedb.install(monkeypatch, driver, str(tmp_path / "db.sqlite"))
    backend = backend_cls(dsn=dsn)
    store = backend.open_session(project="p", provider="openai",
                                 started_at="2026-07-25T10:00:00+00:00")
    try:
        _fill(store)
    finally:
        store.close()

    assert main(["check", "--project", dsn, "--max-context", "1000",
                 "--no-dead-schemas"]) == 1
    out = capsys.readouterr().out
    assert "ctxdiff check · 2 turns" in out
    assert "PASS  max-context" in out
    assert "FAIL  no-dead-schemas" in out
    assert "tool schema 'delete_account' registered but never invoked" in out

    assert main(["check", "--project", dsn, "--max-context", "1000"]) == 0


@pytest.mark.parametrize("driver,backend_cls,dsn", [
    ("psycopg", PostgresStore, PG_DSN),
    ("pymysql", MySQLStore, MY_DSN),
])
def test_check_reads_a_database_configured_by_environment(
        monkeypatch, tmp_path, capsys, driver, backend_cls, dsn):
    """…and with no `--project` at all: a store named by `CTXDIFF_STORE` is
    resolved the same way every other read command resolves it, so a CI job
    that already exports the variable needs no extra flag."""
    fakedb.install(monkeypatch, driver, str(tmp_path / "db.sqlite"))
    backend = backend_cls(dsn=dsn)
    store = backend.open_session(project="p", provider="openai",
                                 started_at="2026-07-25T10:00:00+00:00")
    try:
        _fill(store)
    finally:
        store.close()

    monkeypatch.setenv("CTXDIFF_STORE", dsn)
    monkeypatch.chdir(tmp_path)  # no *.ctrace here — the store must be found
    assert main(["check", "--no-dead-schemas"]) == 1
    assert "FAIL  no-dead-schemas" in capsys.readouterr().out


# --- the numbers agree with `tokens` and `cache` -------------------------------------


def test_check_reports_the_same_peak_that_tokens_prints(tmp_path, capsys):
    """`check`'s "peak N tok at turn M" is the same number `ctxdiff tokens`
    prints for that turn — not a parallel calculation that could drift. Read
    out of the other command's actual output rather than asserted as a
    constant, so the two can never be updated apart."""
    path = _make_clean_trace(str(tmp_path / "run.ctrace"))

    main(["tokens", "--project", path])
    tokens_out = capsys.readouterr().out
    per_turn = dict(re.findall(r"^turn (\d+) · ([\d,]+) tokens", tokens_out, re.M))
    peak_seq = max(per_turn, key=lambda s: int(per_turn[s].replace(",", "")))

    main(["check", "--project", path, "--max-context", "1000"])
    check_out = capsys.readouterr().out
    assert (f"peak {per_turn[peak_seq]} tok at turn {peak_seq}") in check_out


def test_check_reports_the_same_dead_schema_count_that_tokens_prints(tmp_path, capsys):
    """The "N of M registered tools" fraction is derived exactly as `tokens`
    derives it, so the bloat warning and the check violation always agree."""
    path = _make_dead_schema_trace(str(tmp_path / "run.ctrace"))

    main(["tokens", "--project", path])
    bloat_line = next(line for line in capsys.readouterr().out.splitlines()
                      if "schema bloat" in line)
    fraction = re.search(r"(\d+ of \d+) registered tools", bloat_line).group(1)

    main(["check", "--project", path, "--no-dead-schemas"])
    assert f"{fraction} registered tools never used" in capsys.readouterr().out


def test_check_break_frequency_matches_the_cache_report(tmp_path, capsys):
    """`check --require-stable-prefix` collapses breaks with the very same
    grouping `ctxdiff cache` renders through, so both report the same "2/2
    pairs" for the same trace."""
    path = _make_broken_prefix_trace(str(tmp_path / "run.ctrace"))

    main(["cache", "--project", path])
    cache_out = capsys.readouterr().out
    assert "(2/2 pairs)" in cache_out

    main(["check", "--project", path, "--require-stable-prefix"])
    assert "breaks 2/2 pairs" in capsys.readouterr().out

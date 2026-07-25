"""The CI assertion layer (`ctxdiff check`): a THIN threshold layer over the
analyzers that already exist, and deliberately nothing more.

Why it owns no analysis of its own. `check` exists so a context budget can be
asserted unattended on every pull request — which only works if a green `check`
and a hand-read `ctxdiff tokens`/`ctxdiff cache` can never tell two different
stories. So every number this module compares against a threshold is one the
other commands already print: turn totals and schema bloat come from
`tokens.analyze_run`, prefix breaks (including the digest-based attribution for
same-slot image swaps) come from `cache.analyze_cache`, evicted tagged blocks
come from `evictions.analyze_evictions` (the differ's own `evicted`
classification, scoped per agent, narrowed to `label_source == 'tagged'`), and
the "N of M registered tools" denominator comes from
`tokens.registered_tool_names` — the very call `ctxdiff tokens` makes for its
own bloat line. What is genuinely new
here is only: comparison against a user-supplied limit, the selection of which
turn/agent/block to blame, and the wording of the violation.

The one thing computed here that no other command computes is turn-to-turn
GROWTH, and even that is only consecutive subtraction over `analyze_run`'s
per-call totals — with the same per-agent pairing rule `analyze_cache` uses, so
an agent hand-off is never mistaken for a context explosion.

Like the other analyzers this module does no I/O beyond reading the store and
emits no color: it returns frozen dataclasses whose strings `cli/render.py`
lays out into a PASS/FAIL table.

Which number is "the turn's context". `--max-context` is compared against the
same total `ctxdiff tokens` prints as `turn N · X tokens` — the sum of the
call's stored block tokens — NOT the provider's reported prompt count. Two
reasons: it is present for every call (provider usage is optional and often
absent for the newest turn of a streamed run, and a threshold that silently
skips unreported turns is a CI check that passes by not looking), and it is the
number a reader will compare the failure against. Turns whose total mixes in
estimated blocks are marked `(~approx)` — on the PASS summaries as well as the
violation lines, since a high-water mark quoted without that marker reads as a
measurement, which is the same lie one row further up.

A total that is a FLOOR is not a total. Some blocks cost the provider real
tokens that ctxdiff cannot know: an image given as a remote URL or a `file_id`,
or in a format the sniffer does not recognize. Those are stored as zero tokens
under the 'estimate' method (a fabricated guess would be indistinguishable from
a measurement in every view), which means the call's total is a lower bound.
Comparing a lower bound against a budget can only ever produce a false PASS —
1,600 real tokens can sit under a 500-token limit as "8 tok" — so this module
refuses to certify such a turn at all: it reports it as UNMEASURED and fails the
assertion. A gate must never go green on a number it knows is a floor.
"""
from __future__ import annotations

from dataclasses import dataclass

from ctxdiff.analyze.cache import (
    CacheReport,
    analyze_cache,
    group_breaks,
    pairs_denominator,
)
from ctxdiff.analyze.differ import filter_calls
from ctxdiff.analyze.evictions import EvictionReport, analyze_evictions
from ctxdiff.analyze.tokens import (
    CallTokens,
    RunTokens,
    analyze_run,
    registered_tool_names,
)
from ctxdiff.analyze.window import pct_text, round1
from ctxdiff.store.base import Store

# --- the assertion surface ------------------------------------------------------

# The assertion names, in the ONE order they are always reported in — so a
# check's output is a function of which assertions were requested, never of the
# order the flags happened to be typed in. Also the column width the renderer
# pads to (the longest name, `require-stable-prefix`, is 21 characters).
MAX_CONTEXT = "max-context"
MAX_CONTEXT_PCT = "max-context-pct"
REQUIRE_STABLE_PREFIX = "require-stable-prefix"
NO_DEAD_SCHEMAS = "no-dead-schemas"
NO_TAGGED_EVICTION = "no-tagged-eviction"
MAX_GROWTH = "max-growth"
MAX_GROWTH_PCT = "max-growth-pct"

ASSERTION_ORDER = [MAX_CONTEXT, MAX_CONTEXT_PCT, REQUIRE_STABLE_PREFIX,
                   NO_DEAD_SCHEMAS, NO_TAGGED_EVICTION,
                   MAX_GROWTH, MAX_GROWTH_PCT]

# Width the renderer left-justifies an assertion name to: the longest name in
# ASSERTION_ORDER, so every summary starts in the same column.
NAME_WIDTH = max(len(n) for n in ASSERTION_ORDER)


@dataclass(frozen=True)
class Thresholds:
    """Everything the user asked to be asserted. Every field is opt-in: `None`
    (or False) means "this assertion was not requested" and it is left out of
    the report entirely rather than reported as a vacuous pass — a check must
    never imply it verified something nobody asked for.

    `context_window` is not itself an assertion: it is the denominator
    `max_context_pct` is a percentage OF. ctxdiff ships no model→window table
    by design (they change per model and per provider, and a stale table would
    silently move everyone's CI threshold), so the window is supplied by the
    user or the percentage assertion is simply not available. The CLI fills this
    field through `analyze.window.resolve_context_window` — the same
    flag-then-`CTXDIFF_CONTEXT_WINDOW` path `ctxdiff tokens` and the dashboard
    use, so a gate and the report a human reads beside it can never be scored
    against two different windows."""
    max_context: int | None = None
    context_window: int | None = None
    max_context_pct: float | None = None
    require_stable_prefix: bool = False
    no_dead_schemas: bool = False
    no_tagged_eviction: bool = False
    max_growth: int | None = None
    max_growth_pct: float | None = None

    @property
    def any_requested(self) -> bool:
        """Whether at least one assertion was requested. A `check` with none is
        a usage error, not a pass: exiting 0 for "you asked me to verify
        nothing" is precisely the green tick that makes a CI gate worthless."""
        return any([
            self.max_context is not None,
            self.max_context_pct is not None,
            self.require_stable_prefix,
            self.no_dead_schemas,
            self.no_tagged_eviction,
            self.max_growth is not None,
            self.max_growth_pct is not None,
        ])

    @property
    def needs_token_analysis(self) -> bool:
        """Whether any requested assertion needs `analyze_run` (per-turn totals
        or schema bloat). Checked so a prefix-only check never pays for the
        token attribution pass it would not read."""
        return any([
            self.max_context is not None,
            self.max_context_pct is not None,
            self.no_dead_schemas,
            self.max_growth is not None,
            self.max_growth_pct is not None,
        ])


@dataclass(frozen=True)
class AssertionResult:
    """One assertion's verdict. `summary` always states the ACTUAL value next
    to the threshold — on a pass as well as a failure — because "PASS" alone
    tells a reader nothing about how close the run came to the limit, which is
    the number worth watching in a CI log over time. `details` is one line per
    offending turn/agent/block, populated only on a failure."""
    name: str
    passed: bool
    summary: str
    details: list[str]


@dataclass(frozen=True)
class CheckReport:
    """Every requested assertion's verdict, in `ASSERTION_ORDER`, plus the
    scope that was checked. `turns_analyzed` is 0 only when the session (or the
    `--agent` slice of it) holds no calls at all — which the CLI reports as a
    failure rather than a vacuous pass, since a check that verified nothing must
    never be the thing keeping a build green."""
    assertions: list[AssertionResult]
    turns_analyzed: int
    agent: str | None

    @property
    def failed(self) -> list[AssertionResult]:
        """The failing assertions, in report order."""
        return [a for a in self.assertions if not a.passed]

    @property
    def passed(self) -> bool:
        """True iff every requested assertion passed."""
        return not self.failed


# --- shared formatting bits -----------------------------------------------------


def _agent_chip(agent: str | None) -> str:
    """The ` [agent:NAME]` marker appended to a turn reference, or "" for an
    unlabeled/single-agent run. Same spelling `ctxdiff cache` uses for its
    per-agent break warnings, so the two commands name an agent identically."""
    return f" [agent:{agent}]" if agent is not None else ""


def _plural(n: int, word: str) -> str:
    """`n` and `word`, pluralized by appending 's' when n != 1 — used for
    'turn'/'turns', 'break'/'breaks' and friends so the summary lines read as
    English rather than as `1 turns`."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _pct(value: float) -> str:
    """A percentage rendered to one decimal. Values are rounded to 1 decimal
    BEFORE formatting (see `_round1`) so the JS twin's `toFixed(1)` and this
    `:.1f` cannot disagree on a boundary case. Delegates to `analyze.window` so
    the percentage `check` prints and the percentage `tokens` prints go through
    ONE formatter — the two commands quoting the same trace must never differ in
    the last digit."""
    return pct_text(value)


def _round1(value: float) -> float:
    """`round(value, 1)` — the shared rounding step from `analyze.window`, named
    here so every call site in this module reads the same and the JS port has
    one thing to mirror with its `pyRound1` helper."""
    return round1(value)


def _approx_marker(call: CallTokens) -> str:
    """` (~approx)` when this turn's total includes any estimated block, else
    "". The same honesty rule `ctxdiff tokens` follows: a threshold verdict
    computed from a partly-estimated total must say so, so nobody reads a
    failure as an exact overage.

    Applied to every number this module quotes — the PASS summaries' peaks as
    much as the FAIL details. An unmarked `peak 8 tok` on a passing row is the
    same false precision as an unmarked overage, and it is the row a reader
    watches over successive pull requests."""
    return " (~approx)" if call.approximate else ""


def _pair_approx_marker(prev: CallTokens, cur: CallTokens) -> str:
    """` (~approx)` for a GROWTH figure: a difference is only as exact as the
    less exact of the two totals it is taken between, so one estimated block on
    either side marks the delta."""
    return " (~approx)" if (prev.approximate or cur.approximate) else ""


def _flatten(text: str) -> str:
    """Collapse every whitespace run in `text` (newlines included) to a single
    space.

    Applied to the one piece of a violation line that is content rather than
    ctxdiff's own words — a tool schema's `name`, which comes verbatim out of a
    captured payload. The report is pasted into markdown (the GitHub Action
    writes it into a fenced block in the job summary), where a name containing a
    newline and a run of backticks would close the fence and let captured text
    render as markup. Keeping every violation line a single line makes that
    impossible at the source instead of at each display."""
    return " ".join(text.split())


def _signed(value: int) -> str:
    """A growth figure with an explicit sign: `+1,200` / `-40` / `+0`. The
    over-limit lines are always positive (they exceeded a non-negative limit),
    but an UNMEASURED pair is reported whatever its delta, and `+-40` is not a
    number anyone should have to read."""
    return f"+{value:,}" if value >= 0 else f"{value:,}"


# --- unmeasured turns -----------------------------------------------------------


def _unmeasured(calls: list[CallTokens]) -> list[CallTokens]:
    """The turns whose total is a FLOOR: they contain at least one block whose
    token cost could not be determined (see `CallTokens.unmeasured_blocks`)."""
    return [c for c in calls if c.unmeasured_blocks]


def _unmeasured_note(*calls: CallTokens) -> str:
    """The clause explaining why a turn (or a growth pair) cannot be certified:
    how many blocks were never priced, and what that does to the number quoted
    beside it. Takes the calls it is describing so a pair's two turns are summed
    into one count."""
    n = sum(c.unmeasured_blocks for c in calls)
    return (f"{_plural(n, 'block')} of unknown token cost — a floor, "
            f"not a measurement")


def _verdict_summary(over: int, unmeasured: int, tail: str) -> str:
    """The FAIL summary for a threshold assertion: the counts that are non-zero,
    then the run's high-water mark. Written as a joined list of present parts so
    a run that is only over budget, only unmeasured, or both reads correctly
    without three hand-written format strings."""
    parts = []
    if over:
        parts.append(f"{_plural(over, 'turn')} over limit")
    if unmeasured:
        parts.append(f"{_plural(unmeasured, 'turn')} unmeasured")
    parts.append(tail)
    return " · ".join(parts)


def _no_pairs_summary(turns: int, agents: int, what: str) -> str:
    """The summary for an assertion that found no consecutive same-agent pair to
    work on — which is TWO different facts, and reporting the wrong one is how a
    silently no-op assertion goes unnoticed:

    - genuinely fewer than 2 turns: there is nothing a pair could be made of;
    - two or more turns but no two of them belong to the same agent (a four-turn
      run with four per-agent labels). Pairing is per-agent by design — a
      hand-off is not growth and not a cache break — so every pair is
      cross-agent and none is analyzable. Saying "fewer than 2 turns" over four
      turns reads as a typo and hides the fact that the assertion measured
      nothing at all.

    (`turns >= 2` with no pair implies every agent has exactly one turn, since
    any agent with two would supply one — hence the parenthetical.)"""
    if turns < 2:
        return f"fewer than 2 turns — no {what}"
    return (f"no consecutive same-agent pairs ({_plural(agents, 'agent')}, "
            f"1 turn each) — no {what}")


def _agents_in(calls: list[CallTokens]) -> int:
    """How many distinct agent labels the analyzed turns carry (None — an
    unlabeled run — counts as one). The denominator `_no_pairs_summary` names."""
    return len({c.agent for c in calls})


# --- max-context / max-context-pct ----------------------------------------------


def _peak(calls: list[CallTokens]) -> CallTokens:
    """The turn with the largest total. Ties go to the EARLIEST turn (calls
    arrive in seq order and the comparison is strictly greater-than), so the
    reported peak is stable rather than dependent on iteration luck."""
    peak = calls[0]
    for c in calls[1:]:
        if c.total_tokens > peak.total_tokens:
            peak = c
    return peak


def _check_max_context(calls: list[CallTokens], limit: int) -> AssertionResult:
    """Fail when any turn's context total exceeds `limit` tokens — or when a
    turn's total cannot be compared to the limit at all. The peak turn is named
    on a pass too, so the CI log carries the run's actual high-water mark and
    not just a tick.

    Two ways to fail, and the second one is why this check can be trusted:

    - a turn whose total is OVER the limit;
    - a turn whose total is a FLOOR (it contains a block of unknown cost, see
      `_unmeasured`). Such a turn is reported as unmeasured rather than compared,
      because comparing a lower bound to a budget has exactly one possible
      wrong answer — a PASS — and it is silent. A turn that is BOTH over the
      limit and unmeasured is reported once, as over the limit: the overage is
      already proved, and the floor cannot un-prove it."""
    over = [c for c in calls if c.total_tokens > limit]
    unmeasured = [c for c in _unmeasured(calls) if c not in over]
    peak = _peak(calls)
    peak_text = (f"peak {peak.total_tokens:,} tok{_approx_marker(peak)} at turn "
                 f"{peak.seq} · limit {limit:,}")
    if not over and not unmeasured:
        return AssertionResult(MAX_CONTEXT, True, peak_text, [])
    details = [
        f"turn {c.seq}{_agent_chip(c.agent)} · {c.total_tokens:,} tok"
        f"{_approx_marker(c)} · {c.total_tokens - limit:,} over limit"
        for c in over
    ]
    details += [
        f"turn {c.seq}{_agent_chip(c.agent)} · {c.total_tokens:,} tok"
        f"{_approx_marker(c)} · {_unmeasured_note(c)} · limit {limit:,}"
        for c in unmeasured
    ]
    return AssertionResult(
        MAX_CONTEXT, False,
        _verdict_summary(len(over), len(unmeasured), peak_text), details)


def _check_max_context_pct(calls: list[CallTokens], window: int,
                            limit_pct: float) -> AssertionResult:
    """Fail when any turn's context total exceeds `limit_pct` percent of the
    user-supplied `window`. The budget is stated in tokens next to the
    percentage on every line, so a verdict is never something the reader has to
    recompute — and so a turn sitting within a tenth of the limit reads
    unambiguously.

    Comparison is against the exact token budget (`window * pct / 100`), while
    the displayed percentages are rounded to one decimal — the same
    compare-exact/display-rounded split every percentage in ctxdiff uses.

    Unmeasured turns fail here for the same reason they fail `--max-context`: a
    percentage computed from a floor is a floor, and a floor under a budget
    proves nothing."""
    budget = window * limit_pct / 100
    budget_tokens = int(budget)  # truncated: the largest whole token that fits
    over = [c for c in calls if c.total_tokens > budget]
    unmeasured = [c for c in _unmeasured(calls) if c not in over]
    peak = _peak(calls)
    peak_pct = _pct(_round1(peak.total_tokens / window * 100))
    limit_text = f"limit {_pct(limit_pct)}% ({budget_tokens:,} tok)"
    peak_text = (f"peak {peak_pct}%{_approx_marker(peak)} of {window:,} tok "
                 f"window at turn {peak.seq} · {limit_text}")
    if not over and not unmeasured:
        return AssertionResult(MAX_CONTEXT_PCT, True, peak_text, [])
    details = [
        f"turn {c.seq}{_agent_chip(c.agent)} · {c.total_tokens:,} tok"
        f"{_approx_marker(c)} · {_pct(_round1(c.total_tokens / window * 100))}% "
        f"of {window:,} tok window · {limit_text}"
        for c in over
    ]
    details += [
        f"turn {c.seq}{_agent_chip(c.agent)} · {c.total_tokens:,} tok"
        f"{_approx_marker(c)} · {_pct(_round1(c.total_tokens / window * 100))}% "
        f"of {window:,} tok window · {_unmeasured_note(c)} · {limit_text}"
        for c in unmeasured
    ]
    return AssertionResult(
        MAX_CONTEXT_PCT, False,
        _verdict_summary(len(over), len(unmeasured), peak_text), details)


# --- require-stable-prefix -------------------------------------------------------


def _check_stable_prefix(report: CacheReport, turns: int,
                         agents: int) -> AssertionResult:
    """Fail when the prompt-cache prefix breaks anywhere in the run.

    Every word of the explanation is `analyze_cache`'s own: the breaks, their
    attribution (including the sha-digest wording a same-slot image swap gets,
    where a character offset would explain nothing) and the re-billed total all
    come straight from the CacheReport. Breaks are collapsed by culprit with
    the SAME `group_breaks` the `cache` command renders through, so a timestamp
    that breaks the prefix on all twelve turns is one line here and one line
    there.

    `turns`/`agents` describe the analyzed slice and exist only so the
    no-pairs case can say WHICH no-pairs case it is (see `_no_pairs_summary`) —
    the CacheReport carries the pair count but not the reason it is zero."""
    pairs = report.pairs_analyzed
    if pairs == 0:
        return AssertionResult(
            REQUIRE_STABLE_PREFIX, True,
            _no_pairs_summary(turns, agents, "pairs to check"), [])
    if not report.breaks:
        return AssertionResult(
            REQUIRE_STABLE_PREFIX, True,
            f"prefix stable across all {_plural(pairs, 'turn pair')} · "
            f"min stable prefix {report.stable_prefix_tokens_min:,} tok", [])

    details = []
    for group in group_breaks(report.breaks):
        rep = group[0]
        denom = pairs_denominator(report, rep)
        details.append(
            f"turn {rep.seq_prev} → turn {rep.seq}{_agent_chip(rep.agent)} "
            f"[{rep.culprit_label}·{rep.culprit_kind}] breaks {len(group)}/{denom} "
            f"pairs — {rep.detail}")
    return AssertionResult(
        REQUIRE_STABLE_PREFIX, False,
        f"{_plural(len(report.breaks), 'break')} across "
        f"{_plural(pairs, 'turn pair')} · "
        f"{report.rebilled_tokens_total:,} tok re-billed", details)


# --- no-dead-schemas --------------------------------------------------------------


def _check_no_dead_schemas(run_tokens: RunTokens,
                           total_tools: int) -> AssertionResult:
    """Fail when a tool schema is registered but never invoked anywhere in the
    run — the existing bloat detector, with a threshold of zero.

    A run that registers NO tool schemas at all passes and says so: there is
    nothing to be dead, and reporting `0 of 0` would read like a measurement
    that was taken when none was.

    The tool NAME is the one fragment of a violation line that comes from a
    captured payload rather than from ctxdiff, so it goes through `_flatten`:
    a schema named across two lines would otherwise break the line structure of
    the report and, in the GitHub Action's fenced job summary, the fence."""
    bloat = run_tokens.bloat
    if bloat is None:
        return AssertionResult(NO_DEAD_SCHEMAS, True,
                               "no tool schemas registered", [])
    if not bloat.unused_tools:
        return AssertionResult(
            NO_DEAD_SCHEMAS, True,
            f"all {_plural(total_tools, 'registered tool schema')} invoked", [])
    details = [f"tool schema '{_flatten(name)}' registered but never invoked"
               for name in bloat.unused_tools]
    return AssertionResult(
        NO_DEAD_SCHEMAS, False,
        f"{len(bloat.unused_tools)} of {total_tools} registered tools never "
        f"used · {bloat.unused_tokens_per_call:,} tok/call "
        f"({_pct(bloat.pct_of_avg_context)}% of avg context)", details)


# --- no-tagged-eviction ------------------------------------------------------------


def _check_no_tagged_eviction(report: EvictionReport, turns: int,
                              agents: int) -> AssertionResult:
    """Fail when a block the developer TAGGED entered the context and later left
    it for good — the existing eviction detector, with a threshold of zero.

    Every word of every line comes from `analyze_evictions`: the tag, the turn
    the block entered, the turn it disappeared, the per-agent scoping and the
    "it never came back" rule are all decided there, so a green `check` and a
    hand-read `ctxdiff tokens` can never tell two different stories about the
    same trace. What is new here is only the comparison against zero.

    Three states are reported rather than two, because "PASS" has three
    different meanings and only one of them is reassuring:

    - nothing was TAGGED at all — the assertion is structurally vacuous, and
      saying so is what tells a reader to go add a `tracer.tag()` rather than
      trust a tick that measured nothing;
    - nothing could be PAIRED (a single turn, or a fan-out where each agent has
      one turn) — the same no-pairs distinction every other assertion draws;
    - tagged blocks existed, pairs existed, and none of them was evicted, which
      is the pass that actually means something."""
    if report.tagged_blocks == 0:
        return AssertionResult(
            NO_TAGGED_EVICTION, True,
            "no tagged blocks in this run — nothing to lose "
            "(tag load-bearing content with tracer.tag)", [])
    if report.pairs_analyzed == 0:
        return AssertionResult(
            NO_TAGGED_EVICTION, True,
            _no_pairs_summary(turns, agents, "pairs to check"), [])
    tagged_text = (f"all {_plural(report.tagged_blocks, 'tagged block')} "
                   f"survived {_plural(report.pairs_analyzed, 'turn pair')}")
    if not report.evictions:
        return AssertionResult(NO_TAGGED_EVICTION, True, tagged_text, [])
    # The TAG is the one fragment of these lines the developer authored rather
    # than ctxdiff, so it goes through `_flatten` for the same reason a tool
    # schema's name does: one violation, one line, even inside the Action's
    # fenced job summary.
    details = [
        f"the block you tagged '{_flatten(e.label)}' at turn {e.tagged_seq}"
        f"{_agent_chip(e.agent)} was evicted at turn {e.evicted_seq} · "
        f"{e.tokens:,} tok"
        for e in report.evictions
    ]
    # One event per tagged block per agent (`analyze_evictions` dedupes by
    # content hash within each group), so this numerator and `tagged_blocks`
    # count the same kind of thing and the sentence cannot contradict itself.
    return AssertionResult(
        NO_TAGGED_EVICTION, False,
        f"{_plural(len(report.evictions), 'tagged block')} evicted of "
        f"{report.tagged_blocks} across "
        f"{_plural(report.pairs_analyzed, 'turn pair')}", details)


# --- max-growth / max-growth-pct ---------------------------------------------------


def _growth_pairs(calls: list[CallTokens]) -> list[tuple[CallTokens, CallTokens]]:
    """Consecutive (previous, current) turn pairs, paired WITHIN each agent.

    The grouping rule is `analyze_cache`'s, for the same reason: on a
    multi-agent timeline two adjacent calls can belong to different agents, and
    the "growth" between them is not growth at all — it is a hand-off between
    two independent contexts, and flagging it would make `--max-growth`
    unusable on exactly the runs it matters most for. Agents are visited in
    first-appearance order and each agent's calls stay in seq order, so the
    pairs read in a stable, timeline-like order."""
    order: list[str | None] = []
    by_agent: dict[str | None, list[CallTokens]] = {}
    for c in calls:
        if c.agent not in by_agent:
            order.append(c.agent)
            by_agent[c.agent] = []
        by_agent[c.agent].append(c)
    pairs: list[tuple[CallTokens, CallTokens]] = []
    for label in order:
        group = by_agent[label]
        pairs.extend(zip(group, group[1:]))
    return pairs


def _check_max_growth(calls: list[CallTokens], limit: int) -> AssertionResult:
    """Fail when the context grows by more than `limit` tokens between two
    consecutive turns of the same agent. A shrinking context is never a
    violation, so the peak reported on a pass may legitimately be negative —
    that is the run's largest single-turn growth, honestly stated.

    A pair with an UNMEASURED turn on either side fails without being compared:
    a difference between two totals is only knowable when both are, and the
    error runs in both directions (an unmeasured EARLIER turn overstates the
    growth, an unmeasured LATER one understates it), so neither a pass nor a
    numeric violation would be defensible."""
    pairs = _growth_pairs(calls)
    if not pairs:
        return AssertionResult(
            MAX_GROWTH, True,
            _no_pairs_summary(len(calls), _agents_in(calls),
                              "growth to measure"), [])
    growths = [(prev, cur, cur.total_tokens - prev.total_tokens)
               for prev, cur in pairs]
    peak_prev, peak_cur, peak_growth = max(growths, key=lambda g: g[2])
    peak_text = (f"peak growth {peak_growth:,} tok"
                 f"{_pair_approx_marker(peak_prev, peak_cur)} at turn "
                 f"{peak_cur.seq} · limit {limit:,}")
    over = [g for g in growths if g[2] > limit]
    unmeasured = [g for g in growths
                  if (g[0].unmeasured_blocks or g[1].unmeasured_blocks)
                  and g not in over]
    if not over and not unmeasured:
        return AssertionResult(MAX_GROWTH, True, peak_text, [])
    details = [
        f"turn {prev.seq} → turn {cur.seq}{_agent_chip(cur.agent)} · "
        f"+{growth:,} tok{_pair_approx_marker(prev, cur)} "
        f"({prev.total_tokens:,} → {cur.total_tokens:,}) · limit {limit:,}"
        for prev, cur, growth in over
    ]
    details += [
        f"turn {prev.seq} → turn {cur.seq}{_agent_chip(cur.agent)} · "
        f"{_signed(growth)} tok{_pair_approx_marker(prev, cur)} "
        f"({prev.total_tokens:,} → {cur.total_tokens:,}) · "
        f"{_unmeasured_note(prev, cur)} · limit {limit:,}"
        for prev, cur, growth in unmeasured
    ]
    return AssertionResult(
        MAX_GROWTH, False,
        _verdict_summary(len(over), len(unmeasured), peak_text), details)


def _check_max_growth_pct(calls: list[CallTokens],
                          limit_pct: float) -> AssertionResult:
    """Fail when the context grows by more than `limit_pct` percent between two
    consecutive turns of the same agent.

    A pair whose EARLIER turn totalled zero tokens is skipped rather than
    treated as infinite growth: "grew by ∞%" is not a fact anyone can act on,
    and a zero-token turn is a degenerate capture, not a budget regression. When
    every pair is skipped that way the assertion says SO — naming the skip and
    its reason — rather than borrowing the "fewer than 2 turns" wording, which
    over a run that plainly has more than two turns reads as a bug in the
    reader rather than a measurement that never happened.

    Unmeasured pairs are collected from ALL pairs, before the zero-token filter:
    a floor of zero is exactly the shape an unmeasured turn takes (one image, no
    text), and dropping it here is how it would slip through both branches."""
    all_pairs = _growth_pairs(calls)
    if not all_pairs:
        return AssertionResult(
            MAX_GROWTH_PCT, True,
            _no_pairs_summary(len(calls), _agents_in(calls),
                              "growth to measure"), [])

    # The percentage is only defined where the denominator is: pairs whose
    # earlier turn totalled zero are dropped here and accounted for in the
    # summary below.
    pairs = [(prev, cur) for prev, cur in all_pairs if prev.total_tokens > 0]
    growths = [
        (prev, cur,
         _round1((cur.total_tokens - prev.total_tokens) / prev.total_tokens * 100))
        for prev, cur in pairs
    ]
    over = [g for g in growths if g[2] > limit_pct]
    over_pairs = [(prev, cur) for prev, cur, _ in over]
    unmeasured_pairs = [
        (prev, cur) for prev, cur in all_pairs
        if (prev.unmeasured_blocks or cur.unmeasured_blocks)
        and (prev, cur) not in over_pairs
    ]

    if growths:
        peak_prev, peak_cur, peak_growth = max(growths, key=lambda g: g[2])
        tail = (f"peak growth {_pct(peak_growth)}%"
                f"{_pair_approx_marker(peak_prev, peak_cur)} at turn "
                f"{peak_cur.seq} · limit {_pct(limit_pct)}%")
    else:
        tail = (f"all {_plural(len(all_pairs), 'pair')} skipped — the earlier "
                f"turn had 0 tokens")

    if not over and not unmeasured_pairs:
        return AssertionResult(MAX_GROWTH_PCT, True, tail, [])
    details = [
        f"turn {prev.seq} → turn {cur.seq}{_agent_chip(cur.agent)} · "
        f"+{_pct(growth)}%{_pair_approx_marker(prev, cur)} "
        f"({prev.total_tokens:,} → {cur.total_tokens:,} tok) · "
        f"limit {_pct(limit_pct)}%"
        for prev, cur, growth in over
    ]
    # An unmeasured pair is reported WITHOUT a percentage: the number would be
    # derived from a floor, and quoting it beside a limit is the confusion this
    # whole branch exists to avoid.
    details += [
        f"turn {prev.seq} → turn {cur.seq}{_agent_chip(cur.agent)} · "
        f"{prev.total_tokens:,} → {cur.total_tokens:,} tok"
        f"{_pair_approx_marker(prev, cur)} · {_unmeasured_note(prev, cur)} · "
        f"limit {_pct(limit_pct)}%"
        for prev, cur in unmeasured_pairs
    ]
    return AssertionResult(
        MAX_GROWTH_PCT, False,
        _verdict_summary(len(over), len(unmeasured_pairs), tail), details)


# --- the entry point ----------------------------------------------------------------


def analyze_check(ct: Store, thresholds: Thresholds,
                  agent: str | None = None) -> CheckReport:
    """Run every REQUESTED assertion over one session (optionally scoped to one
    agent) and return their verdicts in `ASSERTION_ORDER`.

    How: the two underlying analyzers are run at most once each and only when
    something asks for them — `analyze_run` for the per-turn totals and the
    bloat report, `analyze_cache` for the prefix breaks — then each requested
    assertion is a pure comparison over that output. An empty session produces
    an empty assertion list and `turns_analyzed == 0`; the caller turns that
    into a failure rather than a pass, because a check that looked at nothing
    has proved nothing."""
    calls = filter_calls(ct.get_calls(), agent)
    if not calls:
        return CheckReport(assertions=[], turns_analyzed=0, agent=agent)

    run_tokens = analyze_run(ct, agent=agent) if thresholds.needs_token_analysis else None
    cache_report = analyze_cache(ct, agent=agent) if thresholds.require_stable_prefix else None
    eviction_report = (analyze_evictions(ct, agent=agent)
                       if thresholds.no_tagged_eviction else None)

    results: list[AssertionResult] = []
    if run_tokens is not None and thresholds.max_context is not None:
        results.append(_check_max_context(run_tokens.calls, thresholds.max_context))
    if (run_tokens is not None and thresholds.max_context_pct is not None
            and thresholds.context_window is not None):
        results.append(_check_max_context_pct(
            run_tokens.calls, thresholds.context_window, thresholds.max_context_pct))
    if cache_report is not None:
        # The turn and agent counts come from the raw calls rather than from the
        # CacheReport: a report with zero pairs cannot say WHY it has none, and
        # "fewer than 2 turns" printed over a four-turn run is how a silently
        # no-op assertion stays unnoticed.
        results.append(_check_stable_prefix(
            cache_report, len(calls), len({c.agent for c in calls})))
    if run_tokens is not None and thresholds.no_dead_schemas:
        # The "M" denominator is derived exactly as `ctxdiff tokens` derives it
        # — `registered_tool_names` over EVERY call in the session, not just the
        # --agent slice — so `check` and `tokens` can never print a different
        # "N of M" for the same trace.
        all_blocks = [ct.get_call_blocks(c.id) for c in ct.get_calls()]
        results.append(_check_no_dead_schemas(
            run_tokens, len(registered_tool_names(all_blocks))))
    if eviction_report is not None:
        # Same turn/agent counts the prefix assertion gets, and for the same
        # reason: an EvictionReport with zero pairs cannot say WHY it has none.
        results.append(_check_no_tagged_eviction(
            eviction_report, len(calls), len({c.agent for c in calls})))
    if run_tokens is not None and thresholds.max_growth is not None:
        results.append(_check_max_growth(run_tokens.calls, thresholds.max_growth))
    if run_tokens is not None and thresholds.max_growth_pct is not None:
        results.append(_check_max_growth_pct(run_tokens.calls, thresholds.max_growth_pct))

    return CheckReport(assertions=results, turns_analyzed=len(calls), agent=agent)

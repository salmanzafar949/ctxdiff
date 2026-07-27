"""Git-style colored rendering of analyzer output. Kept separate from
analyze/differ.py (and analyze/tokens.py) so the pure analyzer algorithms
never have to know about ANSI codes or terminals — the viewer (a later
milestone) reuses the same TurnDiff/CallTokens/BloatReport without pulling
any of this in."""
from __future__ import annotations

import os
import sys

from ctxdiff.analyze.cache import CacheReport, group_breaks, pairs_denominator
from ctxdiff.analyze.check import NAME_WIDTH, CheckReport
from ctxdiff.analyze.differ import TurnDiff
from ctxdiff.analyze.evictions import EvictionReport
from ctxdiff.analyze.tokens import BloatReport, CallTokens, RunTokens, UsageTotals
from ctxdiff.analyze.window import format_window_share

# Bare ANSI SGR constants — no colorama/rich, per CLAUDE.md's "runtime deps:
# tiktoken only" rule. Codes are terminated per-segment by _RESET.
_RESET = "\x1b[0m"
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_DIM = "\x1b[2m"


def _color_enabled() -> bool:
    """Color is on only when stdout is a real terminal AND the user hasn't
    opted out via NO_COLOR (https://no-color.org: presence of the env var —
    any value, even empty string is excluded per the convention's own
    wording, but we treat any *set and non-empty* value as opt-out, which
    covers every real-world usage) — and off whenever output is piped/
    captured, so redirected output (files, `| less`, test capture) is always
    plain text."""
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _paint(text: str, color: str, enabled: bool) -> str:
    """Wrap `text` in `color`'s SGR codes, or return it bare when disabled.
    Every colored line below routes through this one function so "is color
    on" is decided once per render call, not re-checked per line."""
    if not enabled:
        return text
    return f"{color}{text}{_RESET}"


def _snippet(text: str, limit: int = 70) -> str:
    """First ~70 chars of `text` for a diff line. Newlines/repeated
    whitespace are flattened to single spaces first so a multi-line block
    still renders as one line; the result is repr-quoted so the snippet's
    own quotes/control characters are visible and unambiguous rather than
    corrupting the terminal line."""
    flat = " ".join(text.split())
    truncated = flat[:limit] + ("…" if len(flat) > limit else "")
    return repr(truncated)


def _tag(label: str, role: str) -> str:
    """Format the `[label·role]` tag shown before a block's snippet."""
    return f"[{label}·{role}]"


def _agent_step_tag(agent: str | None, step: str | None) -> str:
    """Format a trailing `[agent·step]` marker for a turn header, or "" when
    neither is present. Both, one, or none may be set: only the parts that
    exist are shown (e.g. `[researcher]`, `[·retrieve]` never happens — a
    missing agent simply drops that side)."""
    parts = [p for p in (agent, step) if p]
    return f" [{'·'.join(parts)}]" if parts else ""


def _render_inline_diff(segments: list[tuple[str, str]], enabled: bool) -> str:
    """Render a modified block's char-level diff segments as one compact
    line: unchanged text passes through plain, deleted text is bracketed
    `[-...-]` in red, inserted text `{+...+}` in green. This is a compact
    inline convention (not full unified-diff +/- lines) since the entry as a
    whole is already marked '~' in yellow by the caller."""
    parts = []
    for op, seg in segments:
        if op == "equal":
            parts.append(seg)
        elif op == "delete":
            parts.append(_paint(f"[-{seg}-]", _RED, enabled))
        elif op == "insert":
            parts.append(_paint(f"{{+{seg}+}}", _GREEN, enabled))
    return "".join(parts)


def render_turn_diff(diff: TurnDiff) -> str:
    """Render a TurnDiff as compact, git-style colored text (spec §6.2/§7.1):

    - a header line: turn numbers, count of changed blocks, net token delta
    - one line per added ('+', green) / evicted ('−', red) / modified ('~',
      yellow, with an inline char-level diff) entry, in diff order
    - unchanged blocks summarized as a single dim line rather than listed
      individually, so a diff with 40 stable blocks and 2 real changes reads
      as 2 lines, not 42

    Color is auto-disabled per `_color_enabled` (NO_COLOR env var, or stdout
    not a TTY — e.g. piped output or captured test output)."""
    enabled = _color_enabled()
    lines: list[str] = []

    changed = [e for e in diff.entries if e.kind != "unchanged"]
    unchanged = [e for e in diff.entries if e.kind == "unchanged"]

    header = (f"── turn {diff.seq_old} → turn {diff.seq_new} · "
              f"{len(changed)} blocks changed · "
              f"+{diff.tokens_added} −{diff.tokens_evicted} tokens ──")
    lines.append(header)

    for e in diff.entries:
        if e.kind == "added":
            line = (f"+ {_tag(e.label, e.block.role)} {_snippet(e.block.text)}"
                    f"  +{e.block.token_count} tok")
            lines.append(_paint(line, _GREEN, enabled))
        elif e.kind == "evicted":
            line = (f"− {_tag(e.label, e.block.role)} {_snippet(e.block.text)}"
                    f"  −{e.block.token_count} tok")
            lines.append(_paint(line, _RED, enabled))
        elif e.kind == "modified":
            head = _paint(f"~ {_tag(e.label, e.block.role)}", _YELLOW, enabled)
            body = _render_inline_diff(e.inline_diff or [], enabled)
            lines.append(f"{head} {body}")
        # 'unchanged' entries are folded into the single summary line below.

    if unchanged:
        total_tok = sum(e.block.token_count for e in unchanged)
        summary = f"= {len(unchanged)} unchanged blocks · {total_tok} tok"
        lines.append(_paint(summary, _DIM, enabled))

    return "\n".join(lines)


# Bar width in the token heatmap, per spec §7.1 ("proportional bar ... scaled
# to ~30 cols max"). A slice's bar length is its pct of the call's total,
# scaled onto this width, so a 100%-of-call label draws a full-width bar.
_BAR_WIDTH = 30


def _bar(pct: float) -> str:
    """Render a slice's proportional bar: `pct` (0-100, already a share of
    the call's total) scaled onto `_BAR_WIDTH` block characters, rounded to
    the nearest whole char. A near-zero slice can round down to an empty
    bar — that's honest (it really is a sliver), not a bug."""
    length = round(pct / 100 * _BAR_WIDTH)
    return "█" * length


def render_call_tokens(ct: CallTokens, context_window: int | None = None) -> str:
    """Render one call's token attribution (spec §6.3/§7.1): a header line
    naming the turn, its total tokens (thousands-separated), and a `~approx`
    marker IFF `ct.approximate` — the spec's "always marked approximate,
    never fake precision" rule means this marker is never displayed when it
    isn't true and never omitted when it is. Then one bar-chart line per
    label slice, biggest spender first (already sorted by the analyzer).
    When provider usage is available and reconciles to a known prompt-token
    key, a dim line beneath reports it alongside the delta from our own
    count — omitted entirely when there's nothing to reconcile against.

    `context_window`, when the user has stated one (`--context-window` or
    `CTXDIFF_CONTEXT_WINDOW`), turns the bare total into a SHARE — `18,400 /
    200,000 tok · 9.2%`, warning-marked past `CONTEXT_WINDOW_ALARM_PCT` — which
    is the form that answers the question people open this command with. With no
    window the header is byte-for-byte what it always was: ctxdiff ships no
    model→window table, so it renders no percentage it cannot back up.

    The `(~approx)` marker stays where it has always been, immediately after the
    numbers it qualifies, so it keeps qualifying the percentage too: a share
    computed from a partly-estimated total is exactly as approximate as the
    total."""
    enabled = _color_enabled()
    lines: list[str] = []

    approx_marker = " (~approx)" if ct.approximate else ""
    agent_step = _agent_step_tag(ct.agent, ct.step)
    total = (f"{ct.total_tokens:,} tokens" if context_window is None
             else format_window_share(ct.total_tokens, context_window))
    lines.append(f"turn {ct.seq} · {total}{approx_marker}{agent_step}")

    for s in ct.slices:
        bar = _bar(s.pct).ljust(_BAR_WIDTH)
        lines.append(
            f"  {bar} {s.label:<12} {s.tokens:>8,} tok  {s.pct:>5.1f}%"
        )

    if ct.reconciliation_delta is not None:
        for key in ("prompt_tokens", "input_tokens", "prompt_token_count", "inputTokens"):
            value = (ct.provider_usage or {}).get(key)
            if value is not None:
                line = (f"  provider reports {value:,} prompt tokens · "
                        f"Δ {ct.reconciliation_delta:+d}")
                lines.append(_paint(line, _DIM, enabled))
                break

    return "\n".join(lines)


def render_bloat(bloat: BloatReport, total_tools: int | None) -> str:
    """Render the run-level schema-bloat warning (spec §6.3/§7.1): a single
    yellow line naming the unused tools, how many of the registered total
    they are, their per-call token cost, and that cost as a percentage of
    the average call's context. `total_tools` is the count of distinct
    registered tool schemas in the run (used + unused); when the caller
    can't supply it (no registered-tool count available), the unused count
    itself is used as a same-value fallback rather than showing a bogus
    denominator."""
    enabled = _color_enabled()
    n_unused = len(bloat.unused_tools)
    m_total = total_tools if total_tools is not None else n_unused
    tool_list = ", ".join(bloat.unused_tools)
    line = (
        f"⚠ schema bloat: {tool_list} — {n_unused} of {m_total} registered "
        f"tools never used this run — {bloat.unused_tokens_per_call:,} tok "
        f"({bloat.pct_of_avg_context}% of avg context) spent on dead schemas "
        f"every call"
    )
    return _paint(line, _YELLOW, enabled)


def render_evictions(report: EvictionReport) -> str | None:
    """Render the tagged-eviction warning block for `ctxdiff tokens`, or None
    when there is nothing to warn about (the overwhelmingly common case, and the
    reason this returns None rather than a reassuring line: a report that says
    "no evictions" on every run trains people to stop reading it).

    One three-line stanza per eviction, in the analyzer's timeline order:

    - a yellow headline in the words the bug is usually described in — "the
      block you tagged 'rag' at turn 3 was evicted at turn 6" — prefixed with an
      `[agent:NAME]` chip on a multi-agent run, spelled exactly as
      `ctxdiff cache` spells its own. The turn in it is `tagged_seq`, the turn
      the TAG was applied, because that is what the sentence claims happened
      there;
    - the block's snippet, repr-quoted like every other snippet the CLI prints,
      so control characters in captured text are visible rather than acting on
      the terminal;
    - a facts line: the block's cost, the turn the CONTENT entered the context
      (`entered_seq`, which is not always the turn it was tagged on), the last
      turn that still carried it, and the standing reminder that this report only
      ever names blocks that never came back (see `analyze.evictions`).

    Only TAGGED blocks appear here. Every agent loop evicts heuristically
    labeled history by design, so including those would bury this line in the
    ordinary behaviour of every framework there is."""
    if not report.evictions:
        return None
    enabled = _color_enabled()
    lines: list[str] = []
    for e in report.evictions:
        chip = f"[agent:{e.agent}] " if e.agent is not None else ""
        headline = (f"⚠ {chip}the block you tagged '{e.label}' at turn "
                    f"{e.tagged_seq} was evicted at turn {e.evicted_seq}")
        lines.append(_paint(headline, _YELLOW, enabled))
        lines.append(f"  {repr(e.snippet)}")
        lines.append(f"  [{e.label}·{e.role}] {e.tokens:,} tok · entered at "
                     f"turn {e.entered_seq} · last present at turn "
                     f"{e.last_seen_seq} · never returned")
    return "\n".join(lines)


def render_usage_summary(usage: UsageTotals, agent: str | None = None) -> str:
    """Render the provider-usage rollup block for `ctxdiff tokens`. The scope
    label is honest about what was summed: `<agent> total ·` when the analysis
    was filtered to one agent (`agent` given), else `run total ·`. Only claims
    totals from provider-reported numbers: when NO call reported usage it says
    so plainly rather than printing a misleading `in 0 · out 0`. Otherwise it
    reports summed input/output with an honest coverage fraction
    ('5/6 calls reported usage'), then one line per agent when a multi-agent
    breakdown is available (never the case under an --agent filter)."""
    scope = f"{agent} total" if agent is not None else "run total"
    # The remedy line for the one diagnosable cause of missing usage: OpenAI-
    # style streams that never opted into a usage chunk. Printed wherever the
    # count is non-zero (fully-missing AND partial-coverage runs alike) —
    # "no provider usage reported" with no stated cause was a dead end that
    # sent users source-diving (dogfood finding 2026-07-27). ctxdiff will not
    # inject the option itself, so naming the caller-side fix IS the fix.
    hint = ""
    if usage.streamed_without_usage:
        n = usage.streamed_without_usage
        hint = (f"\n  ↳ {n} streamed call{'s' if n != 1 else ''} recorded no "
                f"usage — OpenAI-style streams only report usage when the "
                f"request includes stream_options={{\"include_usage\": true}}")
    if usage.calls_with_usage == 0:
        return f"{scope} · no provider usage reported{hint}"
    coverage = (f"({usage.calls_with_usage}/{usage.calls_total} "
                f"call{'s' if usage.calls_total != 1 else ''} reported usage)")
    lines = [f"{scope} · in {usage.input_tokens:,} tok · "
             f"out {usage.output_tokens:,} tok {coverage}{hint}"]
    if usage.by_agent:
        for name, (inp, outp) in usage.by_agent.items():
            lines.append(f"  {name} · in {inp:,} · out {outp:,}")
    return "\n".join(lines)


def render_agent_summary(run_tokens: RunTokens) -> str | None:
    """Render the per-agent summary block for `ctxdiff tokens` — one line
    listing each agent with its call count and total tokens — or None when the
    run has no multi-agent breakdown (`by_agent` is None: a single-agent or
    --agent-filtered run). Call counts are derived from the CallTokens list
    (each carries its own agent), so the summary needs no extra store read.
    Unlabeled calls appear under the same '(unlabeled)' bucket by_agent uses."""
    by_agent = run_tokens.by_agent
    if not by_agent:
        return None
    counts: dict[str, int] = {}
    for c in run_tokens.calls:
        key = c.agent if c.agent is not None else "(unlabeled)"
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{name} · {counts.get(name, 0)} calls · {tokens:,} tok"
             for name, tokens in by_agent.items()]
    return "agents: " + "   ".join(parts)


def render_run_tokens(calls: list[CallTokens], bloat: BloatReport | None,
                       total_tools: int | None,
                       agent_summary: str | None = None,
                       usage_summary: str | None = None,
                       context_window: int | None = None,
                       evictions: EvictionReport | None = None) -> str:
    """Render `ctxdiff tokens`' full output: the run-level provider-usage rollup
    FIRST (when supplied), then an optional per-agent block-token summary (when
    the run is multi-agent and unfiltered), then one block per selected call
    (already filtered to a single turn by the caller when `--turn` is given),
    then the tagged-eviction warnings, then the bloat warning appended when
    there is one AND it actually names unused tools (a BloatReport with an empty
    unused list — every registered tool got used — has nothing worth printing).

    `context_window` is threaded down to every turn header (see
    `render_call_tokens`); None means no window was stated and every header
    renders exactly as it did before percentages existed.

    Evictions print BEFORE bloat because they are the more expensive fact: dead
    schemas cost tokens, an evicted tagged block cost the agent something it was
    told to remember. Both are omitted entirely when empty."""
    if not calls:
        return "no calls in this run"
    sections: list[str] = []
    if usage_summary:
        sections.append(usage_summary)
    if agent_summary:
        sections.append(agent_summary)
    sections.extend(render_call_tokens(c, context_window) for c in calls)
    if evictions is not None:
        eviction_block = render_evictions(evictions)
        if eviction_block:
            sections.append(eviction_block)
    if bloat is not None and bloat.unused_tools:
        sections.append(render_bloat(bloat, total_tools))
    return "\n\n".join(sections)


def render_cache_report(report: CacheReport) -> str:
    """Render `ctxdiff cache`'s full output (spec §6.4/§7.1):

    - fewer than 2 calls: just the analyzer's own explanatory note, plain.
    - no breaks: a single green "prefix stable" line, plus the waste note
      only when there's something nonzero to report (there won't be, on the
      happy path, but the analyzer's math is trusted over assuming so).
    - breaks: one yellow warning line per DISTINCT culprit (identical
      culprits across pairs — e.g. the same timestamp breaking every turn —
      collapse into one warning carrying a count), each followed by its
      culprit snippet and detail; then a summary block (stable-prefix
      tokens, rebilled total, the waste note); then the fix hint, dimmed,
      when one was detected."""
    enabled = _color_enabled()

    if report.pairs_analyzed == 0:
        return report.estimated_waste_note

    lines: list[str] = []

    if not report.breaks:
        line = (f"✓ prefix stable across all {report.pairs_analyzed} turn pairs "
                f"— minimum stable prefix {report.stable_prefix_tokens_min:,} tokens")
        lines.append(_paint(line, _GREEN, enabled))
        if report.agents_analyzed and report.agents_analyzed > 1:
            lines.append(f"pairs analyzed within {report.agents_analyzed} agents "
                         f"(cross-agent hand-offs are never counted as breaks)")
        if report.rebilled_tokens_total > 0:
            lines.append(report.estimated_waste_note)
        return "\n".join(lines)

    for group in group_breaks(report.breaks):
        rep = group[0]
        count = len(group)
        # Denominator: THAT agent's own pair count when the break is attributed
        # to one and the run was analyzed per-agent, else the run-wide count.
        # Shared with `ctxdiff check` so both report the same frequency.
        denom = pairs_denominator(report, rep)
        frequency = (
            f"breaks the prefix on every turn ({count}/{denom} pairs)"
            if count == denom else
            f"breaks the prefix on {count}/{denom} turn pairs"
        )
        # Prefix the warning with an agent chip when the break is attributed to
        # a named agent (a grouped multi-agent run); unlabeled runs omit it.
        chip = f"[agent:{rep.agent}] " if rep.agent is not None else ""
        header = f"⚠ warning: {chip}[{rep.culprit_label}·{rep.culprit_kind}] {frequency}"
        lines.append(_paint(header, _YELLOW, enabled))
        lines.append(f"  {repr(rep.culprit_snippet)}")
        lines.append(f"  {rep.detail}")

    lines.append("")
    if report.agents_analyzed and report.agents_analyzed > 1:
        lines.append(f"pairs analyzed within {report.agents_analyzed} agents "
                     f"(cross-agent hand-offs are never counted as breaks)")
    lines.append(f"stable prefix (min): {report.stable_prefix_tokens_min:,} tokens")
    lines.append(f"re-billed: {report.rebilled_tokens_total:,} tokens")
    lines.append(report.estimated_waste_note)

    if report.fix_hint:
        lines.append(_paint(f"hint: {report.fix_hint}", _DIM, enabled))

    return "\n".join(lines)


def render_check_report(report: CheckReport, source: str | None = None) -> str:
    """Render `ctxdiff check`'s PASS/FAIL table — the thing a CI log shows and
    a reviewer reads without opening the trace.

    Layout, top to bottom:

    - a scope header naming how many turns were checked, which agent (when the
      check was scoped to one), and WHAT WAS READ — `source`, the caller's
      `session <short id>` label, qualified by the filename when the trace was
      discovered rather than named. That last part is not decoration: with no
      `--project` the CLI reads the most recently modified `*.ctrace` in the
      working directory (the GitHub Action's default), so an unrelated newer
      trace can be checked, pass, and leave a report indistinguishable from one
      over the intended run. A verdict that does not say what it read cannot be
      audited;
    - one status line per REQUESTED assertion, in the analyzer's fixed order,
      as `PASS`/`FAIL` (green/red) + the assertion's name padded to a common
      column + its summary, which always carries the actual value beside the
      threshold — a passing check that reports its high-water mark is what lets
      someone watch a budget approach its limit over successive PRs, instead of
      only finding out on the day it breaks;
    - beneath each FAILING assertion, its violation lines indented two spaces,
      one per offending turn/agent/block;
    - a blank line, then a verdict line (green on a clean pass, red otherwise).

    Every string below the status column is composed by `analyze/check.py`;
    this function only decides columns and color, exactly as the other
    renderers do."""
    enabled = _color_enabled()
    lines: list[str] = []

    scope = f" · agent {report.agent}" if report.agent is not None else ""
    origin = f" · {source}" if source else ""
    turn_word = "turn" if report.turns_analyzed == 1 else "turns"
    lines.append(
        f"ctxdiff check · {report.turns_analyzed} {turn_word}{scope}{origin}")

    for a in report.assertions:
        status = _paint("PASS", _GREEN, enabled) if a.passed else _paint("FAIL", _RED, enabled)
        lines.append(f"{status}  {a.name:<{NAME_WIDTH}}  {a.summary}")
        for detail in a.details:
            lines.append(f"  {detail}")

    lines.append("")
    total = len(report.assertions)
    if report.passed:
        verdict = f"check passed · {total} assertion{'' if total == 1 else 's'}"
        lines.append(_paint(verdict, _GREEN, enabled))
    else:
        failed = len(report.failed)
        verdict = (f"check FAILED · {failed} of {total} "
                   f"assertion{'' if total == 1 else 's'} failed")
        lines.append(_paint(verdict, _RED, enabled))

    return "\n".join(lines)


def render_sessions_list(rows: list[tuple[str, str, str, str, int, str]],
                         empty: str = "no .ctrace files in the current directory") -> str:
    """Render `ctxdiff sessions`' listing (and its hidden `runs` alias). Each
    row is (label, started_local, project, provider, turn_count, agents).

    `label` is deliberately not named after any one thing it can be: a filename
    when listing the `.ctrace` files in a directory, `<filename>#<short id>`
    when one of those files holds several sessions, and a bare short session id
    when listing a configured database (which has no filenames at all).

    `started_local` arrives ALREADY formatted (see `select.format_local`) — this
    module never touches a clock or a timezone, it only lays out columns.

    Prints one line per row, or `empty` when there is nothing to list; the
    caller supplies that message because "no .ctrace files in the current
    directory" would be the wrong answer for a user whose traces live in
    Postgres. `agents` is a comma-joined list of the distinct agent names, or
    '-' when the session has no named agents (single-agent/pre-v2)."""
    if not rows:
        return empty
    lines = [
        f"{label}  {started}  project={project}  provider={provider}"
        f"  turns={n_calls}  agents={agents}"
        for label, started, project, provider, n_calls, agents in rows
    ]
    return "\n".join(lines)


def render_agents_list(rows: list[tuple[str, int, int, str]],
                       empty: str = "no agents in this project") -> str:
    """Render `ctxdiff agents`' listing: one line per agent with how many
    SESSIONS it appears in, how many calls it made in total, and its token
    spend — all aggregated across every session in the project, which is the
    whole point of the command (an agent's cost is a property of the project,
    not of whichever run you happened to open).

    Each row is (name, n_sessions, n_calls, tokens) where `tokens` is already a
    string: the caller formats it as a thousands-separated PROVIDER-REPORTED
    total (input + output), or '-' when not one of that agent's calls carried
    usage — the same "never fake precision" rule the token report follows,
    since printing `tokens=0` for unreported usage would read as free."""
    if not rows:
        return empty
    lines = [
        f"{name}  sessions={n_sessions}  calls={n_calls}  tokens={tokens}"
        for name, n_sessions, n_calls, tokens in rows
    ]
    return "\n".join(lines)

"""Git-style colored rendering of analyzer output. Kept separate from
analyze/differ.py (and analyze/tokens.py) so the pure analyzer algorithms
never have to know about ANSI codes or terminals — the viewer (a later
milestone) reuses the same TurnDiff/CallTokens/BloatReport without pulling
any of this in."""
from __future__ import annotations

import os
import sys

from ctxdiff.analyze.cache import CacheReport, PrefixBreak
from ctxdiff.analyze.differ import TurnDiff
from ctxdiff.analyze.tokens import BloatReport, CallTokens, RunTokens, UsageTotals

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


def render_call_tokens(ct: CallTokens) -> str:
    """Render one call's token attribution (spec §6.3/§7.1): a header line
    naming the turn, its total tokens (thousands-separated), and a `~approx`
    marker IFF `ct.approximate` — the spec's "always marked approximate,
    never fake precision" rule means this marker is never displayed when it
    isn't true and never omitted when it is. Then one bar-chart line per
    label slice, biggest spender first (already sorted by the analyzer).
    When provider usage is available and reconciles to a known prompt-token
    key, a dim line beneath reports it alongside the delta from our own
    count — omitted entirely when there's nothing to reconcile against."""
    enabled = _color_enabled()
    lines: list[str] = []

    approx_marker = " (~approx)" if ct.approximate else ""
    agent_step = _agent_step_tag(ct.agent, ct.step)
    lines.append(f"turn {ct.seq} · {ct.total_tokens:,} tokens{approx_marker}{agent_step}")

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
    if usage.calls_with_usage == 0:
        return f"{scope} · no provider usage reported"
    coverage = (f"({usage.calls_with_usage}/{usage.calls_total} "
                f"call{'s' if usage.calls_total != 1 else ''} reported usage)")
    lines = [f"{scope} · in {usage.input_tokens:,} tok · "
             f"out {usage.output_tokens:,} tok {coverage}"]
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
                       usage_summary: str | None = None) -> str:
    """Render `ctxdiff tokens`' full output: the run-level provider-usage rollup
    FIRST (when supplied), then an optional per-agent block-token summary (when
    the run is multi-agent and unfiltered), then one block per selected call
    (already filtered to a single turn by the caller when `--turn` is given),
    then the bloat warning appended when there is one AND it actually names
    unused tools (a BloatReport with an empty unused list — every registered
    tool got used — has nothing worth printing)."""
    if not calls:
        return "no calls in this run"
    sections: list[str] = []
    if usage_summary:
        sections.append(usage_summary)
    if agent_summary:
        sections.append(agent_summary)
    sections.extend(render_call_tokens(c) for c in calls)
    if bloat is not None and bloat.unused_tools:
        sections.append(render_bloat(bloat, total_tools))
    return "\n\n".join(sections)


def _group_breaks(breaks: list[PrefixBreak]) -> list[list[PrefixBreak]]:
    """Group PrefixBreaks that describe the "same" underlying culprit —
    (agent, culprit_kind, culprit_label, divergent_position) — into one list
    per distinct culprit, in first-seen order. `agent` is part of the key so
    two agents breaking the same way at the same slot stay SEPARATE warnings
    (each carries its own agent chip). Grouping deliberately ignores the
    per-pair `detail`/`culprit_snippet` text (a changing timestamp's exact
    before/after values differ every pair by definition) — what makes two
    breaks "the same warning" is that the same agent's same slot keeps
    breaking the same way, not that the literal diff text is identical."""
    groups: dict[tuple[str | None, str, str, int], list[PrefixBreak]] = {}
    for b in breaks:
        key = (b.agent, b.culprit_kind, b.culprit_label, b.divergent_position)
        groups.setdefault(key, []).append(b)
    return list(groups.values())


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

    for group in _group_breaks(report.breaks):
        rep = group[0]
        count = len(group)
        # Denominator: when the break is attributed to a specific agent and the
        # run was analyzed per-agent, use THAT agent's own pair count (a
        # researcher breaking on both of its 2 pairs is "2/2", not "2/3" of a
        # run that also includes a stable writer). Fall back to the run-wide
        # count for an unlabeled break or a single-timeline run.
        denom = report.pairs_analyzed
        if rep.agent is not None and report.pairs_by_agent:
            denom = report.pairs_by_agent.get(rep.agent, report.pairs_analyzed)
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


def render_runs_list(rows: list[tuple[str, str, str, int, str]]) -> str:
    """Render `ctxdiff runs`' listing. Each row is
    (filename, project, provider, turn_count, agents); prints one line per row,
    or a friendly message when the working directory has no `.ctrace` files.
    `agents` is a comma-joined list of the distinct agent names in the trace,
    or '-' when the run has no named agents (a single-agent/pre-v2 run)."""
    if not rows:
        return "no .ctrace files in the current directory"
    lines = [
        f"{filename}  project={project}  provider={provider}  turns={n_calls}"
        f"  agents={agents}"
        for filename, project, provider, n_calls, agents in rows
    ]
    return "\n".join(lines)

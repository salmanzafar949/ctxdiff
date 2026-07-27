/**
 * Git-style colored rendering of analyzer output. A faithful port of Python
 * `ctxdiff.cli.render` — every line, separator, glyph, padding and number
 * format matches so `ctxdiff diff|tokens|cache|check|runs` output is byte-identical to
 * the Python CLI's (with color auto-disabled off a TTY, exactly as Python does).
 * Kept separate from the analyzers so they never depend on ANSI/terminal code.
 */
import type { TurnDiff, InlineSegment } from "./analyze/diff.js";
import type {
  BloatReport,
  CallTokens,
  RunTokens,
  UsageTotals,
} from "./analyze/tokens.js";
import { groupBreaks, pairsDenominator, type CacheReport } from "./analyze/cache.js";
import type { EvictionReport } from "./analyze/evictions.js";
import { formatWindowShare } from "./analyze/window.js";
import {
  checkPassed,
  failedAssertions,
  NAME_WIDTH,
  type CheckReport,
} from "./analyze/check.js";
import { pyComma as comma, pyRepr, pyRoundHalfEven } from "./analyze/pyround.js";

// Re-exported from its new home so `store/sqlite.ts`, `index.ts` and every
// existing import of `render.pyRepr` keeps working unchanged. It lives in
// `analyze/pyround.ts` now because `analyze/window.ts` needs it too, and an
// analyzer importing the renderer would be a cycle.
export { pyRepr };

// Bare ANSI SGR constants — no color library, matching Python's render.py.
const RESET = "\x1b[0m";
const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const DIM = "\x1b[2m";

/**
 * Color is on only when stdout is a real terminal AND NO_COLOR is unset —
 * off whenever output is piped/captured, so redirected output (files, tests,
 * cross-language comparison) is always plain text. Mirrors Python
 * `_color_enabled`.
 */
function colorEnabled(): boolean {
  if (process.env.NO_COLOR) return false;
  return Boolean(process.stdout.isTTY);
}

/** Wrap `text` in `color`'s SGR codes, or return it bare when disabled. */
function paint(text: string, color: string, enabled: boolean): string {
  return enabled ? `${color}${text}${RESET}` : text;
}

/** First ~70 chars of `text` for a diff line: whitespace flattened to single
 * spaces, truncated with an ellipsis, then repr-quoted. Mirrors Python
 * `_snippet`.
 *
 * The limit counts CODE POINTS, not UTF-16 code units, because Python's
 * `flat[:70]` and `len(flat)` both count code points — so on any text carrying
 * astral characters (emoji, ZWJ sequences, math alphanumerics, astral CJK) a
 * `String.prototype.slice` here would cut the snippet SHORT of Python's, print
 * an ellipsis Python would not, and — worst of all — could cut between the two
 * halves of a surrogate pair, leaving a lone surrogate that `pyRepr` renders as
 * a `\ud83d` escape Python can never produce. `spec/golden/expected/cli/
 * unicode.diff.1-2.txt` is the regression that pins this. The viewer's
 * `sliceCp` solves the same problem the same way for the dashboard's 200-char
 * block snippets. */
function snippet(text: string, limit = 70): string {
  const flat = text.split(/\s+/u).filter((s) => s.length > 0).join(" ");
  const codePoints = Array.from(flat);
  const truncated =
    codePoints.slice(0, limit).join("") + (codePoints.length > limit ? "…" : "");
  return pyRepr(truncated);
}

/** Format the `[label·role]` tag before a block's snippet. */
function tag(label: string, role: string): string {
  return `[${label}·${role}]`;
}

/** Trailing `[agent·step]` marker for a turn header, or "" when neither is set. */
function agentStepTag(agent: string | null, step: string | null): string {
  const parts = [agent, step].filter((p): p is string => Boolean(p));
  return parts.length ? ` [${parts.join("·")}]` : "";
}

/** Render a modified block's char-level diff: unchanged text plain, deletes
 * `[-...-]` red, inserts `{+...+}` green. Mirrors Python `_render_inline_diff`. */
function renderInlineDiff(segments: InlineSegment[], enabled: boolean): string {
  const parts: string[] = [];
  for (const [op, seg] of segments) {
    if (op === "equal") parts.push(seg);
    else if (op === "delete") parts.push(paint(`[-${seg}-]`, RED, enabled));
    else if (op === "insert") parts.push(paint(`{+${seg}+}`, GREEN, enabled));
  }
  return parts.join("");
}

/** Render a TurnDiff as compact git-style text. Mirrors Python
 * `render_turn_diff`: a header, one line per added/evicted/modified entry in
 * diff order, unchanged blocks folded into a single dim summary line. */
export function renderTurnDiff(diff: TurnDiff): string {
  const enabled = colorEnabled();
  const lines: string[] = [];

  const changed = diff.entries.filter((e) => e.kind !== "unchanged");
  const unchanged = diff.entries.filter((e) => e.kind === "unchanged");

  lines.push(
    `── turn ${diff.seqOld} → turn ${diff.seqNew} · ` +
      `${changed.length} blocks changed · ` +
      `+${diff.tokensAdded} −${diff.tokensEvicted} tokens ──`,
  );

  for (const e of diff.entries) {
    if (e.kind === "added") {
      const line =
        `+ ${tag(e.label, e.block.role)} ${snippet(e.block.text)}` +
        `  +${e.block.tokenCount} tok`;
      lines.push(paint(line, GREEN, enabled));
    } else if (e.kind === "evicted") {
      const line =
        `− ${tag(e.label, e.block.role)} ${snippet(e.block.text)}` +
        `  −${e.block.tokenCount} tok`;
      lines.push(paint(line, RED, enabled));
    } else if (e.kind === "modified") {
      const head = paint(`~ ${tag(e.label, e.block.role)}`, YELLOW, enabled);
      const body = renderInlineDiff(e.inlineDiff ?? [], enabled);
      lines.push(`${head} ${body}`);
    }
  }

  if (unchanged.length) {
    const totalTok = unchanged.reduce((acc, e) => acc + e.block.tokenCount, 0);
    lines.push(paint(`= ${unchanged.length} unchanged blocks · ${totalTok} tok`, DIM, enabled));
  }

  return lines.join("\n");
}

const BAR_WIDTH = 30;

/** Render a slice's proportional bar: `pct` scaled onto `BAR_WIDTH` block
 * characters, rounded to the nearest whole char (round-half-to-even). Mirrors
 * Python `_bar`. */
function bar(pct: number): string {
  const length = pyRoundHalfEven((pct / 100) * BAR_WIDTH);
  return "█".repeat(length);
}

/** Left-justify to width `w` (Python `{:<w}`); no truncation if longer. */
function ljust(s: string, w: number): string {
  return s.length >= w ? s : s + " ".repeat(w - s.length);
}
/** Right-justify to width `w` (Python `{:>w}`). */
function rjust(s: string, w: number): string {
  return s.length >= w ? s : " ".repeat(w - s.length) + s;
}

/**
 * Render one call's token attribution. Mirrors Python `render_call_tokens`.
 *
 * `contextWindow`, when the user has stated one (`--context-window` or
 * `CTXDIFF_CONTEXT_WINDOW`), turns the bare total into a SHARE — `18,400 /
 * 200,000 tok · 9.2%`, warning-marked past `CONTEXT_WINDOW_ALARM_PCT`. With no
 * window the header is byte-for-byte what it always was: ctxdiff ships no
 * model→window table, so it renders no percentage it cannot back up.
 *
 * The `(~approx)` marker stays immediately after the numbers it qualifies, so it
 * keeps qualifying the percentage too: a share computed from a partly-estimated
 * total is exactly as approximate as the total.
 */
export function renderCallTokens(ct: CallTokens, contextWindow: number | null = null): string {
  const enabled = colorEnabled();
  const lines: string[] = [];

  const approxMarker = ct.approximate ? " (~approx)" : "";
  const agentStep = agentStepTag(ct.agent, ct.step);
  const total =
    contextWindow === null
      ? `${comma(ct.totalTokens)} tokens`
      : formatWindowShare(ct.totalTokens, contextWindow);
  lines.push(`turn ${ct.seq} · ${total}${approxMarker}${agentStep}`);

  for (const s of ct.slices) {
    const barStr = ljust(bar(s.pct), BAR_WIDTH);
    lines.push(
      `  ${barStr} ${ljust(s.label, 12)} ${rjust(comma(s.tokens), 8)} tok  ` +
        `${rjust(s.pct.toFixed(1), 5)}%`,
    );
  }

  if (ct.reconciliationDelta !== null) {
    for (const key of ["prompt_tokens", "input_tokens", "prompt_token_count", "inputTokens"]) {
      const value = (ct.providerUsage ?? {})[key];
      if (value !== null && value !== undefined) {
        const delta = ct.reconciliationDelta;
        const signed = delta >= 0 ? `+${delta}` : `${delta}`;
        lines.push(
          paint(
            `  provider reports ${comma(value as number)} prompt tokens · Δ ${signed}`,
            DIM,
            enabled,
          ),
        );
        break;
      }
    }
  }

  return lines.join("\n");
}

/** Render the run-level schema-bloat warning. Mirrors Python `render_bloat`. */
export function renderBloat(bloat: BloatReport, totalTools: number | null): string {
  const enabled = colorEnabled();
  const nUnused = bloat.unusedTools.length;
  const mTotal = totalTools !== null ? totalTools : nUnused;
  const toolList = bloat.unusedTools.join(", ");
  const line =
    `⚠ schema bloat: ${toolList} — ${nUnused} of ${mTotal} registered ` +
    `tools never used this run — ${comma(bloat.unusedTokensPerCall)} tok ` +
    `(${bloat.pctOfAvgContext.toFixed(1)}% of avg context) spent on dead schemas ` +
    `every call`;
  return paint(line, YELLOW, enabled);
}

/**
 * Render the tagged-eviction warning block for `ctxdiff tokens`, or null when
 * there is nothing to warn about (the overwhelmingly common case, and the reason
 * this returns null rather than a reassuring line: a report that says "no
 * evictions" on every run trains people to stop reading it). Mirrors Python
 * `render_evictions`.
 *
 * One three-line stanza per eviction, in the analyzer's timeline order: a yellow
 * headline in the words the bug is usually described in — with `taggedSeq`, the
 * turn the TAG was applied, because that is what the sentence claims happened
 * there — the block's snippet repr-quoted like every other snippet the CLI
 * prints, and a facts line carrying the cost, the turn the CONTENT entered
 * (`enteredSeq`, not always the turn it was tagged on), the last turn that still
 * had it, and the standing reminder that this report only ever names blocks that
 * never came back.
 *
 * Only TAGGED blocks appear here. Every agent loop evicts heuristically labeled
 * history by design, so including those would bury this line in the ordinary
 * behaviour of every framework there is.
 */
export function renderEvictions(report: EvictionReport): string | null {
  if (!report.evictions.length) return null;
  const enabled = colorEnabled();
  const lines: string[] = [];
  for (const e of report.evictions) {
    const chip = e.agent !== null ? `[agent:${e.agent}] ` : "";
    const headline =
      `⚠ ${chip}the block you tagged '${e.label}' at turn ` +
      `${e.taggedSeq} was evicted at turn ${e.evictedSeq}`;
    lines.push(paint(headline, YELLOW, enabled));
    lines.push(`  ${pyRepr(e.snippet)}`);
    lines.push(
      `  [${e.label}·${e.role}] ${comma(e.tokens)} tok · entered at ` +
        `turn ${e.enteredSeq} · last present at turn ` +
        `${e.lastSeenSeq} · never returned`,
    );
  }
  return lines.join("\n");
}

/** Render the provider-usage rollup for `ctxdiff tokens`. Mirrors Python
 * `render_usage_summary`. */
export function renderUsageSummary(usage: UsageTotals, agent: string | null = null): string {
  const scope = agent !== null ? `${agent} total` : "run total";
  // The remedy line for the one diagnosable cause of missing usage: OpenAI-
  // style streams that never opted into a usage chunk. Printed wherever the
  // count is non-zero (fully-missing AND partial-coverage runs alike) —
  // "no provider usage reported" with no stated cause was a dead end that
  // sent users source-diving (dogfood finding 2026-07-27). ctxdiff will not
  // inject the option itself, so naming the caller-side fix IS the fix.
  let hint = "";
  if (usage.streamedWithoutUsage > 0) {
    const n = usage.streamedWithoutUsage;
    hint =
      `\n  ↳ ${n} streamed call${n !== 1 ? "s" : ""} recorded no usage — ` +
      `OpenAI-style streams only report usage when the request includes ` +
      `stream_options={"include_usage": true}`;
  }
  if (usage.callsWithUsage === 0) return `${scope} · no provider usage reported${hint}`;
  const coverage =
    `(${usage.callsWithUsage}/${usage.callsTotal} ` +
    `call${usage.callsTotal !== 1 ? "s" : ""} reported usage)`;
  const lines = [
    `${scope} · in ${comma(usage.inputTokens)} tok · ` +
      `out ${comma(usage.outputTokens)} tok ${coverage}${hint}`,
  ];
  if (usage.byAgent) {
    for (const [name, [inp, outp]] of usage.byAgent) {
      lines.push(`  ${name} · in ${comma(inp)} · out ${comma(outp)}`);
    }
  }
  return lines.join("\n");
}

/** Render the per-agent summary block for `ctxdiff tokens`, or null when the
 * run has no multi-agent breakdown. Mirrors Python `render_agent_summary`. */
export function renderAgentSummary(runTokens: RunTokens): string | null {
  const byAgent = runTokens.byAgent;
  if (!byAgent || byAgent.size === 0) return null;
  const counts = new Map<string, number>();
  for (const c of runTokens.calls) {
    const key = c.agent ?? "(unlabeled)";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const parts: string[] = [];
  for (const [name, tokens] of byAgent) {
    parts.push(`${name} · ${counts.get(name) ?? 0} calls · ${comma(tokens)} tok`);
  }
  return "agents: " + parts.join("   ");
}

/**
 * Render `ctxdiff tokens`' full output. Mirrors Python `render_run_tokens`.
 *
 * `contextWindow` is threaded down to every turn header; null means no window
 * was stated and every header renders exactly as it did before percentages
 * existed. Evictions print BEFORE bloat because they are the more expensive
 * fact: dead schemas cost tokens, an evicted tagged block cost the agent
 * something it was told to remember. Both are omitted entirely when empty.
 */
export function renderRunTokens(
  calls: CallTokens[],
  bloat: BloatReport | null,
  totalTools: number | null,
  agentSummary: string | null = null,
  usageSummary: string | null = null,
  contextWindow: number | null = null,
  evictions: EvictionReport | null = null,
): string {
  if (calls.length === 0) return "no calls in this run";
  const sections: string[] = [];
  if (usageSummary) sections.push(usageSummary);
  if (agentSummary) sections.push(agentSummary);
  for (const c of calls) sections.push(renderCallTokens(c, contextWindow));
  if (evictions !== null) {
    const evictionBlock = renderEvictions(evictions);
    if (evictionBlock) sections.push(evictionBlock);
  }
  if (bloat !== null && bloat.unusedTools.length) sections.push(renderBloat(bloat, totalTools));
  return sections.join("\n\n");
}

/** Render `ctxdiff cache`'s full output. Mirrors Python `render_cache_report`. */
export function renderCacheReport(report: CacheReport): string {
  const enabled = colorEnabled();

  if (report.pairsAnalyzed === 0) return report.estimatedWasteNote;

  const lines: string[] = [];

  if (report.breaks.length === 0) {
    lines.push(
      paint(
        `✓ prefix stable across all ${report.pairsAnalyzed} turn pairs ` +
          `— minimum stable prefix ${comma(report.stablePrefixTokensMin)} tokens`,
        GREEN,
        enabled,
      ),
    );
    if (report.agentsAnalyzed && report.agentsAnalyzed > 1) {
      lines.push(
        `pairs analyzed within ${report.agentsAnalyzed} agents ` +
          `(cross-agent hand-offs are never counted as breaks)`,
      );
    }
    if (report.rebilledTokensTotal > 0) lines.push(report.estimatedWasteNote);
    return lines.join("\n");
  }

  for (const group of groupBreaks(report.breaks)) {
    const rep = group[0];
    const count = group.length;
    // Denominator: THAT agent's own pair count when the break is attributed to
    // one and the run was analyzed per-agent, else the run-wide count. Shared
    // with `ctxdiff check` so both report the same frequency.
    const denom = pairsDenominator(report, rep);
    const frequency =
      count === denom
        ? `breaks the prefix on every turn (${count}/${denom} pairs)`
        : `breaks the prefix on ${count}/${denom} turn pairs`;
    const chip = rep.agent !== null ? `[agent:${rep.agent}] ` : "";
    const header = `⚠ warning: ${chip}[${rep.culpritLabel}·${rep.culpritKind}] ${frequency}`;
    lines.push(paint(header, YELLOW, enabled));
    lines.push(`  ${pyRepr(rep.culpritSnippet)}`);
    lines.push(`  ${rep.detail}`);
  }

  lines.push("");
  if (report.agentsAnalyzed && report.agentsAnalyzed > 1) {
    lines.push(
      `pairs analyzed within ${report.agentsAnalyzed} agents ` +
        `(cross-agent hand-offs are never counted as breaks)`,
    );
  }
  lines.push(`stable prefix (min): ${comma(report.stablePrefixTokensMin)} tokens`);
  lines.push(`re-billed: ${comma(report.rebilledTokensTotal)} tokens`);
  lines.push(report.estimatedWasteNote);

  if (report.fixHint) lines.push(paint(`hint: ${report.fixHint}`, DIM, enabled));

  return lines.join("\n");
}

/**
 * Render `ctxdiff check`'s PASS/FAIL table — the thing a CI log shows and a
 * reviewer reads without opening the trace. Mirrors Python
 * `render_check_report`.
 *
 * Layout, top to bottom: a scope header naming how many turns were checked,
 * which agent (when scoped to one) and WHAT WAS READ — `source`, the caller's
 * `session <short id>` label, qualified by the filename when the trace was
 * discovered rather than named. That last part is not decoration: with no
 * `--project` the CLI reads the most recently modified `*.ctrace` in the working
 * directory (the GitHub Action's default), so an unrelated newer trace can be
 * checked, pass, and leave a report indistinguishable from one over the intended
 * run. A verdict that does not say what it read cannot be audited. Then one
 * status line per REQUESTED assertion, in
 * the analyzer's fixed order, as `PASS`/`FAIL` (green/red) + the name padded to
 * a common column + a summary that always carries the actual value beside the
 * threshold — a passing check that reports its high-water mark is what lets
 * someone watch a budget approach its limit over successive PRs, instead of only
 * finding out on the day it breaks; then, beneath each FAILING assertion, its
 * violation lines indented two spaces; then a blank line and a verdict.
 *
 * Every string below the status column is composed by `analyze/check.ts`; this
 * function only decides columns and color, exactly as the other renderers do.
 */
export function renderCheckReport(report: CheckReport, source: string | null = null): string {
  const enabled = colorEnabled();
  const lines: string[] = [];

  const scope = report.agent !== null ? ` · agent ${report.agent}` : "";
  const origin = source ? ` · ${source}` : "";
  const turnWord = report.turnsAnalyzed === 1 ? "turn" : "turns";
  lines.push(`ctxdiff check · ${report.turnsAnalyzed} ${turnWord}${scope}${origin}`);

  for (const a of report.assertions) {
    const status = a.passed ? paint("PASS", GREEN, enabled) : paint("FAIL", RED, enabled);
    lines.push(`${status}  ${ljust(a.name, NAME_WIDTH)}  ${a.summary}`);
    for (const detail of a.details) lines.push(`  ${detail}`);
  }

  lines.push("");
  const total = report.assertions.length;
  const suffix = total === 1 ? "" : "s";
  if (checkPassed(report)) {
    lines.push(paint(`check passed · ${total} assertion${suffix}`, GREEN, enabled));
  } else {
    const failed = failedAssertions(report).length;
    lines.push(
      paint(`check FAILED · ${failed} of ${total} assertion${suffix} failed`, RED, enabled),
    );
  }

  return lines.join("\n");
}

/** One row of the `sessions` listing. `label` is deliberately not named after
 * any one thing it can be: a filename when listing the `.ctrace` files in a
 * directory, `<filename>#<short id>` when one of those files holds several
 * sessions, and a bare short session id when listing a configured database
 * (which has no filenames at all). `started` arrives ALREADY formatted (see
 * `selectors.formatLocal`) — this module never touches a clock or a timezone,
 * it only lays out columns. */
export interface SessionRow {
  label: string;
  started: string;
  project: string;
  provider: string;
  turns: number;
  agents: string;
}

/** Render `ctxdiff sessions`' listing (and its hidden `runs` alias): one line
 * per session, or `empty` when there is nothing to list — the caller supplies
 * that message because "no .ctrace files in the current directory" would be the
 * wrong answer for a user whose traces live in Postgres. Mirrors Python
 * `render_sessions_list`. */
export function renderSessionsList(
  rows: SessionRow[],
  empty = "no .ctrace files in the current directory",
): string {
  if (rows.length === 0) return empty;
  return rows
    .map(
      (r) =>
        `${r.label}  ${r.started}  project=${r.project}  provider=${r.provider}` +
        `  turns=${r.turns}  agents=${r.agents}`,
    )
    .join("\n");
}

/** One row of the `agents` listing. `tokens` is already a string: the caller
 * formats it as a thousands-separated PROVIDER-REPORTED total (input + output),
 * or '-' when not one of that agent's calls carried usage — the same "never fake
 * precision" rule the token report follows, since printing `tokens=0` for
 * unreported usage would read as free. */
export interface AgentRow {
  name: string;
  sessions: number;
  calls: number;
  tokens: string;
}

/** Render `ctxdiff agents`' listing: one line per agent with how many SESSIONS
 * it appears in, how many calls it made in total, and its token spend — all
 * aggregated across every session in the project, which is the whole point of
 * the command (an agent's cost is a property of the project, not of whichever
 * run you happened to open). Mirrors Python `render_agents_list`. */
export function renderAgentsList(rows: AgentRow[], empty = "no agents in this project"): string {
  if (rows.length === 0) return empty;
  return rows
    .map((r) => `${r.name}  sessions=${r.sessions}  calls=${r.calls}  tokens=${r.tokens}`)
    .join("\n");
}

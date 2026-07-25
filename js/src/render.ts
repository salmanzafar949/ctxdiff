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
import {
  checkPassed,
  failedAssertions,
  NAME_WIDTH,
  type CheckReport,
} from "./analyze/check.js";
import { pyComma as comma, pyRoundHalfEven } from "./analyze/pyround.js";

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

// A code point is non-printable (in Python's `str.isprintable` sense) when its
// Unicode category is "Other" (C*: control, format, surrogate, private-use,
// unassigned) or "Separator" (Z*: line/paragraph/space) — EXCEPT the ASCII
// space U+0020, which is printable. This covers C0/C1 controls, DEL, NBSP,
// soft-hyphen, the bidi/zero-width format marks (U+200B–200F, U+202A–202E,
// U+2060, U+FEFF) and the line/paragraph separators, exactly as CPython does.
const NON_PRINTABLE = /[\p{C}\p{Z}]/u;
function isNonPrintable(ch: string): boolean {
  return ch !== " " && NON_PRINTABLE.test(ch);
}

/**
 * Python `repr(str)`: choose single quotes unless the string contains a single
 * quote and no double quote (then double); escape the quote and backslash, use
 * the `\n`/`\r`/`\t` shorthands, escape any non-printable code point (see
 * `isNonPrintable`) using Python's width-appropriate form — `\xNN` below 0x100,
 * `\uNNNN` in the BMP, `\UNNNNNNNN` for astral — and pass every printable
 * character (including café/emoji) through. Iterates code points (`for..of`)
 * like Python, so astral characters are handled as single units.
 *
 * Exported because diff snippets are not the only place Python's repr quoting
 * has to be reproduced: `SQLiteStore.openReader` interpolates a path with `!r`
 * on the Python side, and `JSON.stringify` there would print double quotes
 * where Python prints single ones.
 */
export function pyRepr(s: string): string {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const o = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += "\\" + ch;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (isNonPrintable(ch)) {
      if (o < 0x100) out += "\\x" + o.toString(16).padStart(2, "0");
      else if (o < 0x10000) out += "\\u" + o.toString(16).padStart(4, "0");
      else out += "\\U" + o.toString(16).padStart(8, "0");
    } else out += ch;
  }
  return out + quote;
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

/** Render one call's token attribution. Mirrors Python `render_call_tokens`. */
export function renderCallTokens(ct: CallTokens): string {
  const enabled = colorEnabled();
  const lines: string[] = [];

  const approxMarker = ct.approximate ? " (~approx)" : "";
  const agentStep = agentStepTag(ct.agent, ct.step);
  lines.push(`turn ${ct.seq} · ${comma(ct.totalTokens)} tokens${approxMarker}${agentStep}`);

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

/** Render the provider-usage rollup for `ctxdiff tokens`. Mirrors Python
 * `render_usage_summary`. */
export function renderUsageSummary(usage: UsageTotals, agent: string | null = null): string {
  const scope = agent !== null ? `${agent} total` : "run total";
  if (usage.callsWithUsage === 0) return `${scope} · no provider usage reported`;
  const coverage =
    `(${usage.callsWithUsage}/${usage.callsTotal} ` +
    `call${usage.callsTotal !== 1 ? "s" : ""} reported usage)`;
  const lines = [
    `${scope} · in ${comma(usage.inputTokens)} tok · ` +
      `out ${comma(usage.outputTokens)} tok ${coverage}`,
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

/** Render `ctxdiff tokens`' full output. Mirrors Python `render_run_tokens`. */
export function renderRunTokens(
  calls: CallTokens[],
  bloat: BloatReport | null,
  totalTools: number | null,
  agentSummary: string | null = null,
  usageSummary: string | null = null,
): string {
  if (calls.length === 0) return "no calls in this run";
  const sections: string[] = [];
  if (usageSummary) sections.push(usageSummary);
  if (agentSummary) sections.push(agentSummary);
  for (const c of calls) sections.push(renderCallTokens(c));
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

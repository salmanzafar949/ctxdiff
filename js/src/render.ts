/**
 * Git-style colored rendering of analyzer output. A faithful port of Python
 * `ctxdiff.cli.render` — every line, separator, glyph, padding and number
 * format matches so `ctxdiff diff|tokens|cache|runs` output is byte-identical to
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
import type { CacheReport, PrefixBreak } from "./analyze/cache.js";
import { pyRoundHalfEven } from "./analyze/pyround.js";

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

/** Integer thousands-separator formatting, matching Python's `{n:,}`. */
function comma(n: number): string {
  const neg = n < 0;
  const s = Math.abs(Math.trunc(n)).toString();
  const grouped = s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return neg ? "-" + grouped : grouped;
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
 */
function pyRepr(s: string): string {
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
 * `_snippet`. */
function snippet(text: string, limit = 70): string {
  const flat = text.split(/\s+/u).filter((s) => s.length > 0).join(" ");
  const truncated = flat.slice(0, limit) + (flat.length > limit ? "…" : "");
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

/** Group PrefixBreaks by (agent, culpritKind, culpritLabel, divergentPosition)
 * in first-seen order. Mirrors Python `_group_breaks`. */
function groupBreaks(breaks: PrefixBreak[]): PrefixBreak[][] {
  const groups = new Map<string, PrefixBreak[]>();
  const order: string[] = [];
  for (const b of breaks) {
    const key = JSON.stringify([b.agent, b.culpritKind, b.culpritLabel, b.divergentPosition]);
    let arr = groups.get(key);
    if (!arr) {
      arr = [];
      groups.set(key, arr);
      order.push(key);
    }
    arr.push(b);
  }
  return order.map((k) => groups.get(k)!);
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
    let denom = report.pairsAnalyzed;
    if (rep.agent !== null && report.pairsByAgent) {
      denom = report.pairsByAgent.get(rep.agent) ?? report.pairsAnalyzed;
    }
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

/** Render `ctxdiff runs`' listing. Mirrors Python `render_runs_list`. */
export function renderRunsList(
  rows: { filename: string; project: string; provider: string; turns: number; agents: string }[],
): string {
  if (rows.length === 0) return "no .ctrace files in the current directory";
  return rows
    .map(
      (r) =>
        `${r.filename}  project=${r.project}  provider=${r.provider}  turns=${r.turns}` +
        `  agents=${r.agents}`,
    )
    .join("\n");
}

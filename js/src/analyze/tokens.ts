/**
 * The token attributor: turns a call's blocks into "where did the budget go"
 * data, and cross-references tool_schema blocks against actual tool usage to
 * flag wasted (never-invoked) schema tokens. A faithful port of Python
 * `ctxdiff.analyze.tokens` — same label grouping, same percentage rounding
 * (round-half-to-even at 1 decimal, matching CPython's `round`), same bloat
 * heuristic and thresholds, same provider-usage reconciliation over the four
 * wire shapes. No I/O, no color.
 */
import { distinctAgents, filterCalls } from "./diff.js";
import { pyRound1 } from "./pyround.js";
import type { Call, CallBlock } from "../models.js";
import type { ReadableStore } from "../store/base.js";

// --- value types -------------------------------------------------------------

/** One label's share of a call's token budget. `pct` is rounded to 1 decimal
 * (round-half-to-even), so slices sum to ~100. Mirrors Python `LabelSlice`. */
export interface LabelSlice {
  label: string;
  tokens: number;
  blockCount: number;
  pct: number;
}

/** One call's full token attribution. Mirrors Python `CallTokens`. */
export interface CallTokens {
  seq: number;
  totalTokens: number;
  approximate: boolean;
  slices: LabelSlice[];
  providerUsage: Record<string, unknown> | null;
  reconciliationDelta: number | null;
  agent: string | null;
  step: string | null;
}

/** Cross-run schema-bloat summary. Mirrors Python `BloatReport`. */
export interface BloatReport {
  unusedTools: string[];
  unusedTokensPerCall: number;
  callsAnalyzed: number;
  pctOfAvgContext: number;
}

/** Run-level rollup of PROVIDER-REPORTED usage. Mirrors Python `UsageTotals`. */
export interface UsageTotals {
  inputTokens: number;
  outputTokens: number;
  callsWithUsage: number;
  callsTotal: number;
  byAgent: Map<string, [number, number]> | null;
}

/** A whole run's token attribution. Mirrors Python `RunTokens`. */
export interface RunTokens {
  calls: CallTokens[];
  bloat: BloatReport | null;
  usage: UsageTotals;
  byAgent: Map<string, number> | null;
}

// --- provider usage reconciliation --------------------------------------------

// Prompt-side / output-side token count keys, by provider wire shape. Order
// matters: first present (non-null) wins, since a usage dict carries one
// provider's shape at a time. Mirrors Python's key tuples exactly.
const PROMPT_TOKEN_KEYS = [
  "prompt_tokens",
  "input_tokens",
  "prompt_token_count",
  "inputTokens",
] as const;
const OUTPUT_TOKEN_KEYS = [
  "completion_tokens",
  "output_tokens",
  "candidates_token_count",
  "outputTokens",
] as const;

/** First key in `keys` present in `usage` with a non-null value, or null. A
 * key mapped to null is treated as absent. Mirrors Python `_first_present`. */
function firstPresent(
  usage: Record<string, unknown> | null,
  keys: readonly string[],
): number | null {
  if (!usage) return null;
  for (const key of keys) {
    const value = usage[key];
    if (value !== null && value !== undefined) return value as number;
  }
  return null;
}

/** Provider-reported prompt tokens minus our summed total, or null when no
 * recognizable prompt-token key is present. Mirrors Python
 * `_reconciliation_delta`. */
function reconciliationDelta(
  usage: Record<string, unknown> | null,
  totalTokens: number,
): number | null {
  const value = firstPresent(usage, PROMPT_TOKEN_KEYS);
  return value === null ? null : value - totalTokens;
}

// --- schema-name extraction ---------------------------------------------------

const NESTED_NAME_CONTAINERS = ["function", "toolSpec"] as const;
/** Sentinel for a schema whose name can't be determined. Mirrors Python
 * `UNPARSED_TOOL_NAME`. */
export const UNPARSED_TOOL_NAME = "<unparsed>";

/**
 * Defensively pull a tool's name out of a stored tool_schema block's JSON.
 * Tries, in order: a top-level "name" (Anthropic/Gemini bare shape), then
 * "name" nested under "function" (OpenAI) or "toolSpec" (Bedrock raw). Any
 * parse failure / non-object / unrecognized shape returns the sentinel rather
 * than throwing. Mirrors Python `extract_tool_name`.
 */
export function extractToolName(schemaText: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(schemaText);
  } catch {
    return UNPARSED_TOOL_NAME;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return UNPARSED_TOOL_NAME;
  }
  const obj = parsed as Record<string, unknown>;
  const name = obj["name"];
  if (typeof name === "string" && name) return name;
  for (const key of NESTED_NAME_CONTAINERS) {
    const container = obj[key];
    if (container !== null && typeof container === "object" && !Array.isArray(container)) {
      const nested = (container as Record<string, unknown>)["name"];
      if (typeof nested === "string" && nested) return nested;
    }
  }
  return UNPARSED_TOOL_NAME;
}

/** Every distinct tool name appearing in a tool_schema block anywhere in the
 * run (excluding the sentinel). Mirrors Python `registered_tool_names`. */
export function registeredToolNames(allCallsWithBlocks: CallBlock[][]): Set<string> {
  const names = new Set<string>();
  for (const callBlocks of allCallsWithBlocks) {
    for (const cb of callBlocks) {
      if (cb.label === "tool_schema") {
        const name = extractToolName(cb.block.text);
        if (name !== UNPARSED_TOOL_NAME) names.add(name);
      }
    }
  }
  return names;
}

// --- core analysis -------------------------------------------------------------

/**
 * Attribute one call's blocks to CallTokens: group `tokenCount` by `label`,
 * compute each slice's share of the total (rounded 1 decimal, round-half-to-
 * even), note whether any block used the 'estimate' method, and reconcile
 * against provider usage. Slices are sorted biggest-first. Mirrors Python
 * `analyze_call`.
 */
export function analyzeCall(call: Call, callBlocks: CallBlock[]): CallTokens {
  const labelTokens = new Map<string, number>();
  const labelCounts = new Map<string, number>();
  let approximate = false;
  for (const cb of callBlocks) {
    labelTokens.set(cb.label, (labelTokens.get(cb.label) ?? 0) + cb.block.tokenCount);
    labelCounts.set(cb.label, (labelCounts.get(cb.label) ?? 0) + 1);
    if (cb.block.tokenMethod === "estimate") approximate = true;
  }

  let totalTokens = 0;
  for (const v of labelTokens.values()) totalTokens += v;

  const slices: LabelSlice[] = [];
  for (const [label, tokens] of labelTokens) {
    slices.push({
      label,
      tokens,
      blockCount: labelCounts.get(label)!,
      pct: totalTokens ? pyRound1((tokens / totalTokens) * 100) : 0.0,
    });
  }
  // Sort by tokens descending. Insertion order (Map iteration = first-seen
  // label order) is preserved for equal token counts, matching Python's stable
  // `list.sort(key=..., reverse=True)`.
  stableSortDesc(slices, (s) => s.tokens);

  return {
    seq: call.seq,
    totalTokens,
    approximate,
    slices,
    providerUsage: call.usage,
    reconciliationDelta: reconciliationDelta(call.usage, totalTokens),
    agent: call.agent,
    step: call.step,
  };
}

/** Stable descending sort by a numeric key (Python's `sort(reverse=True)` is
 * stable; JS `Array.sort` is stable in modern V8, but for a strict descending
 * order that keeps equal-key insertion order we sort by [-key, index]). */
function stableSortDesc<T>(arr: T[], key: (t: T) => number): void {
  const indexed = arr.map((v, i) => [v, i] as [T, number]);
  indexed.sort((a, b) => key(b[0]) - key(a[0]) || a[1] - b[1]);
  for (let i = 0; i < arr.length; i++) arr[i] = indexed[i][0];
}

const UNLABELED = "(unlabeled)";

/**
 * Total block tokens per agent label across `calls`, or null when the run spans
 * fewer than 2 distinct agent labels. Null-labeled calls accumulate under
 * '(unlabeled)'. First-appearance order. Mirrors Python `_by_agent_totals`.
 */
function byAgentTotals(
  calls: Call[],
  allBlocks: CallBlock[][],
): Map<string, number> | null {
  if (distinctAgents(calls).length < 2) return null;
  const totals = new Map<string, number>();
  for (let idx = 0; idx < calls.length; idx++) {
    const key = calls[idx].agent ?? UNLABELED;
    const sum = allBlocks[idx].reduce((acc, cb) => acc + cb.block.tokenCount, 0);
    totals.set(key, (totals.get(key) ?? 0) + sum);
  }
  return totals;
}

/**
 * Roll up PROVIDER-REPORTED usage across `calls`. A call yielding neither an
 * input nor an output number is skipped from the sums but still counted in
 * `callsTotal`. Per-agent (input, output) tuples surface only when the run
 * spans ≥2 distinct agents. Mirrors Python `usage_totals`.
 */
export function usageTotals(calls: Call[]): UsageTotals {
  let inputTotal = 0;
  let outputTotal = 0;
  let withUsage = 0;
  const perIn = new Map<string, number>();
  const perOut = new Map<string, number>();
  for (const c of calls) {
    const inV = firstPresent(c.usage, PROMPT_TOKEN_KEYS);
    const outV = firstPresent(c.usage, OUTPUT_TOKEN_KEYS);
    if (inV === null && outV === null) continue;
    withUsage += 1;
    inputTotal += inV ?? 0;
    outputTotal += outV ?? 0;
    const key = c.agent ?? UNLABELED;
    perIn.set(key, (perIn.get(key) ?? 0) + (inV ?? 0));
    perOut.set(key, (perOut.get(key) ?? 0) + (outV ?? 0));
  }

  let byAgent: Map<string, [number, number]> | null = null;
  const labels = distinctAgents(calls);
  if (labels.length >= 2) {
    byAgent = new Map();
    for (const label of labels) {
      const key = label ?? UNLABELED;
      if (perIn.has(key) || perOut.has(key)) {
        byAgent.set(key, [perIn.get(key) ?? 0, perOut.get(key) ?? 0]);
      }
    }
  }

  return {
    inputTokens: inputTotal,
    outputTokens: outputTotal,
    callsWithUsage: withUsage,
    callsTotal: calls.length,
    byAgent,
  };
}

/**
 * Cross-reference every tool_schema block against tool names actually
 * referenced anywhere in the run. How (mirrors Python `detect_bloat`):
 * dedup schemas by content hash; return null if there are none; build a
 * haystack from every assistant-role block and every 'tool_output'-labeled
 * block; a schema is "used" if its name appears as a substring; the unparsed
 * sentinel is skipped (never flagged); unused-token cost is summed ONCE (same
 * schema resent each call) and expressed as a percentage of the average call's
 * total token size (round-half-to-even, 1 decimal).
 */
export function detectBloat(allCallsWithBlocks: CallBlock[][]): BloatReport | null {
  const schemasByHash = new Map<string, [string, number]>();
  for (const callBlocks of allCallsWithBlocks) {
    for (const cb of callBlocks) {
      if (cb.label === "tool_schema" && !schemasByHash.has(cb.block.contentHash)) {
        schemasByHash.set(cb.block.contentHash, [
          extractToolName(cb.block.text),
          cb.block.tokenCount,
        ]);
      }
    }
  }

  if (schemasByHash.size === 0) return null;

  const haystackParts: string[] = [];
  for (const callBlocks of allCallsWithBlocks) {
    for (const cb of callBlocks) {
      if (cb.block.role === "assistant" || cb.label === "tool_output") {
        haystackParts.push(cb.block.text);
      }
    }
  }
  const haystack = haystackParts.join("\n");

  const unusedTools: string[] = [];
  let unusedTokens = 0;
  for (const [name, tokenCount] of schemasByHash.values()) {
    if (name === UNPARSED_TOOL_NAME) continue;
    if (!haystack.includes(name)) {
      unusedTools.push(name);
      unusedTokens += tokenCount;
    }
  }

  const callTotals = allCallsWithBlocks.map((cbs) =>
    cbs.reduce((acc, cb) => acc + cb.block.tokenCount, 0),
  );
  const avgTotal = callTotals.length
    ? callTotals.reduce((a, b) => a + b, 0) / callTotals.length
    : 0;
  const pct = avgTotal ? pyRound1((unusedTokens / avgTotal) * 100) : 0.0;

  return {
    unusedTools,
    unusedTokensPerCall: unusedTokens,
    callsAnalyzed: allCallsWithBlocks.length,
    pctOfAvgContext: pct,
  };
}

/**
 * Load the run's calls (filtered to `agent` when given), attribute each call's
 * tokens, run bloat detection once across the analyzed calls, and compute the
 * per-agent token breakdown. Mirrors Python `analyze_run`.
 */
export function analyzeRun(ct: ReadableStore, agent: string | null = null): RunTokens {
  const calls = filterCalls(ct.getCalls(), agent);
  const allCallsWithBlocks = calls.map((c) => ct.getCallBlocks(c.id));
  const callTokens = calls.map((call, i) => analyzeCall(call, allCallsWithBlocks[i]));
  const bloat = detectBloat(allCallsWithBlocks);
  const byAgent = byAgentTotals(calls, allCallsWithBlocks);
  const usage = usageTotals(calls);
  return { calls: callTokens, bloat, usage, byAgent };
}

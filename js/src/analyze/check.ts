/**
 * The CI assertion layer (`ctxdiff check`): a THIN threshold layer over the
 * analyzers that already exist, and deliberately nothing more. A faithful port
 * of Python `ctxdiff.analyze.check` — same assertions, same fixed report order,
 * same per-agent growth pairing, same wording down to the separators, so the
 * two CLIs' check output is byte-identical.
 *
 * Why it owns no analysis of its own. `check` exists so a context budget can be
 * asserted unattended on every pull request — which only works if a green
 * `check` and a hand-read `ctxdiff tokens`/`ctxdiff cache` can never tell two
 * different stories. So every number compared against a threshold is one the
 * other commands already print: turn totals and schema bloat come from
 * `analyzeRun`, prefix breaks (including the digest-based attribution for
 * same-slot image swaps) come from `analyzeCache`, evicted tagged blocks come
 * from `analyzeEvictions` (the differ's own `evicted` classification, scoped per
 * agent, narrowed to `labelSource === "tagged"`), and the "N of M registered
 * tools" denominator comes from `registeredToolNames` — the very call `ctxdiff
 * tokens` makes for its own bloat line. What is new here is only: comparison
 * against a user-supplied limit, the selection of which turn/agent/block to
 * blame, and the wording of the violation.
 *
 * Which number is "the turn's context". `--max-context` is compared against the
 * same total `ctxdiff tokens` prints as `turn N · X tokens` — the sum of the
 * call's stored block tokens — NOT the provider's reported prompt count. It is
 * present for every call (provider usage is optional, and a threshold that
 * silently skips unreported turns is a CI check that passes by not looking) and
 * it is the number a reader will compare the failure against. Turns whose total
 * mixes in estimated blocks are marked `(~approx)` — on the PASS summaries as
 * well as the violation lines, since a high-water mark quoted without that
 * marker reads as a measurement, which is the same lie one row further up.
 *
 * A total that is a FLOOR is not a total. Some blocks cost the provider real
 * tokens ctxdiff cannot know: an image given as a remote URL or a `file_id`, or
 * in a format the sniffer does not recognize. Those are stored as zero tokens
 * under the 'estimate' method (a fabricated guess would be indistinguishable
 * from a measurement in every view), which makes the call's total a lower
 * bound. Comparing a lower bound against a budget has exactly one possible
 * wrong answer — a PASS — so such a turn is reported as UNMEASURED and fails
 * the assertion instead of being compared.
 */
import {
  analyzeCache,
  groupBreaks,
  pairsDenominator,
  type CacheReport,
} from "./cache.js";
import { filterCalls } from "./diff.js";
import { analyzeEvictions, type EvictionReport } from "./evictions.js";
import { pyComma, pyRound1 } from "./pyround.js";
import {
  analyzeRun,
  registeredToolNames,
  type CallTokens,
  type RunTokens,
} from "./tokens.js";
import type { ReadableStore } from "../store/base.js";

// --- the assertion surface ------------------------------------------------------

// The assertion names, in the ONE order they are always reported in — so a
// check's output is a function of which assertions were requested, never of the
// order the flags happened to be typed in.
export const MAX_CONTEXT = "max-context";
export const MAX_CONTEXT_PCT = "max-context-pct";
export const REQUIRE_STABLE_PREFIX = "require-stable-prefix";
export const NO_DEAD_SCHEMAS = "no-dead-schemas";
export const NO_TAGGED_EVICTION = "no-tagged-eviction";
export const MAX_GROWTH = "max-growth";
export const MAX_GROWTH_PCT = "max-growth-pct";

export const ASSERTION_ORDER = [
  MAX_CONTEXT,
  MAX_CONTEXT_PCT,
  REQUIRE_STABLE_PREFIX,
  NO_DEAD_SCHEMAS,
  NO_TAGGED_EVICTION,
  MAX_GROWTH,
  MAX_GROWTH_PCT,
];

/** Width the renderer left-justifies an assertion name to: the longest name in
 * `ASSERTION_ORDER`, so every summary starts in the same column. */
export const NAME_WIDTH = Math.max(...ASSERTION_ORDER.map((n) => n.length));

/**
 * Everything the user asked to be asserted. Every field is opt-in: `null` (or
 * false) means "this assertion was not requested" and it is left out of the
 * report entirely rather than reported as a vacuous pass — a check must never
 * imply it verified something nobody asked for.
 *
 * `contextWindow` is not itself an assertion: it is the denominator
 * `maxContextPct` is a percentage OF. ctxdiff ships no model→window table by
 * design (they change per model and per provider, and a stale table would
 * silently move everyone's CI threshold), so the window is supplied by the user
 * or the percentage assertion is simply not available. The CLI fills it through
 * `resolveContextWindow` — the same flag-then-`CTXDIFF_CONTEXT_WINDOW` path
 * `ctxdiff tokens` and the dashboard use, so a gate and the report a human reads
 * beside it can never be scored against two different windows. Mirrors Python
 * `Thresholds`.
 */
export interface Thresholds {
  maxContext: number | null;
  contextWindow: number | null;
  maxContextPct: number | null;
  requireStablePrefix: boolean;
  noDeadSchemas: boolean;
  noTaggedEviction: boolean;
  maxGrowth: number | null;
  maxGrowthPct: number | null;
}

/** Whether at least one assertion was requested. A `check` with none is a usage
 * error, not a pass: exiting 0 for "you asked me to verify nothing" is exactly
 * the green tick that makes a CI gate worthless. Mirrors Python
 * `Thresholds.any_requested`. */
export function anyRequested(t: Thresholds): boolean {
  return (
    t.maxContext !== null ||
    t.maxContextPct !== null ||
    t.requireStablePrefix ||
    t.noDeadSchemas ||
    t.noTaggedEviction ||
    t.maxGrowth !== null ||
    t.maxGrowthPct !== null
  );
}

/** Whether any requested assertion needs `analyzeRun` (per-turn totals or
 * schema bloat) — so a prefix-only check never pays for the token attribution
 * pass it would not read. Mirrors Python `Thresholds.needs_token_analysis`. */
function needsTokenAnalysis(t: Thresholds): boolean {
  return (
    t.maxContext !== null ||
    t.maxContextPct !== null ||
    t.noDeadSchemas ||
    t.maxGrowth !== null ||
    t.maxGrowthPct !== null
  );
}

/** One assertion's verdict. `summary` always states the ACTUAL value next to
 * the threshold — on a pass as well as a failure — because "PASS" alone tells a
 * reader nothing about how close the run came to the limit, which is the number
 * worth watching in a CI log over time. `details` is one line per offending
 * turn/agent/block, populated only on a failure. Mirrors Python
 * `AssertionResult`. */
export interface AssertionResult {
  name: string;
  passed: boolean;
  summary: string;
  details: string[];
}

/** Every requested assertion's verdict, in `ASSERTION_ORDER`, plus the scope
 * that was checked. `turnsAnalyzed` is 0 only when the session (or the
 * `--agent` slice of it) holds no calls at all — which the CLI reports as a
 * failure rather than a vacuous pass. Mirrors Python `CheckReport`. */
export interface CheckReport {
  assertions: AssertionResult[];
  turnsAnalyzed: number;
  agent: string | null;
}

/** The failing assertions, in report order. */
export function failedAssertions(report: CheckReport): AssertionResult[] {
  return report.assertions.filter((a) => !a.passed);
}

/** True iff every requested assertion passed. */
export function checkPassed(report: CheckReport): boolean {
  return failedAssertions(report).length === 0;
}

// --- shared formatting bits -----------------------------------------------------

/** The ` [agent:NAME]` marker appended to a turn reference, or "" for an
 * unlabeled/single-agent run. Same spelling `ctxdiff cache` uses for its
 * per-agent break warnings. Mirrors Python `_agent_chip`. */
function agentChip(agent: string | null): string {
  return agent !== null ? ` [agent:${agent}]` : "";
}

/** `n` and `word`, pluralized by appending 's' when n !== 1 — so the summary
 * lines read as English rather than as `1 turns`. Mirrors Python `_plural`. */
function plural(n: number, word: string): string {
  return n === 1 ? `${n} ${word}` : `${n} ${word}s`;
}

/** A percentage rendered to one decimal. Values are rounded to 1 decimal BEFORE
 * formatting (see `pyRound1`) so this `toFixed(1)` and Python's `:.1f` cannot
 * disagree on a boundary case. Mirrors Python `_pct`. */
function pct(value: number): string {
  return value.toFixed(1);
}

/** ` (~approx)` when this turn's total includes any estimated block, else "".
 * The same honesty rule `ctxdiff tokens` follows: a threshold verdict computed
 * from a partly-estimated total must say so. Applied to every number this
 * module quotes — the PASS summaries' peaks as much as the FAIL details, since
 * an unmarked `peak 8 tok` is the same false precision as an unmarked overage.
 * Mirrors Python `_approx_marker`. */
function approxMarker(call: CallTokens): string {
  return call.approximate ? " (~approx)" : "";
}

/** ` (~approx)` for a GROWTH figure: a difference is only as exact as the less
 * exact of the two totals it is taken between, so one estimated block on either
 * side marks the delta. Mirrors Python `_pair_approx_marker`. */
function pairApproxMarker(prev: CallTokens, cur: CallTokens): string {
  return prev.approximate || cur.approximate ? " (~approx)" : "";
}

/** Collapse every whitespace run (newlines included) to a single space.
 *
 * Applied to the one piece of a violation line that is content rather than
 * ctxdiff's own words — a tool schema's `name`, straight out of a captured
 * payload. The report is pasted into markdown (the GitHub Action fences it into
 * the job summary), where a name containing a newline and a run of backticks
 * would close the fence and let captured text render as markup. Mirrors Python
 * `_flatten`. */
function flatten(text: string): string {
  return text.split(/\s+/u).filter((s) => s.length > 0).join(" ");
}

/** A growth figure with an explicit sign: `+1,200` / `-40` / `+0`. The
 * over-limit lines are always positive, but an UNMEASURED pair is reported
 * whatever its delta, and `+-40` is not a number anyone should have to read.
 * Mirrors Python `_signed`. */
function signed(value: number): string {
  return value >= 0 ? `+${pyComma(value)}` : pyComma(value);
}

// --- unmeasured turns -----------------------------------------------------------

/** The turns whose total is a FLOOR: they hold at least one block whose token
 * cost could not be determined. Mirrors Python `_unmeasured`. */
function unmeasuredCalls(calls: CallTokens[]): CallTokens[] {
  return calls.filter((c) => c.unmeasuredBlocks > 0);
}

/** The clause explaining why a turn (or a growth pair) cannot be certified: how
 * many blocks were never priced, and what that does to the number quoted beside
 * it. Mirrors Python `_unmeasured_note`. */
function unmeasuredNote(...calls: CallTokens[]): string {
  const n = calls.reduce((sum, c) => sum + c.unmeasuredBlocks, 0);
  return `${plural(n, "block")} of unknown token cost — a floor, not a measurement`;
}

/** The FAIL summary for a threshold assertion: the counts that are non-zero,
 * then the run's high-water mark — joined from the parts that are present so a
 * run that is only over budget, only unmeasured, or both reads correctly.
 * Mirrors Python `_verdict_summary`. */
function verdictSummary(over: number, unmeasured: number, tail: string): string {
  const parts: string[] = [];
  if (over) parts.push(`${plural(over, "turn")} over limit`);
  if (unmeasured) parts.push(`${plural(unmeasured, "turn")} unmeasured`);
  parts.push(tail);
  return parts.join(" · ");
}

/**
 * The summary for an assertion that found no consecutive same-agent pair to
 * work on — which is TWO different facts, and reporting the wrong one is how a
 * silently no-op assertion goes unnoticed:
 *
 * - genuinely fewer than 2 turns: there is nothing a pair could be made of;
 * - two or more turns but no two of them belong to the same agent (a four-turn
 *   run with four per-agent labels). Pairing is per-agent by design — a hand-off
 *   is not growth and not a cache break — so every pair is cross-agent and none
 *   is analyzable. Saying "fewer than 2 turns" over four turns reads as a typo
 *   and hides the fact that the assertion measured nothing at all.
 *
 * (`turns >= 2` with no pair implies every agent has exactly one turn, since any
 * agent with two would supply one — hence the parenthetical.) Mirrors Python
 * `_no_pairs_summary`.
 */
function noPairsSummary(turns: number, agents: number, what: string): string {
  if (turns < 2) return `fewer than 2 turns — no ${what}`;
  return `no consecutive same-agent pairs (${plural(agents, "agent")}, 1 turn each) — no ${what}`;
}

/** How many distinct agent labels the analyzed turns carry (null — an unlabeled
 * run — counts as one). Mirrors Python `_agents_in`. */
function agentsIn(calls: CallTokens[]): number {
  return new Set(calls.map((c) => c.agent)).size;
}

// --- max-context / max-context-pct ----------------------------------------------

/** The turn with the largest total. Ties go to the EARLIEST turn (calls arrive
 * in seq order and the comparison is strictly greater-than), so the reported
 * peak is stable rather than dependent on iteration luck. Mirrors Python
 * `_peak`. */
function peakCall(calls: CallTokens[]): CallTokens {
  let peak = calls[0];
  for (const c of calls.slice(1)) if (c.totalTokens > peak.totalTokens) peak = c;
  return peak;
}

/**
 * Fail when any turn's context total exceeds `limit` tokens — or when a turn's
 * total cannot be compared to the limit at all. The peak turn is named on a pass
 * too, so the CI log carries the run's actual high-water mark and not just a
 * tick.
 *
 * Two ways to fail, and the second one is why this check can be trusted: a turn
 * OVER the limit, and a turn whose total is a FLOOR (it holds a block of unknown
 * cost). The latter is reported as unmeasured rather than compared, because
 * comparing a lower bound to a budget has exactly one possible wrong answer — a
 * PASS — and it is silent. A turn that is BOTH over the limit and unmeasured is
 * reported once, as over the limit: the overage is already proved, and the floor
 * cannot un-prove it. Mirrors Python `_check_max_context`.
 */
function checkMaxContext(calls: CallTokens[], limit: number): AssertionResult {
  const over = calls.filter((c) => c.totalTokens > limit);
  const unmeasured = unmeasuredCalls(calls).filter((c) => !over.includes(c));
  const peak = peakCall(calls);
  const peakText =
    `peak ${pyComma(peak.totalTokens)} tok${approxMarker(peak)} at turn ${peak.seq} · ` +
    `limit ${pyComma(limit)}`;
  if (!over.length && !unmeasured.length) {
    return { name: MAX_CONTEXT, passed: true, summary: peakText, details: [] };
  }
  const details = over.map(
    (c) =>
      `turn ${c.seq}${agentChip(c.agent)} · ${pyComma(c.totalTokens)} tok` +
      `${approxMarker(c)} · ${pyComma(c.totalTokens - limit)} over limit`,
  );
  details.push(
    ...unmeasured.map(
      (c) =>
        `turn ${c.seq}${agentChip(c.agent)} · ${pyComma(c.totalTokens)} tok` +
        `${approxMarker(c)} · ${unmeasuredNote(c)} · limit ${pyComma(limit)}`,
    ),
  );
  return {
    name: MAX_CONTEXT,
    passed: false,
    summary: verdictSummary(over.length, unmeasured.length, peakText),
    details,
  };
}

/**
 * Fail when any turn's context total exceeds `limitPct` percent of the
 * user-supplied `window`. The budget is stated in tokens next to the percentage
 * on every line, so a verdict is never something the reader has to recompute.
 *
 * Comparison is against the exact token budget (`window * pct / 100`), while
 * the displayed percentages are rounded to one decimal — the same
 * compare-exact/display-rounded split every percentage in ctxdiff uses.
 *
 * Unmeasured turns fail here for the same reason they fail `--max-context`: a
 * percentage computed from a floor is a floor, and a floor under a budget proves
 * nothing. Mirrors Python `_check_max_context_pct`.
 */
function checkMaxContextPct(
  calls: CallTokens[],
  window: number,
  limitPct: number,
): AssertionResult {
  const budget = (window * limitPct) / 100;
  const budgetTokens = Math.trunc(budget); // the largest whole token that fits
  const over = calls.filter((c) => c.totalTokens > budget);
  const unmeasured = unmeasuredCalls(calls).filter((c) => !over.includes(c));
  const peak = peakCall(calls);
  const peakPct = pct(pyRound1((peak.totalTokens / window) * 100));
  const limitText = `limit ${pct(limitPct)}% (${pyComma(budgetTokens)} tok)`;
  const peakText =
    `peak ${peakPct}%${approxMarker(peak)} of ${pyComma(window)} tok window at ` +
    `turn ${peak.seq} · ${limitText}`;
  if (!over.length && !unmeasured.length) {
    return { name: MAX_CONTEXT_PCT, passed: true, summary: peakText, details: [] };
  }
  const details = over.map(
    (c) =>
      `turn ${c.seq}${agentChip(c.agent)} · ${pyComma(c.totalTokens)} tok` +
      `${approxMarker(c)} · ${pct(pyRound1((c.totalTokens / window) * 100))}% ` +
      `of ${pyComma(window)} tok window · ${limitText}`,
  );
  details.push(
    ...unmeasured.map(
      (c) =>
        `turn ${c.seq}${agentChip(c.agent)} · ${pyComma(c.totalTokens)} tok` +
        `${approxMarker(c)} · ${pct(pyRound1((c.totalTokens / window) * 100))}% ` +
        `of ${pyComma(window)} tok window · ${unmeasuredNote(c)} · ${limitText}`,
    ),
  );
  return {
    name: MAX_CONTEXT_PCT,
    passed: false,
    summary: verdictSummary(over.length, unmeasured.length, peakText),
    details,
  };
}

// --- require-stable-prefix -------------------------------------------------------

/**
 * Fail when the prompt-cache prefix breaks anywhere in the run.
 *
 * Every word of the explanation is `analyzeCache`'s own: the breaks, their
 * attribution (including the sha-digest wording a same-slot image swap gets,
 * where a character offset would explain nothing) and the re-billed total all
 * come straight from the CacheReport. Breaks are collapsed by culprit with the
 * SAME `groupBreaks` the `cache` command renders through, so a timestamp that
 * breaks the prefix on all twelve turns is one line here and one line there.
 *
 * `turns`/`agents` describe the analyzed slice and exist only so the no-pairs
 * case can say WHICH no-pairs case it is (see `noPairsSummary`) — the
 * CacheReport carries the pair count but not the reason it is zero. Mirrors
 * Python `_check_stable_prefix`.
 */
function checkStablePrefix(
  report: CacheReport,
  turns: number,
  agents: number,
): AssertionResult {
  const pairs = report.pairsAnalyzed;
  if (pairs === 0) {
    return {
      name: REQUIRE_STABLE_PREFIX,
      passed: true,
      summary: noPairsSummary(turns, agents, "pairs to check"),
      details: [],
    };
  }
  if (!report.breaks.length) {
    return {
      name: REQUIRE_STABLE_PREFIX,
      passed: true,
      summary:
        `prefix stable across all ${plural(pairs, "turn pair")} · ` +
        `min stable prefix ${pyComma(report.stablePrefixTokensMin)} tok`,
      details: [],
    };
  }
  const details = groupBreaks(report.breaks).map((group) => {
    const rep = group[0];
    const denom = pairsDenominator(report, rep);
    return (
      `turn ${rep.seqPrev} → turn ${rep.seq}${agentChip(rep.agent)} ` +
      `[${rep.culpritLabel}·${rep.culpritKind}] breaks ${group.length}/${denom} ` +
      `pairs — ${rep.detail}`
    );
  });
  return {
    name: REQUIRE_STABLE_PREFIX,
    passed: false,
    summary:
      `${plural(report.breaks.length, "break")} across ` +
      `${plural(pairs, "turn pair")} · ` +
      `${pyComma(report.rebilledTokensTotal)} tok re-billed`,
    details,
  };
}

// --- no-dead-schemas --------------------------------------------------------------

/**
 * Fail when a tool schema is registered but never invoked anywhere in the run —
 * the existing bloat detector, with a threshold of zero.
 *
 * A run that registers NO tool schemas at all passes and says so: there is
 * nothing to be dead, and reporting `0 of 0` would read like a measurement that
 * was taken when none was.
 *
 * The tool NAME is the one fragment of a violation line that comes from a
 * captured payload rather than from ctxdiff, so it goes through `flatten`: a
 * schema named across two lines would otherwise break the line structure of the
 * report and, in the GitHub Action's fenced job summary, the fence. Mirrors
 * Python `_check_no_dead_schemas`.
 */
function checkNoDeadSchemas(runTokens: RunTokens, totalTools: number): AssertionResult {
  const bloat = runTokens.bloat;
  if (bloat === null) {
    return {
      name: NO_DEAD_SCHEMAS,
      passed: true,
      summary: "no tool schemas registered",
      details: [],
    };
  }
  if (!bloat.unusedTools.length) {
    return {
      name: NO_DEAD_SCHEMAS,
      passed: true,
      summary: `all ${plural(totalTools, "registered tool schema")} invoked`,
      details: [],
    };
  }
  return {
    name: NO_DEAD_SCHEMAS,
    passed: false,
    summary:
      `${bloat.unusedTools.length} of ${totalTools} registered tools never ` +
      `used · ${pyComma(bloat.unusedTokensPerCall)} tok/call ` +
      `(${pct(bloat.pctOfAvgContext)}% of avg context)`,
    details: bloat.unusedTools.map(
      (name) => `tool schema '${flatten(name)}' registered but never invoked`,
    ),
  };
}

// --- no-tagged-eviction ------------------------------------------------------------

/**
 * Fail when a block the developer TAGGED entered the context and later left it
 * for good — the existing eviction detector, with a threshold of zero. Mirrors
 * Python `_check_no_tagged_eviction`.
 *
 * Every word of every line comes from `analyzeEvictions`: the tag, the turn the
 * block entered, the turn it disappeared, the per-agent scoping and the "it
 * never came back" rule are all decided there, so a green `check` and a
 * hand-read `ctxdiff tokens` can never tell two different stories about the same
 * trace. What is new here is only the comparison against zero.
 *
 * Three states are reported rather than two, because "PASS" has three different
 * meanings and only one of them is reassuring: nothing was TAGGED at all (the
 * assertion is structurally vacuous, and saying so is what tells a reader to go
 * add a `tracer.tag()`); nothing could be PAIRED (a single turn, or a fan-out
 * where each agent has one turn); or tagged blocks existed, pairs existed, and
 * none of them was evicted.
 */
function checkNoTaggedEviction(
  report: EvictionReport,
  turns: number,
  agents: number,
): AssertionResult {
  if (report.taggedBlocks === 0) {
    return {
      name: NO_TAGGED_EVICTION,
      passed: true,
      summary:
        "no tagged blocks in this run — nothing to lose " +
        "(tag load-bearing content with tracer.tag)",
      details: [],
    };
  }
  if (report.pairsAnalyzed === 0) {
    return {
      name: NO_TAGGED_EVICTION,
      passed: true,
      summary: noPairsSummary(turns, agents, "pairs to check"),
      details: [],
    };
  }
  const taggedText =
    `all ${plural(report.taggedBlocks, "tagged block")} survived ` +
    `${plural(report.pairsAnalyzed, "turn pair")}`;
  if (!report.evictions.length) {
    return { name: NO_TAGGED_EVICTION, passed: true, summary: taggedText, details: [] };
  }
  // The TAG is the one fragment of these lines the developer authored rather
  // than ctxdiff, so it goes through `flatten` for the same reason a tool
  // schema's name does: one violation, one line, even inside the Action's fenced
  // job summary.
  const details = report.evictions.map(
    (e) =>
      `the block you tagged '${flatten(e.label)}' at turn ${e.taggedSeq}` +
      `${agentChip(e.agent)} was evicted at turn ${e.evictedSeq} · ` +
      `${pyComma(e.tokens)} tok`,
  );
  // One event per tagged block per agent (`analyzeEvictions` dedupes by content
  // hash within each group), so this numerator and `taggedBlocks` count the same
  // kind of thing and the sentence cannot contradict itself.
  return {
    name: NO_TAGGED_EVICTION,
    passed: false,
    summary:
      `${plural(report.evictions.length, "tagged block")} evicted of ` +
      `${report.taggedBlocks} across ${plural(report.pairsAnalyzed, "turn pair")}`,
    details,
  };
}

// --- max-growth / max-growth-pct ---------------------------------------------------

/**
 * Consecutive (previous, current) turn pairs, paired WITHIN each agent.
 *
 * The grouping rule is `analyzeCache`'s, for the same reason: on a multi-agent
 * timeline two adjacent calls can belong to different agents, and the "growth"
 * between them is not growth at all — it is a hand-off between two independent
 * contexts, and flagging it would make `--max-growth` unusable on exactly the
 * runs it matters most for. Agents are visited in first-appearance order and
 * each agent's calls stay in seq order. Mirrors Python `_growth_pairs`.
 */
function growthPairs(calls: CallTokens[]): [CallTokens, CallTokens][] {
  const order: (string | null)[] = [];
  const byAgent = new Map<string | null, CallTokens[]>();
  for (const c of calls) {
    if (!byAgent.has(c.agent)) {
      order.push(c.agent);
      byAgent.set(c.agent, []);
    }
    byAgent.get(c.agent)!.push(c);
  }
  const pairs: [CallTokens, CallTokens][] = [];
  for (const label of order) {
    const group = byAgent.get(label)!;
    for (let i = 1; i < group.length; i++) pairs.push([group[i - 1], group[i]]);
  }
  return pairs;
}

/** Fail when the context grows by more than `limit` tokens between two
 * consecutive turns of the same agent. A shrinking context is never a
 * violation, so the peak reported on a pass may legitimately be negative — that
 * is the run's largest single-turn growth, honestly stated.
 *
 * A pair with an UNMEASURED turn on either side fails without being compared: a
 * difference between two totals is only knowable when both are, and the error
 * runs in both directions (an unmeasured EARLIER turn overstates the growth, an
 * unmeasured LATER one understates it), so neither a pass nor a numeric
 * violation would be defensible. Mirrors Python `_check_max_growth`. */
function checkMaxGrowth(calls: CallTokens[], limit: number): AssertionResult {
  const pairs = growthPairs(calls);
  if (!pairs.length) {
    return {
      name: MAX_GROWTH,
      passed: true,
      summary: noPairsSummary(calls.length, agentsIn(calls), "growth to measure"),
      details: [],
    };
  }
  const growths: [CallTokens, CallTokens, number][] = pairs.map(([prev, cur]) => [
    prev,
    cur,
    cur.totalTokens - prev.totalTokens,
  ]);
  // Python's `max(..., key=...)` keeps the FIRST maximum; a strict `>` here
  // does the same, so both SDKs name the same peak pair on a tie.
  let peak = growths[0];
  for (const g of growths.slice(1)) if (g[2] > peak[2]) peak = g;
  const peakText =
    `peak growth ${pyComma(peak[2])} tok${pairApproxMarker(peak[0], peak[1])} at ` +
    `turn ${peak[1].seq} · limit ${pyComma(limit)}`;
  const over = growths.filter((g) => g[2] > limit);
  const unmeasured = growths.filter(
    (g) => (g[0].unmeasuredBlocks > 0 || g[1].unmeasuredBlocks > 0) && !over.includes(g),
  );
  if (!over.length && !unmeasured.length) {
    return { name: MAX_GROWTH, passed: true, summary: peakText, details: [] };
  }
  const details = over.map(
    ([prev, cur, growth]) =>
      `turn ${prev.seq} → turn ${cur.seq}${agentChip(cur.agent)} · ` +
      `+${pyComma(growth)} tok${pairApproxMarker(prev, cur)} ` +
      `(${pyComma(prev.totalTokens)} → ${pyComma(cur.totalTokens)}) · limit ${pyComma(limit)}`,
  );
  details.push(
    ...unmeasured.map(
      ([prev, cur, growth]) =>
        `turn ${prev.seq} → turn ${cur.seq}${agentChip(cur.agent)} · ` +
        `${signed(growth)} tok${pairApproxMarker(prev, cur)} ` +
        `(${pyComma(prev.totalTokens)} → ${pyComma(cur.totalTokens)}) · ` +
        `${unmeasuredNote(prev, cur)} · limit ${pyComma(limit)}`,
    ),
  );
  return {
    name: MAX_GROWTH,
    passed: false,
    summary: verdictSummary(over.length, unmeasured.length, peakText),
    details,
  };
}

/**
 * Fail when the context grows by more than `limitPct` percent between two
 * consecutive turns of the same agent.
 *
 * A pair whose EARLIER turn totalled zero tokens is skipped rather than treated
 * as infinite growth: "grew by ∞%" is not a fact anyone can act on, and a
 * zero-token turn is a degenerate capture, not a budget regression. When every
 * pair is skipped that way the assertion says SO — naming the skip and its
 * reason — rather than borrowing the "fewer than 2 turns" wording, which over a
 * run that plainly has more than two turns reads as a bug in the reader rather
 * than a measurement that never happened.
 *
 * Unmeasured pairs are collected from ALL pairs, before the zero-token filter: a
 * floor of zero is exactly the shape an unmeasured turn takes (one image, no
 * text), and dropping it here is how it would slip through both branches.
 * Mirrors Python `_check_max_growth_pct`.
 */
function checkMaxGrowthPct(calls: CallTokens[], limitPct: number): AssertionResult {
  const allPairs = growthPairs(calls);
  if (!allPairs.length) {
    return {
      name: MAX_GROWTH_PCT,
      passed: true,
      summary: noPairsSummary(calls.length, agentsIn(calls), "growth to measure"),
      details: [],
    };
  }

  // The percentage is only defined where the denominator is: pairs whose
  // earlier turn totalled zero are dropped here and accounted for in the
  // summary below.
  const pairs = allPairs.filter(([prev]) => prev.totalTokens > 0);
  const growths: [CallTokens, CallTokens, number][] = pairs.map(([prev, cur]) => [
    prev,
    cur,
    pyRound1(((cur.totalTokens - prev.totalTokens) / prev.totalTokens) * 100),
  ]);
  const over = growths.filter((g) => g[2] > limitPct);
  const overPairs = over.map(([prev, cur]) => [prev, cur] as [CallTokens, CallTokens]);
  const unmeasuredPairs = allPairs.filter(
    ([prev, cur]) =>
      (prev.unmeasuredBlocks > 0 || cur.unmeasuredBlocks > 0) &&
      !overPairs.some(([p, c]) => p === prev && c === cur),
  );

  let tail: string;
  if (growths.length) {
    let peak = growths[0];
    for (const g of growths.slice(1)) if (g[2] > peak[2]) peak = g;
    tail =
      `peak growth ${pct(peak[2])}%${pairApproxMarker(peak[0], peak[1])} at ` +
      `turn ${peak[1].seq} · limit ${pct(limitPct)}%`;
  } else {
    tail = `all ${plural(allPairs.length, "pair")} skipped — the earlier turn had 0 tokens`;
  }

  if (!over.length && !unmeasuredPairs.length) {
    return { name: MAX_GROWTH_PCT, passed: true, summary: tail, details: [] };
  }
  const details = over.map(
    ([prev, cur, growth]) =>
      `turn ${prev.seq} → turn ${cur.seq}${agentChip(cur.agent)} · ` +
      `+${pct(growth)}%${pairApproxMarker(prev, cur)} ` +
      `(${pyComma(prev.totalTokens)} → ${pyComma(cur.totalTokens)} tok) · ` +
      `limit ${pct(limitPct)}%`,
  );
  // An unmeasured pair is reported WITHOUT a percentage: the number would be
  // derived from a floor, and quoting it beside a limit is the confusion this
  // whole branch exists to avoid.
  details.push(
    ...unmeasuredPairs.map(
      ([prev, cur]) =>
        `turn ${prev.seq} → turn ${cur.seq}${agentChip(cur.agent)} · ` +
        `${pyComma(prev.totalTokens)} → ${pyComma(cur.totalTokens)} tok` +
        `${pairApproxMarker(prev, cur)} · ${unmeasuredNote(prev, cur)} · ` +
        `limit ${pct(limitPct)}%`,
    ),
  );
  return {
    name: MAX_GROWTH_PCT,
    passed: false,
    summary: verdictSummary(over.length, unmeasuredPairs.length, tail),
    details,
  };
}

// --- the entry point ----------------------------------------------------------------

/**
 * Run every REQUESTED assertion over one session (optionally scoped to one
 * agent) and return their verdicts in `ASSERTION_ORDER`.
 *
 * How: the two underlying analyzers are run at most once each and only when
 * something asks for them — `analyzeRun` for the per-turn totals and the bloat
 * report, `analyzeCache` for the prefix breaks — then each requested assertion
 * is a pure comparison over that output. An empty session produces an empty
 * assertion list and `turnsAnalyzed === 0`; the caller turns that into a failure
 * rather than a pass, because a check that looked at nothing has proved nothing.
 * Mirrors Python `analyze_check`.
 */
export function analyzeCheck(
  ct: ReadableStore,
  thresholds: Thresholds,
  agent: string | null = null,
): CheckReport {
  const calls = filterCalls(ct.getCalls(), agent);
  if (!calls.length) return { assertions: [], turnsAnalyzed: 0, agent };

  const runTokens = needsTokenAnalysis(thresholds) ? analyzeRun(ct, agent) : null;
  const cacheReport = thresholds.requireStablePrefix ? analyzeCache(ct, agent) : null;
  const evictionReport = thresholds.noTaggedEviction ? analyzeEvictions(ct, agent) : null;

  const results: AssertionResult[] = [];
  if (runTokens !== null && thresholds.maxContext !== null) {
    results.push(checkMaxContext(runTokens.calls, thresholds.maxContext));
  }
  if (
    runTokens !== null &&
    thresholds.maxContextPct !== null &&
    thresholds.contextWindow !== null
  ) {
    results.push(
      checkMaxContextPct(runTokens.calls, thresholds.contextWindow, thresholds.maxContextPct),
    );
  }
  if (cacheReport !== null) {
    // The turn and agent counts come from the raw calls rather than from the
    // CacheReport: a report with zero pairs cannot say WHY it has none, and
    // "fewer than 2 turns" printed over a four-turn run is how a silently no-op
    // assertion stays unnoticed.
    results.push(
      checkStablePrefix(cacheReport, calls.length, new Set(calls.map((c) => c.agent)).size),
    );
  }
  if (runTokens !== null && thresholds.noDeadSchemas) {
    // The "M" denominator is derived exactly as `ctxdiff tokens` derives it —
    // `registeredToolNames` over EVERY call in the session, not just the
    // --agent slice — so `check` and `tokens` can never print a different
    // "N of M" for the same trace.
    const allBlocks = ct.getCalls().map((c) => ct.getCallBlocks(c.id));
    results.push(checkNoDeadSchemas(runTokens, registeredToolNames(allBlocks).size));
  }
  if (evictionReport !== null) {
    // Same turn/agent counts the prefix assertion gets, and for the same reason:
    // an EvictionReport with zero pairs cannot say WHY it has none.
    results.push(
      checkNoTaggedEviction(
        evictionReport,
        calls.length,
        new Set(calls.map((c) => c.agent)).size,
      ),
    );
  }
  if (runTokens !== null && thresholds.maxGrowth !== null) {
    results.push(checkMaxGrowth(runTokens.calls, thresholds.maxGrowth));
  }
  if (runTokens !== null && thresholds.maxGrowthPct !== null) {
    results.push(checkMaxGrowthPct(runTokens.calls, thresholds.maxGrowthPct));
  }

  return { assertions: results, turnsAnalyzed: calls.length, agent };
}

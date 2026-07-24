/**
 * The prompt-cache alignment profiler: finds where a run's provider-side cache
 * prefix breaks between consecutive calls, attributes the break to a specific
 * block, and quantifies the wasted re-billing. A faithful port of Python
 * `ctxdiff.analyze.cache` — same strictly-POSITIONAL prefix walk (distinct from
 * the differ's LCS alignment), same agent-aware grouping (cross-agent hand-offs
 * are never counted as breaks), same culprit classification, same price-free
 * waste note and fix hint. No I/O, no color.
 */
import { diffCalls, distinctAgents, filterCalls, type TurnDiff, type InlineSegment } from "./diff.js";
import type { Call, CallBlock } from "../models.js";
import type { ReadableStore } from "../store/base.js";

// --- value types -------------------------------------------------------------

/** One consecutive-call pair whose cache prefix diverges. Mirrors Python
 * `PrefixBreak`. `culpritKind` is 'modified' | 'added' | 'evicted' |
 * 'reordered' | 'changed'. */
export interface PrefixBreak {
  seqPrev: number;
  seq: number;
  stableBlocks: number;
  stableTokens: number;
  divergentPosition: number;
  culpritKind: string;
  culpritLabel: string;
  culpritSnippet: string;
  detail: string;
  agent: string | null;
}

/** A whole run's cache-prefix analysis. Mirrors Python `CacheReport`. */
export interface CacheReport {
  pairsAnalyzed: number;
  breaks: PrefixBreak[];
  stablePrefixTokensMin: number;
  rebilledTokensTotal: number;
  estimatedWasteNote: string;
  fixHint: string | null;
  agentsAnalyzed: number | null;
  pairsByAgent: Map<string, number> | null;
}

// --- snippet formatting -------------------------------------------------------

/** Collapse text to a single flattened, truncated line (whitespace → single
 * spaces, then hard-cut at `limit` with an ellipsis). Mirrors Python
 * `_flatten_snippet`. */
function flattenSnippet(text: string, limit = 80): string {
  const flat = text.split(/\s+/u).filter((s) => s.length > 0).join(" ");
  return flat.slice(0, limit) + (flat.length > limit ? "…" : "");
}

/** Shorter truncation for a differing substring inside a 'modified' break's
 * detail. Mirrors Python `_truncate`. Slices by code point (like Python). */
function truncate(text: string, limit = 40): string {
  const chars = Array.from(text);
  return chars.slice(0, limit).join("") + (chars.length > limit ? "…" : "");
}

// --- first-difference extraction (from the differ's inline diff) --------------

/**
 * Given a 'modified' entry's char-level inline diff, return [charOffset,
 * oldPart, newPart] for the FIRST run of non-equal segments: charOffset is the
 * summed length (in code points) of the leading equal segments; old/newPart are
 * the deleted/inserted text of that first differing run. Mirrors Python
 * `_first_diff_segment`. Lengths use code-point counts (`Array.from`) to match
 * Python's per-code-point offsets.
 */
function firstDiffSegment(inlineDiff: InlineSegment[]): [number, string, string] {
  let offset = 0;
  let idx = 0;
  while (idx < inlineDiff.length && inlineDiff[idx][0] === "equal") {
    offset += Array.from(inlineDiff[idx][1]).length;
    idx++;
  }
  let oldPart = "";
  let newPart = "";
  while (idx < inlineDiff.length && inlineDiff[idx][0] !== "equal") {
    const [op, seg] = inlineDiff[idx];
    if (op === "delete") oldPart += seg;
    else newPart += seg;
    idx++;
  }
  return [offset, oldPart, newPart];
}

/** Heuristic for 'a small volatile substring inside otherwise-stable text'
 * (e.g. a timestamp): true when there is SOME shared text and the changed text
 * is shorter than the shared text. Mirrors Python `_is_dynamic_change`. Lengths
 * are code-point counts. */
function isDynamicChange(inlineDiff: InlineSegment[]): boolean {
  let equalLen = 0;
  let changedLen = 0;
  for (const [op, seg] of inlineDiff) {
    const len = Array.from(seg).length;
    if (op === "equal") equalLen += len;
    else changedLen += len;
  }
  return equalLen > 0 && changedLen < equalLen;
}

// --- break attribution ---------------------------------------------------------

/**
 * Explain what happened at `position` (first index where old/new hashes
 * diverge), by looking up how the block differ classified that slot. Returns
 * [culpritKind, culpritLabel, culpritSnippet, detail, isDynamicChange]. Mirrors
 * Python `_attribute_break`: a same-slot 'modified' at this position → modified;
 * else an 'added' at this new-position → added; else an 'evicted' at this
 * old-position → evicted; else a moved 'unchanged' touching this position →
 * reordered; else a defensive 'changed' fallback.
 */
function attributeBreak(
  turnDiff: TurnDiff,
  position: number,
  old: CallBlock[],
  next: CallBlock[],
): [string, string, string, string, boolean] {
  const modified = turnDiff.entries.find(
    (e) => e.kind === "modified" && e.positionOld === position && e.positionNew === position,
  );
  if (modified) {
    const [offset, oldPart, newPart] = firstDiffSegment(modified.inlineDiff ?? []);
    const detail =
      `modified ${modified.label} block — first difference at ` +
      `char ${offset}: '${truncate(oldPart)}' → '${truncate(newPart)}'`;
    const isDynamic = isDynamicChange(modified.inlineDiff ?? []);
    return ["modified", modified.label, flattenSnippet(modified.block.text), detail, isDynamic];
  }

  const added = turnDiff.entries.find((e) => e.kind === "added" && e.positionNew === position);
  if (added) {
    const detail =
      `block inserted at position ${position} — ${added.label}/` +
      `${added.block.role} block not present in the previous turn`;
    return ["added", added.label, flattenSnippet(added.block.text), detail, false];
  }

  const evicted = turnDiff.entries.find((e) => e.kind === "evicted" && e.positionOld === position);
  if (evicted) {
    const detail =
      `block evicted at position ${position} — ${evicted.label}/` +
      `${evicted.block.role} block from the previous turn is missing here`;
    return ["evicted", evicted.label, flattenSnippet(evicted.block.text), detail, false];
  }

  const reordered = turnDiff.entries.find(
    (e) =>
      e.kind === "unchanged" &&
      e.positionOld !== e.positionNew &&
      (e.positionNew === position || e.positionOld === position),
  );
  if (reordered) {
    const detail =
      `block reordered — ${reordered.label}/${reordered.block.role} moved ` +
      `from position ${reordered.positionOld} to ${reordered.positionNew}, ` +
      `breaking the byte-for-byte prefix match at position ${position}`;
    return ["reordered", reordered.label, flattenSnippet(reordered.block.text), detail, false];
  }

  // Final defensive fallback — no known fixture reaches this.
  const newSide = position < next.length ? next[position] : null;
  const oldSide = position < old.length ? old[position] : null;
  const side = newSide ?? oldSide;
  const text = side ? side.block.text : "";
  const label = side ? side.label : "unknown";
  const detail = `context diverges at position ${position} (not a simple modify/insert/evict/reorder)`;
  return ["changed", label, flattenSnippet(text), detail, false];
}

// --- waste note + fix hint ------------------------------------------------------

/** Compose the neutral, price-free wasted-spend note — no hardcoded prices.
 * Mirrors Python `_waste_note`. */
function wasteNote(rebilledTokensTotal: number, pairsAnalyzed: number): string {
  const turnWord = pairsAnalyzed === 1 ? "turn" : "turns";
  return (
    `${rebilledTokensTotal} tokens re-billed across ${pairsAnalyzed} ${turnWord} ` +
    "that a stable prefix would have served from cache (cached input is " +
    "typically billed at a fraction of the full input price — check your " +
    "provider's current rates)"
  );
}

const DYNAMIC_FIELD_HINT =
  "a dynamic value inside an early system block breaks the prefix every " +
  "turn — move volatile content below the stable blocks";

/**
 * Detect the one actionable pattern: a dynamic value baked into an early system
 * block, breaking the prefix identically every turn. All must hold: ≥1 break;
 * every break shares the same divergent position AND culprit kind; that kind is
 * 'modified'; the position is among the first three (0/1/2); the label is
 * 'system'; and every modification looks dynamic. Mirrors Python `_fix_hint`.
 */
function fixHint(breaks: PrefixBreak[], dynamicFlags: boolean[]): string | null {
  if (breaks.length === 0) return null;
  const first = breaks[0];
  const sameCulpritEveryTime = breaks.every(
    (b) => b.divergentPosition === first.divergentPosition && b.culpritKind === first.culpritKind,
  );
  if (!sameCulpritEveryTime) return null;
  if (first.culpritKind !== "modified") return null;
  if (first.culpritLabel !== "system") return null;
  if (first.divergentPosition > 2) return null;
  if (!dynamicFlags.every((f) => f)) return null;
  return DYNAMIC_FIELD_HINT;
}

// --- core algorithm --------------------------------------------------------------

/**
 * Walk one agent's calls (in seq order) for prefix stability. Returns
 * [breaks, dynamicFlags, stableTokensPerPair, rebilledTotal]. For each
 * consecutive pair, walk both block-hash lists position-by-position from index
 * 0 (strictly positional — cache stability is "does byte K match byte K"), stop
 * at the first divergence or the shorter length. Reaching the shorter length is
 * an exact leading prefix (pure growth/truncation) → NOT a break. Otherwise the
 * blocks before the divergence are the stable prefix; everything in the newer
 * call from the divergence onward is rebilled; `diffCalls` explains the slot.
 * Mirrors Python `_analyze_group`.
 */
function analyzeGroup(
  calls: Call[],
  blocksByCallId: Map<string, CallBlock[]>,
  agentLabel: string | null,
): [PrefixBreak[], boolean[], number[], number] {
  const breaks: PrefixBreak[] = [];
  const dynamicFlags: boolean[] = [];
  const stableTokensPerPair: number[] = [];
  let rebilledTotal = 0;

  for (let idx = 0; idx + 1 < calls.length; idx++) {
    const prevCall = calls[idx];
    const call = calls[idx + 1];
    const old = blocksByCallId.get(prevCall.id)!;
    const next = blocksByCallId.get(call.id)!;
    const oldHashes = old.map((cb) => cb.block.contentHash);
    const newHashes = next.map((cb) => cb.block.contentHash);

    const shorterLen = Math.min(oldHashes.length, newHashes.length);
    let i = 0;
    while (i < shorterLen && oldHashes[i] === newHashes[i]) i++;

    const stableTokens = next.slice(0, i).reduce((acc, cb) => acc + cb.block.tokenCount, 0);
    stableTokensPerPair.push(stableTokens);

    if (i === shorterLen) continue; // exact leading prefix — not a break

    const rebilled = next.slice(i).reduce((acc, cb) => acc + cb.block.tokenCount, 0);
    rebilledTotal += rebilled;

    const turnDiff = diffCalls(old, next, prevCall.seq, call.seq);
    const [culpritKind, culpritLabel, culpritSnippet, detail, isDynamic] = attributeBreak(
      turnDiff,
      i,
      old,
      next,
    );

    breaks.push({
      seqPrev: prevCall.seq,
      seq: call.seq,
      stableBlocks: i,
      stableTokens,
      divergentPosition: i,
      culpritKind,
      culpritLabel,
      culpritSnippet,
      detail,
      agent: agentLabel,
    });
    dynamicFlags.push(isDynamic);
  }

  return [breaks, dynamicFlags, stableTokensPerPair, rebilledTotal];
}

/**
 * Analyze a run for cache-prefix stability, agent-aware. Grouping (the
 * correctness fix): cache stability is only meaningful BETWEEN CALLS OF THE
 * SAME AGENT. `agent` given → only that agent's calls. `agent` null with a
 * single distinct agent → one global timeline. `agent` null with multiple
 * agents → split into per-agent groups, analyze within each, merge; a
 * cross-agent adjacent pair is never counted. `stablePrefixTokensMin` is the
 * smallest stable-prefix token count across every analyzed pair. Mirrors Python
 * `analyze_cache`.
 */
export function analyzeCache(ct: ReadableStore, agent: string | null = null): CacheReport {
  const calls = filterCalls(ct.getCalls(), agent);
  const blocksByCallId = new Map<string, CallBlock[]>();
  for (const c of calls) blocksByCallId.set(c.id, ct.getCallBlocks(c.id));

  const labels = distinctAgents(calls);
  const grouped = agent === null && labels.length > 1;

  let groups: [string | null, Call[]][];
  let agentsAnalyzed: number | null;
  if (grouped) {
    groups = labels.map((lbl) => [lbl, calls.filter((c) => c.agent === lbl)]);
    agentsAnalyzed = labels.length;
  } else {
    const sole = agent !== null ? agent : labels.length ? labels[0] : null;
    groups = [[sole, calls]];
    agentsAnalyzed = null;
  }

  const allBreaks: PrefixBreak[] = [];
  const allDynamic: boolean[] = [];
  const allStable: number[] = [];
  let rebilledTotal = 0;
  const pairsByAgent = new Map<string, number>();
  for (const [label, groupCalls] of groups) {
    const [breaks, dyn, stable, rebilled] = analyzeGroup(groupCalls, blocksByCallId, label);
    allBreaks.push(...breaks);
    allDynamic.push(...dyn);
    allStable.push(...stable);
    rebilledTotal += rebilled;
    if (grouped) pairsByAgent.set(label ?? "(unlabeled)", stable.length);
  }

  const pairsAnalyzed = allStable.length;
  if (pairsAnalyzed === 0) {
    return {
      pairsAnalyzed: 0,
      breaks: [],
      stablePrefixTokensMin: 0,
      rebilledTokensTotal: 0,
      estimatedWasteNote: "fewer than 2 calls in this run — nothing to analyze",
      fixHint: null,
      agentsAnalyzed,
      pairsByAgent: pairsByAgent.size ? pairsByAgent : null,
    };
  }

  // Fix hint BEFORE reordering (it's order-independent: reads all(dynamic) and
  // compares every break to breaks[0]).
  const hint = fixHint(allBreaks, allDynamic);

  // Merge order: by the newer call's seq, so breaks read in timeline order.
  allBreaks.sort((a, b) => a.seq - b.seq);

  return {
    pairsAnalyzed,
    breaks: allBreaks,
    stablePrefixTokensMin: Math.min(...allStable),
    rebilledTokensTotal: rebilledTotal,
    estimatedWasteNote: wasteNote(rebilledTotal, pairsAnalyzed),
    fixHint: hint,
    agentsAnalyzed,
    pairsByAgent: pairsByAgent.size ? pairsByAgent : null,
  };
}

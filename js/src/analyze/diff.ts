/**
 * The block differ: turns two calls' ordered block lists into a structured,
 * git-style diff. A faithful port of Python `ctxdiff.analyze.differ` — same
 * SequenceMatcher alignment, same added/evicted/modified/unchanged
 * classification, same move-reconciliation, same token deltas — so `ctxdiff
 * diff` produces byte-identical output across the two SDKs. No I/O, no color.
 */
import { SequenceMatcher } from "./sequence-matcher.js";
import type { Block, CallBlock } from "../models.js";
import type { Call } from "../models.js";
import type { ReadableStore } from "../store/base.js";

// --- agent-awareness helpers -------------------------------------------------

/** Filter a call list to one agent's calls. `agent === null` means NO filter
 * (every call). A concrete name returns only calls whose `.agent` equals it.
 * Mirrors Python `filter_calls`. */
export function filterCalls(calls: Call[], agent: string | null): Call[] {
  if (agent === null) return [...calls];
  return calls.filter((c) => c.agent === agent);
}

/** Load every call from `ct` and filter to `agent` (null = whole run). Mirrors
 * Python `agent_calls`. */
export function agentCalls(ct: ReadableStore, agent: string | null): Call[] {
  return filterCalls(ct.getCalls(), agent);
}

/** Distinct agent labels across `calls`, in first-appearance order. A call with
 * no agent contributes a single `null` entry. Mirrors Python `distinct_agents`.
 */
export function distinctAgents(calls: Call[]): (string | null)[] {
  const seen: (string | null)[] = [];
  for (const c of calls) {
    if (!seen.includes(c.agent)) seen.push(c.agent);
  }
  return seen;
}

// --- value types -------------------------------------------------------------

export type DiffKind = "added" | "evicted" | "modified" | "unchanged";
/** One char-level diff segment: [op, text], op in 'equal' | 'delete' | 'insert'. */
export type InlineSegment = [string, string];

/** One block's fate between two turns. See Python `DiffEntry`: `block` is the
 * side worth showing (new for added/modified/unchanged, old for evicted);
 * `oldBlock`/`inlineDiff` are populated only for 'modified'. */
export interface DiffEntry {
  kind: DiffKind;
  block: Block;
  label: string;
  positionOld: number | null;
  positionNew: number | null;
  inlineDiff: InlineSegment[] | null;
  oldBlock: Block | null;
}

/** The full diff between two calls: every block's DiffEntry plus token deltas.
 * A 'modified' block counts as evict-old + add-new for budgeting. Mirrors
 * Python `TurnDiff`. */
export interface TurnDiff {
  seqOld: number;
  seqNew: number;
  entries: DiffEntry[];
  tokensAdded: number;
  tokensEvicted: number;
}

// --- inline (char-level) diff -------------------------------------------------

/**
 * Char-level diff between a modified block's old and new text, via
 * SequenceMatcher over CODE POINTS (`Array.from`, matching Python's per-code-
 * point string handling). Returns a flat list of [op, text] segments in
 * old→new order; a 'replace' opcode is split into a delete-then-insert pair so
 * every segment maps to exactly one side. Mirrors Python `_inline_diff`.
 */
function inlineDiff(oldText: string, newText: string): InlineSegment[] {
  const o = Array.from(oldText);
  const n = Array.from(newText);
  const sm = new SequenceMatcher(o, n);
  const segments: InlineSegment[] = [];
  for (const op of sm.getOpcodes()) {
    if (op.tag === "equal") {
      segments.push(["equal", o.slice(op.i1, op.i2).join("")]);
    } else if (op.tag === "delete") {
      segments.push(["delete", o.slice(op.i1, op.i2).join("")]);
    } else if (op.tag === "insert") {
      segments.push(["insert", n.slice(op.j1, op.j2).join("")]);
    } else if (op.tag === "replace") {
      segments.push(["delete", o.slice(op.i1, op.i2).join("")]);
      segments.push(["insert", n.slice(op.j1, op.j2).join("")]);
    }
  }
  return segments;
}

// --- moved-block reconciliation ----------------------------------------------

/**
 * Fold pure moves that the LCS alignment reports as evict+add of identical
 * content into single 'unchanged' (moved) entries. How: index every 'evicted'
 * and 'added' entry by content hash; wherever the same hash appears on both
 * sides, pair them FIFO (in encounter order — identical-content blocks are
 * interchangeable) and replace each pair with one 'unchanged' entry carrying
 * both positions. Count mismatches (a real net add/removal of repeated content)
 * leave their leftovers as genuine evicted/added. Mirrors Python
 * `_reconcile_moves` (its `id()`-based consumed set becomes an object-identity
 * Set here).
 */
function reconcileMoves(entries: DiffEntry[]): DiffEntry[] {
  const evictedByHash = new Map<string, DiffEntry[]>();
  const addedByHash = new Map<string, DiffEntry[]>();
  const push = (m: Map<string, DiffEntry[]>, h: string, e: DiffEntry): void => {
    const arr = m.get(h);
    if (arr) arr.push(e);
    else m.set(h, [e]);
  };
  for (const e of entries) {
    if (e.kind === "evicted") push(evictedByHash, e.block.contentHash, e);
    else if (e.kind === "added") push(addedByHash, e.block.contentHash, e);
  }

  const moved: DiffEntry[] = [];
  const consumed = new Set<DiffEntry>();
  for (const [h, evictedList] of evictedByHash) {
    const addedList = addedByHash.get(h) ?? [];
    const pairs = Math.min(evictedList.length, addedList.length);
    for (let k = 0; k < pairs; k++) {
      const ev = evictedList[k];
      const ad = addedList[k];
      consumed.add(ev);
      consumed.add(ad);
      moved.push({
        kind: "unchanged",
        block: ad.block,
        label: ad.label,
        positionOld: ev.positionOld,
        positionNew: ad.positionNew,
        inlineDiff: null,
        oldBlock: null,
      });
    }
  }

  const kept = entries.filter((e) => !consumed.has(e));
  kept.push(...moved);
  return kept;
}

// --- core algorithm ------------------------------------------------------------

/**
 * Diff two ordered CallBlock lists into a TurnDiff. Algorithm (mirrors Python
 * `diff_calls`): SequenceMatcher-align the two ordered hash lists; 'equal'
 * opcodes → unchanged (recording both positions, so a moved-but-identical block
 * shows a position delta); 'delete' → evicted; 'insert' → added; 'replace' →
 * pair the two spans positionally — a pair in the same logical slot (same role
 * AND kind) is a 'modified' with a char-level inline diff, otherwise
 * evicted/added; length-mismatch leftovers in a replace span are straight
 * evict/add. A move-reconciliation post-pass folds spurious evict+add of
 * identical content back into unchanged (moved) entries. Token deltas count a
 * modification as evict-old + add-new. Entries are sorted by new position
 * (falling back to old for pure evictions).
 */
export function diffCalls(
  old: CallBlock[],
  next: CallBlock[],
  seqOld: number,
  seqNew: number,
): TurnDiff {
  const oldHashes = old.map((cb) => cb.block.contentHash);
  const newHashes = next.map((cb) => cb.block.contentHash);
  const sm = new SequenceMatcher(oldHashes, newHashes);

  let entries: DiffEntry[] = [];
  for (const op of sm.getOpcodes()) {
    if (op.tag === "equal") {
      for (let i = op.i1, j = op.j1; i < op.i2 && j < op.j2; i++, j++) {
        const o = old[i];
        const n = next[j];
        entries.push({
          kind: "unchanged",
          block: n.block,
          label: n.label,
          positionOld: o.position,
          positionNew: n.position,
          inlineDiff: null,
          oldBlock: null,
        });
      }
    } else if (op.tag === "delete") {
      for (let i = op.i1; i < op.i2; i++) {
        const o = old[i];
        entries.push({
          kind: "evicted",
          block: o.block,
          label: o.label,
          positionOld: o.position,
          positionNew: null,
          inlineDiff: null,
          oldBlock: null,
        });
      }
    } else if (op.tag === "insert") {
      for (let j = op.j1; j < op.j2; j++) {
        const n = next[j];
        entries.push({
          kind: "added",
          block: n.block,
          label: n.label,
          positionOld: null,
          positionNew: n.position,
          inlineDiff: null,
          oldBlock: null,
        });
      }
    } else if (op.tag === "replace") {
      const oldSlice = old.slice(op.i1, op.i2);
      const newSlice = next.slice(op.j1, op.j2);
      const paired = Math.min(oldSlice.length, newSlice.length);
      for (let k = 0; k < paired; k++) {
        const o = oldSlice[k];
        const n = newSlice[k];
        const sameSlot =
          o.block.role === n.block.role && o.block.kind === n.block.kind;
        if (sameSlot) {
          entries.push({
            kind: "modified",
            block: n.block,
            label: n.label,
            positionOld: o.position,
            positionNew: n.position,
            inlineDiff: inlineDiff(o.block.text, n.block.text),
            oldBlock: o.block,
          });
        } else {
          entries.push({
            kind: "evicted",
            block: o.block,
            label: o.label,
            positionOld: o.position,
            positionNew: null,
            inlineDiff: null,
            oldBlock: null,
          });
          entries.push({
            kind: "added",
            block: n.block,
            label: n.label,
            positionOld: null,
            positionNew: n.position,
            inlineDiff: null,
            oldBlock: null,
          });
        }
      }
      for (const o of oldSlice.slice(paired)) {
        entries.push({
          kind: "evicted",
          block: o.block,
          label: o.label,
          positionOld: o.position,
          positionNew: null,
          inlineDiff: null,
          oldBlock: null,
        });
      }
      for (const n of newSlice.slice(paired)) {
        entries.push({
          kind: "added",
          block: n.block,
          label: n.label,
          positionOld: null,
          positionNew: n.position,
          inlineDiff: null,
          oldBlock: null,
        });
      }
    }
  }

  entries = reconcileMoves(entries);

  // Modified counts as evict-old + add-new for budgeting (see TurnDiff).
  let tokensAdded = 0;
  let tokensEvicted = 0;
  for (const e of entries) {
    if (e.kind === "added" || e.kind === "modified") {
      tokensAdded += e.block.tokenCount;
    }
    if (e.kind === "evicted") tokensEvicted += e.block.tokenCount;
    if (e.kind === "modified" && e.oldBlock) tokensEvicted += e.oldBlock.tokenCount;
  }

  // Stable render order: by new position, falling back to old for pure
  // evictions. Node's Array.sort is not guaranteed stable across all keys, but
  // the fallback key makes every entry's sort key well-defined, matching the
  // Python `sort(key=...)` which is a total order on the same values.
  entries.sort((a, b) => {
    const ka = a.positionNew !== null ? a.positionNew : (a.positionOld as number);
    const kb = b.positionNew !== null ? b.positionNew : (b.positionOld as number);
    return ka - kb;
  });

  return { seqOld, seqNew, entries, tokensAdded, tokensEvicted };
}

/**
 * Resolve two turn numbers (call.seq) to their calls in `ct`, load each call's
 * blocks, and delegate to `diffCalls`. Throws a clear Error if either turn has
 * no matching call. Mirrors Python `diff_turns`.
 *
 * `labels` spells the two turns as the USER typed them, for the not-found
 * message only. Python's `--turn` is an arbitrary-precision int, so it echoes
 * `1000000000000000000000` verbatim; the same value as a JS number renders
 * `1e+21`, which is not what anyone typed. Callers that already hold the raw
 * text (the CLI does) pass it; anything else gets the numbers, unchanged.
 */
export function diffTurns(
  ct: ReadableStore,
  turnOld: number,
  turnNew: number,
  labels?: [string, string],
): TurnDiff {
  const calls = ct.getCalls();
  const bySeq = new Map<number, Call>(calls.map((c) => [c.seq, c]));
  const shown = labels ?? [String(turnOld), String(turnNew)];
  const missing = [turnOld, turnNew]
    .map((s, i) => ({ seq: s, text: shown[i] }))
    .filter((t) => !bySeq.has(t.seq));
  if (missing.length) {
    const available = calls.map((c) => c.seq).sort((a, b) => a - b);
    // Python list repr: `[1, 2, 3]` (space after each comma), not JSON's `[1,2,3]`.
    const pyList = (arr: Array<number | string>) => "[" + arr.join(", ") + "]";
    throw new Error(
      `turn(s) ${pyList(missing.map((t) => t.text))} not found in this run ` +
        `(available turns: ${pyList(available)})`,
    );
  }
  const oldCall = bySeq.get(turnOld)!;
  const newCall = bySeq.get(turnNew)!;
  const oldBlocks = ct.getCallBlocks(oldCall.id);
  const newBlocks = ct.getCallBlocks(newCall.id);
  return diffCalls(oldBlocks, newBlocks, turnOld, turnNew);
}

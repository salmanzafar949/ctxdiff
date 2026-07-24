/**
 * A faithful port of Python's `difflib.SequenceMatcher` (the subset ctxdiff
 * uses: `getOpcodes`, via `getMatchingBlocks`/`findLongestMatch`), with
 * `autojunk` disabled and no `isjunk` — exactly how the Python analyzers
 * construct it (`SequenceMatcher(None, a, b, autojunk=False)`).
 *
 * WHY a hand port: the diff and cache analyzers must produce byte-identical
 * classifications to the Python SDK on the same `.ctrace`. The whole
 * added/evicted/modified/reordered vocabulary flows from these exact opcodes,
 * so the alignment — including its tie-breaking — must match CPython's
 * algorithm move for move, not merely "some LCS diff".
 *
 * Generic over element type `T` (string hashes for block alignment, single
 * code points for char-level inline diffs). Elements are compared with `===`
 * and used as Map keys, so `T` must be a primitive (string/number) — which is
 * all the analyzers ever pass.
 */

/** A maximal matching block: `a[a0 .. a0+size) === b[b0 .. b0+size)`. Field
 * names mirror difflib's `Match(a, b, size)` triple. */
export interface Match {
  a: number;
  b: number;
  size: number;
}

/** One opcode describing how to turn `a[i1:i2]` into `b[j1:j2]`. `tag` is one
 * of 'equal' | 'replace' | 'delete' | 'insert' — the same vocabulary
 * `SequenceMatcher.get_opcodes()` returns. */
export interface Opcode {
  tag: "equal" | "replace" | "delete" | "insert";
  i1: number;
  i2: number;
  j1: number;
  j2: number;
}

export class SequenceMatcher<T extends string | number> {
  private a: T[];
  private b: T[];
  private b2j: Map<T, number[]>;

  /**
   * Index `b` once: `b2j` maps each element of `b` to the ascending list of
   * positions it occurs at. This is difflib's `__chain_b` with `autojunk=False`
   * and `isjunk=None`, so there is no popular-element pruning and no junk set —
   * every element contributes every one of its indices.
   */
  constructor(a: T[], b: T[]) {
    this.a = a;
    this.b = b;
    this.b2j = new Map();
    b.forEach((el, idx) => {
      const arr = this.b2j.get(el);
      if (arr) arr.push(idx);
      else this.b2j.set(el, [idx]);
    });
  }

  /**
   * Find the longest matching block in `a[alo:ahi]` vs `b[blo:bhi]`, ported
   * verbatim from difflib. How: a rolling `j2len` dict where `j2len[j]` is the
   * length of the longest match ending at `a[i], b[j]`; each row `i` is rebuilt
   * from the previous row's `j-1` entries, so a run of equal elements grows by
   * one each step. The earliest longest match wins (`k > bestsize`, strict),
   * then it is extended outward over adjacent equal elements — matching
   * CPython's tie-breaking, which the analyzers depend on. The junk-extension
   * loops difflib has after these are omitted because there is no junk here.
   */
  findLongestMatch(alo: number, ahi: number, blo: number, bhi: number): Match {
    const { a, b, b2j } = this;
    let besti = alo;
    let bestj = blo;
    let bestsize = 0;
    let j2len = new Map<number, number>();

    for (let i = alo; i < ahi; i++) {
      const newj2len = new Map<number, number>();
      const js = b2j.get(a[i]);
      if (js) {
        for (const j of js) {
          if (j < blo) continue;
          if (j >= bhi) break;
          const k = (j2len.get(j - 1) ?? 0) + 1;
          newj2len.set(j, k);
          if (k > bestsize) {
            besti = i - k + 1;
            bestj = j - k + 1;
            bestsize = k;
          }
        }
      }
      j2len = newj2len;
    }

    // Extend the best match by equal elements on each end (no junk to skip).
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) {
      besti--;
      bestj--;
      bestsize++;
    }
    while (
      besti + bestsize < ahi &&
      bestj + bestsize < bhi &&
      a[besti + bestsize] === b[bestj + bestsize]
    ) {
      bestsize++;
    }

    return { a: besti, b: bestj, size: bestsize };
  }

  /**
   * Return the list of maximal matching blocks, terminated by the sentinel
   * `(len(a), len(b), 0)` — difflib's `get_matching_blocks`. How: divide and
   * conquer via an explicit LIFO stack (matching CPython's `queue.pop()`), find
   * the longest match in each sub-range, then recurse on the ranges to its left
   * and right. The collected blocks are sorted and adjacent blocks coalesced,
   * exactly as difflib does, so the output order/coalescing is identical.
   */
  getMatchingBlocks(): Match[] {
    const la = this.a.length;
    const lb = this.b.length;
    const queue: [number, number, number, number][] = [[0, la, 0, lb]];
    const matching: Match[] = [];

    while (queue.length) {
      const [alo, ahi, blo, bhi] = queue.pop()!;
      const m = this.findLongestMatch(alo, ahi, blo, bhi);
      if (m.size) {
        matching.push(m);
        if (alo < m.a && blo < m.b) queue.push([alo, m.a, blo, m.b]);
        if (m.a + m.size < ahi && m.b + m.size < bhi) {
          queue.push([m.a + m.size, ahi, m.b + m.size, bhi]);
        }
      }
    }

    // Sort by (a, b, size) — CPython sorts the Match tuples lexicographically.
    matching.sort((x, y) => x.a - y.a || x.b - y.b || x.size - y.size);

    // Coalesce adjacent equal blocks into one.
    let i1 = 0;
    let j1 = 0;
    let k1 = 0;
    const nonAdjacent: Match[] = [];
    for (const m of matching) {
      if (i1 + k1 === m.a && j1 + k1 === m.b) {
        k1 += m.size;
      } else {
        if (k1) nonAdjacent.push({ a: i1, b: j1, size: k1 });
        i1 = m.a;
        j1 = m.b;
        k1 = m.size;
      }
    }
    if (k1) nonAdjacent.push({ a: i1, b: j1, size: k1 });
    nonAdjacent.push({ a: la, b: lb, size: 0 });
    return nonAdjacent;
  }

  /**
   * Turn the matching blocks into edit opcodes — difflib's `get_opcodes`. Each
   * gap between consecutive matching blocks becomes a 'replace' (both sides
   * non-empty), 'delete' (only `a` advanced), or 'insert' (only `b` advanced);
   * each matching block itself becomes an 'equal'. Identical control flow to
   * CPython, so the emitted opcode sequence matches exactly.
   */
  getOpcodes(): Opcode[] {
    let i = 0;
    let j = 0;
    const answer: Opcode[] = [];
    for (const m of this.getMatchingBlocks()) {
      const ai = m.a;
      const bj = m.b;
      let tag: Opcode["tag"] | "" = "";
      if (i < ai && j < bj) tag = "replace";
      else if (i < ai) tag = "delete";
      else if (j < bj) tag = "insert";
      if (tag) answer.push({ tag, i1: i, i2: ai, j1: j, j2: bj });
      i = ai + m.size;
      j = bj + m.size;
      if (m.size) answer.push({ tag: "equal", i1: ai, i2: i, j1: bj, j2: j });
    }
    return answer;
  }
}

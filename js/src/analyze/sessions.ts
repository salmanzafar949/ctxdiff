/**
 * The discovery scanner behind `ctxdiff sessions` and `ctxdiff agents` when no
 * project has been named: walk every `*.ctrace` in a directory and read what is
 * in it. A faithful port of the Python CLI's `_cmd_sessions`/`_cmd_agents`
 * gathering loops — sorted by filename, one bad file skipped (never fatal),
 * every session of every file surfaced. Read-only.
 *
 * Why a directory scan exists at all when every other command narrows to ONE
 * project: discovery is the one job where narrowing defeats the purpose. You
 * cannot pass `--project` to a project you were never shown.
 */
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { CTrace } from "../store/ctrace.js";
import type { Call, Session } from "../models.js";

/** One `.ctrace` file in the scanned directory, with every session it holds
 * (oldest first, exactly as `listSessions` returns them). */
export interface FileSessions {
  filename: string;
  sessions: Session[];
}

/**
 * Whether a directory entry is a trace a scan should SEE.
 *
 * The dot-prefix exclusion is not a style choice: the Python CLI scans with
 * `glob("*.ctrace")`, and a `*` wildcard never matches a leading dot. Without it
 * the two CLIs list different files — and, through `findDefaultRun`, ANALYZE
 * different files, so `ctxdiff tokens` reports numbers for a project the other
 * SDK would never have opened.
 */
export function isTraceFile(filename: string): boolean {
  return !filename.startsWith(".") && filename.endsWith(".ctrace");
}

/**
 * Compare two filenames by CODE POINT, which is what Python's `sorted()` does.
 *
 * JS's default array comparator orders by UTF-16 code UNIT, so every astral
 * character (encoded as a surrogate pair starting at 0xD800) sorts as if it were
 * below U+E000 — `𝐀` (U+1D400) lands before `豈` (U+F900) here and after it in
 * Python. Comparing `Array.from(...)`'s code points removes the surrogate
 * encoding from the question entirely.
 */
export function compareCodePoints(a: string, b: string): number {
  const x = Array.from(a);
  const y = Array.from(b);
  const shared = Math.min(x.length, y.length);
  for (let i = 0; i < shared; i++) {
    const ca = x[i].codePointAt(0) as number;
    const cb = y[i].codePointAt(0) as number;
    if (ca !== cb) return ca < cb ? -1 : 1;
  }
  return x.length - y.length;
}

/** The `*.ctrace` files in `dir`, sorted by filename exactly as Python's
 * `sorted(glob(...))` sorts them. A shared helper so the two scanners below can
 * never disagree about which files exist or in what order they are reported. An
 * unreadable directory yields nothing rather than throwing — the caller prints
 * an empty listing, which is the truth. */
function traceFiles(dir: string): string[] {
  try {
    return readdirSync(dir).filter(isTraceFile).sort(compareCodePoints);
  } catch {
    return [];
  }
}

/**
 * Every session of every `*.ctrace` in `dir`, grouped by file and sorted by
 * filename. A file that fails to open (corrupt, wrong schema version, not
 * actually a ctrace) is SKIPPED rather than aborting the listing — one bad file
 * shouldn't hide every good one. Mirrors the Python `_cmd_sessions` cwd branch.
 */
export function listFileSessions(dir: string): FileSessions[] {
  const out: FileSessions[] = [];
  for (const filename of traceFiles(dir)) {
    let ct: CTrace;
    try {
      ct = CTrace.open(join(dir, filename));
    } catch {
      continue; // skip unreadable files, don't crash the listing
    }
    try {
      out.push({ filename, sessions: ct.listSessions() });
    } finally {
      ct.close();
    }
  }
  return out;
}

/**
 * Every session's calls across every `*.ctrace` in `dir`, one array per session
 * — the raw material `ctxdiff agents` aggregates over. Same skip-a-bad-file
 * rule as `listFileSessions`. Mirrors the Python `_cmd_agents` cwd branch.
 */
export function listFileCalls(dir: string): Call[][] {
  const out: Call[][] = [];
  for (const filename of traceFiles(dir)) {
    let ct: CTrace;
    try {
      ct = CTrace.open(join(dir, filename));
    } catch {
      continue;
    }
    try {
      for (const s of ct.listSessions()) out.push(ct.getCalls(s.id));
    } finally {
      ct.close();
    }
  }
  return out;
}

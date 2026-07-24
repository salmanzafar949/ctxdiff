/**
 * The `runs` lister: enumerate every `*.ctrace` in a directory with its
 * project / provider / turn-count / distinct-agent summary. A faithful port of
 * the Python CLI's `_cmd_runs` gathering logic — sorted by filename, one bad
 * file skipped (not fatal), agents joined in first-appearance order. Read-only.
 */
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { CTrace } from "../store/ctrace.js";

/** One row of the runs listing: (filename, project, provider, turns, agents).
 * `agents` is a comma-joined list of distinct named agents, or '-' when none. */
export interface RunRow {
  filename: string;
  project: string;
  provider: string;
  turns: number;
  agents: string;
}

/**
 * List every `*.ctrace` file in `dir` as RunRows, sorted by filename. How:
 * glob the directory for `.ctrace` files, open each read-only, read its run row
 * and calls; a file that fails to open (corrupt, wrong schema version, not a
 * ctrace) is SKIPPED rather than aborting the listing. `agents` collects each
 * call's non-empty agent in first-appearance order. Mirrors Python `_cmd_runs`.
 */
export function listRuns(dir: string): RunRow[] {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  const candidates = entries.filter((f) => f.endsWith(".ctrace")).sort();

  const rows: RunRow[] = [];
  for (const filename of candidates) {
    let ct: CTrace;
    try {
      ct = CTrace.open(join(dir, filename));
    } catch {
      continue; // skip unreadable files, don't crash the listing
    }
    try {
      const run = ct.getRun();
      const calls = ct.getCalls();
      const names: string[] = [];
      for (const c of calls) {
        if (c.agent && !names.includes(c.agent)) names.push(c.agent);
      }
      rows.push({
        filename,
        project: run.project,
        provider: run.provider,
        turns: calls.length,
        agents: names.length ? names.join(", ") : "-",
      });
    } finally {
      ct.close();
    }
  }
  return rows;
}

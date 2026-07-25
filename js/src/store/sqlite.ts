/**
 * The SQLite backend — ctxdiff's zero-config default, and the reference
 * implementation of `StoreBackend`.
 *
 * This module is a thin, connection-less FACTORY around `CTrace` (which is the
 * actual `Store` implementation, unchanged): it exists so the local-first path
 * travels through exactly the same seam as Postgres/MySQL, instead of `trace.ts`
 * special-casing "no backend configured -> reach for the CTrace class". Nothing
 * about the `.ctrace` file, its schema, its DDL or its write path changes — a
 * user who never calls `configure()` gets byte-identical files to before, from
 * byte-identical code.
 *
 * It is also the only backend that is SYNCHRONOUS end to end (`node:sqlite` has
 * no async API), which is deliberate rather than incidental: opening a local
 * file costs microseconds and no network round trip, so the existing inline-open
 * behavior — and every existing test that calls `wrap()` then reads the file —
 * stays exactly as it is. See `store/base.ts` for how one protocol accommodates
 * both that and the promise-based network stores.
 */
import { statSync } from "node:fs";
import { join } from "node:path";
import { CTrace } from "./ctrace.js";
import { pyRepr } from "../render.js";
import type { FileStoreBackend, OpenSessionArgs } from "./base.js";

/** Whether `path` exists AND is a directory. `statSync` with `throwIfNoEntry`
 * off makes "missing" and "not a directory" the same answer — false — which is
 * what both callers want: a path that is not an existing directory is treated as
 * a file name. */
function isDirectory(path: string): boolean {
  const st = statSync(path, { throwIfNoEntry: false });
  return st !== undefined && st.isDirectory();
}

/**
 * Points ctxdiff at a local `.ctrace` file (or a directory of them).
 *
 * `path` resolution, in the order a user expects:
 * - `undefined` (the default) — one file per project in the CURRENT directory:
 *   `./<project>.ctrace`, exactly what `trace.init(project)` has always done.
 * - an existing DIRECTORY — `<dir>/<project>.ctrace`, so `CTXDIFF_STORE=~/traces`
 *   keeps every project's DB in one place while preserving the one-file-per-
 *   project model.
 * - anything else — that exact file, for every project (an explicit
 *   `trace.init(project, { path })` or `CTXDIFF_STORE=./my.ctrace`).
 *
 * Constructing this touches no disk: the file is created/opened only when
 * `openSession()`/`openReader()` is called, matching the `StoreBackend` contract
 * that a backend is inert until used.
 */
export class SQLiteStore implements FileStoreBackend {
  readonly path: string | null;

  /**
   * Record where traces should live. `path` may be omitted (a per-project file
   * in the cwd), a directory, or a file — see the class docstring; nothing is
   * resolved or created here, since the project name (needed for the
   * per-project default) isn't known yet.
   */
  constructor(opts: { path?: string | null } = {}) {
    this.path = opts.path ?? null;
  }

  /**
   * Resolve the concrete `.ctrace` file this backend uses for `project`. How:
   * applies the three rules in the class docstring — null and an existing
   * directory both expand to `<dir>/<project>.ctrace` (cwd for null), anything
   * else is returned verbatim. Public (rather than private to `openSession`)
   * because `Tracer.path` reports it to the user and existing tests assert on
   * it.
   */
  pathFor(project: string): string {
    if (this.path === null) return `${project}.ctrace`;
    if (isDirectory(this.path)) return join(this.path, `${project}.ctrace`);
    return this.path;
  }

  /**
   * Start a NEW session in this project's `.ctrace`, creating the file if
   * absent. A straight delegation to `CTrace.openOrCreateSession` — which
   * already does the schema-if-not-exists, the v1->v2 column upgrade, the
   * append-a-run-row, the WAL/busy_timeout write configuration and the bounded
   * locked-retry — so the backend seam adds exactly zero behavior to the local
   * path.
   */
  openSession(args: OpenSessionArgs): CTrace {
    return CTrace.openOrCreateSession(
      this.pathFor(args.project),
      args.project,
      args.provider,
      args.model ?? "",
      args.startedAt ?? "",
    );
  }

  /**
   * Open this backend's file read-side, bound to its NEWEST session. Requires a
   * concrete file: with no path, or a directory, there is no single file to read
   * (the project name is a write-time input only), so this throws a message
   * pointing at `--run`, which is how the CLI already selects a file in that
   * case.
   *
   * The path is quoted with `pyRepr`, not `JSON.stringify`: the Python twin of
   * this message interpolates `{path!r}`, which prefers SINGLE quotes and only
   * switches to double ones when the path itself contains a single quote. The
   * message reaches the user through both CLIs, so it has to be the same bytes.
   */
  openReader(): CTrace {
    if (this.path === null || isDirectory(this.path)) {
      throw new Error(
        "ctxdiff: SQLiteStore has no single file to read " +
          `(path=${this.path === null ? "None" : pyRepr(this.path)}); ` +
          "pass an explicit .ctrace path",
      );
    }
    return CTrace.open(this.path);
  }
}

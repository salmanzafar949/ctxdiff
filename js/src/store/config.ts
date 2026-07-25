/**
 * Configure once, then everything just works.
 *
 * Two ways to point ctxdiff at a database instead of a local file:
 *
 *     import { configure, PostgresStore } from "ctxdiff";
 *     configure({ store: new PostgresStore({ dsn: "postgresql://user@host/db" }) });
 *
 * ...or, with no code change at all, the `CTXDIFF_STORE` environment variable:
 *
 *     CTXDIFF_STORE=postgresql://user@host/db   # Postgres
 *     CTXDIFF_STORE=mysql://user@host/db        # MySQL
 *     CTXDIFF_STORE=sqlite:///abs/path.ctrace   # SQLite, explicit
 *     CTXDIFF_STORE=~/traces                    # a directory of .ctrace files
 *
 * Every subsequent `trace.init(project)` uses whatever is configured. Resolution
 * order is explicit-beats-ambient: an explicit `trace.init(project, { store })`
 * argument, then `configure()`, then `CTXDIFF_STORE`, then — when NOTHING is
 * configured — null, which callers read as "the zero-config default": a local
 * `./<project>.ctrace`, byte-identical to ctxdiff's behavior before backends
 * existed. Local-first is the default and stays the default.
 *
 * The env var is read at resolution time (not import time) so a test or a
 * subprocess can set it after ctxdiff has already been imported.
 */
import { homedir } from "node:os";
import { join } from "node:path";
import type { StoreBackend } from "./base.js";
import { SQLiteStore } from "./sqlite.js";
import { PostgresStore } from "./postgres.js";
import { MySQLStore } from "./mysql.js";

export const ENV_VAR = "CTXDIFF_STORE";

// Every backend name a scheme can carry. A value with NO `://` that is EXACTLY
// one of these is a typo for the URL form, never a filename — see `fromDsn`,
// where treating `CTXDIFF_STORE=postgres` as a path would silently write a local
// SQLite file called "postgres" while the user believed they had a database.
const BACKEND_NAMES = new Set([
  "postgres",
  "postgresql",
  "mysql",
  "mariadb",
  "sqlite",
  "sqlite3",
]);

// The process-wide default set by `configure()`. null means "not configured",
// which is deliberately distinct from `new SQLiteStore()`: it lets `resolve()`
// fall through to the env var before defaulting, and lets `configure()` reset
// ctxdiff back to the zero-config local path.
let configuredStore: StoreBackend | null = null;

/** Options for `configure`. */
export interface ConfigureOptions {
  /** The process-wide default backend, or null/omitted to clear it. */
  store?: StoreBackend | null;
}

/**
 * Set (or clear) the process-wide default backend. `configure({ store: X })`
 * makes every later `trace.init(project)` open its session in X; `configure()`
 * (or `configure({ store: null })`) clears it, restoring the zero-config local
 * `.ctrace` default. Deliberately a plain module-level global rather than
 * async-context state: it is a once-at-startup deployment choice, and a backend
 * is immutable and connection-less, so sharing one across concurrent work is
 * safe (each `openSession()` makes its own connection).
 */
export function configure(opts: ConfigureOptions = {}): void {
  configuredStore = opts.store ?? null;
}

/** Whatever `configure()` last set (null if never set / cleared) — without
 * consulting the environment. Used by tests and by `resolve()`. */
export function configured(): StoreBackend | null {
  return configuredStore;
}

/**
 * Resolve the backend to use, explicit-beats-ambient: an explicitly passed
 * `store`, else the `configure()`d default, else `CTXDIFF_STORE` parsed via
 * `fromDsn`, else null ("nothing configured — use the zero-config local
 * default"). May THROW for an unparseable `CTXDIFF_STORE`; callers on the
 * capture path resolve inside their fail-open guard so a typo'd env var degrades
 * capture with a warning instead of breaking the host.
 */
export function resolve(store?: StoreBackend | null): StoreBackend | null {
  if (store) return store;
  if (configuredStore) return configuredStore;
  const dsn = (process.env[ENV_VAR] ?? "").trim();
  if (dsn) return fromDsn(dsn);
  return null;
}

/**
 * Build a backend from a `CTXDIFF_STORE`-style string.
 *
 * How: the URL scheme picks the adapter — `postgres`/`postgresql` (plus
 * `+driver` suffixes like `postgresql+psycopg`, which are tolerated and ignored
 * so a SQLAlchemy-shaped URL pasted from an existing app just works), `mysql`,
 * and `sqlite`/`sqlite3`. A string with NO recognizable scheme is treated as a
 * filesystem path (that is what a bare `~/traces` or `./my.ctrace` obviously
 * means) — UNLESS it is exactly a backend NAME (`postgres`, `mysql`, `sqlite`,
 * ...), which is a typo for the URL form and is REFUSED: writing a local SQLite
 * file literally named "postgres" for a user who asked for Postgres is the one
 * lie this module exists to prevent. A Windows drive letter (`C:\...`) has no
 * `://` and so is never mistaken for a scheme. An unknown scheme throws, naming
 * what IS supported, rather than silently falling back to a local file the user
 * never asked for.
 */
export function fromDsn(dsn: string): StoreBackend {
  const scheme = dsn.includes("://") ? dsn.split("://", 1)[0].toLowerCase() : "";
  const base = scheme.split("+", 1)[0]; // tolerate SQLAlchemy-style 'mysql+pymysql'

  if (base === "postgres" || base === "postgresql") {
    return new PostgresStore({ dsn: stripDriverSuffix(dsn, base) });
  }
  if (base === "mysql" || base === "mariadb") {
    return new MySQLStore({ dsn: stripDriverSuffix(dsn, base) });
  }
  if (base === "sqlite" || base === "sqlite3") {
    return new SQLiteStore({ path: sqlitePath(dsn) });
  }
  if (base) {
    throw new Error(
      `ctxdiff: unsupported ${ENV_VAR} scheme '${base}://' — expected ` +
        "postgresql://, mysql://, sqlite:// or a filesystem path",
    );
  }
  // No scheme, but the whole value IS a backend name: the user meant the URL
  // form and dropped the `://` (or their shell ate it). Refuse loudly — the
  // alternative is a local `./postgres` SQLite file, silently, for a user who
  // asked for a database and would be told `tracer.path` is "postgres".
  const bare = dsn.trim().toLowerCase();
  if (BACKEND_NAMES.has(bare)) {
    throw new Error(
      `ctxdiff: ${ENV_VAR}='${dsn}' looks like a backend name, not a location — ` +
        `it is missing '://'. Use e.g. ${bare}://user:password@host:port/database ` +
        "(or pass a filesystem path for a local .ctrace)",
    );
  }
  // A plain path. `~` is expanded here so the common `CTXDIFF_STORE=~/traces`
  // (which a shell may or may not expand) works.
  return new SQLiteStore({ path: expandUser(dsn) });
}

/** Rewrite `postgresql+psycopg://...` to `postgresql://...` (and the MySQL
 * equivalent), leaving a plain DSN untouched. Why: users paste DSNs out of
 * existing SQLAlchemy config, where the `+driver` part names a PYTHON driver —
 * meaningless (and rejected) at the libpq/mysql level. */
function stripDriverSuffix(dsn: string, base: string): string {
  const idx = dsn.indexOf("://");
  const head = dsn.slice(0, idx);
  return head.includes("+") ? `${base}${dsn.slice(idx)}` : dsn;
}

/** Expand a leading `~` to the user's home directory; anything else is returned
 * unchanged. */
function expandUser(path: string): string {
  if (path === "~") return homedir();
  if (path.startsWith("~/")) return join(homedir(), path.slice(2));
  return path;
}

/**
 * Turn a `sqlite://` URL into a filesystem path, accepting every spelling people
 * actually write:
 *
 * - `sqlite:///var/traces/a.ctrace` — three slashes, the plain URL reading of an
 *   absolute path;
 * - `sqlite:////var/traces/a.ctrace` — four slashes, which is how SQLAlchemy
 *   spells the SAME absolute path (it treats the first `/` as a separator). Both
 *   land on `/var/traces/a.ctrace`, because a leading run of slashes is collapsed
 *   — `//var` is never a different place from `/var`;
 * - `sqlite://rel/dir` — two slashes, where the first segment would parse as a
 *   host and is rejoined onto the path;
 * - `sqlite://~/traces` — `~` is expanded.
 *
 * Percent-escapes are decoded, so a path with a space or `#` survives. Parsed by
 * hand rather than with `URL` because `URL` lowercases the authority component,
 * which would silently rewrite a case-sensitive directory name.
 */
export function sqlitePath(dsn: string): string {
  const rest = dsn.slice(dsn.indexOf("://") + 3);
  // Strip a query/fragment the way a URL parser would, so `?mode=ro` never ends
  // up in a filename.
  const raw = rest.split(/[?#]/, 1)[0];
  const collapsed = raw.startsWith("//") ? "/" + raw.replace(/^\/+/, "") : raw;
  let decoded: string;
  try {
    decoded = decodeURIComponent(collapsed);
  } catch {
    decoded = collapsed; // a stray '%' is a literal, not an escape
  }
  return expandUser(decoded);
}

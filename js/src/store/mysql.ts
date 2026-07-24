/**
 * The MySQL/MariaDB backend — on `mysql2`, an OPTIONAL peer dependency.
 *
 *     import { configure, MySQLStore } from "ctxdiff";
 *     configure({ store: new MySQLStore({ dsn: "mysql://user:pw@host:3306/agents" }) });
 *
 * ...or `CTXDIFF_STORE=mysql://user:pw@host/agents`. Tables are created on first
 * connect if missing; there is no migration step.
 *
 * WHY `mysql2` over `mysql`: it is the maintained driver, it has a first-class
 * promise API (`mysql2/promise`) so the shared layer's `async` write path needs
 * no callback shimming, it supports `rowsAsArray` — which is what lets the
 * shared SELECTs read positionally, exactly like the Python port's tuples — and
 * it is pure JavaScript with no build step, so adding it to an existing agent
 * never drags in a compiler.
 *
 * The driver is imported lazily inside `connect()` so core stays
 * dependency-light and a missing peer becomes a clear install hint at connect
 * time (inside the tracer's fail-open guard) rather than an import crash. Its
 * types are described structurally here so neither the build nor the published
 * `.d.ts` depends on `mysql2` being installed.
 */
import {
  BLOCK_TABLE,
  CALL_BLOCK_TABLE,
  CALL_TABLE,
  DEFAULT_CONNECT_TIMEOUT,
  DEFAULT_STATEMENT_TIMEOUT,
  RUN_TABLE,
  SQLBackend,
  DetachableSocket,
  withDeadline,
  type Dialect,
  type SqlConn,
} from "./sql.js";

// MySQL server error codes worth one bounded retry: deadlock found, lock wait
// timeout (both routine under concurrent writers touching the same block rows),
// and the two "connection went away" codes a proxy/pooler can produce mid-write.
const TRANSIENT_ERRNOS = new Set([1213, 1205, 2006, 2013]);

// Errnos meaning THIS CONNECTION is gone, so the retry must reopen before it can
// succeed: 2006 (server has gone away — a restart, a `KILL`, or wait_timeout),
// 2013 (lost connection during the query), 2003/2002 (cannot connect at all,
// which is what a reconnect attempt itself reports).
const CONNECTION_LOST_ERRNOS = new Set([2002, 2003, 2006, 2013]);

// The same facts as `mysql2` spells them: it reports a dead peer with a string
// `code` far more often than with an errno, so both are matched.
const CONNECTION_LOST_CODES = new Set([
  "PROTOCOL_CONNECTION_LOST",
  "PROTOCOL_ENQUEUE_AFTER_QUIT",
  "PROTOCOL_ENQUEUE_AFTER_FATAL_ERROR",
  "PROTOCOL_ENQUEUE_HANDSHAKE_TWICE",
  "ECONNRESET",
  "EPIPE",
  "ETIMEDOUT",
  "ECONNREFUSED",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "ENOTFOUND",
]);

// Errnos raised when two clients create the same object at once. MySQL
// serializes DDL, so `CREATE TABLE IF NOT EXISTS` rarely collides this way and
// these mostly exist so the two adapters classify the same situations, rather
// than leaving MySQL a silent exception to the rule.
//
// What a concurrent MySQL cold start DOES produce is a plain DEADLOCK (1213) or
// lock wait timeout (1205) out of the DDL — observed on MySQL 9.x with several
// containers creating the tables at the same moment. That is NOT a schema race
// (nothing was created twice; the server picked a victim among concurrent
// transactions) and is deliberately kept out of this set: it is already
// `isTransient`, and `SQLBackend.ensureSchema` retries on either condition.
const SCHEMA_RACE_ERRNOS = new Set([1050, 1061, 1022, 1062]);

// Identifier lengths. Ids and content hashes are hex strings; VARCHAR(64) covers
// both with room to spare and — unlike TEXT — can be a PRIMARY KEY / FOREIGN KEY
// in MySQL without a prefix length, which is the crux of the DDL difference from
// Postgres. Free-text columns use LONGTEXT (a prompt can be megabytes).
const ID_LEN = 64;
// `started_at` is an ISO-8601 timestamp (~32 chars); the slack is for callers
// that pass their own spelling.
const TIMESTAMP_LEN = 64;

// Byte-exact comparison, pinned per column on every id/hash/timestamp.
//
// MySQL's default collation for utf8mb4 (`utf8mb4_0900_ai_ci` on 8.0+) is accent-
// and CASE-INSENSITIVE, which quietly changes what this schema MEANS rather than
// merely how it sorts:
//   - `content_hash` is the primary key of a CONTENT-ADDRESSED table. Under a
//     *_ci collation two hashes differing only in case are one key, so the second
//     block is deduped away and its call reads back the FIRST block's text —
//     corruption with no error anywhere.
//   - `started_at` compares by collation rather than by bytes, so an ISO-8601
//     string stops behaving like the byte-ordered value every other backend
//     treats it as.
// These columns are hex ids and ISO timestamps — ASCII by construction — so
// `ascii_bin` is both the narrowest and the most exact choice, and it makes MySQL
// match SQLite's BINARY and Postgres' C-locale TEXT. It is spelled on each
// COLUMN, not as a table default, so it cannot be changed out from under the
// schema by an `ALTER TABLE ... CONVERT TO CHARACTER SET`.
const BIN = "CHARACTER SET ascii COLLATE ascii_bin";

// TCP keepalive probing for a connection sitting IDLE between an agent's calls,
// so a silently-dead peer is noticed rather than discovered at the next write.
const KEEPALIVE_INITIAL_DELAY_MS = 10_000;

/** The `mysql2/promise` surface this module uses, described structurally so the
 * driver's types are never required to build or consume ctxdiff. */
interface My2Connection {
  query(sql: string, values?: unknown[]): Promise<[unknown, unknown]>;
  end(): Promise<void>;
  destroy(): void;
  on(event: "error" | "end", handler: (err?: unknown) => void): void;
  connection?: { stream?: { destroyed?: boolean; unref?(): void }; _closing?: boolean };
}
interface My2Module {
  createConnection(config: Record<string, unknown>): Promise<My2Connection>;
}

/** The MySQL server error code carried by a driver error, or null. `mysql2`
 * attaches `errno` for a server error; anything else (a bug in ctxdiff, a bad
 * value, a plain `TypeError`) has none and is classified by every caller as "not
 * retryable", which is what makes an unrecognized failure re-raise immediately
 * into the fail-open guard instead of being retried. */
function errnoOf(err: unknown): number | null {
  const errno = (err as { errno?: unknown } | null)?.errno;
  return typeof errno === "number" ? errno : null;
}

/** The driver's string `code` (`PROTOCOL_CONNECTION_LOST`, `ER_DUP_ENTRY`,
 * `ECONNRESET`), or null. */
function codeOf(err: unknown): string | null {
  const code = (err as { code?: unknown } | null)?.code;
  return typeof code === "string" ? code : null;
}

/**
 * Everything about MySQL that the shared SQL layer cannot assume.
 *
 * Types: MySQL cannot index a `TEXT` column without a prefix length, so every
 * column that is a primary key, foreign key or unique key must be a bounded
 * `VARCHAR(64)` — and ONLY those. Everything else is `LONGTEXT`, because a
 * bounded VARCHAR on a free-text column is not a size limit, it is a data loss:
 * under STRICT mode (the default since 5.7) a 400-character provider error, tag
 * label or agent name raises error 1406 and, since the call and its blocks are
 * written in ONE transaction, drops the WHOLE turn — on MySQL alone, where
 * SQLite and Postgres store it fine. `latency_ms` is `BIGINT` for the same
 * reason: `INT` milliseconds overflow after ~24 days. Tables are explicitly
 * `ENGINE=InnoDB` because the foreign keys and transactional writes below are
 * meaningless on MyISAM, and `CHARSET=utf8mb4` so prompts containing emoji/CJK
 * round-trip byte-exactly — with `ascii_bin` pinned on the id/hash/timestamp
 * columns (see `BIN`).
 *
 * Ordering: `insert_order` is an `AUTO_INCREMENT` column — MySQL's portable
 * equivalent of SQLite's implicit `rowid`, and the ONLY total order that
 * survives clocks disagreeing between the containers sharing this database.
 * MySQL requires an AUTO_INCREMENT column to be the first column of a key, which
 * a plain `UNIQUE KEY` satisfies without disturbing the `id` primary key.
 *
 * Dedup: `ON DUPLICATE KEY UPDATE content_hash = content_hash` — a no-op
 * assignment that makes a duplicate hash silently succeed. Deliberately NOT
 * `INSERT IGNORE`, which downgrades EVERY error (truncation, bad values, FK
 * violations) to a warning and would let a malformed block be silently
 * half-stored; this form no-ops on exactly the duplicate-key case and lets every
 * other error raise.
 */
export const MySQLDialect: Dialect = {
  name: "mysql",

  /** The `CREATE TABLE IF NOT EXISTS` statements, in dependency order (parents
   * first, since the child tables declare foreign keys), one per statement
   * because multi-statement scripts are disabled. Re-running them is a no-op,
   * which is what makes auto-create safe on every connect. */
  ddl(): string[] {
    return [
      `CREATE TABLE IF NOT EXISTS ${RUN_TABLE} (
  id              VARCHAR(${ID_LEN}) ${BIN} NOT NULL,
  insert_order    BIGINT NOT NULL AUTO_INCREMENT,
  project         LONGTEXT NOT NULL,
  started_at      VARCHAR(${TIMESTAMP_LEN}) ${BIN} NOT NULL,
  provider        LONGTEXT NOT NULL,
  models          LONGTEXT NOT NULL,
  ctxdiff_version LONGTEXT NOT NULL,
  schema_version  INT NOT NULL,
  UNIQUE KEY ctxdiff_run_insert_order (insert_order),
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,
      `CREATE TABLE IF NOT EXISTS ${CALL_TABLE} (
  id          VARCHAR(${ID_LEN}) ${BIN} NOT NULL,
  run_id      VARCHAR(${ID_LEN}) ${BIN} NOT NULL,
  seq         INT NOT NULL,
  params      LONGTEXT NOT NULL,
  usage_json  LONGTEXT,
  latency_ms  BIGINT,
  error       LONGTEXT,
  agent       LONGTEXT,
  step        LONGTEXT,
  provider    LONGTEXT,
  PRIMARY KEY (id),
  UNIQUE KEY ctxdiff_call_run_seq (run_id, seq),
  FOREIGN KEY (run_id) REFERENCES ${RUN_TABLE}(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,
      `CREATE TABLE IF NOT EXISTS ${BLOCK_TABLE} (
  content_hash VARCHAR(${ID_LEN}) ${BIN} NOT NULL,
  role         LONGTEXT NOT NULL,
  kind         LONGTEXT NOT NULL,
  body         LONGTEXT NOT NULL,
  token_count  INT NOT NULL,
  token_method LONGTEXT NOT NULL,
  PRIMARY KEY (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,
      `CREATE TABLE IF NOT EXISTS ${CALL_BLOCK_TABLE} (
  call_id      VARCHAR(${ID_LEN}) ${BIN} NOT NULL,
  block_id     VARCHAR(${ID_LEN}) ${BIN} NOT NULL,
  pos          INT NOT NULL,
  label        LONGTEXT NOT NULL,
  label_source LONGTEXT NOT NULL,
  PRIMARY KEY (call_id, pos),
  FOREIGN KEY (call_id) REFERENCES ${CALL_TABLE}(id),
  FOREIGN KEY (block_id) REFERENCES ${BLOCK_TABLE}(content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,
    ];
  },

  /** Insert a block, or no-op if this content hash is already stored — the
   * content-addressed dedup primitive, in MySQL syntax. See the dialect
   * docstring for why this and not `INSERT IGNORE`. */
  insertBlockDedup(): string {
    return (
      `INSERT INTO ${BLOCK_TABLE} ` +
      "(content_hash, role, kind, body, token_count, token_method) " +
      "VALUES (?, ?, ?, ?, ?, ?) " +
      "ON DUPLICATE KEY UPDATE content_hash = content_hash"
    );
  },

  /** True when a failed write is worth ONE bounded retry: the
   * deadlock/lock-wait/connection-lost codes. Anything without a recognizable
   * errno or code is NOT transient and re-raises immediately into the fail-open
   * guard. */
  isTransient(err: unknown): boolean {
    const errno = errnoOf(err);
    if (errno !== null && TRANSIENT_ERRNOS.has(errno)) return true;
    const code = codeOf(err);
    return code !== null && CONNECTION_LOST_CODES.has(code);
  },

  /** True when the error means the CONNECTION died rather than the statement
   * being refused — a `KILL`, a server restart, a pooler recycle, `wait_timeout`
   * — so the retry reopens before trying again instead of re-running against a
   * dead socket. */
  isConnectionLost(err: unknown): boolean {
    const errno = errnoOf(err);
    if (errno !== null && CONNECTION_LOST_ERRNOS.has(errno)) return true;
    const code = codeOf(err);
    if (code !== null && CONNECTION_LOST_CODES.has(code)) return true;
    const message = (err as { message?: unknown } | null)?.message;
    return (
      typeof message === "string" &&
      /can't add new command when connection is in closed state|connection lost|server has gone away/i.test(
        message,
      )
    );
  },

  /** True when the auto-create DDL collided with another client creating the
   * same object. See `SCHEMA_RACE_ERRNOS`: MySQL serializes DDL, so this is a
   * parity hook rather than a live code path. */
  isSchemaRace(err: unknown): boolean {
    const errno = errnoOf(err);
    return errno !== null && SCHEMA_RACE_ERRNOS.has(errno);
  },
};

/** A `mysql2` connection wrapped as the shared layer's `SqlConn`: array rows, a
 * hard client-side deadline per statement, and a liveness flag the retry path
 * can trust. */
class My2Conn implements SqlConn {
  private readonly conn: My2Connection;
  private readonly deadline: number;
  // Referenced while a statement is in flight, released when idle — see
  // `DetachableSocket` and `SqlConn.unref`.
  private readonly socket: DetachableSocket;
  private dead = false;

  constructor(conn: My2Connection, deadlineMs: number) {
    this.conn = conn;
    this.deadline = deadlineMs;
    this.socket = new DetachableSocket(() => conn.connection?.stream);
    // `mysql2` emits 'error' asynchronously when the server goes away; an
    // unhandled one would crash the HOST process, which is the opposite of
    // fail-open. Latching `dead` is also what makes the reopen path work after
    // the driver stops attaching codes to its rejections.
    conn.on("error", () => {
      this.dead = true;
    });
    conn.on("end", () => {
      this.dead = true;
    });
  }

  async query(sql: string, params: unknown[] = []): Promise<unknown[][]> {
    const work = this.conn.query(sql, params);
    const [rows] = await this.socket.during(
      withDeadline(work, this.deadline, "mysql query", () => this.destroy()),
    );
    // A DDL/INSERT/UPDATE resolves to a ResultSetHeader, not rows; normalizing
    // to an empty list here means callers never branch on statement kind.
    return Array.isArray(rows) ? (rows as unknown[][]) : [];
  }

  isDead(): boolean {
    return (
      this.dead ||
      this.conn.connection?._closing === true ||
      this.conn.connection?.stream?.destroyed === true
    );
  }

  /** Release the socket whenever this connection is idle — see `SqlConn.unref`
   * and `DetachableSocket`. */
  unref(): void {
    this.socket.detach();
  }

  /** Rip the socket down without waiting for a protocol-level goodbye — used
   * when a deadline fires, precisely because the server is not answering and a
   * graceful `end()` would wait for a reply that is never coming. */
  private destroy(): void {
    this.dead = true;
    try {
      this.conn.destroy();
    } catch {
      /* already gone */
    }
  }

  async close(): Promise<void> {
    this.dead = true;
    try {
      // Bounded: `end()` sends COM_QUIT and waits, which a wedged server never
      // answers — and close() must not be the thing that hangs a shutdown.
      await this.socket.during(
        withDeadline(this.conn.end(), this.deadline, "mysql close", () => this.destroy()),
      );
    } catch {
      /* closing is best-effort on the way out */
    }
  }
}

/** Options for `MySQLStore`. A DSN, individual fields, or both — explicit fields
 * win, so a DSN from the environment can be overridden in code without string
 * surgery. */
export interface MySQLStoreOptions {
  dsn?: string;
  host?: string;
  port?: number;
  user?: string;
  password?: string;
  database?: string;
  /** Seconds for the TCP connect/handshake (default 5). */
  connectTimeout?: number;
  /** Seconds any single statement may run (default 10). */
  statementTimeout?: number;
}

/**
 * Points ctxdiff at a MySQL/MariaDB database. Connection-less until used, so
 * `configure({ store: new MySQLStore({ dsn }) })` at import time is free and an
 * unreachable server surfaces inside the tracer's fail-open guard.
 */
export class MySQLStore extends SQLBackend {
  readonly dialect = MySQLDialect;
  readonly dsn: string | null;
  readonly connectTimeout: number;
  readonly statementTimeout: number;
  private readonly overrides: Omit<
    MySQLStoreOptions,
    "dsn" | "connectTimeout" | "statementTimeout"
  >;

  /**
   * Record how to connect, plus the two bounds that stop a slow/dead server from
   * ever delaying the host: `connectTimeout` seconds for the TCP
   * connect/handshake, and `statementTimeout` seconds for any single statement —
   * applied BOTH server-side (`max_execution_time`) and, plus a margin, as the
   * client-side deadline, so a server that accepts a connection and then goes
   * silent cannot wedge the writer. Nothing is connected or validated here.
   */
  constructor(opts: MySQLStoreOptions = {}) {
    super();
    this.dsn = opts.dsn ?? null;
    this.overrides = {
      host: opts.host,
      port: opts.port,
      user: opts.user,
      password: opts.password,
      database: opts.database,
    };
    this.connectTimeout = opts.connectTimeout ?? DEFAULT_CONNECT_TIMEOUT;
    this.statementTimeout = opts.statementTimeout ?? DEFAULT_STATEMENT_TIMEOUT;
  }

  /**
   * Resolve the DSN and the explicit fields into driver connect arguments. How:
   * parses the URL (percent-decoding user/password, which routinely contain `@`
   * or `/`), applies sane defaults (localhost:3306), then overlays any
   * explicitly-passed field so code beats the DSN. Public because it is exactly
   * what a test wants to assert on without opening a connection.
   */
  connectOptions(): {
    host: string;
    port: number;
    user?: string;
    password?: string;
    database?: string;
  } {
    let parsed: URL | null = null;
    if (this.dsn) {
      try {
        parsed = new URL(this.dsn);
      } catch {
        parsed = null; // not a URL: fall back to the explicit fields below
      }
    }
    const resolved = {
      host: parsed?.hostname || "localhost",
      port: parsed?.port ? Number(parsed.port) : 3306,
      user: parsed?.username ? decodeURIComponent(parsed.username) : undefined,
      password: parsed?.password ? decodeURIComponent(parsed.password) : undefined,
      database: parsed?.pathname ? parsed.pathname.replace(/^\//, "") || undefined : undefined,
    };
    for (const [key, value] of Object.entries(this.overrides)) {
      if (value !== undefined) (resolved as Record<string, unknown>)[key] = value;
    }
    return resolved;
  }

  /**
   * Open one bounded `mysql2` connection.
   *
   * How: imports the driver lazily (missing peer -> actionable install hint, not
   * an import crash), connects with `connectTimeout` so an unreachable host
   * fails in seconds, keeps autocommit at the server default with explicit
   * `BEGIN`/`COMMIT` so `recordCall` is one real transaction, forces utf8mb4 so
   * prompt text round-trips byte-exactly, asks for positional rows
   * (`rowsAsArray`) so the shared SELECTs read the same way they do on Postgres,
   * and enables TCP keepalive so a silently-dead peer is noticed while idle.
   *
   * Statements are bounded twice. Server-side `max_execution_time` is applied
   * best-effort: it exists on MySQL 5.7.8+ but not on MariaDB (which spells it
   * `max_statement_time`, in seconds), so an unknown-variable error there is
   * swallowed rather than failing an otherwise good connection. The real
   * guarantee is the CLIENT-side deadline in `My2Conn`, because a server-side
   * bound can never fire when the packets carrying its verdict are the ones
   * being dropped.
   */
  protected async connect(): Promise<SqlConn> {
    const mysql = await loadMysql2();
    const options = this.connectOptions();
    // If the deadline wins, whatever the driver eventually hands back must be
    // destroyed: it would be a live, REFERENCED socket that nothing owns, never
    // closed and never released — exactly the handle that keeps a finished
    // process alive forever.
    let abandoned = false;
    const connecting = mysql
      .createConnection({
        ...options,
        connectTimeout: this.connectTimeout * 1000,
        charset: "utf8mb4",
        rowsAsArray: true,
        multipleStatements: false,
        enableKeepAlive: true,
        keepAliveInitialDelay: KEEPALIVE_INITIAL_DELAY_MS,
      })
      .then((connection) => {
        if (abandoned) {
          try {
            connection.destroy();
          } catch {
            /* already gone */
          }
        }
        return connection;
      });
    const raw = await withDeadline(
      connecting,
      (this.connectTimeout + 1) * 1000,
      "mysql connect",
      () => {
        abandoned = true;
      },
    );
    const conn = new My2Conn(raw, this.deadlineMs());
    try {
      await conn.query(`SET SESSION max_execution_time = ${this.statementTimeout * 1000}`);
    } catch {
      /* MariaDB / old MySQL: no such variable — never fail a good connection
         over a knob whose job the client-side deadline already does */
    }
    return conn;
  }
}

/**
 * Import `mysql2/promise` lazily, turning a missing optional peer dependency
 * into an actionable one-line install hint instead of a module-resolution crash
 * at `import "ctxdiff"`. The error surfaces at CONNECT time, which is inside the
 * tracer's fail-open guard, so a user who configured MySQL without installing
 * the driver gets a warning and an untouched host program.
 */
async function loadMysql2(): Promise<My2Module> {
  try {
    const mod = (await import("mysql2/promise")) as unknown as {
      default?: My2Module;
    } & My2Module;
    return mod.default ?? mod;
  } catch (err) {
    throw new Error(
      "ctxdiff: MySQLStore needs the 'mysql2' driver — install it with " +
        "`npm install mysql2`",
      { cause: err },
    );
  }
}

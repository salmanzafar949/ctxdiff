/**
 * Backend behavior that needs NO server: the dialects' DDL and SQL, their error
 * classification, and — the part that matters most — the promise that a
 * networked store can never delay, freeze or hang the host.
 *
 * The bounding tests here use a REAL TCP server that accepts connections and
 * then says nothing ever again (a "blackhole": a wedged box, a hung pooler, a
 * network partition). That is the case no driver knob covers on its own — a
 * connect timeout ends at the handshake, and a SERVER-side statement timeout
 * cannot fire when the packets carrying its verdict are the ones being dropped —
 * so it is the case worth proving locally rather than only against a live
 * database.
 */
import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { createServer, type Server, type Socket } from "node:net";
import type { AddressInfo } from "node:net";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { init } from "../src/trace.js";
import { configure, ENV_VAR } from "../src/store/config.js";
import { PostgresDialect, PostgresStore, toDollarPlaceholders } from "../src/store/postgres.js";
import { MySQLDialect, MySQLStore } from "../src/store/mysql.js";
import {
  DeadlineExceededError,
  SQLBackend,
  withDeadline,
  type Dialect,
  type SqlConn,
} from "../src/store/sql.js";
import { EmptyStoreError } from "../src/store/base.js";
import { snapshotStore } from "../src/store/snapshot.js";
import type {
  Call,
  CallBlock,
  OpenSessionArgs,
  RecordCallArgs,
  Run,
  Session,
  Store,
  StoreBackend,
} from "../src/store/base.js";

// --- helpers ------------------------------------------------------------------

const cleanups: (() => void)[] = [];
const dirs: string[] = [];

beforeEach(() => {
  configure();
  delete process.env[ENV_VAR];
});

afterEach(() => {
  configure();
  delete process.env[ENV_VAR];
  for (const fn of cleanups.splice(0)) fn();
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

function tempDir(): string {
  const d = mkdtempSync(join(tmpdir(), "ctxdiff-backends-"));
  dirs.push(d);
  return d;
}

/** A TCP server that ACCEPTS and then goes silent forever — the failure mode a
 * connect timeout cannot see and a server-side statement timeout cannot report.
 * Sockets are kept referenced so nothing is closed early, and torn down in
 * `afterEach`. */
async function blackhole(): Promise<number> {
  const sockets: Socket[] = [];
  const server: Server = createServer((socket) => {
    sockets.push(socket);
    socket.on("error", () => undefined);
    // Deliberately no reads, no writes, no close: the peer simply never answers.
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  cleanups.push(() => {
    for (const s of sockets) s.destroy();
    server.close();
  });
  return port;
}

/** A port nothing is listening on — connection REFUSED, the "database is simply
 * not there" case (as opposed to the blackhole's "there but mute"). */
async function closedPort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}

/** The minimal OpenAI-shaped client the tracer knows how to wrap. */
function stubClient(): object {
  return {
    constructor: { name: "OpenAI" },
    chat: {
      completions: {
        create: async (): Promise<unknown> => ({
          choices: [{ message: { role: "assistant", content: "ok" } }],
          usage: { prompt_tokens: 5, completion_tokens: 2 },
        }),
      },
    },
    responses: {},
  };
}

/** Count event-loop ticks while `fn` runs. A frozen loop registers ~0 ticks; a
 * healthy one registers many, which is the difference between "the host is
 * waiting on a promise" and "the host is BLOCKED". */
async function ticksDuring(fn: () => Promise<void>): Promise<number> {
  let ticks = 0;
  const timer = setInterval(() => {
    ticks += 1;
  }, 2);
  try {
    await fn();
  } finally {
    clearInterval(timer);
  }
  return ticks;
}

// --- dialects -----------------------------------------------------------------

describe("Postgres dialect", () => {
  const ddl = PostgresDialect.ddl().join("\n");

  it("creates four tables, all IF NOT EXISTS, parents first", () => {
    const statements = PostgresDialect.ddl();
    expect(statements.length).toBe(4);
    expect(statements.every((s) => s.includes("CREATE TABLE IF NOT EXISTS"))).toBe(true);
    // ctxdiff_run must be created before the table whose FK references it.
    expect(ddl.indexOf("ctxdiff_run (")).toBeLessThan(ddl.indexOf("ctxdiff_call ("));
    expect(ddl.indexOf("ctxdiff_block (")).toBeLessThan(ddl.indexOf("ctxdiff_call_block ("));
  });

  it("gives sessions a monotonic insert_order and widens latency_ms to BIGINT", () => {
    // Sessions order by insert_order, never by the clock — see store/sql.ts.
    expect(ddl).toContain("insert_order    BIGSERIAL NOT NULL UNIQUE");
    // INTEGER milliseconds overflow after ~24 days, and an overflow REJECTS the
    // whole call rather than storing an odd number.
    expect(ddl).toContain("latency_ms  BIGINT");
  });

  it("dedups blocks with ON CONFLICT on the content hash ALONE", () => {
    const sql = PostgresDialect.insertBlockDedup();
    expect(sql).toContain("ON CONFLICT (content_hash) DO NOTHING");
    // Naming the column means any OTHER constraint violation still raises,
    // which is what content-addressed dedup should mean.
    expect(sql).not.toContain("DO UPDATE");
  });

  it("rewrites the shared `?` placeholders to $1..$n in order", () => {
    expect(toDollarPlaceholders("INSERT INTO t VALUES (?, ?, ?)")).toBe(
      "INSERT INTO t VALUES ($1, $2, $3)",
    );
    expect(toDollarPlaceholders("SELECT 1")).toBe("SELECT 1");
  });

  it("classifies transient, connection-lost and schema-race errors", () => {
    expect(PostgresDialect.isTransient({ code: "40001" })).toBe(true); // serialization
    expect(PostgresDialect.isTransient({ code: "40P01" })).toBe(true); // deadlock
    expect(PostgresDialect.isTransient({ code: "08006" })).toBe(true); // connection class
    expect(PostgresDialect.isTransient({ code: "23505" })).toBe(false); // a real violation
    expect(PostgresDialect.isTransient(new TypeError("bug"))).toBe(false);

    expect(PostgresDialect.isConnectionLost({ code: "57P01" })).toBe(true); // terminated
    expect(PostgresDialect.isConnectionLost({ code: "08003" })).toBe(true);
    expect(PostgresDialect.isConnectionLost({ code: "ECONNRESET" })).toBe(true);
    // The driver stops attaching codes after the first post-kill failure — this
    // message shape is what every LATER statement actually rejects with.
    expect(
      PostgresDialect.isConnectionLost(
        new Error("Client has encountered a connection error and is not queryable"),
      ),
    ).toBe(true);
    expect(PostgresDialect.isConnectionLost({ code: "23505" })).toBe(false);

    // `pg`'s OWN timers reject with a bare Error carrying no code at all —
    // `query_timeout` with "Query read timeout", the connect timeout with
    // "timeout expired". The read timeout in particular leaves the client
    // BUSY (the query is off the queue but still active on the wire), so a
    // connection not classified as lost here is one ctxdiff would keep using:
    // every following statement rejects with "already executing a query" and
    // burns its full deadline doing it.
    expect(PostgresDialect.isConnectionLost(new Error("Query read timeout"))).toBe(true);
    expect(PostgresDialect.isConnectionLost(new Error("timeout expired"))).toBe(true);
    // A SERVER-side statement_timeout (57014) is a different animal: the
    // connection is fine and the statement was cancelled, so it stays classified
    // as a statement failure, not a lost connection.
    expect(PostgresDialect.isConnectionLost({ code: "57014" })).toBe(false);

    // The cold-start catalog race, and ONLY from the schema path.
    for (const code of ["23505", "42P07", "42710"]) {
      expect(PostgresDialect.isSchemaRace({ code })).toBe(true);
    }
    expect(PostgresDialect.isSchemaRace({ code: "40001" })).toBe(false);
  });
});

describe("MySQL dialect", () => {
  const ddl = MySQLDialect.ddl().join("\n");

  it("pins ascii_bin on every id/hash/timestamp column", () => {
    // The default utf8mb4 collation is case-INSENSITIVE: two content hashes
    // differing only in case would collapse into one primary key, and the second
    // block would silently read back the first one's text.
    for (const column of [
      "id              VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin",
      "started_at      VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin",
      "run_id      VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin",
      "content_hash VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin",
      "call_id      VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin",
      "block_id     VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin",
    ]) {
      expect(ddl, column).toContain(column);
    }
  });

  it("keeps free text unbounded (LONGTEXT) so STRICT mode cannot drop a turn", () => {
    // A bounded VARCHAR on a free-text column is data loss, not a size limit.
    for (const column of ["error", "agent", "step", "provider", "label", "label_source", "body"]) {
      expect(ddl, column).toMatch(new RegExp(`${column}\\s+LONGTEXT`));
    }
    expect(ddl).toContain("latency_ms  BIGINT");
    // Only the indexable key columns are bounded.
    expect(ddl).not.toMatch(/error\s+VARCHAR/);
  });

  it("uses InnoDB + utf8mb4 and an AUTO_INCREMENT insert_order key", () => {
    expect(MySQLDialect.ddl().every((s) => s.includes("ENGINE=InnoDB"))).toBe(true);
    expect(MySQLDialect.ddl().every((s) => s.includes("CHARSET=utf8mb4"))).toBe(true);
    expect(ddl).toContain("insert_order    BIGINT NOT NULL AUTO_INCREMENT");
    expect(ddl).toContain("UNIQUE KEY ctxdiff_run_insert_order (insert_order)");
  });

  it("dedups with ON DUPLICATE KEY UPDATE, never INSERT IGNORE", () => {
    const sql = MySQLDialect.insertBlockDedup();
    expect(sql).toContain("ON DUPLICATE KEY UPDATE content_hash = content_hash");
    // INSERT IGNORE downgrades EVERY error (truncation, bad values, FK
    // violations) to a warning and would half-store a malformed block.
    expect(sql).not.toContain("INSERT IGNORE");
  });

  it("classifies transient, connection-lost and schema-race errors", () => {
    expect(MySQLDialect.isTransient({ errno: 1213 })).toBe(true); // deadlock
    expect(MySQLDialect.isTransient({ errno: 1205 })).toBe(true); // lock wait
    expect(MySQLDialect.isTransient({ code: "PROTOCOL_CONNECTION_LOST" })).toBe(true);
    expect(MySQLDialect.isTransient({ errno: 1406 })).toBe(false); // truncation
    expect(MySQLDialect.isTransient(new TypeError("bug"))).toBe(false);

    expect(MySQLDialect.isConnectionLost({ errno: 2006 })).toBe(true);
    expect(MySQLDialect.isConnectionLost({ errno: 2013 })).toBe(true);
    expect(MySQLDialect.isConnectionLost({ code: "ECONNRESET" })).toBe(true);
    // What mysql2 actually rejects with once the socket is gone.
    expect(
      MySQLDialect.isConnectionLost(
        new Error("Can't add new command when connection is in closed state"),
      ),
    ).toBe(true);
    expect(MySQLDialect.isConnectionLost({ errno: 1213 })).toBe(false);

    for (const errno of [1050, 1061, 1022, 1062]) {
      expect(MySQLDialect.isSchemaRace({ errno })).toBe(true);
    }
    expect(MySQLDialect.isSchemaRace({ errno: 1213 })).toBe(false);
  });

  it("parses a DSN into connect options, with explicit fields winning", () => {
    const store = new MySQLStore({ dsn: "mysql://u:p@host:3307/db", user: "override" });
    expect(store.connectOptions()).toMatchObject({
      host: "host",
      port: 3307,
      user: "override",
      password: "p",
      database: "db",
    });
    // No DSN at all: sane localhost defaults.
    expect(new MySQLStore().connectOptions()).toMatchObject({ host: "localhost", port: 3306 });
  });
});

// --- bounding -----------------------------------------------------------------

describe("withDeadline", () => {
  it("rejects with DeadlineExceededError and runs the expiry hook", async () => {
    let expired = false;
    const never = new Promise<void>(() => undefined);
    await expect(
      withDeadline(never, 20, "test", () => {
        expired = true;
      }),
    ).rejects.toBeInstanceOf(DeadlineExceededError);
    expect(expired).toBe(true);
  });

  it("passes a value through untouched when the work wins", async () => {
    await expect(withDeadline(Promise.resolve(7), 1000, "test")).resolves.toBe(7);
  });

  it("contains a LATE rejection from the abandoned work (no unhandled rejection)", async () => {
    // A socket that errors AFTER we gave up waiting on it must not crash the
    // host process, so the abandoned promise gets a no-op handler. Asserted by
    // listening for the process-level event directly rather than trusting the
    // test runner to notice.
    const caught: unknown[] = [];
    const listener = (err: unknown): void => {
      caught.push(err);
    };
    process.on("unhandledRejection", listener);
    try {
      let reject!: (err: Error) => void;
      const late = new Promise<void>((_, r) => {
        reject = r;
      });
      await expect(withDeadline(late, 10, "test")).rejects.toBeInstanceOf(DeadlineExceededError);
      reject(new Error("socket died later"));
      // Unhandled rejections are reported on a later turn of the loop.
      await new Promise((r) => setTimeout(r, 50));
      expect(caught).toEqual([]);
    } finally {
      process.off("unhandledRejection", listener);
    }
  });
});

describe("a wedged database never reaches the host", () => {
  it("wrap() does NO I/O and returns immediately against a blackholed server", async () => {
    const port = await blackhole();
    configure({
      store: new PostgresStore({
        dsn: `postgresql://u:p@127.0.0.1:${port}/db`,
        connectTimeout: 1,
        statementTimeout: 1,
      }),
    });
    const tracer = init("wedged");
    const started = performance.now();
    const client = tracer.wrap(stubClient());
    const elapsed = performance.now() - started;
    // wrap() constructs a recipe and an empty queue. Nothing else.
    expect(elapsed).toBeLessThan(50);
    expect(client).not.toBe(null);
    await tracer.close();
  });

  it("the host call completes at full speed and the event loop keeps ticking", async () => {
    const port = await blackhole();
    configure({
      store: new PostgresStore({
        dsn: `postgresql://u:p@127.0.0.1:${port}/db`,
        connectTimeout: 1,
        statementTimeout: 1,
      }),
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const tracer = init("wedged");
    const client = tracer.wrap(stubClient()) as {
      chat: { completions: { create(a: unknown): Promise<{ choices: { message: { content: string } }[] }> } };
    };

    let result: string | null = null;
    // The window spans the host call AND the 300ms that follow, during which the
    // store's connect to the mute peer is still pending — precisely when a
    // blocking implementation would have the loop frozen.
    const ticks = await ticksDuring(async () => {
      const started = performance.now();
      const res = await client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      });
      result = res.choices[0].message.content;
      // The host's own call must not have waited on the database at all.
      expect(performance.now() - started).toBeLessThan(100);
      await new Promise((r) => setTimeout(r, 300));
    });

    expect(result).toBe("ok");
    // A frozen loop would register ~0 ticks in 300ms; a healthy one, ~150.
    expect(ticks).toBeGreaterThan(50);
    await tracer.close();
    warn.mockRestore();
  });

  it("close() is BOUNDED against a blackholed server (never hangs)", async () => {
    const port = await blackhole();
    configure({
      store: new PostgresStore({
        dsn: `postgresql://u:p@127.0.0.1:${port}/db`,
        connectTimeout: 1,
        statementTimeout: 1,
      }),
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const tracer = init("wedged");
    const client = tracer.wrap(stubClient()) as {
      chat: { completions: { create(a: unknown): Promise<unknown> } };
    };
    await client.chat.completions.create({ model: "m", messages: [] });

    const started = performance.now();
    let ticks = 0;
    const timer = setInterval(() => {
      ticks += 1;
    }, 2);
    await tracer.close();
    clearInterval(timer);
    const elapsed = performance.now() - started;

    // The bound is the store's own: connect (1s + 1s slack) then the close
    // budget (statementTimeout + 2s). Comfortably under ten seconds, and NOT
    // "forever", which is what an unbounded await against a mute peer means.
    expect(elapsed).toBeLessThan(10_000);
    expect(ticks).toBeGreaterThan(0); // the loop ran the whole time
    warn.mockRestore();
  }, 20_000);
});

// --- snapshotting one run out of a shared database ----------------------------

/** A `Store` that counts what was asked of it — enough to prove a snapshot's
 * round trips scale with the RUN and not with the database it lives in. */
class CountingStore implements Store {
  readonly calls = { getRun: 0, getCalls: 0, getCallBlocks: 0, listSessions: 0, close: 0 };
  /** The session ids the snapshot asked for, so `--session` reaching a network
   * store can be asserted rather than assumed. */
  readonly asked: { getRun: (string | undefined)[]; getCalls: (string | undefined)[] } = {
    getRun: [],
    getCalls: [],
  };
  async recordCall(): Promise<string> {
    return "c";
  }
  async noteModel(): Promise<void> {}
  async listSessions(): Promise<Session[]> {
    this.calls.listSessions += 1;
    return [
      {
        id: "r1",
        project: "p",
        startedAt: "2026-01-01",
        provider: "openai",
        models: [],
        agents: [],
        turnCount: 1,
      },
    ];
  }
  async getRun(sessionId?: string): Promise<Run> {
    this.calls.getRun += 1;
    this.asked.getRun.push(sessionId);
    return {
      id: sessionId ?? "r1",
      project: "p",
      startedAt: "2026-01-01",
      provider: "openai",
      models: [],
      ctxdiffVersion: "0.1.0",
    };
  }
  async getCalls(sessionId?: string): Promise<Call[]> {
    this.calls.getCalls += 1;
    this.asked.getCalls.push(sessionId);
    return [
      {
        id: "call-1",
        runId: sessionId ?? "r1",
        seq: 1,
        params: {},
        usage: null,
        latencyMs: null,
        error: null,
        agent: null,
        step: null,
        provider: null,
      },
    ];
  }
  async getCallBlocks(): Promise<CallBlock[]> {
    this.calls.getCallBlocks += 1;
    return [];
  }
  async close(): Promise<void> {
    this.calls.close += 1;
  }
}

describe("snapshotStore reads the RUN, not the database", () => {
  it("skips the session listing unless it is asked for", async () => {
    // `listSessions()` is a COUNT and a GROUP BY over every call row in the
    // database — a shared database, which is the entire reason these backends
    // exist. Taking it on every snapshot made `diff`, `tokens`, `cache` and
    // `export` — all of which analyze exactly ONE run — pay for every run
    // anybody else had ever recorded: measured at 2ms for the run itself
    // against 202ms for the listing at 2001 sessions / 200k calls, and growing
    // with the database forever. Only `ctxdiff runs` needs it, and it has its
    // own path to it.
    const store = new CountingStore();
    const snapshot = await snapshotStore(store);
    expect(store.calls.listSessions).toBe(0);
    // Everything the analyzers actually read is still there, in a fixed number
    // of round trips: one run, one call list, one block list per call.
    expect(snapshot.getRun().id).toBe("r1");
    expect(snapshot.getCalls().map((c) => c.seq)).toEqual([1]);
    expect(store.calls).toMatchObject({ getRun: 1, getCalls: 1, getCallBlocks: 1, close: 1 });
    // And asking for it anyway says so, rather than answering "no sessions".
    expect(() => snapshot.listSessions()).toThrow(/sessions: true/);
  });

  it("still materializes the session list when asked", async () => {
    const store = new CountingStore();
    const snapshot = await snapshotStore(store, { sessions: true });
    expect(store.calls.listSessions).toBe(1);
    expect(snapshot.listSessions().map((s) => s.id)).toEqual(["r1"]);
  });

  it("snapshots a CHOSEN session, not just the handle's binding", async () => {
    // How `--session` reaches a network store: a `CTrace` can be pinned with a
    // forwarding view, but a snapshot is materialized once and cannot be
    // re-pointed — so the session id has to travel INTO the reads.
    const store = new CountingStore();
    const snapshot = await snapshotStore(store, { sessionId: "chosen" });
    expect(store.asked).toEqual({ getRun: ["chosen"], getCalls: ["chosen"] });
    expect(snapshot.getRun().id).toBe("chosen");
  });

  it("reuses a session list the caller already fetched", async () => {
    // The CLI must list sessions to resolve `--session` at all; passing that
    // list in means the expensive listing query is paid once per command, not
    // twice.
    const store = new CountingStore();
    const sessionList = await store.listSessions();
    store.calls.listSessions = 0;
    const snapshot = await snapshotStore(store, { sessionId: "r1", sessionList });
    expect(store.calls.listSessions).toBe(0); // not re-queried
    expect(snapshot.listSessions()).toBe(sessionList);
  });

  it("closes the store even when the read fails", async () => {
    const store = new CountingStore();
    store.getCalls = async (): Promise<Call[]> => {
      throw new Error("read failed");
    };
    await expect(snapshotStore(store)).rejects.toThrow("read failed");
    expect(store.calls.close).toBe(1);
  });
});

// --- reading with a role that cannot CREATE -----------------------------------

/** A scripted `SqlConn`: answers from `responses` (first matching prefix wins)
 * and records every statement, so a test can make exactly one kind of statement
 * fail the way a server would. */
class ScriptedConn implements SqlConn {
  readonly seen: string[] = [];
  constructor(private readonly responses: [RegExp, unknown[][] | Error][]) {}
  async query(sql: string): Promise<unknown[][]> {
    this.seen.push(sql);
    for (const [pattern, answer] of this.responses) {
      if (pattern.test(sql)) {
        if (answer instanceof Error) throw answer;
        return answer;
      }
    }
    return [];
  }
  isDead(): boolean {
    return false;
  }
  unref(): void {}
  async close(): Promise<void> {}
}

/** A `SQLBackend` over a `ScriptedConn` — everything the base class does
 * (ensureSchema, the version gate, binding the newest session) with no server. */
class ScriptedBackend extends SQLBackend {
  readonly dialect: Dialect;
  readonly statementTimeout = 10;
  constructor(
    readonly conn: ScriptedConn,
    dialect: Dialect = PostgresDialect,
  ) {
    super();
    this.dialect = dialect;
  }
  protected async connect(): Promise<SqlConn> {
    return this.conn;
  }
}

/** What Postgres actually raises when a SELECT-only role runs CREATE TABLE. */
function permissionDenied(): Error {
  return Object.assign(new Error("permission denied for schema public"), { code: "42501" });
}

describe("a read-only role can still READ", () => {
  it("opens a reader when ensureSchema is refused but the tables are there", async () => {
    // `openReader()` runs `ensureSchema()` first so that reading a database
    // ctxdiff has never written says "no sessions" instead of "no such table".
    // But a SELECT-only role — the RIGHT role to hand an analyst pointing
    // `ctxdiff diff` at production — cannot CREATE, so that convenience turned
    // into "permission denied for schema public" on a database whose ctxdiff
    // tables were sitting right there, with nothing in the message hinting that
    // ctxdiff had tried to create anything.
    const conn = new ScriptedConn([
      [/^CREATE TABLE/, permissionDenied()],
      [/FROM ctxdiff_run ORDER BY insert_order DESC/, [["run-1", 2]]],
      [/^SELECT id, project/, [["run-1", "p", "2026-01-01", "openai", "[]", "0.1.0"]]],
    ]);
    const reader = await new ScriptedBackend(conn).openReader();
    try {
      // It tried to create, was refused, and read anyway.
      expect(conn.seen.some((s) => s.startsWith("CREATE TABLE"))).toBe(true);
      expect((await reader.getRun()).id).toBe("run-1");
    } finally {
      await reader.close();
    }
  });

  it("names BOTH causes when the read fails too", async () => {
    // The refusal is only ignorable while the read works. When it does not, the
    // error has to say what ctxdiff tried, what the database said to each half,
    // and what the operator can do — not just re-raise whichever half failed
    // last.
    const conn = new ScriptedConn([
      [/^CREATE TABLE/, permissionDenied()],
      [
        /FROM ctxdiff_run/,
        Object.assign(new Error('relation "ctxdiff_run" does not exist'), { code: "42P01" }),
      ],
    ]);
    await expect(new ScriptedBackend(conn).openReader()).rejects.toThrow(
      /permission denied for schema public[\s\S]*ctxdiff_run" does not exist/,
    );
    await expect(new ScriptedBackend(conn).openReader()).rejects.toThrow(/SELECT/);
  });

  it("still reports a plain read failure on its own when creating SUCCEEDED", async () => {
    // No schema failure to blame: the error must stay exactly what the database
    // said, or a real bug would be buried under a permissions story.
    const conn = new ScriptedConn([
      [/FROM ctxdiff_run/, Object.assign(new Error("boom"), { code: "XX000" })],
    ]);
    await expect(new ScriptedBackend(conn).openReader()).rejects.toThrow(/^boom$/);
  });

  it("keeps the version gate and the empty-store signal intact", async () => {
    const newer = new ScriptedConn([
      [/^CREATE TABLE/, permissionDenied()],
      [/FROM ctxdiff_run ORDER BY insert_order DESC/, [["run-1", 99]]],
    ]);
    await expect(new ScriptedBackend(newer).openReader()).rejects.toThrow(
      /schema version 99 is newer/,
    );
    const empty = new ScriptedConn([[/^CREATE TABLE/, permissionDenied()]]);
    await expect(new ScriptedBackend(empty).openReader()).rejects.toBeInstanceOf(EmptyStoreError);
  });
});

// --- the auto-create DDL failing transiently ----------------------------------

/** A `ScriptedConn` whose `CREATE TABLE` statements fail the first `failures`
 * times they are attempted and succeed afterwards — one client's view of a
 * server that transiently refuses the auto-create DDL, which is what several
 * containers cold-starting against the same empty database produces. */
class FlakyDDLConn extends ScriptedConn {
  /** Every `CREATE TABLE` attempted, failed ones included, so a test can tell a
   * retry apart from a give-up. */
  ddlAttempts = 0;
  private failuresLeft: number;
  constructor(
    failures: number,
    private readonly error: Error,
  ) {
    super([]);
    this.failuresLeft = failures;
  }
  async query(sql: string): Promise<unknown[][]> {
    const rows = await super.query(sql); // records into `seen` either way
    if (sql.startsWith("CREATE TABLE")) {
      this.ddlAttempts += 1;
      if (this.failuresLeft > 0) {
        this.failuresLeft -= 1;
        throw this.error;
      }
    }
    return rows;
  }
}

/** What MySQL raises when it picks this transaction as the victim — seen from
 * `CREATE TABLE IF NOT EXISTS` on a concurrent cold start, not just from writes. */
function deadlock(): Error {
  return Object.assign(
    new Error("Deadlock found when trying to get lock; try restarting transaction"),
    { errno: 1213, code: "ER_LOCK_DEADLOCK", sqlState: "40001" },
  );
}

describe("ensureSchema survives a TRANSIENT failure of the auto-create DDL", () => {
  it("retries a MySQL deadlock (1213) and opens the session", async () => {
    // A deadlock out of `CREATE TABLE IF NOT EXISTS` is not a schema race — the
    // server picked a victim among concurrent transactions and nothing is wrong
    // with the schema — so `isSchemaRace` does not (and should not) cover it.
    // Un-retried it cost a whole run's capture on a live MySQL 9.x cold start,
    // which is exactly what the catalog-race retry exists to prevent, so the
    // retry predicate has to admit `isTransient` too.
    const conn = new FlakyDDLConn(1, deadlock());
    const store = await new ScriptedBackend(conn, MySQLDialect).openSession({
      project: "p",
      provider: "openai",
    });
    try {
      // It retried the DDL rather than propagating: the failed statement plus a
      // full clean pass of all four tables.
      expect(conn.ddlAttempts).toBe(5);
      expect(conn.seen.some((s) => s.startsWith("INSERT INTO ctxdiff_run"))).toBe(true);
    } finally {
      await store.close();
    }
  });

  it("does NOT retry a genuinely non-retryable DDL failure", async () => {
    // The widened predicate must stay a predicate: a permission error is not
    // transient and not a race, so it propagates on the FIRST attempt — no
    // pointless retries, and the operator still gets the real message.
    const denied = Object.assign(new Error("CREATE command denied to user 'ro'@'%'"), {
      errno: 1142,
      code: "ER_TABLEACCESS_DENIED_ERROR",
    });
    const conn = new FlakyDDLConn(99, denied);
    await expect(
      new ScriptedBackend(conn, MySQLDialect).openSession({ project: "p", provider: "openai" }),
    ).rejects.toThrow(/CREATE command denied/);
    expect(conn.ddlAttempts).toBe(1);
  });
});

describe("fail-open with an unreachable database", () => {
  it("the host call is unaffected, ONE warning is logged, and NO local file appears", async () => {
    const port = await closedPort();
    const dir = tempDir();
    const cwd = process.cwd();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    process.chdir(dir);
    try {
      process.env[ENV_VAR] = `postgresql://u:p@127.0.0.1:${port}/db`;
      const tracer = init("dead");
      // A networked backend has no path — and must never invent one.
      expect(tracer.path).toBeNull();
      const client = tracer.wrap(stubClient()) as {
        chat: {
          completions: {
            create(a: unknown): Promise<{ choices: { message: { content: string } }[] }>;
          };
        };
      };

      for (let i = 0; i < 3; i++) {
        const res = await client.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: `turn ${i}` }],
        });
        expect(res.choices[0].message.content).toBe("ok");
      }
      await tracer.close();

      const degraded = warn.mock.calls.filter((c) =>
        String(c[0]).includes("capture degraded (store setup failed)"),
      );
      // Exactly one, however many calls were dropped.
      expect(degraded).toHaveLength(1);
      // THE POINT: no surprise `.ctrace` in the working directory. A user who
      // asked for Postgres and got a local file has been lied to.
      expect(readdirSync(dir)).toEqual([]);
    } finally {
      process.chdir(cwd);
      warn.mockRestore();
    }
  }, 20_000);
});

// --- the writer ---------------------------------------------------------------

/** A `Store` that resolves after `delayMs`, recording the order it was called
 * in — enough to prove the writer's serialization and flush without a server. */
class SlowStore implements Store {
  readonly persisted: number[] = [];
  private inFlight = 0;
  maxConcurrent = 0;
  constructor(private readonly delayMs: number) {}

  async recordCall(args: RecordCallArgs): Promise<string> {
    this.inFlight += 1;
    this.maxConcurrent = Math.max(this.maxConcurrent, this.inFlight);
    await new Promise((r) => setTimeout(r, this.delayMs));
    this.inFlight -= 1;
    this.persisted.push(args.seq);
    return `call-${args.seq}`;
  }
  async noteModel(): Promise<void> {}
  async listSessions(): Promise<never[]> {
    return [];
  }
  async getRun(): Promise<never> {
    throw new Error("not used");
  }
  async getCalls(): Promise<never[]> {
    return [];
  }
  async getCallBlocks(): Promise<never[]> {
    return [];
  }
  closed = false;
  async close(): Promise<void> {
    // Deliberately slower than the drain, so a close() that returns before the
    // connection is released is observable.
    await new Promise((r) => setTimeout(r, 10));
    this.closed = true;
  }
}

/** A `Store` that accepts everything and NEVER settles anything — a database
 * that has stopped answering, in both of `close()`'s phases at once. */
class HangingStore implements Store {
  closeAttempted = false;
  recordCall(): Promise<string> {
    return new Promise(() => undefined);
  }
  noteModel(): Promise<void> {
    return new Promise(() => undefined);
  }
  async listSessions(): Promise<never[]> {
    return [];
  }
  async getRun(): Promise<never> {
    throw new Error("not used");
  }
  async getCalls(): Promise<never[]> {
    return [];
  }
  async getCallBlocks(): Promise<never[]> {
    return [];
  }
  close(): Promise<void> {
    this.closeAttempted = true;
    return new Promise(() => undefined);
  }
}

/** A networked-looking backend around a `HangingStore`, publishing a tiny
 * statement bound so `close()`'s budget is small enough to measure. */
class HangingBackend implements StoreBackend {
  readonly statementTimeout = 0.05;
  readonly store = new HangingStore();
  async openSession(_args: OpenSessionArgs): Promise<Store> {
    return this.store;
  }
  async openReader(): Promise<Store> {
    return this.store;
  }
}

/** A networked-looking backend (no `pathFor`) around a `SlowStore`. */
class SlowBackend implements StoreBackend {
  readonly statementTimeout = 10;
  constructor(readonly store: SlowStore, private readonly openDelayMs = 0) {}
  async openSession(_args: OpenSessionArgs): Promise<Store> {
    await new Promise((r) => setTimeout(r, this.openDelayMs));
    return this.store;
  }
  async openReader(): Promise<Store> {
    return this.store;
  }
}

describe("the serial writer", () => {
  it("never delays the host, then flushes every call IN SEQ ORDER on close", async () => {
    const store = new SlowStore(15);
    configure({ store: new SlowBackend(store, 40) });
    const tracer = init("writer");
    const client = tracer.wrap(stubClient()) as {
      chat: { completions: { create(a: unknown): Promise<unknown> } };
    };

    const started = performance.now();
    for (let i = 0; i < 5; i++) {
      await client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: `turn ${i}` }],
      });
    }
    // Five calls against a store whose open alone takes 40ms and whose write
    // takes 15ms: if the host waited on ANY of that it could not be this fast.
    expect(performance.now() - started).toBeLessThan(40);
    expect(store.persisted.length).toBeLessThan(5); // still draining

    await tracer.close();
    // close() is the flush point: everything is persisted, in call order.
    expect(store.persisted).toEqual([1, 2, 3, 4, 5]);
    // ONE statement at a time on the one connection, always.
    expect(store.maxConcurrent).toBe(1);
  });

  it("records a call that lands DURING close()'s flush", async () => {
    // Our `close()` yields to the event loop for the whole flush, so a host LLM
    // call that was already in flight lands inside that window. Dropping it
    // would lose a turn the user watched happen — Python recovers exactly this
    // job in `_drain_stragglers`, and so do we.
    const warns: string[] = [];
    const warn = vi.spyOn(console, "warn").mockImplementation((...a: unknown[]) => {
      warns.push(String(a[0]));
    });
    const store = new SlowStore(20);
    configure({ store: new SlowBackend(store, 10) });
    const tracer = init("straggler");
    const client = tracer.wrap(stubClient()) as {
      chat: { completions: { create(a: unknown): Promise<unknown> } };
    };
    await client.chat.completions.create({ model: "m", messages: [] });

    // Start the flush, then let a second call complete while it is running.
    const closing = tracer.close();
    await client.chat.completions.create({ model: "m", messages: [] });
    await closing;

    expect(store.persisted).toEqual([1, 2]);
    expect(warns.some((w) => w.includes("drained 1 write(s) enqueued during close"))).toBe(true);
    expect(warns.some((w) => w.includes("will not be recorded"))).toBe(false);
    warn.mockRestore();
  });

  it("bounds close() by ONE deadline shared by the flush AND the release", async () => {
    // `close()` promises a bound. It has two phases — wait for the drain, then
    // release the connection — and giving each phase the FULL budget makes the
    // real ceiling twice what is documented: 24 seconds on the defaults, not 12,
    // for a store that has stopped answering in both phases. A shutdown bound
    // that is quietly double is not a bound.
    //
    // A store that never settles ANYTHING is the only way to charge both phases
    // their maximum, which is exactly the wedged-database case the bound exists
    // for.
    const backend = new HangingBackend();
    configure({ store: backend });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const tracer = init("hang");
    const client = tracer.wrap(stubClient()) as {
      chat: { completions: { create(a: unknown): Promise<unknown> } };
    };
    await client.chat.completions.create({ model: "m", messages: [] });

    const started = performance.now();
    await tracer.close();
    const elapsed = performance.now() - started;
    warn.mockRestore();

    // The backend publishes a 0.05s statement bound, so close()'s budget is
    // 50ms + the 2s margin = 2050ms TOTAL.
    const bound = 0.05 * 1000 + 2_000;
    // It really did wait — this is a bound, not a "give up immediately".
    expect(elapsed).toBeGreaterThan(bound * 0.8);
    // And it waited ONCE. Charging each phase separately lands near 2x.
    expect(elapsed, `close() took ${Math.round(elapsed)}ms against a ${bound}ms bound`).toBeLessThan(
      bound * 1.4,
    );
    // The release was still ATTEMPTED, just not waited on past the deadline.
    expect(backend.store.closeAttempted).toBe(true);
  }, 20_000);

  it("close() is idempotent and safe to call without awaiting the first one", async () => {
    const store = new SlowStore(5);
    configure({ store: new SlowBackend(store) });
    const tracer = init("writer");
    const client = tracer.wrap(stubClient()) as {
      chat: { completions: { create(a: unknown): Promise<unknown> } };
    };
    await client.chat.completions.create({ model: "m", messages: [] });
    await Promise.all([tracer.close(), tracer.close()]);
    expect(store.persisted).toEqual([1]);
    // A second close must await the FIRST one's full shutdown — including the
    // connection release — not just its drain.
    expect(store.closed).toBe(true);
  });
});

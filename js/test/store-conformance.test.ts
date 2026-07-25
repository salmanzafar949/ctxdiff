/**
 * HEADLINE STORAGE TEST: every backend means the SAME THING.
 *
 * One parametrized suite runs identical assertions against each `StoreBackend`:
 *
 * - `node:sqlite` — ALWAYS, since it is the zero-config default and needs no
 *   server;
 * - `postgres` — when `CTXDIFF_TEST_POSTGRES_DSN` names a real server;
 * - `mysql`    — when `CTXDIFF_TEST_MYSQL_DSN` does.
 *
 * The live suites are SKIPPED (visibly, by name) rather than silently passing
 * when those variables are unset, because a stub driver structurally cannot
 * catch what these tests exist to catch: a collation that folds two content
 * hashes into one, a VARCHAR that truncates a long error under STRICT mode, a
 * catalog race on a cold start, a connection killed mid-run. Those are server
 * behaviors, and only a server can prove them.
 *
 * Everything asserted here is a SEMANTIC promise, not an implementation detail:
 * a `.ctrace` and a Postgres database recording the same run must read back the
 * same calls, the same blocks in the same order, the same dedup, the same
 * session ordering — otherwise "pluggable storage" would mean "your traces mean
 * something slightly different now".
 */
import { describe, it, expect, beforeEach, afterEach, afterAll } from "vitest";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { connect, createServer, type AddressInfo, type Socket } from "node:net";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { CallBlock } from "../src/models.js";
import type { Store, StoreBackend } from "../src/store/base.js";
import { EmptyStoreError } from "../src/store/base.js";
import { SQLiteStore } from "../src/store/sqlite.js";
import { PostgresStore } from "../src/store/postgres.js";
import { MySQLStore } from "../src/store/mysql.js";
import { configure, ENV_VAR } from "../src/store/config.js";
import { init } from "../src/trace.js";
import { main } from "../src/cli.js";

const PG_DSN = process.env.CTXDIFF_TEST_POSTGRES_DSN;
const MYSQL_DSN = process.env.CTXDIFF_TEST_MYSQL_DSN;

const tempDirs: string[] = [];
afterAll(() => {
  for (const d of tempDirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

/** One block, content-addressed by an explicit hash so a test can control dedup
 * and byte-exactness directly instead of going through the hasher. */
function block(
  hash: string,
  text: string,
  position: number,
  opts: { role?: string; kind?: string; label?: string; tokens?: number } = {},
): CallBlock {
  return {
    block: {
      contentHash: hash,
      role: opts.role ?? "user",
      kind: opts.kind ?? "text",
      text,
      tokenCount: opts.tokens ?? text.length,
      tokenMethod: "exact",
    },
    position,
    label: opts.label ?? `label-${position}`,
    labelSource: "heuristic",
  };
}

/** A description of one backend under test, plus how to hand each test a store
 * that starts EMPTY — a fresh file for SQLite, dropped tables for a server. */
interface BackendCase {
  name: string;
  enabled: boolean;
  skipHint: string;
  create(): StoreBackend;
  reset(): Promise<void>;
}

/** Drop ctxdiff's four tables so a live backend starts each test empty. Uses the
 * driver directly (not the store) so a bug in the store cannot hide behind its
 * own cleanup. */
async function dropPostgresTables(dsn: string): Promise<void> {
  const pg = await import("pg");
  const client = new pg.default.Client({ connectionString: dsn });
  await client.connect();
  try {
    await client.query(
      "DROP TABLE IF EXISTS ctxdiff_call_block, ctxdiff_call, ctxdiff_block, ctxdiff_run CASCADE",
    );
  } finally {
    await client.end();
  }
}

async function dropMysqlTables(dsn: string): Promise<void> {
  const mysql = await import("mysql2/promise");
  const conn = await mysql.default.createConnection(dsn);
  try {
    await conn.query("SET FOREIGN_KEY_CHECKS = 0");
    for (const t of ["ctxdiff_call_block", "ctxdiff_call", "ctxdiff_block", "ctxdiff_run"]) {
      await conn.query(`DROP TABLE IF EXISTS ${t}`);
    }
    await conn.query("SET FOREIGN_KEY_CHECKS = 1");
  } finally {
    await conn.end();
  }
}

const CASES: BackendCase[] = [
  {
    name: "node:sqlite",
    enabled: true,
    skipHint: "",
    // A fresh file per store: SQLite's isolation is free, and this is exactly
    // the shape a real user gets from the zero-config default.
    create() {
      const dir = mkdtempSync(join(tmpdir(), "ctxdiff-conf-"));
      tempDirs.push(dir);
      return new SQLiteStore({ path: join(dir, "project.ctrace") });
    },
    async reset() {
      /* a new file per create(); nothing to clean */
    },
  },
  {
    name: "postgres",
    enabled: PG_DSN !== undefined,
    skipHint: "set CTXDIFF_TEST_POSTGRES_DSN to run against a real PostgreSQL server",
    create: () => new PostgresStore({ dsn: PG_DSN! }),
    reset: () => dropPostgresTables(PG_DSN!),
  },
  {
    name: "mysql",
    enabled: MYSQL_DSN !== undefined,
    skipHint: "set CTXDIFF_TEST_MYSQL_DSN to run against a real MySQL server",
    create: () => new MySQLStore({ dsn: MYSQL_DSN! }),
    reset: () => dropMysqlTables(MYSQL_DSN!),
  },
];

for (const backend of CASES) {
  const title = backend.enabled
    ? `store conformance: ${backend.name}`
    : `store conformance: ${backend.name} — SKIPPED (${backend.skipHint})`;

  describe.skipIf(!backend.enabled)(title, () => {
    let store: StoreBackend;
    // Everything a test opened, closed in reverse so a live server is never left
    // holding connections between cases.
    let open: Store[] = [];

    beforeEach(async () => {
      await backend.reset();
      store = backend.create();
      open = [];
    });

    /** Open a session and remember it for teardown. */
    async function session(opts: { project?: string; provider?: string; startedAt?: string } = {}) {
      const s = await store.openSession({
        project: opts.project ?? "conf",
        provider: opts.provider ?? "openai",
        model: "",
        startedAt: opts.startedAt ?? new Date().toISOString(),
      });
      open.push(s);
      return s;
    }

    // Release every connection a case opened, in reverse order, even when the
    // case failed — a live server must never be left holding handles.
    afterEach(async () => {
      for (const s of open.splice(0).reverse()) {
        try {
          await s.close();
        } catch {
          /* teardown is best-effort */
        }
      }
    });

    it("appends calls to one session and reads them back field-for-field", async () => {
      const s = await session();
      const id = await s.recordCall({
        seq: 1,
        params: { model: "gpt-4o", temperature: 0.2 },
        usage: { input_tokens: 11, output_tokens: 3 },
        latencyMs: 1234,
        error: null,
        callBlocks: [block("h-sys", "you are helpful", 0, { role: "system" })],
        agent: "planner",
        step: "plan",
        provider: "openai",
      });
      await s.recordCall({
        seq: 2,
        params: { model: "gpt-4o" },
        usage: null,
        latencyMs: null,
        error: "RateLimitError",
        callBlocks: [block("h-sys", "you are helpful", 0, { role: "system" })],
      });

      const calls = await s.getCalls();
      expect(calls.map((c) => c.seq)).toEqual([1, 2]);
      expect(calls[0].id).toBe(id);
      expect(calls[0].params).toEqual({ model: "gpt-4o", temperature: 0.2 });
      expect(calls[0].usage).toEqual({ input_tokens: 11, output_tokens: 3 });
      // A driver that returned BIGINT as a string would break every consumer
      // that does arithmetic on latency.
      expect(calls[0].latencyMs).toBe(1234);
      expect(typeof calls[0].latencyMs).toBe("number");
      expect(calls[0].agent).toBe("planner");
      expect(calls[0].step).toBe("plan");
      expect(calls[0].provider).toBe("openai");
      // Nullable columns come back as null, never undefined or "".
      expect(calls[1].usage).toBeNull();
      expect(calls[1].latencyMs).toBeNull();
      expect(calls[1].agent).toBeNull();
      expect(calls[1].error).toBe("RateLimitError");
    });

    it("keeps sessions isolated: each reads only its own calls", async () => {
      const a = await session({ project: "alpha" });
      const b = await session({ project: "beta" });
      await a.recordCall({
        seq: 1,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block("h-a", "alpha text", 0)],
      });
      await b.recordCall({
        seq: 1,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block("h-b", "beta text", 0)],
      });
      await b.recordCall({
        seq: 2,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block("h-b", "beta text", 0)],
      });

      expect((await a.getCalls()).length).toBe(1);
      expect((await b.getCalls()).length).toBe(2);
      expect((await a.getRun()).project).toBe("alpha");
      expect((await b.getRun()).project).toBe("beta");
      // A session can still be named explicitly, and reads only that session.
      const bId = (await b.getRun()).id;
      expect((await a.getCalls(bId)).length).toBe(2);
    });

    it("dedups blocks by content hash — stored once, referenced by many calls", async () => {
      const s = await session();
      const shared = block("h-shared", "the shared system prompt", 0, { role: "system" });
      const c1 = await s.recordCall({
        seq: 1,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [shared, block("h-1", "first question", 1)],
      });
      const c2 = await s.recordCall({
        seq: 2,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [shared, block("h-2", "second question", 1)],
      });

      const b1 = await s.getCallBlocks(c1);
      const b2 = await s.getCallBlocks(c2);
      // Same hash, same text on both sides: the second write no-oped rather
      // than failing or overwriting.
      expect(b1[0].block.contentHash).toBe(b2[0].block.contentHash);
      expect(b1[0].block.text).toBe("the shared system prompt");
      expect(b2[0].block.text).toBe("the shared system prompt");
      expect(b1[1].block.text).toBe("first question");
      expect(b2[1].block.text).toBe("second question");
    });

    it("returns a call's blocks in POSITION order, with each call's own labels", async () => {
      const s = await session();
      const id = await s.recordCall({
        seq: 1,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        // Deliberately inserted out of order: position, not insertion, decides.
        callBlocks: [
          block("h-c", "third", 2, { label: "tail" }),
          block("h-a", "first", 0, { label: "head" }),
          block("h-b", "second", 1, { label: "middle" }),
        ],
      });
      const blocks = await s.getCallBlocks(id);
      expect(blocks.map((b) => b.position)).toEqual([0, 1, 2]);
      expect(blocks.map((b) => b.block.text)).toEqual(["first", "second", "third"]);
      expect(blocks.map((b) => b.label)).toEqual(["head", "middle", "tail"]);
      expect(blocks.every((b) => b.labelSource === "heuristic")).toBe(true);
    });

    it("listSessions reports agents in first-appearance order with turn counts", async () => {
      const s = await session({ project: "multi" });
      const agents = ["researcher", "writer", "researcher", "critic"];
      for (const [i, agent] of agents.entries()) {
        await s.recordCall({
          seq: i + 1,
          params: { model: "m" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block(`h-${i}`, `turn ${i}`, 0)],
          agent,
        });
      }
      const sessions = await s.listSessions();
      expect(sessions.length).toBe(1);
      expect(sessions[0].project).toBe("multi");
      expect(sessions[0].turnCount).toBe(4);
      // Deduped, in FIRST-appearance order — not sorted, not insertion-deduped.
      expect(sessions[0].agents).toEqual(["researcher", "writer", "critic"]);
      expect(sessions[0].models).toEqual(["m"]);
    });

    it("rolls models up in first-seen order, deduped, ignoring blanks", async () => {
      const s = await session();
      for (const [i, model] of ["gpt-4o", "gpt-4o", "o3", "gpt-4o"].entries()) {
        await s.recordCall({
          seq: i + 1,
          params: { model },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block(`h-${i}`, `t${i}`, 0)],
        });
      }
      await s.noteModel(null);
      await s.noteModel("");
      expect((await s.getRun()).models).toEqual(["gpt-4o", "o3"]);
    });

    it("binds a reader to the NEWEST session even when clocks run backwards", async () => {
      // The whole reason `insert_order` exists: with several containers on one
      // database the clocks disagree, so the session written LAST is not the one
      // with the latest `started_at`. Ordering must follow the WRITE, not the
      // clock — otherwise `openReader()` binds somebody else's run.
      const first = await session({ project: "p1", startedAt: "2030-01-01T00:00:00.000Z" });
      const second = await session({ project: "p2", startedAt: "2020-01-01T00:00:00.000Z" });
      const third = await session({ project: "p3", startedAt: "2025-01-01T00:00:00.000Z" });

      const sessions = await third.listSessions();
      expect(sessions.map((s) => s.project)).toEqual(["p1", "p2", "p3"]);

      const reader = await store.openReader();
      open.push(reader);
      expect((await reader.getRun()).project).toBe("p3");
      expect((await reader.getRun()).id).toBe((await third.getRun()).id);
      // ...and the two earlier sessions are unaffected.
      expect((await first.getRun()).project).toBe("p1");
      expect((await second.getRun()).project).toBe("p2");
    });

    it("compares content hashes BYTE-EXACTLY (case-sensitively)", async () => {
      // MySQL's default utf8mb4 collation is case-INSENSITIVE, which would fold
      // these two hashes into one primary key: the second block would be deduped
      // away and its call would read back the FIRST block's text. Silent
      // corruption, no error anywhere — hence `ascii_bin` on every hash column.
      const s = await session();
      const upper = "AABBCCDD";
      const lower = "aabbccdd";
      const c1 = await s.recordCall({
        seq: 1,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block(upper, "UPPERCASE HASH CONTENT", 0)],
      });
      const c2 = await s.recordCall({
        seq: 2,
        params: { model: "m" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block(lower, "lowercase hash content", 0)],
      });
      expect((await s.getCallBlocks(c1))[0].block.text).toBe("UPPERCASE HASH CONTENT");
      expect((await s.getCallBlocks(c2))[0].block.text).toBe("lowercase hash content");
      expect((await s.getCallBlocks(c1))[0].block.contentHash).toBe(upper);
      expect((await s.getCallBlocks(c2))[0].block.contentHash).toBe(lower);
    });

    it("stores long values whole — a big prompt and long error/label/agent", async () => {
      // A bounded VARCHAR on a free-text column is not a size limit, it is data
      // loss: under MySQL's STRICT mode an over-long value raises 1406 and, since
      // the call and its blocks are ONE transaction, drops the WHOLE turn.
      const s = await session();
      const body = "x".repeat(200_000);
      const longError = "ProviderError: " + "e".repeat(2_000);
      const longLabel = "label-" + "l".repeat(1_000);
      const longAgent = "agent-" + "a".repeat(1_000);
      const id = await s.recordCall({
        seq: 1,
        params: { model: "m", prompt: "p".repeat(50_000) },
        usage: null,
        latencyMs: 9_999_999_999, // > 2^32 ms: INTEGER would overflow
        error: longError,
        callBlocks: [block("h-long", body, 0, { label: longLabel, tokens: 50_000 })],
        agent: longAgent,
        step: "s".repeat(500),
      });

      const call = (await s.getCalls())[0];
      expect(call.error).toBe(longError);
      expect(call.agent).toBe(longAgent);
      expect(call.step!.length).toBe(500);
      expect(call.latencyMs).toBe(9_999_999_999);
      expect((call.params.prompt as string).length).toBe(50_000);
      const blocks = await s.getCallBlocks(id);
      expect(blocks[0].block.text.length).toBe(200_000);
      expect(blocks[0].label).toBe(longLabel);
      expect(blocks[0].block.tokenCount).toBe(50_000);
    });

    it("round-trips unicode (emoji/CJK/RTL) byte-exactly", async () => {
      const s = await session();
      const text = "🙈 中文 مرحبا ​‮ ok";
      const id = await s.recordCall({
        seq: 1,
        params: { model: "m", note: text },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block("h-uni", text, 0)],
      });
      expect((await s.getCallBlocks(id))[0].block.text).toBe(text);
      expect((await s.getCalls())[0].params.note).toBe(text);
    });

    it("close() is idempotent and never throws", async () => {
      const s = await session();
      await s.close();
      // `Promise.resolve` normalizes the two legal shapes: the SQLite store
      // closes synchronously (returns undefined), a networked one returns a
      // promise. Neither may throw on a second close.
      await expect(Promise.resolve(s.close())).resolves.toBeUndefined();
    });

    it("openSession auto-creates the schema — a second cold open just works", async () => {
      // The "no migration step" promise: nothing has ever run against this
      // store, and two independent opens must both succeed.
      const a = await session({ project: "cold-1" });
      const b = await session({ project: "cold-2" });
      expect((await b.listSessions()).map((x) => x.project)).toEqual(["cold-1", "cold-2"]);
      expect((await a.getRun()).ctxdiffVersion).toBeTruthy();
    });
  });
}

describe.skipIf(PG_DSN === undefined)(
  PG_DSN === undefined
    ? "postgres live-only — SKIPPED (set CTXDIFF_TEST_POSTGRES_DSN)"
    : "postgres live-only",
  () => {
    beforeEach(() => dropPostgresTables(PG_DSN!));

    it("survives a CONCURRENT COLD START against an empty database", async () => {
      // `CREATE TABLE IF NOT EXISTS` is a check, not a lock: replicas that pass
      // it in the same instant all insert into pg_type/pg_class and all but one
      // fail on a system index (23505 / 42P07 / 42710). That is not a rare
      // corner, it is the shape of a service scaling up — un-retried, every
      // replica but one records NOTHING for its whole run.
      const backend = new PostgresStore({ dsn: PG_DSN! });
      const sessions = await Promise.all(
        Array.from({ length: 8 }, (_, i) =>
          backend.openSession({
            project: `cold-${i}`,
            provider: "openai",
            model: "",
            startedAt: new Date().toISOString(),
          }),
        ),
      );
      try {
        expect(sessions.length).toBe(8);
        const listed = await sessions[0].listSessions();
        expect(listed.length).toBe(8);
        expect(new Set(listed.map((s) => s.project)).size).toBe(8);
      } finally {
        for (const s of sessions) await s.close();
      }
    });

    it("BOUNDS a statement blocked on someone else's lock, then keeps recording", async () => {
      // The "alive but not answering" case: another session holds a lock on the
      // row ctxdiff wants to update. Unbounded, that write would hold the run's
      // single writer — and therefore `close()` — for as long as the other
      // transaction lasts. Bounded, it costs ONE write and capture continues.
      const backend = new PostgresStore({ dsn: PG_DSN!, statementTimeout: 1 });
      const s = await backend.openSession({
        project: "locked",
        provider: "openai",
        model: "",
        startedAt: new Date().toISOString(),
      });
      const pg = await import("pg");
      const blocker = new pg.default.Client({ connectionString: PG_DSN! });
      await blocker.connect();
      try {
        await s.recordCall({
          seq: 1,
          params: { model: "m" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block("l-1", "before the lock", 0)],
        });
        const runId = (await s.getRun()).id;
        await blocker.query("BEGIN");
        await blocker.query("SELECT * FROM ctxdiff_run WHERE id = $1 FOR UPDATE", [runId]);

        const started = Date.now();
        // The model roll-up UPDATEs the locked row, so this write waits — and
        // must be given up on, fast, rather than pinning the writer.
        await expect(s.noteModel("blocked-model")).rejects.toThrow();
        const elapsed = Date.now() - started;
        expect(elapsed).toBeLessThan(5_000);
        expect(elapsed).toBeGreaterThan(500); // it really did wait on the lock

        await blocker.query("ROLLBACK");
        // Capture is NOT over: the next call records, and the roll-up that
        // failed is retried by the next call carrying a model (commit-then-mark).
        await s.recordCall({
          seq: 2,
          params: { model: "after" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block("l-2", "after the lock", 0)],
        });
        expect((await s.getCalls()).map((c) => c.seq)).toEqual([1, 2]);
        expect((await s.getRun()).models).toEqual(["m", "after"]);
      } finally {
        await blocker.end();
        await s.close();
      }
    }, 20_000);

    it("REOPENS a connection killed mid-run and keeps recording", async () => {
      // A retry against a dead socket is inert: without the reopen, one pooler
      // recycle / failover / restart ends capture for the whole process, with
      // one warning, even though the database is healthy a second later.
      const backend = new PostgresStore({ dsn: PG_DSN!, statementTimeout: 5 });
      const s = await backend.openSession({
        project: "kill",
        provider: "openai",
        model: "",
        startedAt: new Date().toISOString(),
      });
      try {
        await s.recordCall({
          seq: 1,
          params: { model: "m" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block("k-1", "before the kill", 0)],
        });
        // A handle WITHOUT a reconnect callable proves the kill was real: after
        // it, `pg` rejects every statement with a message carrying no SQLSTATE
        // at all, which is exactly why the store also consults the connection's
        // own liveness flag.
        const doomed = await backend.openReader();
        await killPostgresBackends(PG_DSN!);
        await expect(doomed.getCalls()).rejects.toThrow();
        await doomed.close();
        // The write below lands on the same dead socket, is classified as
        // connection loss, REOPENS, and succeeds.
        await s.recordCall({
          seq: 2,
          params: { model: "m" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block("k-2", "after the kill", 0)],
        });
        const calls = await s.getCalls();
        expect(calls.map((c) => c.seq)).toEqual([1, 2]);
      } finally {
        await s.close();
      }
    });
  },
);

describe.skipIf(MYSQL_DSN === undefined)(
  MYSQL_DSN === undefined
    ? "mysql live-only — SKIPPED (set CTXDIFF_TEST_MYSQL_DSN)"
    : "mysql live-only",
  () => {
    beforeEach(() => dropMysqlTables(MYSQL_DSN!));

    it("survives a CONCURRENT COLD START against an empty database", async () => {
      const backend = new MySQLStore({ dsn: MYSQL_DSN! });
      const sessions = await Promise.all(
        Array.from({ length: 8 }, (_, i) =>
          backend.openSession({
            project: `cold-${i}`,
            provider: "openai",
            model: "",
            startedAt: new Date().toISOString(),
          }),
        ),
      );
      try {
        const listed = await sessions[0].listSessions();
        expect(listed.length).toBe(8);
      } finally {
        for (const s of sessions) await s.close();
      }
    });

    it("REOPENS a connection killed mid-run and keeps recording", async () => {
      const backend = new MySQLStore({ dsn: MYSQL_DSN!, statementTimeout: 5 });
      const s = await backend.openSession({
        project: "kill",
        provider: "openai",
        model: "",
        startedAt: new Date().toISOString(),
      });
      try {
        await s.recordCall({
          seq: 1,
          params: { model: "m" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block("k-1", "before the kill", 0)],
        });
        // A handle WITHOUT a reconnect callable proves the kill was real.
        const doomed = await backend.openReader();
        await killMysqlConnections(MYSQL_DSN!);
        await expect(doomed.getCalls()).rejects.toThrow();
        await doomed.close();
        await s.recordCall({
          seq: 2,
          params: { model: "m" },
          usage: null,
          latencyMs: null,
          error: null,
          callBlocks: [block("k-2", "after the kill", 0)],
        });
        expect((await s.getCalls()).map((c) => c.seq)).toEqual([1, 2]);
      } finally {
        await s.close();
      }
    });

    it("pins ascii_bin on every id/hash/timestamp column and LONGTEXT elsewhere", async () => {
      // Asserted against the SERVER's own catalog, not against our DDL string:
      // what matters is the collation the table ENDED UP with.
      const backend = new MySQLStore({ dsn: MYSQL_DSN! });
      const s = await backend.openSession({
        project: "ddl",
        provider: "openai",
        model: "",
        startedAt: new Date().toISOString(),
      });
      await s.close();
      const mysql = await import("mysql2/promise");
      const conn = await mysql.default.createConnection(MYSQL_DSN!);
      try {
        const [rows] = (await conn.query(
          "SELECT table_name, column_name, data_type, collation_name " +
            "FROM information_schema.columns WHERE table_schema = DATABASE() " +
            "AND table_name LIKE 'ctxdiff\\_%'",
        )) as [Record<string, string>[], unknown];
        const by = new Map(
          rows.map((r) => [
            `${r.TABLE_NAME ?? r.table_name}.${r.COLUMN_NAME ?? r.column_name}`,
            r,
          ]),
        );
        const binary = [
          "ctxdiff_run.id",
          "ctxdiff_run.started_at",
          "ctxdiff_call.id",
          "ctxdiff_call.run_id",
          "ctxdiff_block.content_hash",
          "ctxdiff_call_block.call_id",
          "ctxdiff_call_block.block_id",
        ];
        for (const key of binary) {
          const row = by.get(key)!;
          expect(row, key).toBeDefined();
          expect(row.COLLATION_NAME ?? row.collation_name, key).toBe("ascii_bin");
          expect((row.DATA_TYPE ?? row.data_type).toLowerCase(), key).toBe("varchar");
        }
        for (const key of [
          "ctxdiff_call.error",
          "ctxdiff_call.agent",
          "ctxdiff_call.step",
          "ctxdiff_call.provider",
          "ctxdiff_call_block.label",
          "ctxdiff_block.body",
          "ctxdiff_run.project",
        ]) {
          const row = by.get(key)!;
          expect((row.DATA_TYPE ?? row.data_type).toLowerCase(), key).toBe("longtext");
        }
        const latency = by.get("ctxdiff_call.latency_ms")!;
        expect((latency.DATA_TYPE ?? latency.data_type).toLowerCase()).toBe("bigint");
      } finally {
        await conn.end();
      }
    });
  },
);

describe.skipIf(PG_DSN === undefined)(
  PG_DSN === undefined
    ? "end-to-end through a configured database — SKIPPED (set CTXDIFF_TEST_POSTGRES_DSN)"
    : "end-to-end through a configured database",
  () => {
    beforeEach(() => dropPostgresTables(PG_DSN!));

    it("records a real run through configure() and reads it back via the CLI", async () => {
      // The whole feature in one test: point ctxdiff at a database with ONE call,
      // trace an agent, then have `npx ctxdiff` analyze it with no file anywhere.
      const tracer = init("e2e", { store: new PostgresStore({ dsn: PG_DSN! }) });
      expect(tracer.path).toBeNull(); // a database has no path, and never invents one
      const client = tracer.wrap(stubOpenAI(), { agent: "planner" }) as {
        chat: { completions: { create(a: unknown): Promise<unknown> } };
      };
      for (const q of ["what is ctxdiff?", "and how does it store traces?"]) {
        await client.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are a careful assistant." },
            { role: "user", content: q },
          ],
        });
      }
      await tracer.close(); // the flush point

      const previous = process.env[ENV_VAR];
      process.env[ENV_VAR] = PG_DSN!;
      try {
        // `tokens` and `runs` go through the same reader path `diff`/`view` use.
        const tokens = await runCli(["tokens"]);
        expect(tokens.code).toBe(0);
        expect(tokens.out).toContain("turn 1");
        expect(tokens.out).toContain("turn 2");

        const runs = await runCli(["runs"]);
        expect(runs.code).toBe(0);
        expect(runs.out).toContain("project=e2e");
        expect(runs.out).toContain("turns=2");
        expect(runs.out).toContain("agents=planner");

        const diff = await runCli(["diff", "--turn", "1", "--turn", "2"]);
        expect(diff.code).toBe(0);
        expect(diff.out).toContain("turn 1 → turn 2");

        // `export` has no filename to derive, so it says so rather than guessing.
        const noOut = await runCli(["export"]);
        expect(noOut.code).toBe(1);
        expect(noOut.err).toContain("pass --out FILE.html");

        const out = join(mkdtempSync(join(tmpdir(), "ctxdiff-e2e-")), "dash.html");
        const exported = await runCli(["export", "--out", out]);
        expect(exported.code).toBe(0);
        expect(readFileSync(out, "utf-8")).toContain("<!DOCTYPE html>");
      } finally {
        if (previous === undefined) delete process.env[ENV_VAR];
        else process.env[ENV_VAR] = previous;
      }
    });

    it("survives four concurrent tracers writing overlapping blocks", async () => {
      // The shape a fleet actually produces: several agents in several
      // containers, all sharing one system prompt, all upserting the SAME block
      // rows at the same moment. That is where a deadlock or a serialization
      // failure appears — and where the bounded write retry has to earn its
      // keep, since dropping the call would be needless capture loss.
      const tracers = Array.from({ length: 4 }, (_, i) =>
        init(`fleet-${i}`, { store: new PostgresStore({ dsn: PG_DSN! }) }),
      );
      const clients = tracers.map(
        (t, i) =>
          t.wrap(stubOpenAI(), { agent: `agent-${i}` }) as {
            chat: { completions: { create(a: unknown): Promise<unknown> } };
          },
      );
      await Promise.all(
        clients.map(async (client) => {
          for (let turn = 0; turn < 5; turn++) {
            await client.chat.completions.create({
              model: "gpt-4o",
              messages: [
                // Identical across every tracer: the contended block.
                { role: "system", content: "One shared system prompt for the fleet." },
                { role: "user", content: `turn ${turn}` },
              ],
            });
          }
        }),
      );
      await Promise.all(tracers.map((t) => t.close()));

      const reader = await new PostgresStore({ dsn: PG_DSN! }).openReader();
      try {
        const sessions = await reader.listSessions();
        expect(sessions.length).toBe(4);
        // Every call of every run persisted — no silent drops under contention.
        expect(sessions.map((s) => s.turnCount)).toEqual([5, 5, 5, 5]);
        expect(new Set(sessions.map((s) => s.project)).size).toBe(4);
      } finally {
        await reader.close();
      }
    });

    it("a host that FORGETS tracer.close() can still exit", async () => {
      // Node keeps a process alive for every referenced handle, and a live TCP
      // socket is one — so an open tracing connection would hold a finished
      // script open FOREVER. A debugging tool that prevents a program from
      // exiting is a worse failure than losing the last queued write, which is
      // exactly why Python's writer thread is a daemon and why ours detaches its
      // socket (`SqlConn.unref`). Proven in a real subprocess, because the
      // failure IS "the process does not exit".
      const built = freshBuild();
      const script = `
        const { init, configure, PostgresStore } = await import(${JSON.stringify(built)});
        configure({ store: new PostgresStore({ dsn: ${JSON.stringify(PG_DSN!)} }) });
        const t = init("forgot-close");
        const c = t.wrap({ constructor: { name: "OpenAI" },
          chat: { completions: { create: async () => ({ choices: [{ message: { content: "ok" } }], usage: {} }) } },
          responses: {} });
        await c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] });
        console.log("HOST_DONE");
      `;
      const started = Date.now();
      const proc = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
        encoding: "utf8",
        timeout: 15_000,
      });
      expect(proc.stdout).toContain("HOST_DONE");
      // `spawnSync` reports a timeout kill as a signal; an exit code means the
      // process ended on its own, which is the whole assertion.
      expect(proc.signal, "the process had to be KILLED — it never exited").toBeNull();
      expect(proc.status).toBe(0);
      expect(Date.now() - started).toBeLessThan(10_000);
    }, 30_000);

    it("a script driving the Store API DIRECTLY runs to completion", async () => {
      // The mirror image of the test above, and the trap the detached socket
      // sets if it is applied unconditionally: with no other referenced handle,
      // Node does not WAIT for an unref'd socket — it fires `beforeExit` and
      // terminates, so `await store.recordCall(...)` would never settle and the
      // rest of the user's script would silently vanish (exit 13). The socket is
      // therefore re-referenced for the duration of every statement. Both halves
      // have to hold at once, so both are tested in a real subprocess.
      const built = freshBuild();
      const script = `
        const { PostgresStore } = await import(${JSON.stringify(built)});
        const b = new PostgresStore({ dsn: ${JSON.stringify(PG_DSN!)} });
        const s = await b.openSession({ project: "direct", provider: "openai", model: "", startedAt: new Date().toISOString() });
        console.log("OPENED");
        await s.recordCall({ seq: 1, params: { model: "m" }, usage: null, latencyMs: 5, error: null,
          callBlocks: [{ block: { contentHash: "d1", role: "user", kind: "text", text: "t", tokenCount: 1, tokenMethod: "exact" },
                        position: 0, label: "l", labelSource: "heuristic" }] });
        console.log("RECORDED");
        console.log("READ", JSON.stringify((await s.getCalls()).map((c) => c.seq)));
        await s.close();
        console.log("CLOSED");
      `;
      const proc = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
        encoding: "utf8",
        timeout: 20_000,
      });
      expect(proc.stdout, proc.stderr).toContain("OPENED");
      expect(proc.stdout, proc.stderr).toContain("RECORDED");
      expect(proc.stdout, proc.stderr).toContain("READ [1]");
      // The one that fails when an idle-unref is applied unconditionally.
      expect(proc.stdout, "the script was cut short mid-await").toContain("CLOSED");
      expect(proc.status).toBe(0);
    }, 30_000);

    it("six processes cold-starting the SAME empty database each persist every call", async () => {
      // THE COLD START, in the shape a deployment actually produces it: a service
      // scales up and every replica points at the same empty database in the same
      // instant. `CREATE TABLE IF NOT EXISTS` is a check, not a lock, so all but
      // one lose the catalog race and go into `ensureSchema`'s bounded retry —
      // and that retry's backoff is the moment the process holds NO referenced
      // handle at all (the writer's socket is detached while idle). Node then
      // exits, abandoning the half-open session with no warning of any kind.
      //
      // Run as REAL processes with no `tracer.close()`, because `close()`'s own
      // referenced timer masks the bug entirely and only a real process can
      // observe "the process exited instead of finishing its writes".
      const built = freshBuild();
      const N = 6;
      const CALLS = 3;
      const results = await Promise.all(
        Array.from({ length: N }, (_, i) =>
          runScript(`
            const { init, configure, PostgresStore } = await import(${JSON.stringify(built)});
            configure({ store: new PostgresStore({ dsn: ${JSON.stringify(PG_DSN!)} }) });
            const t = init("cold-${i}");
            const c = t.wrap(${STUB_CLIENT_SRC});
            for (let k = 0; k < ${CALLS}; k++) {
              await c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "turn " + k }] });
            }
            console.log("HOST_DONE");
          `),
        ),
      );
      for (const [i, r] of results.entries()) {
        expect(r.out, `process ${i} stderr: ${r.err}`).toContain("HOST_DONE");
        expect(r.signal, `process ${i} was KILLED — it never exited`).toBeNull();
        expect(r.code, `process ${i} stderr: ${r.err}`).toBe(0);
      }

      const reader = await new PostgresStore({ dsn: PG_DSN! }).openReader();
      try {
        const sessions = await reader.listSessions();
        // Every replica's session, and every one of its calls. A losing replica
        // that died in the schema backoff shows up here as a missing session or
        // as a session with fewer turns than it made.
        expect(sessions.map((s) => s.project).sort()).toEqual(
          Array.from({ length: N }, (_, i) => `cold-${i}`),
        );
        expect(sessions.map((s) => s.turnCount)).toEqual(Array.from({ length: N }, () => CALLS));
      } finally {
        await reader.close();
      }
    }, 90_000);

    it("keeps every call across a connection RST mid-run, without tracer.close()", async () => {
      // The other half of the same bug, on the WRITE retry: a pooler recycle, a
      // failover or a restart resets the connection mid-run. `withWriteRetry`
      // reopens — correctly — and then backs off, and that backoff was the window
      // in which the process, holding only the fresh (already detached) socket,
      // simply exited. Every call after the reset was lost, silently.
      //
      // The reset is delivered by a TCP proxy rather than `pg_terminate_backend`
      // so it lands at an exact, reproducible point: the third call's INSERT,
      // destroyed before it ever reaches the server. Only that one connection is
      // cut; the reopen goes through, so a correct writer loses nothing.
      const built = freshBuild();
      const CALLS = 6;
      const proxy = await rstProxy(PG_DSN!, "INSERT INTO ctxdiff_call (", 3);
      const r = await runScript(`
        const { init, configure, PostgresStore } = await import(${JSON.stringify(built)});
        configure({ store: new PostgresStore({ dsn: ${JSON.stringify(proxy.dsn)} }) });
        const t = init("rst");
        const c = t.wrap(${STUB_CLIENT_SRC});
        for (let k = 0; k < ${CALLS}; k++) {
          await c.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "turn " + k }] });
        }
        console.log("HOST_DONE");
      `);
      expect(r.out, r.err).toContain("HOST_DONE");
      expect(r.signal, "the process had to be KILLED — it never exited").toBeNull();
      // The RST has to have actually happened, or this test proves nothing.
      expect(proxy.fired(), "the proxy never saw the third call INSERT").toBe(true);

      const reader = await new PostgresStore({ dsn: PG_DSN! }).openReader();
      try {
        // Read through the REAL dsn: the proxy is only how the failure was
        // injected. Every call is here, including the one whose INSERT was
        // destroyed in flight — the retry re-ran it on the reopened connection.
        expect((await reader.getCalls()).map((c) => c.seq)).toEqual([1, 2, 3, 4, 5, 6]);
      } finally {
        await reader.close();
      }
    }, 90_000);

    it("recovers when pg's OWN query timeout fires, instead of reusing a busy client", async () => {
      // `pg` enforces `query_timeout` client-side with a plain timer: it rejects
      // with a bare `Error: Query read timeout` — no SQLSTATE, no errno — and
      // leaves the socket open AND the client still executing that query. So the
      // connection is unusable while looking perfectly healthy: the rollback and
      // then every retry land on a busy client ("Calling client.query() when the
      // client is already executing a query"), each burning the full deadline,
      // and the call is dropped even though the database is fine.
      //
      // Injected with a proxy that stops relaying ONE connection's traffic at the
      // second call's BEGIN — a wedged box or a partition, the exact case
      // `query_timeout` exists for and the one a SERVER-side statement_timeout
      // structurally cannot report.
      const proxy = await stallProxy(PG_DSN!, "BEGIN", 3);
      const backend = new PostgresStore({ dsn: proxy.dsn, statementTimeout: 1 });
      const s = await backend.openSession({
        project: "qtimeout",
        provider: "openai",
        model: "",
        startedAt: new Date().toISOString(),
      });
      const failures: string[] = [];
      try {
        for (const seq of [1, 2, 3]) {
          try {
            await s.recordCall({
              seq,
              params: { model: "m" },
              usage: null,
              latencyMs: null,
              error: null,
              callBlocks: [block(`qt-${seq}`, `turn ${seq}`, 0)],
            });
          } catch (err) {
            failures.push(`seq ${seq}: ${(err as Error).message}`);
          }
        }
      } finally {
        await s.close();
      }
      expect(proxy.fired(), "the proxy never stalled a connection").toBe(true);
      expect(failures).toEqual([]);

      // Read back through the REAL dsn: the stall was only how the failure was
      // injected. Every call is here, re-run on a reopened connection.
      const reader = await new PostgresStore({ dsn: PG_DSN! }).openReader();
      try {
        expect((await reader.getCalls()).map((c) => c.seq)).toEqual([1, 2, 3]);
      } finally {
        await reader.close();
      }
    }, 60_000);

    it("reads with a real SELECT-only role that cannot CREATE", async ({ skip }) => {
      // The credential an operator SHOULD hand somebody pointing `ctxdiff diff`
      // at production: SELECT and nothing else. `openReader()` runs
      // `ensureSchema()` for the convenience of reporting "no sessions" instead
      // of "no such table", and that convenience used to be fatal here —
      // "permission denied for schema public", with no hint that ctxdiff had
      // tried to CREATE and did not actually need to.
      //
      // Needs a role, so it self-skips where the test DSN cannot make one.
      const tracer = init("ro", { store: new PostgresStore({ dsn: PG_DSN! }) });
      const client = tracer.wrap(stubOpenAI()) as {
        chat: { completions: { create(a: unknown): Promise<unknown> } };
      };
      await client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "written by a role that CAN create" }],
      });
      await tracer.close();

      const readOnly = await createReadOnlyRole(PG_DSN!);
      if (readOnly === null) skip("the test DSN's role cannot CREATE ROLE");
      try {
        const reader = await new PostgresStore({ dsn: readOnly.dsn }).openReader();
        try {
          expect((await reader.getRun()).project).toBe("ro");
          expect((await reader.getCalls()).length).toBe(1);
        } finally {
          await reader.close();
        }
      } finally {
        await readOnly.drop();
      }
    }, 60_000);

    it("leaves a session behind even for a run that records nothing", async () => {
      // An agent that wraps a client and then errors before its first LLM call
      // still ran. The local `.ctrace` records that (the run row is written at
      // wrap time), and a database must say the same thing rather than making
      // the run invisible — hence the writer opens the session eagerly.
      const tracer = init("silent", { store: new PostgresStore({ dsn: PG_DSN! }) });
      tracer.wrap(stubOpenAI());
      await tracer.close();

      const reader = await new PostgresStore({ dsn: PG_DSN! }).openReader();
      try {
        const run = await reader.getRun();
        expect(run.project).toBe("silent");
        expect(run.provider).toBe("openai");
        expect(run.models).toEqual([]); // no call, so no model was ever seen
        expect(await reader.getCalls()).toEqual([]);
      } finally {
        await reader.close();
      }
    });

    it("prints an empty listing (exit 0) for a configured store with no sessions", async () => {
      const previous = process.env[ENV_VAR];
      process.env[ENV_VAR] = PG_DSN!;
      try {
        const runs = await runCli(["runs"]);
        expect(runs.code).toBe(0);
        expect(runs.out).toContain("no sessions in the configured store");
      } finally {
        if (previous === undefined) delete process.env[ENV_VAR];
        else process.env[ENV_VAR] = previous;
      }
    });
  },
);

/**
 * The built bundle the subprocess tests import — and proof it is not STALE.
 *
 * These two tests are the only ones that can observe process EXIT, so they must
 * run a subprocess, and the only importable artifact is `dist/`. A stale bundle
 * would let them pass while testing code nobody changed, which is worse than not
 * having them: comparing mtimes turns that silent pass into a clear failure.
 */
function freshBuild(): string {
  const built = join(process.cwd(), "dist", "index.js");
  if (!existsSync(built)) {
    throw new Error("run `npm run build` first: the process-exit tests need dist/");
  }
  const builtAt = statSync(built).mtimeMs;
  const newest = newestMtime(join(process.cwd(), "src"));
  if (newest > builtAt) {
    throw new Error(
      "dist/ is older than src/ — run `npm run build`: the process-exit tests " +
        "would otherwise pass against stale code",
    );
  }
  return built;
}

/**
 * The stub OpenAI-shaped client as SOURCE, for the subprocess tests to inline.
 *
 * A subprocess gets its script as a string, so it cannot import a helper from
 * this file; spelling the client out at each call site instead would drift.
 * Deliberately answers from memory: these tests are about what the STORE does,
 * so the host's own LLM call must cost nothing and, crucially, must schedule no
 * timer of its own — a referenced timer anywhere in the host would hold the
 * process open and hide the very bug being tested.
 */
const STUB_CLIENT_SRC = `{
  constructor: { name: "OpenAI" },
  chat: { completions: { create: async () => ({ choices: [{ message: { content: "ok" } }], usage: {} }) } },
  responses: {},
}`;

/** Run `script` as a real ESM subprocess and resolve how it ended. Async (not
 * `spawnSync`) so several can race the same database at once, which is what a
 * cold start is. */
function runScript(
  script: string,
  timeoutMs = 30_000,
): Promise<{ code: number | null; signal: string | null; out: string; err: string }> {
  return new Promise((resolve) => {
    const proc = spawn(process.execPath, ["--input-type=module", "-e", script], {
      timeout: timeoutMs,
    });
    let out = "";
    let err = "";
    proc.stdout.on("data", (d) => (out += String(d)));
    proc.stderr.on("data", (d) => (err += String(d)));
    proc.on("close", (code, signal) => resolve({ code, signal, out, err }));
  });
}

/**
 * A TCP proxy in front of a real database that RESETS one connection at an
 * exact point in the protocol stream, then behaves normally forever after.
 *
 * How: it forwards bytes both ways verbatim until the `nth` client->server
 * chunk containing `marker` (a statement's text — both drivers send it in the
 * clear), at which point it destroys BOTH sockets WITHOUT forwarding that chunk.
 * The write therefore never reached the server, so a writer that retries
 * correctly loses nothing, and one that dies in its backoff loses everything
 * after this point. Killing on a statement rather than on a timer is what makes
 * the injection reproducible instead of a race with the test's own clock.
 *
 * Returns the DSN to point ctxdiff at and a `fired()` the test asserts on, so a
 * proxy that never triggered can never be mistaken for a passing test.
 */
function rstProxy(
  dsn: string,
  marker: string,
  nth: number,
): Promise<{ dsn: string; fired: () => boolean }> {
  return interceptProxy(dsn, marker, nth, "rst");
}

/**
 * The same proxy, injecting the OTHER failure: it stops relaying that
 * connection's traffic in both directions and holds the sockets OPEN forever.
 *
 * That is a wedged box or a partition rather than a reset — the case where no
 * server-side timeout can report itself, because the packets carrying its
 * verdict are the ones being dropped, and therefore the case `pg`'s own
 * `query_timeout` is the only thing that fires on. Every LATER connection is
 * relayed normally, so a store that reopens recovers and one that keeps using
 * the busy client does not.
 */
function stallProxy(
  dsn: string,
  marker: string,
  nth: number,
): Promise<{ dsn: string; fired: () => boolean }> {
  return interceptProxy(dsn, marker, nth, "stall");
}

/** The shared machinery behind `rstProxy`/`stallProxy`: relay verbatim until the
 * `nth` client->server chunk containing `marker`, then apply `mode` to that one
 * connection and relay every later connection normally. */
async function interceptProxy(
  dsn: string,
  marker: string,
  nth: number,
  mode: "rst" | "stall",
): Promise<{ dsn: string; fired: () => boolean }> {
  const target = new URL(dsn);
  const targetPort = Number(target.port || "5432");
  const needle = Buffer.from(marker);
  const sockets: Socket[] = [];
  let seen = 0;
  let fired = false;

  const server = createServer((client) => {
    const upstream = connect(targetPort, target.hostname);
    sockets.push(client, upstream);
    // A deliberately broken connection errors on both ends; that is the point.
    client.on("error", () => undefined);
    upstream.on("error", () => undefined);
    let stalled = false;
    client.on("data", (chunk) => {
      if (stalled) return;
      if (!fired && chunk.includes(needle) && ++seen === nth) {
        fired = true;
        if (mode === "rst") {
          // Destroyed WITHOUT forwarding: the statement never reached the
          // server, so a writer that retries correctly loses nothing.
          client.destroy();
          upstream.destroy();
        } else {
          // Held open and mute: the client waits for an answer that will never
          // come, which is what a partitioned peer looks like.
          stalled = true;
        }
        return;
      }
      upstream.write(chunk);
    });
    upstream.on("data", (chunk) => {
      if (!stalled) client.write(chunk);
    });
    client.on("close", () => upstream.destroy());
    upstream.on("close", () => {
      if (!stalled) client.destroy();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  proxyCleanups.push(() => {
    for (const s of sockets) s.destroy();
    server.close();
  });

  const proxied = new URL(dsn);
  proxied.hostname = "127.0.0.1";
  proxied.port = String(port);
  return { dsn: proxied.toString(), fired: () => fired };
}

/** Every proxy this file opened, torn down after each case so a failed test
 * never leaves a listening socket behind. */
const proxyCleanups: (() => void)[] = [];
afterEach(() => {
  for (const fn of proxyCleanups.splice(0)) fn();
});

/** Newest mtime anywhere under `dir`. */
function newestMtime(dir: string): number {
  let newest = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    newest = Math.max(newest, entry.isDirectory() ? newestMtime(full) : statSync(full).mtimeMs);
  }
  return newest;
}

/** The minimal OpenAI-shaped client the tracer knows how to wrap. */
function stubOpenAI(): object {
  return {
    constructor: { name: "OpenAI" },
    chat: {
      completions: {
        create: async (): Promise<unknown> => ({
          choices: [{ message: { role: "assistant", content: "an answer" } }],
          usage: { prompt_tokens: 12, completion_tokens: 4 },
        }),
      },
    },
    responses: {},
  };
}

/** Run the real CLI in-process, capturing stdout/stderr — the same harness the
 * analyzer-conformance suite uses, so "works against a database" means the
 * actual `npx ctxdiff` code path and not a test-only shortcut. */
async function runCli(argv: string[]): Promise<{ code: number; out: string; err: string }> {
  const out: string[] = [];
  const err: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  // @ts-expect-error narrow override of the write signature for capture
  process.stdout.write = (s: string) => (out.push(String(s)), true);
  // @ts-expect-error narrow override of the write signature for capture
  process.stderr.write = (s: string) => (err.push(String(s)), true);
  try {
    const code = await main(argv);
    return { code, out: out.join(""), err: err.join("") };
  } finally {
    process.stdout.write = origOut;
    process.stderr.write = origErr;
  }
}

describe.skipIf(PG_DSN === undefined)(
  PG_DSN === undefined
    ? "empty store — SKIPPED (set CTXDIFF_TEST_POSTGRES_DSN)"
    : "empty store",
  () => {
    it("openReader on a store with no sessions throws EmptyStoreError", async () => {
      await dropPostgresTables(PG_DSN!);
      const backend = new PostgresStore({ dsn: PG_DSN! });
      await expect(backend.openReader()).rejects.toBeInstanceOf(EmptyStoreError);
    });
  },
);

/**
 * Create a throwaway Postgres role that can SELECT ctxdiff's tables and NOTHING
 * else — no CREATE on the schema — and return a DSN for it plus its cleanup.
 *
 * Resolves null when the test DSN's own role may not create roles, so the test
 * skips honestly instead of failing on somebody's locked-down CI database. The
 * password is random and the role is dropped in the caller's `finally`, so a
 * failed test leaves nothing behind that could log in.
 */
async function createReadOnlyRole(
  dsn: string,
): Promise<{ dsn: string; drop: () => Promise<void> } | null> {
  const pg = await import("pg");
  const name = `ctxdiff_ro_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const password = randomUUID().replace(/-/g, "");
  const admin = new pg.default.Client({ connectionString: dsn });
  await admin.connect();
  try {
    await admin.query(`CREATE ROLE ${name} LOGIN PASSWORD '${password}'`);
  } catch {
    await admin.end();
    return null; // not allowed to create roles here
  }
  try {
    await admin.query(`GRANT CONNECT ON DATABASE ${quoteIdent(currentDatabase(dsn))} TO ${name}`);
    await admin.query(`GRANT USAGE ON SCHEMA public TO ${name}`);
    // SELECT on ctxdiff's tables, and explicitly NOT create.
    await admin.query(`GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${name}`);
    await admin.query(`REVOKE CREATE ON SCHEMA public FROM ${name}`);
    await admin.query(`REVOKE CREATE ON SCHEMA public FROM PUBLIC`);
  } finally {
    await admin.end();
  }

  const url = new URL(dsn);
  url.username = name;
  url.password = password;
  return {
    dsn: url.toString(),
    async drop(): Promise<void> {
      const cleanup = new pg.default.Client({ connectionString: dsn });
      await cleanup.connect();
      try {
        await cleanup.query(`GRANT CREATE ON SCHEMA public TO PUBLIC`);
        await cleanup.query(`REASSIGN OWNED BY ${name} TO CURRENT_USER`);
        await cleanup.query(`DROP OWNED BY ${name}`);
        await cleanup.query(`DROP ROLE IF EXISTS ${name}`);
      } finally {
        await cleanup.end();
      }
    },
  };
}

/** The database a DSN names, for the one GRANT that needs it by name. */
function currentDatabase(dsn: string): string {
  return new URL(dsn).pathname.replace(/^\//, "");
}

/** Double-quote an identifier for a statement that cannot be parameterized
 * (GRANT takes no placeholders). */
function quoteIdent(name: string): string {
  return `"${name.replace(/"/g, '""')}"`;
}

/** Terminate every OTHER backend connected to this database — the test-side
 * stand-in for a failover, a pooler recycle or a restart. */
async function killPostgresBackends(dsn: string): Promise<void> {
  const pg = await import("pg");
  const client = new pg.default.Client({ connectionString: dsn });
  await client.connect();
  try {
    await client.query(
      "SELECT pg_terminate_backend(pid) FROM pg_stat_activity " +
        "WHERE datname = current_database() AND pid <> pg_backend_pid()",
    );
  } finally {
    await client.end();
  }
}

/** The MySQL equivalent: KILL every other connection to this schema. */
async function killMysqlConnections(dsn: string): Promise<void> {
  const mysql = await import("mysql2/promise");
  const conn = await mysql.default.createConnection(dsn);
  try {
    const [rows] = (await conn.query(
      "SELECT id FROM information_schema.processlist " +
        "WHERE db = DATABASE() AND id <> CONNECTION_ID()",
    )) as [{ ID?: number; id?: number }[], unknown];
    for (const row of rows) {
      const id = row.ID ?? row.id;
      try {
        await conn.query(`KILL ${id}`);
      } catch {
        /* the connection may have gone on its own */
      }
    }
  } finally {
    await conn.end();
  }
}

describe("model roll-up fallback (Python `or`, not JS `??`)", () => {
  it("falls through an EMPTY model to the Bedrock `modelId` spelling", async () => {
    // `params.get("model") or params.get("modelId")` in Python: an empty string
    // is falsy there, so it falls through. `??` would have rolled up "" — which
    // `noteModel` then ignores — and the run would report no model at all.
    const dir = mkdtempSync(join(tmpdir(), "ctxdiff-rollup-"));
    tempDirs.push(dir);
    const store = new SQLiteStore({ path: join(dir, "rollup.ctrace") });
    const s = store.openSession({ project: "rollup", provider: "bedrock", model: "" });
    try {
      s.recordCall({
        seq: 1,
        params: { model: "", modelId: "anthropic.claude-sonnet-4" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [block("r-1", "hi", 0)],
      });
      expect(s.getRun().models).toEqual(["anthropic.claude-sonnet-4"]);
    } finally {
      s.close();
    }
  });
});

/**
 * The session/agent CLI, against EVERY backend.
 *
 * The selectors (`--project`/`--session`/`--agent`) are the one place the CLI
 * reaches past "the newest session" and asks a store to answer for a NAMED one.
 * That path is trivially right for `node:sqlite` (one file, one connection,
 * every read takes a session id) and genuinely different for a network store,
 * where the synchronous analyzers consume an in-memory SNAPSHOT taken of a
 * chosen session rather than of the handle's own binding — and where a
 * cross-session diff has to materialize two of them. So the whole surface is
 * asserted against every backend, with the live ones skipping loudly by name.
 */
for (const backendCase of CASES) {
  const title = backendCase.enabled
    ? `session/agent CLI: ${backendCase.name}`
    : `session/agent CLI: ${backendCase.name} — SKIPPED (${backendCase.skipHint})`;

  describe.skipIf(!backendCase.enabled)(title, () => {
    let store: StoreBackend;
    /** The two seeded session ids, oldest ("good") first. */
    let good: string;
    let bad: string;
    let cwd: string;
    let originalCwd: string;

    beforeEach(async () => {
      await backendCase.reset();
      store = backendCase.create();
      // Seed the PROJECT fixture: two sessions, each with a researcher and a
      // writer, differing in exactly one block of turn 3.
      const ids: string[] = [];
      for (const [startedAt, tail] of [
        ["2026-07-20T09:15:00+00:00", "good"],
        ["2026-07-21T18:42:30+00:00", "bad"],
      ] as [string, string][]) {
        const s = await store.openSession({
          project: "pipeline", provider: "openai", model: "", startedAt,
        });
        try {
          await s.recordCall({
            seq: 1, params: { model: "gpt-4o" },
            usage: { prompt_tokens: 100, completion_tokens: 20 },
            latencyMs: 5, error: null,
            callBlocks: [block("r-sys", "you are the researcher", 0, { role: "system" }),
                         block("r-q", "find facts", 1)],
            agent: "researcher",
          });
          await s.recordCall({
            seq: 2, params: { model: "gpt-4o" },
            usage: { prompt_tokens: 40, completion_tokens: 8 },
            latencyMs: 6, error: null,
            callBlocks: [block("w-sys", "you are the writer", 0, { role: "system" }),
                         block("w-q", "write it up", 1)],
            agent: "writer",
          });
          await s.recordCall({
            seq: 3, params: { model: "gpt-4o" }, usage: null, latencyMs: 7, error: null,
            callBlocks: [block("r-sys", "you are the researcher", 0, { role: "system" }),
                         block("r-q", "find facts", 1),
                         block(`r-more-${tail}`, `more detail (${tail})`, 2)],
            agent: "researcher",
          });
          ids.push((await s.getRun()).id);
        } finally {
          await s.close();
        }
      }
      [good, bad] = ids;
      // An EMPTY cwd, so the CLI can only resolve the configured store — a
      // stray *.ctrace would let the file fallback answer instead.
      cwd = mkdtempSync(join(tmpdir(), "ctxdiff-sel-cwd-"));
      tempDirs.push(cwd);
      originalCwd = process.cwd();
      process.chdir(cwd);
      configure({ store });
    });

    afterEach(() => {
      configure({ store: null });
      process.chdir(originalCwd);
    });

    it("sessions lists every session with its local time and agents", async () => {
      const r = await runCli(["sessions"]);
      expect(r.code).toBe(0);
      const lines = r.out.trimEnd().split("\n");
      expect(lines).toHaveLength(2);
      // The label differs by backend on purpose — a file store labels by
      // filename, a database by short session id — but the id is always
      // selectable from the row.
      expect(lines[0]).toContain(good.slice(0, 12));
      expect(lines[1]).toContain(bad.slice(0, 12));
      for (const line of lines) {
        expect(line).toContain("project=pipeline");
        expect(line).toContain("turns=3");
        expect(line).toContain("agents=researcher, writer");
      }
    });

    it("agents rolls every session in the store up per agent", async () => {
      const r = await runCli(["agents"]);
      expect(r.code).toBe(0);
      const lines = r.out.trimEnd().split("\n");
      expect(lines[0]).toBe("researcher  sessions=2  calls=4  tokens=240");
      expect(lines[1]).toBe("writer  sessions=2  calls=2  tokens=96");
    });

    it("--session resolves what a bare command refuses to guess", async () => {
      const ambiguous = await runCli(["tokens"]);
      expect(ambiguous.code).toBe(2);
      expect(ambiguous.out).toBe("");
      expect(ambiguous.err).toContain("pass --session to pick one");
      expect(ambiguous.err).toContain(good.slice(0, 12));
      expect(ambiguous.err).toContain(bad.slice(0, 12));

      const scoped = await runCli(["tokens", "--session", bad.slice(0, 12), "--agent", "writer"]);
      expect(scoped.code).toBe(0);
      expect(scoped.out).toContain("turn 2 ·");
      expect(scoped.out).not.toContain("turn 1 ·");
    });

    it("cross-session diff compares one agent's turn across two runs", async () => {
      const r = await runCli([
        "diff", "--session", `${good}:3`, "--session", `${bad}:3`, "--agent", "researcher",
      ]);
      expect(r.code).toBe(0);
      const lines = r.out.trimEnd().split("\n");
      expect(lines[0]).toBe(
        `── ${good.slice(0, 12)} · researcher · turn 3  →  ` +
          `${bad.slice(0, 12)} · researcher · turn 3 ──`,
      );
      // Char-level inline diff: the shared trailing "d" of good/bad stays equal.
      expect(r.out).toContain("[-goo-]");
      expect(r.out).toContain("{+ba+}");
    });
  });
}

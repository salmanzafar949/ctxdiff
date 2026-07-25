/**
 * Hardening regressions for the project-scoped store — the defects that only
 * became reachable once ONE stable `./<project>.ctrace` started accumulating
 * many sessions and many concurrent writers.
 *
 * Each test pins a property that is invisible in normal single-writer use but
 * user-visible the moment a file is contended, reopened, or pre-existing:
 *   1. a contended write must not freeze Node's single event loop for seconds,
 *      and a failed store setup must be latched (never re-retried per wrap);
 *   2. `close()` must be idempotent — the one public method on a fail-open
 *      library that must never throw into a `finally`/exit handler;
 *   4. a model roll-up whose write FAILED must be retried by a later call
 *      (never marked seen before it is committed), and a roll-up failure must
 *      not escape `recordCall` after the call itself is already persisted;
 *   5. appending a v2 session into a physically-v1 file must upgrade the file
 *      in place rather than silently dropping agent/step/provider;
 *   6. a persistently failing recorder must warn ONCE, not per call.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { DatabaseSync } from "node:sqlite";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import OpenAI from "openai";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";
import { Recorder } from "../src/capture/recorder.js";
import { OpenAIAdapter } from "../src/capture/openai.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-hard-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  vi.restoreAllMocks();
  for (const p of created.splice(0)) {
    for (const suffix of ["", "-wal", "-shm"]) {
      try {
        rmSync(p + suffix, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    }
  }
});

/**
 * Seed a real project file, then hold its WRITE lock open on a SECOND
 * connection — the in-process stand-in for "another process is mid-commit".
 * `BEGIN EXCLUSIVE` takes the WAL write lock immediately, so every subsequent
 * write from any other connection hits SQLITE_BUSY until `release()` runs.
 */
function holdWriteLock(path: string): () => void {
  const seed = CTrace.create(path, "held", "openai", "");
  seed.close();
  const holder = new DatabaseSync(path);
  holder.exec("PRAGMA journal_mode = WAL");
  holder.exec("BEGIN EXCLUSIVE");
  let released = false;
  return () => {
    if (released) return;
    released = true;
    try {
      holder.exec("ROLLBACK");
    } catch {
      /* ignore */
    }
    holder.close();
  };
}

/** A real OpenAI client whose HTTP layer is a canned stub, so `wrap()` +
 * `create()` exercise the full record path with no network. */
function stubClient(): OpenAI {
  const fetchFn = async () =>
    new Response(
      JSON.stringify({
        id: "cmpl",
        object: "chat.completion",
        model: "gpt-4o",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
      { headers: { "content-type": "application/json" } },
    );
  return new OpenAI({ apiKey: "test", fetch: fetchFn as unknown as typeof fetch });
}

describe("store: contended writes stay off the event loop's critical path", () => {
  it("gives up on a held lock in about a second, not tens of seconds", () => {
    const path = tmpTrace();
    const release = holdWriteLock(path);
    try {
      const t0 = performance.now();
      // Every attempt blocks inside node:sqlite SYNCHRONOUSLY (no await is
      // possible), so this elapsed time is time the Node event loop is FROZEN.
      // The retry budget (busy_timeout x attempts + backoff) must therefore stay
      // small: ~1s total, never the ~32s the 5000ms/6-attempt budget produced.
      expect(() =>
        CTrace.openOrCreateSession(path, "proj", "openai", "", new Date().toISOString()),
      ).toThrow();
      const elapsed = performance.now() - t0;
      expect(elapsed).toBeLessThan(2000);
    } finally {
      release();
    }
  });

  it("latches a failed setup so a second wrap() does not re-run the retry loop", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const release = holdWriteLock(path);
    try {
      const tracer = init("proj", { path });

      const t0 = performance.now();
      tracer.wrap(stubClient(), { agent: "a" });
      const first = performance.now() - t0;

      const t1 = performance.now();
      tracer.wrap(stubClient(), { agent: "b" });
      const second = performance.now() - t1;

      // The first wrap really did pay the (bounded) retry budget...
      expect(first).toBeGreaterThan(100);
      // ...and the second paid NOTHING: the setup failure is latched, so we
      // never re-enter the blocking retry loop once per wrap for the whole run.
      expect(second).toBeLessThan(100);
      tracer.close();
    } finally {
      release();
    }
  });
});

describe("store: close() is idempotent", () => {
  it("closing a CTrace twice does not throw", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    ct.close();
    expect(() => ct.close()).not.toThrow();
  });

  it("closing a Tracer twice does not throw", async () => {
    const path = tmpTrace();
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(stubClient(), { agent: "a" }) as OpenAI;
    await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    tracer.close();
    // Double-close is a normal JS shape (`finally` + `process.on('exit')`); a
    // fail-open library must survive it.
    expect(() => tracer.close()).not.toThrow();
  });
});

describe("store: the model roll-up survives a failed write", () => {
  it("retries a model whose UPDATE failed on the next call that sees it", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    const holder = new DatabaseSync(path);
    holder.exec("PRAGMA journal_mode = WAL");
    holder.exec("BEGIN EXCLUSIVE");
    // Contended: the UPDATE exhausts its retries and raises.
    expect(() => ct.noteModel("gpt-4o")).toThrow();
    holder.exec("ROLLBACK");
    holder.close();

    // The model must NOT have been marked seen by the failed attempt, so this
    // second sighting still writes it. (Before the fix `models` stayed [] for
    // the life of the run.)
    ct.noteModel("gpt-4o");
    expect(ct.getRun().models).toEqual(["gpt-4o"]);
    ct.close();
  });

  it("does not throw out of recordCall when the roll-up fails after the call is committed", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    vi.spyOn(ct, "noteModel").mockImplementation(() => {
      throw new Error("roll-up boom");
    });
    // The call itself was already persisted; a roll-up failure must not surface
    // as "failed to record call" for a call that WAS recorded.
    expect(() =>
      ct.recordCall({
        seq: 1,
        params: { model: "gpt-4o" },
        usage: null,
        latencyMs: 1,
        error: null,
        callBlocks: [],
        agent: "a",
      }),
    ).not.toThrow();
    expect(ct.getCalls()).toHaveLength(1);
    ct.close();
  });
});

// A verbatim copy of the v1 `call` DDL (no agent/step/provider), used to forge a
// REAL physically-v1 file so the append-into-v1 path runs against actual on-disk
// v1 bytes rather than a mock. Mirrors tests/test_store.py's `_V1_DDL`.
const V1_DDL = `
CREATE TABLE run (
  id TEXT PRIMARY KEY, project TEXT NOT NULL, started_at TEXT NOT NULL,
  provider TEXT NOT NULL, models TEXT NOT NULL, ctxdiff_version TEXT NOT NULL,
  schema_version INTEGER NOT NULL);
CREATE TABLE call (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL,
  params TEXT NOT NULL, usage TEXT, latency_ms INTEGER, error TEXT,
  UNIQUE(run_id, seq));
CREATE TABLE block (
  content_hash TEXT PRIMARY KEY, role TEXT NOT NULL, kind TEXT NOT NULL,
  text TEXT NOT NULL, token_count INTEGER NOT NULL, token_method TEXT NOT NULL);
CREATE TABLE call_block (
  call_id TEXT NOT NULL, block_id TEXT NOT NULL, position INTEGER NOT NULL,
  label TEXT NOT NULL, label_source TEXT NOT NULL, PRIMARY KEY (call_id, position));
`;

function buildV1File(path: string): void {
  const db = new DatabaseSync(path);
  db.exec(V1_DDL);
  db.exec(
    "INSERT INTO run VALUES ('run1','proj','2026-01-01T00:00:00Z','openai'," +
      "'[\"gpt-4o\"]','0.1.0',1)",
  );
  db.exec("INSERT INTO call VALUES ('call1','run1',1,'{\"model\":\"m\"}',NULL,10,NULL)");
  db.close();
}

describe("store: appending into a physically-v1 file", () => {
  it("upgrades the call table in place so agent/step/provider are preserved", () => {
    const path = tmpTrace();
    buildV1File(path);

    const ct = CTrace.openOrCreateSession(
      path,
      "proj",
      "openai",
      "",
      new Date().toISOString(),
    );
    ct.recordCall({
      seq: 1,
      params: { model: "gpt-4o" },
      usage: null,
      latencyMs: 5,
      error: null,
      callBlocks: [],
      agent: "planner",
      step: "retrieve",
      provider: "openai",
    });
    ct.close();

    const r = CTrace.open(path);
    const sessions = r.listSessions();
    expect(sessions).toHaveLength(2); // the v1 run plus the appended session
    const appended = sessions[sessions.length - 1];
    const calls = r.getCalls(appended.id);
    expect(calls).toHaveLength(1);
    // The whole point: attribution written under a schema_version=2 run row is
    // actually STORED, not silently dropped by the v1 7-column write shape.
    expect(calls[0].agent).toBe("planner");
    expect(calls[0].step).toBe("retrieve");
    expect(calls[0].provider).toBe("openai");
    expect(appended.agents).toEqual(["planner"]);
    // The pre-existing v1 call is untouched and still readable.
    const legacy = r.getCalls("run1");
    expect(legacy).toHaveLength(1);
    expect(legacy[0].agent).toBeNull();
    r.close();
  });
});

describe("recorder: a persistently failing record warns once", () => {
  it("logs the record failure once, not on every call", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    // A store whose write always fails: every record() takes the catch branch.
    vi.spyOn(ct, "recordCall").mockImplementation(() => {
      throw new Error("store boom");
    });
    const rec = new Recorder(ct, new OpenAIAdapter(), null);
    for (let i = 1; i <= 3; i++) {
      rec.record({
        seq: i,
        kwargs: { model: "gpt-4o", messages: [{ role: "user", content: "hi" }] },
        response: null,
        latencyMs: 1,
        error: null,
        tagged: [],
      });
    }
    const failures = warn.mock.calls.filter((c) =>
      String(c[0]).includes("failed to record call"),
    );
    expect(failures).toHaveLength(1);
    ct.close();
  });
});

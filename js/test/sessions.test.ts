/**
 * Project-scoped storage tests — the JS parity for the Python 2311181 change:
 * one `.ctrace` per PROJECT accumulating MANY sessions, appended multi-writer-
 * safely, with UTC-canonical timestamps and a back-compat multi-session reader
 * whose session-less reads default to the NEWEST session.
 *
 * Covers: two `init`s append two isolated sessions to one file; back-compat of a
 * single-session file; `listSessions` agents/turn counts (oldest-first);
 * default-read == newest; UTC timestamp round-trip; and — gated on a built dist,
 * like the venv-gated conformance suite — genuine multi-PROCESS concurrent
 * writers not corrupting or losing sessions.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync, existsSync, writeFileSync, mkdtempSync } from "node:fs";
import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";
import OpenAI from "openai";
import { init } from "../src/trace.js";
import { CTrace, parseStartedAt } from "../src/store/ctrace.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-sess-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  vi.restoreAllMocks();
  for (const p of created.splice(0)) {
    // Remove the main db plus any WAL/SHM sidecars. `recursive` matters: the
    // fail-open test registers a temp DIRECTORY here, and a non-recursive
    // rmSync silently no-ops on it, leaking a tmpdir per run.
    for (const suffix of ["", "-wal", "-shm"]) {
      try {
        rmSync(p + suffix, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    }
  }
});

/** A real OpenAI client whose HTTP layer is a canned stub echoing a fixed
 * completion, so a full `wrap()` → `create()` records one call with no network. */
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

/** Run one wrapped completion through a fresh Tracer bound to `path`, under
 * `agent`, recording `n` calls into one appended session. Returns after close. */
async function recordSession(path: string, agent: string, n = 1): Promise<void> {
  const tracer = init("proj", { path });
  const wrapped = tracer.wrap(stubClient(), { agent }) as OpenAI;
  for (let i = 0; i < n; i++) {
    await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: `hi ${agent} ${i}` }],
    });
  }
  tracer.close();
}

describe("project-scoped store: appending sessions", () => {
  it("two init()s append two isolated sessions to one project file", async () => {
    const path = tmpTrace();
    await recordSession(path, "alpha", 1);
    await recordSession(path, "beta", 2);

    const ct = CTrace.open(path);
    const sessions = ct.listSessions();
    expect(sessions).toHaveLength(2);

    // Oldest-first ordering, distinct run ids.
    expect(sessions[0].id).not.toBe(sessions[1].id);
    expect(sessions[0].agents).toEqual(["alpha"]);
    expect(sessions[1].agents).toEqual(["beta"]);
    expect(sessions[0].turnCount).toBe(1);
    expect(sessions[1].turnCount).toBe(2);

    // Sessions are isolated: each session's calls belong only to it.
    expect(ct.getCalls(sessions[0].id)).toHaveLength(1);
    expect(ct.getCalls(sessions[1].id)).toHaveLength(2);
    for (const c of ct.getCalls(sessions[0].id)) expect(c.agent).toBe("alpha");
    for (const c of ct.getCalls(sessions[1].id)) expect(c.agent).toBe("beta");
    ct.close();
  });

  it("a session-less read defaults to the NEWEST session", async () => {
    const path = tmpTrace();
    await recordSession(path, "old", 1);
    await recordSession(path, "new", 3);

    const ct = CTrace.open(path);
    // No selector → newest session (the one just made): 3 calls, all agent "new".
    expect(ct.getCalls()).toHaveLength(3);
    for (const c of ct.getCalls()) expect(c.agent).toBe("new");
    // getRun() with no selector is the newest run too.
    const run = ct.getRun();
    const sessions = ct.listSessions();
    expect(run.id).toBe(sessions[sessions.length - 1].id);
    ct.close();
  });

  it("a single-session file reads identically (back-compat)", async () => {
    const path = tmpTrace();
    await recordSession(path, "solo", 2);

    const ct = CTrace.open(path);
    const sessions = ct.listSessions();
    expect(sessions).toHaveLength(1);
    expect(sessions[0].turnCount).toBe(2);
    // Session-less reads == the only session.
    expect(ct.getCalls()).toHaveLength(2);
    expect(ct.getRun().project).toBe("proj");
    expect(ct.getCalls(sessions[0].id)).toEqual(ct.getCalls());
    ct.close();
  });

  it("listSessions collects distinct agents in first-appearance order", () => {
    const path = tmpTrace();
    // Drive the store directly to place several agents on one session's calls.
    const ct = CTrace.create(path, "multi", "openai", "", new Date().toISOString());
    const order = ["planner", "worker", "planner", "critic", "worker"];
    order.forEach((agent, i) => {
      ct.recordCall({
        seq: i + 1,
        params: { model: "gpt-4o" },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [],
        agent,
      });
    });
    ct.close();

    const r = CTrace.open(path);
    const sessions = r.listSessions();
    expect(sessions).toHaveLength(1);
    expect(sessions[0].turnCount).toBe(5);
    // Deduped, first-appearance order — not sorted, not per-call.
    expect(sessions[0].agents).toEqual(["planner", "worker", "critic"]);
    r.close();
  });
});

describe("project-scoped store: UTC-canonical timestamps", () => {
  it("init writes a UTC ...Z started_at that round-trips", async () => {
    const path = tmpTrace();
    await recordSession(path, "a", 1);
    const ct = CTrace.open(path);
    const startedAt = ct.listSessions()[0].startedAt;
    ct.close();
    // Canonical UTC with a trailing Z (from Date.prototype.toISOString()).
    expect(startedAt).toMatch(/^\d{4}-\d{2}-\d{2}T.*Z$/);
    const d = parseStartedAt(startedAt);
    expect(d.getTime()).toBe(new Date(startedAt).getTime());
  });

  it("parseStartedAt tolerates Z, an explicit offset, and a naive UTC value", () => {
    // Trailing Z.
    expect(parseStartedAt("2026-07-24T12:00:00Z").getTime()).toBe(
      Date.UTC(2026, 6, 24, 12, 0, 0),
    );
    // Explicit +04:00 offset resolves to the same instant 4h earlier in UTC.
    expect(parseStartedAt("2026-07-24T16:00:00+04:00").getTime()).toBe(
      Date.UTC(2026, 6, 24, 12, 0, 0),
    );
    // Legacy naive value (no zone) is assumed UTC, NOT local — the whole point.
    expect(parseStartedAt("2026-07-24T12:00:00").getTime()).toBe(
      Date.UTC(2026, 6, 24, 12, 0, 0),
    );
    // Fractional seconds + Z, as toISOString() emits.
    expect(parseStartedAt("2026-07-24T12:00:00.123Z").getTime()).toBe(
      Date.UTC(2026, 6, 24, 12, 0, 0, 123),
    );
  });
});

describe("project-scoped store: fail-open on setup failure", () => {
  it("a store-setup failure never breaks the host call (records nothing)", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    // Point the store at a DIRECTORY path: `new DatabaseSync(dir)` cannot open it
    // and throws, forcing the wrap()-time setup fail-open branch.
    const dir = mkdtempSync(join(tmpdir(), "ctxdiff-faildir-"));
    created.push(dir);

    const tracer = init("failopen", { path: dir });
    const wrapped = tracer.wrap(stubClient(), { agent: "a" }) as OpenAI;

    // The host call still runs and returns its real response.
    const res = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.choices[0].message.content).toBe("ok");
    // A second call also works — capture stays degraded, host stays intact.
    const res2 = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "again" }],
    });
    expect(res2.choices[0].message.content).toBe("ok");

    tracer.close();

    // Degradation was warned exactly once (one-time guard), never thrown.
    const degraded = warn.mock.calls.filter((c) =>
      String(c[0]).includes("capture degraded (store setup failed)"),
    );
    expect(degraded).toHaveLength(1);
  });
});

// Genuine multi-PROCESS concurrent writers. Gated on a built dist (like the
// venv-gated conformance suite): the workers are separate `node` processes that
// import the compiled CTrace, so this runs green under `npm run build && npm
// test` and is reported as an explicit SKIP if dist is absent.
const distIndex = join(process.cwd(), "dist", "index.js");
const hasDist = existsSync(distIndex);

describe("project-scoped store: multi-writer safety", () => {
  it.skipIf(!hasDist)(
    "concurrent Node processes appending to one file don't corrupt or lose sessions",
    async () => {
      const path = tmpTrace();
      const workerPath = join(tmpdir(), `ctxdiff-worker-${randomUUID()}.mjs`);
      created.push(workerPath);
      // Worker: open/create a session in the shared project file and record a few
      // calls, exercising the real busy_timeout-first + WAL + bounded-retry
      // append path under cross-process lock contention.
      writeFileSync(
        workerPath,
        `
const { CTrace } = await import(process.argv[2]);
const [, , , path, agent] = process.argv;
const ct = CTrace.openOrCreateSession(path, "proj", "openai", "", new Date().toISOString());
for (let s = 1; s <= 3; s++) {
  ct.recordCall({ seq: s, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [], agent });
}
ct.close();
`,
      );

      const N = 6;
      const distUrl = pathToFileURL(distIndex).href;
      // Spawn them all at once (a barrier) so they genuinely race on file locks.
      const runs = Array.from({ length: N }, (_, i) => {
        return new Promise<number>((resolve, reject) => {
          const child = spawn(
            process.execPath,
            [workerPath, distUrl, path, `agent-${i}`],
            { stdio: ["ignore", "ignore", "pipe"] },
          );
          let stderr = "";
          child.stderr.on("data", (d) => (stderr += d));
          child.on("exit", (code) =>
            code === 0
              ? resolve(0)
              : reject(new Error(`worker ${i} exited ${code}: ${stderr}`)),
          );
          child.on("error", reject);
        });
      });
      await Promise.all(runs);

      // Every session survived, none corrupted: N sessions, each with 3 calls,
      // each carrying its own distinct agent.
      const ct = CTrace.open(path);
      const sessions = ct.listSessions();
      expect(sessions).toHaveLength(N);
      const agents = new Set<string>();
      for (const s of sessions) {
        expect(s.turnCount).toBe(3);
        expect(s.agents).toHaveLength(1);
        agents.add(s.agents[0]);
      }
      // All N distinct agents present — no session's writes were lost or merged.
      expect(agents.size).toBe(N);
      ct.close();
    },
  );
});

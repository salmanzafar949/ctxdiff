/**
 * Viewer tests: the self-contained HTML dashboard. Assert (a) ZERO external
 * requests (no http/https/protocol-relative/cdn refs), (b) an XSS payload in
 * trace text is neutralized (escaped in the JSON island, never live markup,
 * title HTML-escaped), (c) cross-language: the JS-rendered HTML is byte-
 * identical to the Python viewer's on a shared `.ctrace`, and (d) the demo
 * generator + exporter produce a file. Keeps the venv-spawn discipline.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { CTrace } from "../src/store/ctrace.js";
import { contentHash, basicLabel } from "../src/models.js";
import { countTokens } from "../src/tokenize.js";
import {
  buildPayload,
  buildProjectPayload,
  exportHtml,
  type ProjectPayloadOptions,
} from "../src/viewer/export.js";
import { PAGE, renderPage } from "../src/viewer/template.js";
import { pyJsonDumps } from "../src/viewer/pyjson.js";
import { buildDemoTrace } from "../src/demo.js";
import { makeFixtures } from "./helpers/fixtures.js";
import { bootPage } from "./helpers/page.js";
import { formatLocal } from "../src/selectors.js";

const repoRoot = resolve(process.cwd(), "..");
const venvPython = join(repoRoot, "venv", "bin", "python");
const pySrc = join(repoRoot, "src");
const hasVenv = existsSync(venvPython);
/** The built CLI, spawned by the one parity case that has to go through
 * `--session` (only the CLI can pin an old run as the dashboard's focus). */
const distCli = resolve(process.cwd(), "dist", "cli.js");
const hasBuild = existsSync(distCli);

let dir: string;
let fx: ReturnType<typeof makeFixtures>;

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-viewer-"));
  fx = makeFixtures(dir);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Build a CallBlock from [role, kind, text], like the recorder. */
function blk(role: string, kind: string, text: string, position: number) {
  const [tokenCount, tokenMethod] = countTokens(text, "openai");
  const [label, labelSource] = basicLabel(role, kind, text, []);
  return {
    block: { contentHash: contentHash(role, kind, text), role, kind, text, tokenCount, tokenMethod },
    position,
    label,
    labelSource,
  };
}

/** Extract the JSON island text and everything outside it from a rendered page. */
function splitIsland(html: string): { island: string; outside: string } {
  const open = '<script id="ctxdiff-data" type="application/json">';
  const start = html.indexOf(open) + open.length;
  const end = html.indexOf("</script>", start);
  return { island: html.slice(start, end), outside: html.slice(0, start) + html.slice(end) };
}

describe("viewer: self-contained (zero external requests)", () => {
  it("emits no http/https/protocol-relative/cdn references", () => {
    const out = join(dir, "mt.html");
    exportHtml(fx.multiturn, out);
    const html = readFileSync(out, "utf-8");
    expect(html).not.toMatch(/https?:\/\//);
    expect(html).not.toMatch(/src=["']\/\//);
    expect(html).not.toMatch(/href=["']\/\//);
    expect(html).not.toMatch(/\/\/cdn/);
    // Sanity: it IS a full self-contained document with the inline data island.
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain('<script id="ctxdiff-data" type="application/json">');
  });
});

describe("viewer: XSS-safe", () => {
  it("neutralizes markup/script payloads in trace text and escapes the title", () => {
    const path = join(dir, "xss.ctrace");
    const ct = CTrace.create(path, "xss-<script>proj", "openai", "", "2026-07-24T00:00:00Z");
    const evilClose = "</script><script>alert('pwned')</script>";
    const evilAttr = '"><img src=x onerror=alert(1)>';
    ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("system", "message", evilClose, 0), blk("user", "message", evilAttr, 1)] });
    ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("system", "message", evilClose, 0), blk("user", "message", "safe", 1)] });
    ct.close();

    const out = join(dir, "xss.html");
    exportHtml(path, out);
    const html = readFileSync(out, "utf-8");
    const { island, outside } = splitIsland(html);

    // The data island cannot be broken out of: every `</` (incl. `</script>`)
    // in trace text is escaped to `<\/`.
    expect(island).not.toContain("</script");
    expect(island).toContain("<\\/script");
    // The onerror payload exists ONLY inside the island (as escaped JSON), never
    // as live markup elsewhere in the document.
    expect(outside).not.toContain("onerror=alert");
    expect(island).toContain('\\"><img'); // JSON-escaped quote — inert
    // The project name in <title> is HTML-escaped.
    expect(html).toContain("<title>ctxdiff — xss-&lt;script&gt;proj</title>");
    // The payload DOES appear inside the JSON island (as inert JSON-string text
    // the browser reads via textContent, never parses as HTML) — that's fine —
    // but there is NO live `<img ...onerror>` element anywhere OUTSIDE it.
    expect(outside).not.toMatch(/<img[^>]*onerror/i);
    expect(outside).not.toContain("<script>alert");
  });
});

describe("viewer: XSS-safe against $-expansion in the marker replacement", () => {
  // `String.prototype.replace(str, replacement)` expands `$'`/`$\``/`$&`/`$$`
  // when the replacement is a STRING. Since `dataJson` (block text) and the
  // project title are attacker-influenced replacements, a `$'` would expand to
  // the page tail (which has a real `</script>`), closing the JSON island early
  // and rendering following markup live. renderPage must use function replacers.

  it("a $'-prefixed <img onerror> in block text does not break out of the island", () => {
    const path = join(dir, "xss-dollar.ctrace");
    const ct = CTrace.create(path, "proj", "openai", "", "2026-07-24T00:00:00Z");
    // All four expansion forms plus the live-markup payload after $'.
    const evil = "BREAK$'<img src=x onerror=alert(document.domain)>END $$ $& $` tail";
    ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("user", "message", evil, 0)] });
    ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("user", "message", "safe", 0)] });
    ct.close();

    const out = join(dir, "xss-dollar.html");
    exportHtml(path, out);
    const html = readFileSync(out, "utf-8");

    // Exactly two real </script> closers (the data island + the page script).
    // A `$'`-driven early close would push this higher.
    expect((html.match(/<\/script>/g) ?? []).length).toBe(2);
    // The onerror payload must NOT leak outside the JSON island as live markup.
    const { outside } = splitIsland(html);
    expect(outside).not.toMatch(/<img[^>]*onerror/i);
    expect(outside).not.toContain("onerror");
  });

  it("a project title containing $' / $& / $$ / $` does not break out", () => {
    const path = join(dir, "xss-title.ctrace");
    const ct = CTrace.create(path, "p$'$&$$$`<script>x</script>", "openai", "", "2026-07-24T00:00:00Z");
    ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("user", "message", "hi", 0)] });
    ct.close();

    const out = join(dir, "xss-title.html");
    exportHtml(path, out);
    const html = readFileSync(out, "utf-8");

    // The `$`-forms are preserved verbatim (no expansion), and the <script> in
    // the title is HTML-escaped — so no live element and exactly two closers.
    expect(html).toContain("<title>ctxdiff — p$&#x27;$&amp;$$$`&lt;script&gt;x&lt;/script&gt;</title>");
    expect((html.match(/<\/script>/g) ?? []).length).toBe(2);
    expect(html).not.toMatch(/<script>x<\/script>/); // not injected live
  });

  it.skipIf(!hasVenv)("stays byte-identical to Python's export for a $-payload trace", () => {
    const path = join(dir, "xss-parity.ctrace");
    const ct = CTrace.create(path, "proj$'$&", "openai", "", "2026-07-24T00:00:00Z");
    const evil = "BREAK$'<img src=x onerror=alert(1)>END $$ $& $` tail";
    ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("user", "message", evil, 0)] });
    ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [blk("user", "message", "safe", 0)] });
    ct.close();

    const jsOut = join(dir, "xss-parity-js.html");
    exportHtml(path, jsOut);
    const jsHtml = readFileSync(jsOut, "utf-8");

    const pyOut = join(dir, "xss-parity-py.html");
    const proc = spawnSync(
      venvPython,
      ["-c", `from ctxdiff.viewer.export import export_html; export_html(${JSON.stringify(path)}, ${JSON.stringify(pyOut)})`],
      { encoding: "utf8", env: { ...process.env, PYTHONPATH: pySrc } },
    );
    expect(proc.status, `python export failed: ${proc.stderr}`).toBe(0);
    expect(jsHtml).toBe(readFileSync(pyOut, "utf-8"));
  });
});

describe("viewer: payload shape", () => {
  it("reduces params to model only and embeds precomputed analyzer output", () => {
    const ct = CTrace.open(fx.multiturn);
    try {
      const p = buildPayload(ct) as {
        run: { project: string };
        calls: { params: Record<string, unknown>; usage: unknown }[];
        diffs: unknown[];
        tokens: { bloat: unknown };
        cache: unknown;
        stats: { distinct_blocks: number };
      };
      expect(p.run.project).toBe("research");
      // params carry ONLY the model (temperature etc. stripped from the artifact)
      expect(Object.keys(p.calls[0].params)).toEqual(["model"]);
      expect(p.calls[0].params.model).toBe("gpt-4o");
      expect(p.diffs.length).toBe(2); // adjacent-pair diffs
      expect(p.tokens.bloat).not.toBeNull(); // unused schemas present
      expect(p.stats.distinct_blocks).toBeGreaterThan(0);
    } finally {
      ct.close();
    }
  });
});

describe.skipIf(!hasVenv)("viewer: cross-language HTML byte-identical to Python", () => {
  it("renders identical HTML for the same trace", () => {
    // `project` and `edge` are the MULTI-SESSION / MULTI-AGENT cases: they are
    // what exercise the three-level payload — the level-1 agent index, the
    // level-2 session index, and the per-session `details` map — where a
    // divergence in key order, aggregate arithmetic, session ORDERING or the
    // detail cap between the two SDKs would show up as different bytes.
    for (const ctrace of [fx.multiturn, fx.multiagent, fx.dynamic, fx.project, fx.edge]) {
      const jsOut = join(dir, "js-render.html");
      exportHtml(ctrace, jsOut);
      const jsHtml = readFileSync(jsOut, "utf-8");

      const pyOut = join(dir, "py-render.html");
      const proc = spawnSync(
        venvPython,
        ["-c", `from ctxdiff.viewer.export import export_html; export_html(${JSON.stringify(ctrace)}, ${JSON.stringify(pyOut)})`],
        { encoding: "utf8", env: { ...process.env, PYTHONPATH: pySrc } },
      );
      expect(proc.status, `python export failed: ${proc.stderr}`).toBe(0);
      const pyHtml = readFileSync(pyOut, "utf-8");

      expect(jsHtml, `HTML mismatch for ${ctrace}`).toBe(pyHtml);
    }
  });

  it("renders identical HTML for the names that break a name-keyed object", () => {
    // The three shapes where the two SDKs' object handling can diverge even
    // though the trace is the same file: an agent named `__proto__` (which a
    // plain JS object swallows, dropping a whole `by_agent` entry the Python
    // dict keeps), a PROJECT named `__CTXDIFF_DATA__` (whose title substitution
    // collides with the page's own data marker — differently in each language),
    // and sessions written out of chronological order (where "first/last seen"
    // must be a min/max, not the ends of the insert-ordered list).
    for (const ctrace of [
      writeProtoAgentSession(dir, "proto-parity.ctrace"),
      writeMarkerNamedProject(dir, "marker-parity.ctrace"),
      writeSkewedSessions(dir, "skew-parity.ctrace"),
    ]) {
      const jsOut = join(dir, "js-hard.html");
      exportHtml(ctrace, jsOut);
      const pyOut = join(dir, "py-hard.html");
      const proc = spawnSync(
        venvPython,
        ["-c", `from ctxdiff.viewer.export import export_html; export_html(${JSON.stringify(ctrace)}, ${JSON.stringify(pyOut)})`],
        { encoding: "utf8", env: { ...process.env, PYTHONPATH: pySrc } },
      );
      expect(proc.status, `python export failed: ${proc.stderr}`).toBe(0);
      expect(readFileSync(jsOut, "utf-8"), `HTML mismatch for ${ctrace}`).toBe(
        readFileSync(pyOut, "utf-8"),
      );
    }
  });

  it.skipIf(!hasBuild)("renders identical HTML when --session names a run older than the cap", () => {
    // Driven through both CLIs, because `--session` is what pins the focus and
    // the focus is what `start.session` must name. A 30-session project puts the
    // oldest run well outside the 25-session detail cap, so this also pins the
    // "focus is embedded even when it missed the cut" behavior in both SDKs.
    const path = writeProjectSessions(dir, "oldsession-parity.ctrace", 30);
    const ct = CTrace.open(path);
    const oldest = (() => {
      try {
        return ct.listSessions()[0].id;
      } finally {
        ct.close();
      }
    })();

    const jsOut = join(dir, "js-oldsession.html");
    const pyOut = join(dir, "py-oldsession.html");
    const argv = ["export", "--project", path, "--session", oldest, "--out"];
    const jsProc = spawnSync(process.execPath, [distCli, ...argv, jsOut], {
      encoding: "utf8",
      env: { ...process.env, NO_COLOR: "1" },
    });
    expect(jsProc.status, `js export failed: ${jsProc.stderr}`).toBe(0);
    const pyProc = spawnSync(
      venvPython,
      ["-c", "from ctxdiff.cli import main; import sys; sys.exit(main(sys.argv[1:]))", ...argv, pyOut],
      { encoding: "utf8", env: { ...process.env, PYTHONPATH: pySrc, NO_COLOR: "1" } },
    );
    expect(pyProc.status, `python export failed: ${pyProc.stderr}`).toBe(0);

    const jsHtml = readFileSync(jsOut, "utf-8");
    expect(jsHtml).toBe(readFileSync(pyOut, "utf-8"));
    // ...and both opened ON the named run, not on the newest one.
    expect(jsHtml).toContain(`"start": {"level": 3, "agent": null, "session": "${oldest}"}`);
  });

  it("the JS demo trace matches the Python demo trace block-for-block", () => {
    const jsDemo = join(dir, "js-demo.ctrace");
    buildDemoTrace(jsDemo);
    const ct = CTrace.open(jsDemo);
    const jsBlocks = ct
      .getCalls()
      .flatMap((c) => ct.getCallBlocks(c.id).map((cb) => `${cb.block.contentHash} ${cb.label}`));
    ct.close();

    const pyDemo = join(dir, "py-demo.ctrace");
    const proc = spawnSync(
      venvPython,
      [
        "-c",
        "from ctxdiff.demo import build_demo_trace\n" +
          "from ctxdiff.store.ctrace import CTrace\n" +
          `build_demo_trace(${JSON.stringify(pyDemo)})\n` +
          `ct = CTrace.open(${JSON.stringify(pyDemo)})\n` +
          "print('\\n'.join(cb.block.content_hash + ' ' + cb.label for c in ct.get_calls() for cb in ct.get_call_blocks(c.id)))",
      ],
      { encoding: "utf8", env: { ...process.env, PYTHONPATH: pySrc } },
    );
    expect(proc.status, `python demo failed: ${proc.stderr}`).toBe(0);
    expect(jsBlocks).toEqual(proc.stdout.trim().split("\n"));
  });
});

describe("viewer: exporter + demo write files", () => {
  it("exportHtml writes an .html next to the trace by default", () => {
    const written = exportHtml(fx.dynamic);
    expect(written.endsWith("dynamic.html")).toBe(true);
    expect(existsSync(written)).toBe(true);
    rmSync(written, { force: true });
  });

  it("buildDemoTrace writes a six-call trace with two agents", () => {
    const p = join(dir, "demo.ctrace");
    buildDemoTrace(p);
    const ct = CTrace.open(p);
    try {
      const calls = ct.getCalls();
      expect(calls.length).toBe(6);
      expect(new Set(calls.map((c) => c.agent))).toEqual(new Set(["researcher", "writer"]));
    } finally {
      ct.close();
    }
  });
});

// --- the three-level project dashboard ------------------------------------------
//
// LEVEL 1 lists every agent across every session, LEVEL 2 the sessions one agent
// appeared in, LEVEL 3 that session's turn-by-turn detail. These assert the
// PAYLOAD (must match Python's field-for-field) and then BOOT THE PAGE and drive
// it, because only executing the shipped script says what the dashboard does.

/** A PROJECT trace: one file, several sessions, two agents in each, fixed UTC
 * timestamps a day apart. The writer sits out the LAST session, so an agent's
 * session count is genuinely not the project's. Mirrors the Python fixture. */
function writeProjectSessions(dir: string, name: string, sessions: number): string {
  const path = join(dir, name);
  for (let i = 0; i < sessions; i++) {
    const ct = CTrace.openOrCreateSession(
      path, "pipeline", "openai", "",
      `2026-07-${String(i + 1).padStart(2, "0")}T09:15:00+00:00`,
    );
    ct.recordCall({
      seq: 1, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 100, completion_tokens: 20 },
      latencyMs: 10, error: null,
      callBlocks: [blk("system", "message", "sys R", 0), blk("user", "message", `research ${i}`, 1)],
      agent: "researcher", step: "gather",
    });
    ct.recordCall({
      seq: 2, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 40, completion_tokens: 8 },
      latencyMs: 10, error: null,
      callBlocks: [
        blk("system", "message", "sys R", 0),
        blk("user", "message", `research ${i}`, 1),
        blk("assistant", "message", "more", 2),
      ],
      agent: "researcher", step: "gather",
    });
    if (i < sessions - 1) {
      ct.recordCall({
        seq: 3, params: { model: "gpt-4o" }, usage: null, latencyMs: 10, error: null,
        callBlocks: [blk("system", "message", "sys W", 0), blk("user", "message", `write ${i}`, 1)],
        agent: "writer", step: "compose",
      });
    }
    ct.close();
  }
  return path;
}

/** A project whose sessions were written OUT of chronological order — the shape
 * two capturing machines with disagreeing clocks (or a backfill) produce. One
 * agent runs in all three, so its span is unambiguous. */
function writeSkewedSessions(dir: string, name: string): string {
  const path = join(dir, name);
  for (const stamp of ["2026-05-01", "2026-01-01", "2026-03-01"]) {
    const ct = CTrace.openOrCreateSession(
      path, "skew", "openai", "", `${stamp}T09:00:00+00:00`,
    );
    ct.recordCall({
      seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null,
      callBlocks: [blk("user", "message", "hi", 0)], agent: "researcher",
    });
    ct.close();
  }
  return path;
}

/** One session with two agents, the FIRST of them named `__proto__` — the name
 * that silently disappears from a plain object used as a name-keyed map. Both
 * report provider usage, so `stats.usage.by_agent` is populated for both. */
function writeProtoAgentSession(dir: string, name: string): string {
  const path = join(dir, name);
  const ct = CTrace.create(path, "pipeline", "openai", "", "2026-05-01T09:00:00+00:00");
  ["__proto__", "writer"].forEach((agent, i) => {
    ct.recordCall({
      seq: i + 1, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 100, completion_tokens: 20 },
      latencyMs: 10, error: null,
      callBlocks: [blk("system", "message", "sys", 0), blk("user", "message", `t ${i}`, 1)],
      agent, step: "s",
    });
  });
  ct.close();
  return path;
}

/** One turn whose blocks carry a KNOWN label and a label named `__proto__` —
 * the string a plain object used as a known-label SET reports as present even
 * though it was never put there. Tagged, because that is how such a label
 * really arrives: `tracer.tag()` takes arbitrary user text. */
function writeProtoLabelSession(dir: string, name: string): string {
  const path = join(dir, name);
  const ct = CTrace.create(path, "labels", "openai", "", "2026-05-01T09:00:00+00:00");
  const tagged = { ...blk("user", "message", "retrieved chunk", 1), label: "__proto__", labelSource: "tagged" };
  ct.recordCall({
    seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 10, error: null,
    callBlocks: [blk("system", "message", "sys", 0), tagged],
  });
  ct.close();
  return path;
}

/** A one-session project whose NAME is the page's own data marker — the string
 * that, once substituted into `<title>`, collides with the marker the JSON
 * island is still waiting for. */
function writeMarkerNamedProject(dir: string, name: string): string {
  const path = join(dir, name);
  const ct = CTrace.create(path, "__CTXDIFF_DATA__", "openai", "", "2026-05-01T09:00:00+00:00");
  ct.recordCall({
    seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null,
    callBlocks: [blk("user", "message", "hi", 0)],
  });
  ct.close();
  return path;
}

/** Build a project payload straight off a `.ctrace`. */
function projectPayload(path: string, opts: ProjectPayloadOptions = {}) {
  const ct = CTrace.open(path);
  try {
    return buildProjectPayload(ct, opts) as {
      project: {
        name: string; sessions_total: number; detail_cap: number; focus: string;
        start: { level: number; agent: string | null; session: string | null };
        agents: { name: string; sessions: number; calls: number; input: number; output: number; reported: number; first_seen: string; last_seen: string }[];
        sessions: { id: string; started_at: string; provider: string; models: string[]; turn_count: number; detail: boolean; agents: { name: string; turns: number; reported: number }[] }[];
        details: Record<string, unknown>;
      };
    };
  } finally {
    ct.close();
  }
}

describe("viewer: three-level payload", () => {
  it("appends `project` to an otherwise untouched single-session payload", () => {
    // The project payload is a strict SUPERSET of the single-session one, which
    // is what lets one page render both shapes and keeps every existing
    // assertion about run/calls/diffs/tokens/cache/stats meaningful.
    const path = writeProjectSessions(dir, "shape.ctrace", 2);
    const full = projectPayload(path) as unknown as Record<string, unknown>;
    const ct = CTrace.open(path);
    try {
      const { project, ...rest } = full;
      expect(project).toBeDefined();
      expect(rest).toEqual(buildPayload(ct));
    } finally {
      ct.close();
    }
  });

  it("level 1 aggregates each agent across every session", () => {
    const p = projectPayload(writeProjectSessions(dir, "l1.ctrace", 3)).project;
    const agents = Object.fromEntries(p.agents.map((a) => [a.name, a]));
    expect(p.agents.map((a) => a.name)).toEqual(["researcher", "writer"]);
    expect(agents.researcher).toMatchObject({
      sessions: 3, calls: 6, input: 3 * 140, output: 3 * 28, reported: 6,
    });
    // The writer sat out the newest session and reported NO provider usage —
    // `reported: 0` is what makes the page print "-" instead of a false 0.
    expect(agents.writer).toMatchObject({ sessions: 2, calls: 2, reported: 0 });
    expect(agents.researcher.first_seen.startsWith("2026-07-01")).toBe(true);
    expect(agents.researcher.last_seen.startsWith("2026-07-03")).toBe(true);
    expect(agents.writer.last_seen.startsWith("2026-07-02")).toBe(true);
  });

  it("level 2 lists every session newest first with per-agent turns", () => {
    const p = projectPayload(writeProjectSessions(dir, "l2.ctrace", 3)).project;
    expect(p.sessions.map((s) => s.started_at.slice(0, 10))).toEqual([
      "2026-07-03", "2026-07-02", "2026-07-01",
    ]);
    expect(p.sessions[0].turn_count).toBe(2); // writer sat this one out
    expect(p.sessions[0].agents.map((a) => a.name)).toEqual(["researcher"]);
    expect(p.sessions[1].agents.map((a) => [a.name, a.turns])).toEqual([
      ["researcher", 2], ["writer", 1],
    ]);
  });

  it("embeds timestamps as raw UTC for the browser to localize", () => {
    // The file is meant to be shared, so the conversion to a wall clock happens
    // in the VIEWER's zone at render time, never baked in at export.
    const p = projectPayload(writeProjectSessions(dir, "tz.ctrace", 2)).project;
    for (const s of p.sessions) expect(s.started_at.endsWith("+00:00")).toBe(true);
    for (const a of p.agents) {
      expect(a.first_seen.endsWith("+00:00")).toBe(true);
      expect(a.last_seen.endsWith("+00:00")).toBe(true);
    }
    expect(PAGE).toContain("function localTime(value)");
    expect(PAGE).toContain("getTimezoneOffset()");
  });

  it("opens straight on level 3 for a single-agent single-session project", () => {
    // The fast path: nothing to choose between at either level above, so the
    // common case never clicks twice.
    const p = projectPayload(fx.multiturn).project;
    expect(p.sessions_total).toBe(1);
    expect(p.agents.length).toBe(1); // the (unlabeled) bucket
    expect(p.start).toEqual({ level: 3, agent: null, session: p.focus });
  });

  it("opens on the agent listing when there is a choice to make", () => {
    const p = projectPayload(fx.multiagent).project;
    expect(p.start).toEqual({ level: 1, agent: null, session: null });
  });

  it("lets --session/--agent preselect the level", () => {
    const path = writeProjectSessions(dir, "presel.ctrace", 3);
    expect(projectPayload(path, { agent: "researcher" }).project.start).toEqual({
      level: 2, agent: "researcher", session: null,
    });
    const bySession = projectPayload(path, { sessionSelected: true }).project.start;
    expect(bySession.level).toBe(3);
    expect(bySession.agent).toBeNull();
    const both = projectPayload(path, { agent: "writer", sessionSelected: true }).project.start;
    expect(both).toMatchObject({ level: 3, agent: "writer" });
    // An agent with a single session has no listing worth showing.
    const solo = writeProjectSessions(dir, "presel-solo.ctrace", 1);
    expect(projectPayload(solo, { agent: "researcher" }).project.start.level).toBe(3);
  });

  it("embeds detail only for the most recent sessions, listing them all", () => {
    // The scale decision, asserted: levels 1 and 2 are aggregates and are never
    // capped, so no session or agent is hidden; only the block-level detail —
    // the part with size — is limited.
    const total = 28; // > the 25-session cap
    const p = projectPayload(writeProjectSessions(dir, "big.ctrace", total)).project;
    expect(p.sessions_total).toBe(total);
    expect(p.sessions.length).toBe(total);
    const detailed = p.sessions.filter((s) => s.detail);
    expect(detailed.length).toBe(p.detail_cap);
    expect(p.sessions.slice(0, p.detail_cap).map((s) => s.id)).toEqual(
      detailed.map((s) => s.id),
    );
    // The focus session's detail IS the payload's top level, never duplicated.
    expect(p.focus in p.details).toBe(false);
    expect(Object.keys(p.details).length).toBe(p.detail_cap - 1);
  });

  it("always embeds the detail of an explicitly named OLD session", () => {
    const path = writeProjectSessions(dir, "oldfocus.ctrace", 27);
    const ct = CTrace.open(path);
    let p;
    try {
      const oldest = ct.listSessions()[0].id;
      p = (buildProjectPayload(ct, { focusSessionId: oldest, sessionSelected: true }) as {
        project: { focus: string; detail_cap: number; start: { session: string | null }; sessions: { id: string; detail: boolean }[]; details: Record<string, unknown> };
      }).project;
      expect(p.focus).toBe(oldest);
    } finally {
      ct.close();
    }
    expect(p.sessions.find((s) => s.id === p.focus)!.detail).toBe(true);
    // ...and the page OPENS on it: embedding the detail is only half the job if
    // `start` sends the page somewhere else (see the test below).
    expect(p.start.session).toBe(p.focus);
    // One MORE session than the cap is embedded, since the focus is extra.
    expect(Object.keys(p.details).length).toBe(p.detail_cap);
  });

  it("opens on the NAMED session, not on whichever is listed first", () => {
    // `project.sessions` is newest-first, so reading its head named the newest
    // run for every project with more than one session: the page boots with
    // `openSession(start.session)`, which repoints the whole level-3 view —
    // breadcrumb, header and blocks table — at a run the user never asked for,
    // while the focus session's embedded detail sits unread. Checked for
    // `--session` alone and `--agent` + `--session`, two branches to level 3.
    const path = writeProjectSessions(dir, "namedfocus.ctrace", 4);
    const ct = CTrace.open(path);
    try {
      const sessions = ct.listSessions();
      const oldest = sessions[0].id;
      const newest = sessions[sessions.length - 1].id;
      expect(oldest).not.toBe(newest);
      for (const opts of [
        { focusSessionId: oldest, sessionSelected: true },
        { agent: "researcher", focusSessionId: oldest, sessionSelected: true },
      ]) {
        const p = (buildProjectPayload(ct, opts) as {
          project: { focus: string; start: { level: number; session: string | null } };
        }).project;
        expect(p.focus).toBe(oldest);
        expect(p.start.level).toBe(3);
        expect(p.start.session).toBe(oldest);
        expect(p.start.session).not.toBe(newest);
      }
    } finally {
      ct.close();
    }
  });

  it("reads an agent's span chronologically, not in write order", () => {
    // `first_seen`/`last_seen` are the OLDEST and NEWEST timestamps the agent
    // ran at, which is not the same as the first and last session WRITTEN once
    // two capturing machines' clocks disagree. Sessions written 2026-05-01,
    // 2026-01-01, 2026-03-01 reported `first seen 2026-05-01 ... last seen
    // 2026-03-01` — a span running backwards past its own earliest run.
    const path = writeSkewedSessions(dir, "skew.ctrace");
    const p = projectPayload(path).project;
    expect(p.agents[0].first_seen.startsWith("2026-01-01")).toBe(true);
    expect(p.agents[0].last_seen.startsWith("2026-05-01")).toBe(true);
    // Session ORDERING is a separate question and deliberately unchanged: rows
    // stay in reverse INSERT order, the only order a store reliably knows.
    expect(p.sessions.map((s) => s.started_at.slice(0, 10))).toEqual([
      "2026-03-01", "2026-01-01", "2026-05-01",
    ]);
  });

  it("keeps an agent named __proto__ in the per-agent usage map", () => {
    // `by_agent` was a plain `{}`, so `byAgent["__proto__"] = io` invoked the
    // prototype SETTER and created no own property at all: `Object.keys` (and
    // therefore `pyJsonDumps`) omitted the agent entirely, and the page showed
    // its chip as `in 0 · out 0` — a fabricated "this agent was free" for a run
    // that reported real spend. A null-prototype map takes the name as data.
    const path = writeProtoAgentSession(dir, "proto-payload.ctrace");
    const ct = CTrace.open(path);
    try {
      const p = buildPayload(ct) as {
        stats: { usage: { by_agent: Record<string, [number, number]> | null } };
      };
      const byAgent = p.stats.usage.by_agent!;
      expect(Object.keys(byAgent)).toEqual(["__proto__", "writer"]);
      expect(byAgent["__proto__"]).toEqual([100, 20]);
      // The serialized island must carry it too — that is the byte that differed.
      expect(pyJsonDumps(byAgent)).toBe(
        '{"__proto__": [100, 20], "writer": [100, 20]}',
      );
    } finally {
      ct.close();
    }
  });

  it("still exports from a reader that cannot list sessions", () => {
    // A reader materialized for ONE session degrades to a one-session project
    // rather than failing — a dashboard of what it has beats no dashboard.
    const ct = CTrace.open(fx.multiturn);
    try {
      const noListing = {
        getRun: (s?: string) => ct.getRun(s),
        getCalls: (s?: string) => ct.getCalls(s),
        getCallBlocks: (id: string) => ct.getCallBlocks(id),
      };
      const p = (buildProjectPayload(noListing) as {
        project: { sessions_total: number; start: { level: number } };
      }).project;
      expect(p.sessions_total).toBe(1);
      expect(p.start.level).toBe(3);
    } finally {
      ct.close();
    }
  });
});

describe("viewer: XSS-safe at every level", () => {
  // An agent NAME is attacker-influenced text that levels 1 and 2 and the
  // breadcrumb all render, and a session's provider/model strings are rendered
  // by level 2 — none of them may become live markup anywhere in the document.
  let evilPath: string;
  let evilHtml: string;
  const evilAgent = "</script><img src=x onerror=alert('L1')>";
  const evilProvider = "</script><svg onload=alert('L2')>";

  beforeAll(() => {
    evilPath = join(dir, "evil-levels.ctrace");
    ["plain", evilAgent].forEach((agent, i) => {
      const ct = CTrace.openOrCreateSession(
        evilPath, "proj", evilProvider, "</script><script>alert('model')</script>",
        `2026-07-0${i + 1}T00:00:00+00:00`,
      );
      ct.recordCall({
        seq: 1, params: { model: "m" }, usage: null, latencyMs: 1, error: null,
        callBlocks: [blk("user", "message", "hi", 0)], agent,
      });
      ct.close();
    });
    const out = join(dir, "evil-levels.html");
    exportHtml(evilPath, out);
    evilHtml = readFileSync(out, "utf-8");
  });

  it("keeps hostile agent names and session labels inside the JSON island", () => {
    const { island, outside } = splitIsland(evilHtml);
    // They DID reach the payload — levels 1 and 2 render them...
    const unescaped = island.replace(/<\\\//g, "</");
    expect(unescaped).toContain(evilAgent);
    expect(unescaped).toContain(evilProvider);
    // ...as inert JSON only: nothing closes the island early, and no live
    // element appears anywhere outside it.
    expect(island).not.toContain("</script");
    expect(outside).not.toContain("onerror");
    expect(outside).not.toContain("onload");
    expect(outside).not.toMatch(/<img[^>]*onerror/i);
    expect(outside).not.toMatch(/<svg[^>]*onload/i);
    // Exactly two real closers: the data island and the page script.
    expect((evilHtml.match(/<\/script>/g) ?? []).length).toBe(2);
  });

  it("renders hostile names as TEXT, never markup, on all three levels", () => {
    // The payload assertions above prove nothing became live markup at BUILD
    // time. This proves the same at RENDER time, which is where it could still
    // go wrong: the page is booted and walked, and every string it ever assigns
    // as innerHTML is checked for the payload.
    const page = bootPage(evilHtml);
    const seenAsMarkup: string[] = [];
    const walk = () => {
      seenAsMarkup.push(page.byId.get("l1")!.markup(), page.byId.get("l2")!.markup(),
                        page.byId.get("l3")!.markup(), page.byId.get("crumbs")!.markup());
    };
    expect(page.visibleLevel()).toBe(1);
    // The hostile agent name is on screen — as readable text.
    expect(page.level(1).text()).toContain(evilAgent);
    walk();
    const evilRow = page.rows().find((r) => r.text().includes(evilAgent))!;
    evilRow.click();
    expect(page.visibleLevel()).toBe(2);
    expect(page.byId.get("crumbs")!.text()).toContain(evilAgent);
    walk();
    page.rows()[0].click();
    expect(page.visibleLevel()).toBe(3);
    walk();
    // Not one of those innerHTML assignments carried any of it.
    for (const markup of seenAsMarkup) {
      expect(markup).not.toContain("onerror");
      expect(markup).not.toContain("onload");
      expect(markup).not.toContain("<img");
      expect(markup).not.toContain("<svg");
    }
  });

  it("stays self-contained with a multi-session project embedded", () => {
    expect(evilHtml).not.toMatch(/https?:\/\//);
    expect(evilHtml).not.toMatch(/src=["']\/\//);
    expect(evilHtml).not.toMatch(/href=["']\/\//);
    expect(evilHtml).not.toContain("//cdn");
  });
});

describe("viewer: the page drives all three levels", () => {
  // These BOOT the exported dashboard's own inline script (see helpers/page.ts)
  // and drive it, because the payload only says what the page COULD show.
  const TZ = "Asia/Dubai";
  const originalTz = process.env.TZ;
  let html: string;

  beforeAll(() => {
    process.env.TZ = TZ;
    const path = writeProjectSessions(dir, "nav.ctrace", 3);
    const out = join(dir, "nav.html");
    exportHtml(path, out);
    html = readFileSync(out, "utf-8");
  });
  afterAll(() => {
    if (originalTz === undefined) delete process.env.TZ;
    else process.env.TZ = originalTz;
  });

  it("lands on level 1, drills agent -> sessions -> turns, and walks back up", () => {
    const page = bootPage(html);
    expect(page.visibleLevel()).toBe(1);
    expect(page.byId.get("crumbs")!.text()).toBe("all agents");
    expect(page.rows().map((r) => r.text())).toEqual(["researcher", "writer"]);

    page.rows()[0].click();                       // -> level 2
    expect(page.visibleLevel()).toBe(2);
    expect(page.byId.get("crumbs")!.text()).toBe("all agents›researcher");
    expect(page.rows()).toHaveLength(3);          // researcher ran in all three

    page.rows()[0].click();                       // -> level 3
    expect(page.visibleLevel()).toBe(3);
    expect(page.byId.get("crumbs")!.text()).toContain("researcher›");
    expect(page.byId.get("blocks")!.text()).toContain("sys R");

    page.crumbs()[1].click();                     // back to level 2
    expect(page.visibleLevel()).toBe(2);
    page.crumbs()[0].click();                     // back to level 1
    expect(page.visibleLevel()).toBe(1);
  });

  it("scopes level 3 to the agent that was drilled into", () => {
    // "the agent's runs with traces" means THAT agent's turns, not the whole
    // session's — the scrubber, the arrow keys and the growth chart all follow
    // the scope, and clearing the chip restores the full session.
    const page = bootPage(html);
    page.rows()[1].click();                       // writer
    page.rows()[0].click();                       // its newest session (2026-07-02)
    expect(page.visibleLevel()).toBe(3);
    // That session holds 3 turns; the writer made exactly one of them.
    expect(page.bars()).toHaveLength(1);
    const chips = page.chips();
    expect(chips.map((c) => c.getAttribute("aria-pressed"))).toEqual(["false", "true"]);
    chips[1].click();                             // clear the scope
    expect(page.bars()).toHaveLength(3);
  });

  it("renders every timestamp in the VIEWER's local zone", () => {
    // Stored 09:15 UTC; Asia/Dubai is +04:00, so the page must say 13:15 +04:00
    // — the whole point of converting client-side rather than at export.
    const page = bootPage(html);
    expect(page.level(1).text()).toContain("2026-07-01 13:15:00 +04:00");
    page.rows()[0].click();
    expect(page.level(2).text()).toContain("2026-07-03 13:15:00 +04:00");
    // ...and it agrees with the CLI's own local-time column for the same instant.
    expect(page.level(2).text()).toContain(formatLocal("2026-07-03T09:15:00+00:00"));
  });

  it("hides the breadcrumb and opens on the detail for a lone session", () => {
    const out = join(dir, "solo-nav.html");
    exportHtml(fx.multiturn, out);
    const page = bootPage(readFileSync(out, "utf-8"));
    expect(page.visibleLevel()).toBe(3);
    expect(page.byId.get("crumbs")!.hidden).toBe(true);
    expect(page.bars()).toHaveLength(3);
  });

  it("marks sessions whose detail was not embedded and names the cap", () => {
    const path = writeProjectSessions(dir, "capped-nav.ctrace", 28);
    const out = join(dir, "capped-nav.html");
    exportHtml(path, out);
    const page = bootPage(readFileSync(out, "utf-8"));
    page.rows()[0].click();                       // researcher: all 28 sessions
    const text = page.level(2).text();
    expect(text).toContain("detail not embedded");
    expect(text).toContain("25 most recent sessions");
    // Every session is LISTED even so — nothing is hidden.
    expect(page.level(2).find((e) => e.className === "rowlink")).toHaveLength(28);
    // ...but the three beyond the cap are inert rather than dead links.
    const inert = page.level(2).find((e) => e.className === "rowlink" && e.disabled);
    expect(inert).toHaveLength(3);
  });
});

describe("viewer: names that a plain JS object would swallow", () => {
  // Agent names and project names are arbitrary user text. Two of those strings
  // mean something to JavaScript itself — `__proto__` as an object key, and the
  // page's own `__CTXDIFF_DATA__` marker as a substitution target — and both
  // used to change what the exported page SHOWS. These boot the real page.

  it("gives an agent named __proto__ its palette color and its true spend", () => {
    // `AGENT_COLOR[name] = "#3987e5"` on a plain `{}` invokes the __proto__
    // SETTER, which ignores strings: no own property appeared, `agentColor`
    // returned Object.prototype, and the browser dropped it as a style value —
    // that agent rendered with no color dot at all. The tooltip's `by_agent`
    // lookup had the mirror-image problem: with the key missing from the JS
    // payload it INHERITED a truthy Object.prototype and printed `in 0 · out 0`,
    // inventing "this agent was free" for a run that reported real spend.
    const path = writeProtoAgentSession(dir, "proto-page.ctrace");
    const out = join(dir, "proto-page.html");
    exportHtml(path, out);
    const page = bootPage(readFileSync(out, "utf-8"));

    expect(page.visibleLevel()).toBe(1);
    const [protoRow, writerRow] = page.rows();
    expect(protoRow.text()).toBe("__proto__");
    // The palette is assigned by order of first appearance and neither agent
    // may fall through to the unknown color.
    expect(protoRow.children[0].className).toBe("agent-dot");
    expect(protoRow.children[0].style.background).toBe("#3987e5");
    expect(writerRow.children[0].style.background).toBe("#d95926");

    protoRow.click();                             // -> level 2
    page.rows()[0].click();                       // -> level 3
    expect(page.visibleLevel()).toBe(3);
    const [protoChip] = page.chips();
    expect(protoChip.children[0].style.background).toBe("#3987e5");
    // ...and the chip quotes THIS agent's provider usage, not a prototype's.
    expect(protoChip.getAttribute("title")).toBe("__proto__ · in 100 · out 20");
  });

  it("gives a block labeled __proto__ the unknown color, not a dead CSS var", () => {
    // `KNOWN_LABELS` is a plain object literal used as a set, so
    // `KNOWN_LABELS["__proto__"]` INHERITED a truthy value (Object.prototype)
    // and the label passed as known: `labelColor` returned
    // `var(--c-__proto__)`, a custom property the page's CSS never declares, so
    // the swatch rendered with no color while every real label kept its hue.
    // `constructor` and `toString` inherit functions the same way. An OWN
    // property check sends anything custom to the unknown color, which exists.
    const path = writeProtoLabelSession(dir, "proto-label.ctrace");
    const out = join(dir, "proto-label.html");
    exportHtml(path, out);
    const page = bootPage(readFileSync(out, "utf-8"));
    expect(page.visibleLevel()).toBe(3);

    // The blocks table: the known label keeps its own hue, the hostile one falls
    // through to the unknown color rather than to a variable that is not there.
    const chips = page.byId.get("blocks")!.find((e) => e.className === "label-chip");
    // children: [dot, label text, ("tagged" badge)].
    expect(chips.map((c) => c.children[1].text())).toEqual(["system", "__proto__"]);
    expect(chips[0].children[0].style.background).toBe("var(--c-system)");
    expect(chips[1].children[0].style.background).toBe("var(--c-unknown)");

    // ...and the same for the token-allocation legend, which colors by label too.
    const legend = page.byId.get("alloc")!.find((e) => e.className === "chip");
    expect(legend.map((c) => c.style.background)).not.toContain("var(--c-__proto__)");
  });

  it("boots a project named __CTXDIFF_DATA__ with an intact title and island", () => {
    // Two sequential single-marker replacements cannot survive a title that
    // CONTAINS the other marker: JS replaced only the FIRST `__CTXDIFF_DATA__`,
    // which after the title pass was the one now sitting inside `<title>`, so
    // the island kept the literal marker, `JSON.parse` threw and the page
    // rendered NOTHING. One pass over both markers has no such ordering.
    const path = writeMarkerNamedProject(dir, "marker-page.ctrace");
    const out = join(dir, "marker-page.html");
    exportHtml(path, out);
    const rendered = readFileSync(out, "utf-8");

    expect(rendered).toContain("<title>ctxdiff — __CTXDIFF_DATA__</title>");
    expect(rendered).not.toContain("__CTXDIFF_TITLE__");
    // The island is the payload, not a leftover marker...
    const { island } = splitIsland(rendered);
    expect(island.startsWith("{")).toBe(true);
    expect(JSON.parse(island.replace(/<\\\//g, "</")).run.project).toBe("__CTXDIFF_DATA__");
    // ...and the page it boots is a working dashboard, not an empty document.
    const page = bootPage(rendered);
    expect(page.visibleLevel()).toBe(3);
    expect(page.byId.get("h-project")!.text()).toContain("__CTXDIFF_DATA__");
  });

  it("substitutes each marker exactly once, whatever the other one contains", () => {
    // renderPage's own contract, isolated from the exporter: neither argument
    // may be re-scanned as part of filling the other, in either direction.
    const filled = renderPage("__CTXDIFF_DATA__", '{"x": "__CTXDIFF_TITLE__"}');
    expect(filled).toContain("<title>__CTXDIFF_DATA__</title>");
    expect(splitIsland(filled).island).toBe('{"x": "__CTXDIFF_TITLE__"}');
  });
});

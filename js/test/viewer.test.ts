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
import { exportHtml, buildPayload } from "../src/viewer/export.js";
import { buildDemoTrace } from "../src/demo.js";
import { makeFixtures } from "./helpers/fixtures.js";

const repoRoot = resolve(process.cwd(), "..");
const venvPython = join(repoRoot, "venv", "bin", "python");
const pySrc = join(repoRoot, "src");
const hasVenv = existsSync(venvPython);

let dir: string;
let fx: { multiturn: string; multiagent: string; dynamic: string; bidi: string };

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
    for (const ctrace of [fx.multiturn, fx.multiagent, fx.dynamic]) {
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

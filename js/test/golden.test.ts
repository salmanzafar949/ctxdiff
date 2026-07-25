/**
 * CROSS-SDK GOLDEN CHECK (JS side). Every fixture in `spec/golden/corpus/` is
 * rebuilt with THIS SDK's tokenizer, hasher and labeler, rendered through THIS
 * SDK's CLI and viewer, and compared to the committed expectations under
 * `spec/golden/expected/` — the very same files `tests/test_golden.py` compares
 * against on the Python side.
 *
 * The property this defends, which nothing else in the suite defends: the two
 * SDKs count tokens with two independent libraries (`gpt-tokenizer` here,
 * `tiktoken` in Python) on independent release cadences. The live conformance
 * suites compare the SDKs to EACH OTHER, so they pass happily if both drift the
 * same way, and they degrade to nothing on a machine where only one SDK is
 * installed. A committed golden cannot drift with the tools: if either SDK's
 * numbers move, this fails, and if BOTH move it still fails until someone runs
 * `npm run golden:regen` and reviews the diff.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/cli.js";
import { exportHtml } from "../src/viewer/export.js";
import { CTrace } from "../src/store/ctrace.js";
import {
  buildAll,
  caseArgv,
  hashHtml,
  installedTokenizerVersion,
  loadManifest,
  readCliGolden,
  readHtmlGolden,
  type Manifest,
} from "./helpers/golden.js";

const manifest: Manifest = loadManifest();

let dir: string;
let traces: Record<string, string>;
const savedTz = process.env.TZ;
const savedNoColor = process.env.NO_COLOR;

beforeAll(() => {
  // Pin the two ambient inputs that would otherwise make the goldens depend on
  // the machine: TZ, because `ctxdiff sessions` renders each session's start
  // time in the VIEWER's local zone (see `formatLocal`), and NO_COLOR, because
  // the renderers emit ANSI escapes on a TTY while the goldens are plain text.
  process.env.TZ = manifest.tz;
  if (manifest.no_color !== false) process.env.NO_COLOR = "1";
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-golden-"));
  traces = buildAll(manifest, dir);
});

afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
  if (savedTz === undefined) delete process.env.TZ;
  else process.env.TZ = savedTz;
  if (savedNoColor === undefined) delete process.env.NO_COLOR;
  else process.env.NO_COLOR = savedNoColor;
});

/** Run the CLI in-process with both streams captured. In-process rather than
 * spawned: ~30x faster across the case matrix, and it exercises the very
 * `main` the published bin wraps. */
async function runCli(argv: string[]): Promise<{ code: number; out: string; err: string }> {
  const outChunks: string[] = [];
  const errChunks: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  // @ts-expect-error narrow override of the write signature for capture
  process.stdout.write = (s: string) => (outChunks.push(String(s)), true);
  // @ts-expect-error narrow override of the write signature for capture
  process.stderr.write = (s: string) => (errChunks.push(String(s)), true);
  let code: number;
  try {
    code = await main(argv);
  } finally {
    process.stdout.write = origOut;
    process.stderr.write = origErr;
  }
  return { code, out: outChunks.join(""), err: errChunks.join("") };
}

describe("golden corpus — the tokenizer pin is what the environment actually has", () => {
  it("gpt-tokenizer is installed at the version the manifest pins", () => {
    // A FAILURE, never a skip. The whole point of the pin is that the numbers
    // below were produced under a known tokenizer; comparing them under an
    // unknown one would either pass by luck or fail with a misleading message
    // about the CLI. Naming the real cause here is what makes a resolver drift
    // diagnosable in one line of CI output.
    expect(installedTokenizerVersion()).toBe(manifest.tokenizers.javascript.pinned_version);
  });

  it("the fixtures really run the exact tokenizer (no silent estimate fallback)", () => {
    // Guards against the most dangerous way a golden suite goes vacuous: if
    // `countTokens` started falling back to the character heuristic for every
    // block, the goldens would still be internally consistent and every
    // comparison would still pass — while measuring nothing about the
    // tokenizer. Every openai-provider block must be counted by the real one.
    let exact = 0;
    for (const path of Object.values(traces)) {
      const ct = CTrace.open(path);
      try {
        for (const session of ct.listSessions()) {
          for (const call of ct.getCalls(session.id)) {
            for (const cb of ct.getCallBlocks(call.id)) {
              if ((call.provider ?? session.provider) === "openai") {
                expect(cb.block.tokenMethod).toBe("tiktoken");
                exact += 1;
              }
            }
          }
        }
      } finally {
        ct.close();
      }
    }
    expect(exact).toBeGreaterThan(50);
  });
});

describe("golden corpus — CLI stdout matches the committed expectations", () => {
  for (const c of manifest.cli_cases) {
    it(`${c.name}`, async () => {
      const { code, out, err } = await runCli(caseArgv(c, traces));
      expect(code, `case exited ${code}\nstderr:\n${err}`).toBe(0);
      expect(out).toBe(readCliGolden(c.name));
    });
  }
});

describe("golden corpus — HTML dashboards match the committed hashes", () => {
  for (const c of manifest.html_cases) {
    it(`${c.name}`, () => {
      const outPath = join(dir, `${c.name}.js.html`);
      exportHtml(traces[c.fixture], outPath);
      expect(hashHtml(outPath)).toEqual(readHtmlGolden(c.name));
    });
  }
});

describe("golden corpus — the harness itself catches drift", () => {
  /**
   * The meta-test. A golden suite that compares against an empty string, a
   * missing file, or its own output is worse than no suite at all: it is a
   * green tick that means nothing. This mutates a committed expectation
   * IN MEMORY — one digit of one number — and asserts the comparison rejects
   * it. If this ever passes, every assertion above is decorative.
   */
  it("a one-character mutation of an expected number is rejected", async () => {
    const c = manifest.cli_cases.find((x) => x.name === "round-numbers.tokens")!;
    const { out } = await runCli(caseArgv(c, traces));
    const expected = readCliGolden(c.name);
    expect(out).toBe(expected);

    const mutated = expected.replace(/(\d)/, (d) => String((Number(d) + 1) % 10));
    expect(mutated, "the fixture golden has no digit to mutate").not.toBe(expected);
    expect(out).not.toBe(mutated);
  });

  it("a missing expectation fails loudly instead of comparing against nothing", () => {
    expect(() => readCliGolden("no-such-case")).toThrow(/no golden for CLI case/);
    expect(() => readHtmlGolden("no-such-case")).toThrow(/no golden for HTML case/);
  });
});

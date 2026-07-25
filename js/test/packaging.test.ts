/**
 * Packaging tests: `npm pack` the tarball, `npm install` it into a throwaway
 * project, and drive the INSTALLED binary — `node_modules/.bin/ctxdiff` and
 * `npx ctxdiff` — exactly as a user does.
 *
 * Why this exists, and why `cli-smoke.test.ts` is not enough: every other test
 * in this suite runs the CLI at its real path (`node dist/cli.js`) or imports
 * `main` in-process. npm installs `bin` as a SYMLINK at `node_modules/.bin/
 * ctxdiff`, so a real invocation enters the bundle with `process.argv[1]`
 * pointing at the shim, not at `dist/cli.js`. ctxdiff 0.2.0 shipped an
 * entry-point guard that string-matched `process.argv[1].endsWith("cli.js")`;
 * under the shim that test is false, `main()` never ran, and `npx ctxdiff
 * <anything>` printed nothing and exited 0 for every user — while the whole
 * suite stayed green, because testing the BUILT artifact is not the same as
 * testing the INSTALLED artifact. This file closes that gap: it is the only
 * test that would have failed on 0.2.0.
 *
 * Cost: one `npm pack` + one `npm install` of a dependency-light tarball,
 * shared by every case via module-scope setup. Skipped by name (never
 * vacuously passed) when dist isn't built or the environment can't pack/install
 * — e.g. an offline box with a cold npm cache.
 */
import { describe, it, expect, afterAll } from "vitest";
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const pkgRoot = process.cwd();
const isWindows = process.platform === "win32";
const npm = isWindows ? "npm.cmd" : "npm";

let stagingDir = "";
let projectDir = "";
/** The installed shim under test: the symlink/`.cmd` npm creates, NOT dist/cli.js. */
let binPath = "";

function sh(cmd: string, args: string[], cwd: string, timeout = 180_000) {
  return spawnSync(cmd, args, {
    cwd,
    encoding: "utf8",
    timeout,
    shell: isWindows, // npm on Windows is a .cmd and needs a shell
    env: { ...process.env, NO_COLOR: "1", npm_config_update_notifier: "false" },
  });
}

/**
 * Pack + install once. Returns null when the installed binary is ready, or a
 * human-readable reason to skip. Runs at module scope rather than in
 * `beforeAll` so the decision is known at collection time and surfaces as a
 * NAMED skipped suite in the reporter instead of a silent pass.
 */
function setUpInstalledPackage(): string | null {
  if (!existsSync(resolve(pkgRoot, "dist", "cli.js"))) {
    return "dist not built — run `npm run build` first";
  }
  try {
    stagingDir = mkdtempSync(join(tmpdir(), "ctxdiff-pack-"));
    projectDir = join(stagingDir, "consumer");

    const packed = sh(npm, ["pack", "--pack-destination", stagingDir, "--loglevel=error"], pkgRoot);
    if (packed.status !== 0) {
      return `npm pack failed: ${(packed.stderr || packed.error?.message || "").trim().slice(0, 200)}`;
    }
    // npm pack prints the tarball filename on stdout (last non-empty line).
    const name = (packed.stdout ?? "").trim().split("\n").filter(Boolean).pop()?.trim();
    const tarball = name ? join(stagingDir, name) : "";
    if (!tarball || !existsSync(tarball)) return `npm pack produced no tarball (got ${name})`;

    mkdirSync(projectDir, { recursive: true });
    writeFileSync(
      join(projectDir, "package.json"),
      JSON.stringify({ name: "ctxdiff-packaging-probe", version: "0.0.0", private: true }) + "\n",
    );
    // --prefer-offline so a warm npm cache is enough; the only runtime dep is
    // gpt-tokenizer, which this repo's own install already cached.
    const installed = sh(
      npm,
      ["install", "--no-audit", "--no-fund", "--prefer-offline", "--loglevel=error", tarball],
      projectDir,
    );
    if (installed.status !== 0) {
      return `npm install of the tarball failed (offline with a cold cache?): ${(installed.stderr || installed.error?.message || "").trim().slice(0, 200)}`;
    }

    binPath = join(projectDir, "node_modules", ".bin", isWindows ? "ctxdiff.cmd" : "ctxdiff");
    if (!existsSync(binPath)) return `npm install did not create ${binPath}`;
    return null;
  } catch (err) {
    return `packaging setup threw: ${(err as Error).message}`;
  }
}

const skipReason = setUpInstalledPackage();

const suiteName = skipReason
  ? `ctxdiff installed package — SKIPPED: ${skipReason}`
  : "ctxdiff installed package (npm pack -> npm install -> node_modules/.bin/ctxdiff)";

/** Execute the INSTALLED shim itself — the whole point of this file. */
function runBin(argv: string[], cwd = projectDir) {
  const proc = sh(binPath, argv, cwd, 60_000);
  return { code: proc.status ?? -1, out: proc.stdout ?? "", err: proc.stderr ?? "" };
}

afterAll(() => {
  if (stagingDir) rmSync(stagingDir, { recursive: true, force: true });
});

describe.skipIf(skipReason !== null)(suiteName, () => {
  it("`.bin/ctxdiff --help` prints usage on stdout (exit 2, the JS CLI's convention)", () => {
    const r = runBin(["--help"]);
    // The regression that shipped as 0.2.0 was EMPTY stdout and exit 0 — a
    // silent no-op. Assert the non-emptiness first, because that is the bug.
    expect(r.out.trim()).not.toBe("");
    expect(r.out).toContain("usage: ctxdiff");
    expect(r.out).toContain("demo");
    // `--help` is not a command, so it takes the usage branch: usage on stdout,
    // exit 2. Same as the in-tree `dist/cli.js` smoke test's "no command" case.
    expect(r.code).toBe(2);
  });

  it("`.bin/ctxdiff` with no command prints usage and exits 2", () => {
    const r = runBin([]);
    expect(r.code).toBe(2);
    expect(r.out).toContain("usage: ctxdiff");
  });

  it("`.bin/ctxdiff demo --keep --no-open` writes a real .ctrace + .html (exit 0)", () => {
    const r = runBin(["demo", "--keep", "--no-open"]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("sample trace  ->");
    expect(r.out).toContain("dashboard     ->");

    const ctrace = join(projectDir, "ctxdiff-demo.ctrace");
    const html = join(projectDir, "ctxdiff-demo.html");
    expect(existsSync(ctrace)).toBe(true);
    expect(existsSync(html)).toBe(true);
    const page = readFileSync(html, "utf-8");
    expect(page).toContain("<!DOCTYPE html>");
    expect(page).not.toMatch(/https?:\/\//); // self-contained dashboard
  });

  it("the installed binary reads back the trace it just wrote", () => {
    // Sequenced after the demo case above (vitest runs a file's tests in order),
    // so the .ctrace exists; assert it rather than assume it.
    const ctrace = join(projectDir, "ctxdiff-demo.ctrace");
    expect(existsSync(ctrace)).toBe(true);

    const r = runBin(["tokens", ctrace]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("% of avg context");

    const sessions = runBin(["sessions", ctrace]);
    expect(sessions.code).toBe(0);
    expect(sessions.out).toContain("ctxdiff-demo.ctrace");
  });

  it("a bad flag through the installed binary is a usage error (exit 2, stderr)", () => {
    const r = runBin(["tokens", "--bogus", "x"]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("unrecognized arguments: --bogus");
    expect(r.err).not.toMatch(/at \w+.*\(.*:\d+:\d+\)/); // no stack frames
  });

  it("`npx ctxdiff --help` resolves the local install and runs it", () => {
    // `--no` = never fetch from the registry, so this must resolve the ctxdiff
    // we just installed into projectDir — the `npx ctxdiff` path users hit. The
    // `--` separator keeps `--help` from being eaten by npx's own arg parser.
    const proc = sh(npm.replace(/^npm/, "npx"), ["--no", "--", "ctxdiff", "--help"], projectDir, 60_000);
    expect(proc.stdout ?? "").toContain("usage: ctxdiff");
    expect(proc.status).toBe(2);
  });

  it("the published bin is the ESM build, and the CJS twin runs identically", () => {
    // Both dist entries ship in the tarball and both carry the entry guard;
    // `require()`-ing the CJS one must produce the same CLI, not a silent no-op.
    const installedRoot = join(projectDir, "node_modules", "ctxdiff");
    const esm = join(installedRoot, "dist", "cli.js");
    const cjs = join(installedRoot, "dist", "cli.cjs");
    expect(existsSync(esm)).toBe(true);
    expect(existsSync(cjs)).toBe(true);

    for (const entry of [esm, cjs]) {
      const direct = sh(process.execPath, [entry, "--help"], projectDir, 60_000);
      expect(direct.stdout ?? "", `${entry} --help`).toContain("usage: ctxdiff");
      expect(direct.status, `${entry} --help`).toBe(2);
      // And through a symlink pointing at it, which is all `.bin/ctxdiff` is —
      // the CJS build has no `.bin` entry of its own, so this is the only way
      // to prove its guard survives the indirection too.
      const link = join(projectDir, `link-${entry.endsWith(".cjs") ? "cjs" : "js"}`);
      rmSync(link, { force: true });
      symlinkSync(entry, link);
      const viaLink = sh(process.execPath, [link, "sessions"], projectDir, 60_000);
      expect(viaLink.stdout ?? "", `${link} sessions`).toContain("ctxdiff-demo.ctrace");
      expect(viaLink.status, `${link} sessions`).toBe(0);
    }
  });
});

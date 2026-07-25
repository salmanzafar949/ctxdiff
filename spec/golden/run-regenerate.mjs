/**
 * `npm run golden:regen` — the JS toolchain's entry point to the ONE golden
 * regenerator, `spec/golden/regenerate.py`.
 *
 * Why a shim rather than a second regenerator in TypeScript: two regenerators
 * would be two sources of truth for the committed expectations, and the first
 * time they disagreed the goldens would silently become "whichever one you
 * happened to run". There is exactly one writer (Python) and exactly two
 * checkers (both SDKs' test suites) — so a regenerated golden is only blessed
 * after the JS side independently reproduces it, which `regenerate.py` enforces
 * before it exits 0.
 *
 * Interpreter resolution, in order: the repo's own `venv/bin/python` (what
 * CONTRIBUTING tells contributors to create, and what CI builds), then `python3`
 * and `python` on PATH. Nothing is installed and no network is touched; if none
 * of them can import `ctxdiff` the underlying script says so.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const goldenDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(goldenDir, "..", "..");

/** The first usable Python: the repo venv if present, else PATH. Returns the
 * command to spawn — resolution failures surface from the spawn itself, with
 * the interpreter named, rather than as a silent skip. */
function pickPython() {
  const venv = join(repoRoot, "venv", "bin", "python");
  if (existsSync(venv)) return venv;
  const venvWin = join(repoRoot, "venv", "Scripts", "python.exe");
  if (existsSync(venvWin)) return venvWin;
  return process.platform === "win32" ? "python" : "python3";
}

const python = pickPython();
const script = join(goldenDir, "regenerate.py");
const proc = spawnSync(python, [script, ...process.argv.slice(2)], {
  stdio: "inherit",
  cwd: repoRoot,
});
if (proc.error) {
  console.error(
    `golden:regen could not run ${python}: ${proc.error.message}\n` +
      "Create the repo venv first: python -m venv venv && ./venv/bin/pip install -e .",
  );
  process.exit(2);
}
process.exit(proc.status ?? 1);

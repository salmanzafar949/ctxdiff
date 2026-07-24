/**
 * Post-build fixup: esbuild (via tsup) strips the `node:` prefix from builtin
 * specifiers. That's harmless for `crypto` (the bare name is a valid builtin)
 * but fatal for `sqlite` — there is NO bare `sqlite` builtin, only
 * `node:sqlite` — so the emitted `require("sqlite")` / `from "sqlite"` throws
 * MODULE_NOT_FOUND at import. We restore the exact specifier in the built
 * bundles. Narrowly targets the sqlite import only.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");
const files = ["index.js", "index.cjs"];

for (const f of files) {
  const p = join(dist, f);
  let src;
  try {
    src = readFileSync(p, "utf8");
  } catch {
    continue;
  }
  const fixed = src
    .replace(/require\((["'])sqlite\1\)/g, 'require("node:sqlite")')
    .replace(/from\s+(["'])sqlite\1/g, 'from "node:sqlite"');
  if (fixed !== src) {
    writeFileSync(p, fixed);
    console.log(`fix-node-sqlite: restored node:sqlite in dist/${f}`);
  }
}

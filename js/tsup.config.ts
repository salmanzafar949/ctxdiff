import { defineConfig } from "tsup";

// Dual ESM + CJS build. Externals: the provider SDKs (`openai`, `@anthropic-ai/
// sdk`, `@google/genai`) are optional peers we never import (only duck-type via
// a Proxy), and `gpt-tokenizer` is a real runtime dependency resolved from
// node_modules — keeping it external avoids bundling a second copy into dist, so
// the published tarball stays lean. `pg` and `mysql2` are the optional database
// drivers: they are `await import()`ed only when a Postgres/MySQL connection is
// actually opened, and marking them external keeps that import DYNAMIC in the
// bundle — inlining them would turn "you configured Postgres without installing
// pg" from a one-line install hint into a build/resolve failure for every user,
// including the overwhelming majority who only ever write a local `.ctrace`. `node:sqlite` is a built-in; esbuild strips
// its `node:` prefix on output (a documented esbuild behavior), restored by
// scripts/fix-node-sqlite.mjs in the build script (the bare `sqlite` name is not
// a valid builtin, so the prefix is mandatory). Sourcemaps are omitted from the
// published build to keep the tarball small. Type declarations are emitted by
// `tsc --emitDeclarationOnly` in the build script rather than tsup's
// rollup-plugin-dts, which is incompatible with the installed TypeScript.
export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts"],
  format: ["esm", "cjs"],
  dts: false,
  clean: true,
  sourcemap: false,
  target: "node22",
  platform: "node",
  // `src/cli.ts` decides whether it is the process entry point by comparing
  // `realpath(process.argv[1])` against its own `fileURLToPath(import.meta.url)`
  // — the only check that survives npm publishing `bin` as a SYMLINK
  // (`node_modules/.bin/ctxdiff` → `dist/cli.js`). `import.meta` does not exist
  // in CJS, and esbuild leaves it EMPTY there with only a warning, which would
  // silently give `dist/cli.cjs` a broken guard. `shims` makes tsup rewrite it
  // to a `__filename`-derived URL in the CJS output, so both builds behave
  // identically; it only injects into files that actually reference
  // `import.meta.url`/`__dirname`, so the other entries are untouched.
  shims: true,
  external: ["openai", "@anthropic-ai/sdk", "@google/genai", "gpt-tokenizer", "pg", "mysql2", "mysql2/promise"],
});

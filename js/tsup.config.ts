import { defineConfig } from "tsup";

// Dual ESM + CJS build. `openai` is a peer (user supplies it) and is never
// imported by our code — we only duck-type the client via a Proxy — so it stays
// external. `node:sqlite` is a built-in; esbuild strips its `node:` prefix on
// output (a documented esbuild behavior), which is then restored by
// scripts/fix-node-sqlite.mjs in the build script (the bare `sqlite` name is not
// a valid builtin, so the prefix is mandatory). Type declarations are emitted by
// `tsc --emitDeclarationOnly` in the build script rather than tsup's
// rollup-plugin-dts, which is incompatible with the installed TypeScript.
export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: false,
  clean: true,
  sourcemap: true,
  target: "node22",
  platform: "node",
  external: ["openai"],
});

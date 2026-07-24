import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    // The cross-language conformance test spawns Python and can be slow on a
    // cold venv; give the suite generous headroom.
    testTimeout: 30000,
  },
});

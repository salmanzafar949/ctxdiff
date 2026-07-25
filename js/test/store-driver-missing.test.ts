/**
 * The optional peer dependencies are OPTIONAL, and their absence must read like
 * a fixable mistake rather than a crash.
 *
 * `pg` and `mysql2` are imported lazily, at CONNECT time, for two reasons that
 * this file pins down: ctxdiff's core stays dependency-light, and a user who
 * configured a database but never installed its driver gets a one-line install
 * hint from inside the tracer's fail-open guard — not a module-resolution error
 * thrown out of `import "ctxdiff"`, which would break a program that merely
 * mentioned ctxdiff.
 *
 * Both modules are mocked to fail on import, which is exactly what Node does
 * when the package is not there.
 */
import { describe, it, expect, vi } from "vitest";
import { PostgresStore } from "../src/store/postgres.js";
import { MySQLStore } from "../src/store/mysql.js";

vi.mock("pg", () => {
  throw new Error("Cannot find package 'pg'");
});
vi.mock("mysql2/promise", () => {
  throw new Error("Cannot find package 'mysql2'");
});

describe("a missing driver", () => {
  it("is not an import-time crash: constructing a backend still works", () => {
    // The whole point of the lazy import — a module that merely CONFIGURES
    // Postgres must load fine on a machine without the driver.
    expect(() => new PostgresStore({ dsn: "postgresql://u@h/db" })).not.toThrow();
    expect(() => new MySQLStore({ dsn: "mysql://u@h/db" })).not.toThrow();
  });

  it("PostgresStore fails at connect time with an actionable install hint", async () => {
    const backend = new PostgresStore({ dsn: "postgresql://u@h/db" });
    await expect(
      backend.openSession({ project: "p", provider: "openai", model: "", startedAt: "" }),
    ).rejects.toThrow(/needs the 'pg' driver — install it with `npm install pg`/);
    await expect(backend.openReader()).rejects.toThrow(/npm install pg/);
  });

  it("MySQLStore fails at connect time with an actionable install hint", async () => {
    const backend = new MySQLStore({ dsn: "mysql://u@h/db" });
    await expect(
      backend.openSession({ project: "p", provider: "openai", model: "", startedAt: "" }),
    ).rejects.toThrow(/needs the 'mysql2' driver — install it with `npm install mysql2`/);
    await expect(backend.openReader()).rejects.toThrow(/npm install mysql2/);
  });
});

/**
 * `configure()` / `CTXDIFF_STORE` resolution — the seam a user actually touches.
 *
 * Every assertion here is pure: resolving a backend must connect to nothing,
 * create nothing and touch no disk, which is exactly what makes
 * `configure({ store })` safe to evaluate at import time in someone's module.
 * The one thing this file is most concerned with is the LIE ctxdiff must never
 * tell — silently writing a local `.ctrace` for a user who asked for a database
 * (`CTXDIFF_STORE=postgres`, a typo for `postgres://...`).
 */
import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { init } from "../src/trace.js";
import { configure, configured, ENV_VAR, fromDsn, resolve } from "../src/store/config.js";
import { SQLiteStore } from "../src/store/sqlite.js";
import { PostgresStore } from "../src/store/postgres.js";
import { MySQLStore } from "../src/store/mysql.js";

const dirs: string[] = [];

beforeEach(() => {
  delete process.env[ENV_VAR];
  configure();
});

afterEach(() => {
  delete process.env[ENV_VAR];
  configure();
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

function tempDir(): string {
  const d = mkdtempSync(join(tmpdir(), "ctxdiff-configure-"));
  dirs.push(d);
  return d;
}

describe("configure()", () => {
  it("is empty until set, and configure() clears it back to the local default", () => {
    expect(configured()).toBeNull();
    const store = new PostgresStore({ dsn: "postgresql://localhost/x" });
    configure({ store });
    expect(configured()).toBe(store);
    configure();
    expect(configured()).toBeNull();
    // Nothing configured resolves to null — "use the zero-config local default".
    expect(resolve()).toBeNull();
  });

  it("configure({ store: null }) clears it too", () => {
    configure({ store: new MySQLStore({ dsn: "mysql://localhost/x" }) });
    configure({ store: null });
    expect(configured()).toBeNull();
  });

  it("constructing a backend touches nothing — no connection, no file", () => {
    const before = readdirSync(process.cwd());
    const pg = new PostgresStore({ dsn: "postgresql://user@nowhere.invalid:1/db" });
    const my = new MySQLStore({ dsn: "mysql://user@nowhere.invalid:1/db" });
    configure({ store: pg });
    expect(resolve()).toBe(pg);
    expect(my.connectOptions().host).toBe("nowhere.invalid");
    expect(readdirSync(process.cwd())).toEqual(before);
  });

  it("resolution is explicit-beats-ambient: argument > configure() > env > null", () => {
    const explicit = new SQLiteStore({ path: "/tmp/explicit.ctrace" });
    const ambient = new SQLiteStore({ path: "/tmp/ambient.ctrace" });
    process.env[ENV_VAR] = "/tmp/env.ctrace";
    configure({ store: ambient });

    expect(resolve(explicit)).toBe(explicit);
    expect(resolve()).toBe(ambient);
    configure();
    expect((resolve() as SQLiteStore).path).toBe("/tmp/env.ctrace");
    delete process.env[ENV_VAR];
    expect(resolve()).toBeNull();
  });

  it("reads CTXDIFF_STORE at resolution time, not import time", () => {
    expect(resolve()).toBeNull();
    process.env[ENV_VAR] = "postgresql://user@host/db";
    expect(resolve()).toBeInstanceOf(PostgresStore);
  });
});

describe("CTXDIFF_STORE parsing", () => {
  it("maps postgres/postgresql (and a SQLAlchemy +driver suffix) to PostgresStore", () => {
    for (const dsn of [
      "postgres://u:p@h:5432/db",
      "postgresql://u:p@h:5432/db",
      "postgresql+psycopg://u:p@h:5432/db",
    ]) {
      const backend = fromDsn(dsn);
      expect(backend, dsn).toBeInstanceOf(PostgresStore);
      // The `+driver` part is SQLAlchemy-only syntax libpq rejects — it must be
      // normalized away, not passed through.
      expect((backend as PostgresStore).dsn).not.toContain("+");
    }
  });

  it("maps mysql/mariadb to MySQLStore, stripping a +driver suffix", () => {
    expect(fromDsn("mysql://u@h/db")).toBeInstanceOf(MySQLStore);
    expect(fromDsn("mariadb://u@h/db")).toBeInstanceOf(MySQLStore);
    expect((fromDsn("mysql+pymysql://u@h/db") as MySQLStore).dsn).toBe("mysql://u@h/db");
  });

  it("percent-decodes a MySQL user/password containing @ and /", () => {
    const store = fromDsn("mysql://us%40er:p%2Fw@db.host:3307/agents") as MySQLStore;
    expect(store.connectOptions()).toMatchObject({
      host: "db.host",
      port: 3307,
      user: "us@er",
      password: "p/w",
      database: "agents",
    });
  });

  it("accepts every sqlite:// spelling, including SQLAlchemy's four slashes", () => {
    expect((fromDsn("sqlite:///var/traces/a.ctrace") as SQLiteStore).path).toBe(
      "/var/traces/a.ctrace",
    );
    // Four slashes is how SQLAlchemy spells the SAME absolute path.
    expect((fromDsn("sqlite:////var/traces/a.ctrace") as SQLiteStore).path).toBe(
      "/var/traces/a.ctrace",
    );
    expect((fromDsn("sqlite://rel/dir") as SQLiteStore).path).toBe("rel/dir");
    expect((fromDsn("sqlite3:///abs/x.ctrace") as SQLiteStore).path).toBe("/abs/x.ctrace");
    expect((fromDsn("sqlite:///with%20space/a.ctrace") as SQLiteStore).path).toBe(
      "/with space/a.ctrace",
    );
    expect((fromDsn("sqlite://~/traces") as SQLiteStore).path).toBe(join(homedir(), "traces"));
  });

  it("treats a scheme-less value as a filesystem path, expanding ~", () => {
    expect((fromDsn("./my.ctrace") as SQLiteStore).path).toBe("./my.ctrace");
    expect((fromDsn("~/traces") as SQLiteStore).path).toBe(join(homedir(), "traces"));
    // A Windows drive letter has no '://' and must not be read as a scheme.
    expect((fromDsn("C:\\traces\\a.ctrace") as SQLiteStore).path).toBe("C:\\traces\\a.ctrace");
  });

  it("REFUSES a bare backend name rather than writing a file called 'postgres'", () => {
    for (const name of ["postgres", "postgresql", "mysql", "mariadb", "sqlite", "sqlite3"]) {
      expect(() => fromDsn(name), name).toThrow(/looks like a backend name/);
      expect(() => fromDsn(name.toUpperCase()), name).toThrow(/missing ':\/\/'/);
    }
  });

  it("rejects an unknown scheme instead of silently falling back to a local file", () => {
    expect(() => fromDsn("oracle://u@h/db")).toThrow(/unsupported CTXDIFF_STORE scheme 'oracle/);
    expect(() => fromDsn("mongodb://h/db")).toThrow(/expected postgresql:\/\/, mysql:\/\//);
  });
});

describe("SQLiteStore path resolution", () => {
  it("defaults to ./<project>.ctrace, uses <dir>/<project>.ctrace for a directory", () => {
    expect(new SQLiteStore().pathFor("agents")).toBe("agents.ctrace");
    const dir = tempDir();
    expect(new SQLiteStore({ path: dir }).pathFor("agents")).toBe(join(dir, "agents.ctrace"));
    expect(new SQLiteStore({ path: "/x/one.ctrace" }).pathFor("agents")).toBe("/x/one.ctrace");
  });

  it("refuses to open a reader when there is no single file to read", () => {
    expect(() => new SQLiteStore().openReader()).toThrow(/no single file to read/);
    expect(() => new SQLiteStore({ path: tempDir() }).openReader()).toThrow(/no single file/);
  });
});

describe("Tracer backend resolution", () => {
  it("honours CTXDIFF_STORE pointing at a directory of .ctrace files", () => {
    const dir = tempDir();
    process.env[ENV_VAR] = dir;
    const tracer = init("envproj");
    expect(tracer.path).toBe(join(dir, "envproj.ctrace"));
  });

  it("an explicit path beats an ambient configure()/env setting", () => {
    const dir = tempDir();
    configure({ store: new PostgresStore({ dsn: "postgresql://user@host/db" }) });
    const explicit = join(dir, "explicit.ctrace");
    expect(init("p", { path: explicit }).path).toBe(explicit);
  });

  it("an explicit store option beats everything, and reports no path", () => {
    process.env[ENV_VAR] = "/tmp/ignored.ctrace";
    const tracer = init("p", { store: new MySQLStore({ dsn: "mysql://u@h/db" }) });
    expect(tracer.path).toBeNull();
  });

  it("a networked backend reports tracer.path === null", () => {
    configure({ store: new PostgresStore({ dsn: "postgresql://user@host/db" }) });
    expect(init("p").path).toBeNull();
  });

  it("a bad CTXDIFF_STORE degrades init() rather than throwing into the host", async () => {
    const cwd = tempDir();
    const before = process.cwd();
    process.chdir(cwd);
    try {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
      process.env[ENV_VAR] = "postgres"; // the bare-name typo
      // init() must NOT throw: a misconfigured trace destination is a tracing
      // problem, and tracing problems never take down the traced program.
      const tracer = init("typo");
      expect(tracer.path).toBeNull();

      // ...and neither must wrap() or the host's own calls. This is what drives
      // `UnavailableBackend.openSession`, which re-throws the resolution error
      // into the writer's fail-open guard.
      const client = tracer.wrap({
        constructor: { name: "OpenAI" },
        chat: {
          completions: {
            create: async (): Promise<unknown> => ({
              choices: [{ message: { content: "ok" } }],
              usage: {},
            }),
          },
        },
        responses: {},
      }) as {
        chat: {
          completions: {
            create(a: unknown): Promise<{ choices: { message: { content: string } }[] }>;
          };
        };
      };
      const res = await client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      });
      expect(res.choices[0].message.content).toBe("ok");
      await tracer.close();

      // One warning carrying the REAL cause — the typo, not a generic failure.
      const degraded = warn.mock.calls.filter((c) =>
        String(c[0]).includes("capture degraded (store setup failed)"),
      );
      expect(degraded).toHaveLength(1);
      expect(String(degraded[0][1])).toMatch(/looks like a backend name/);
      warn.mockRestore();

      // ...and above all, NO local file named after the backend.
      expect(existsSync(join(cwd, "postgres"))).toBe(false);
      expect(readdirSync(cwd)).toEqual([]);
    } finally {
      process.chdir(before);
    }
  });
});

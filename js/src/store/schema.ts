/**
 * The `.ctrace` SQLite schema. One file = one run. `SCHEMA_VERSION` is written
 * into every run row so an old/foreign file can be rejected with a clear message
 * instead of crashing a reader.
 *
 * This DDL is byte-identical (semantically — same tables, columns, constraints)
 * to the Python SDK's `ctxdiff.store.schema`, so a file written by either SDK
 * opens in the other's reader. SCHEMA_VERSION MUST stay at 2 to match Python's
 * v2 layout (the `call` table's nullable agent/step/provider attribution
 * columns).
 */

export const SCHEMA_VERSION = 2;

export const DDL = `
CREATE TABLE IF NOT EXISTS run (
  id              TEXT PRIMARY KEY,
  project         TEXT NOT NULL,
  started_at      TEXT NOT NULL,
  provider        TEXT NOT NULL,
  models          TEXT NOT NULL,   -- JSON array of model ids seen
  ctxdiff_version TEXT NOT NULL,
  schema_version  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS call (
  id          TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL REFERENCES run(id),
  seq         INTEGER NOT NULL,
  params      TEXT NOT NULL,       -- JSON
  usage       TEXT,                -- JSON, nullable
  latency_ms  INTEGER,
  error       TEXT,
  agent       TEXT,                -- v2: which agent made this call (nullable)
  step        TEXT,                -- v2: sticky step label active at call time (nullable)
  provider    TEXT,                -- v2: provider this call went through (nullable)
  UNIQUE(run_id, seq)
);

CREATE TABLE IF NOT EXISTS block (
  content_hash TEXT PRIMARY KEY,
  role         TEXT NOT NULL,
  kind         TEXT NOT NULL,
  text         TEXT NOT NULL,
  token_count  INTEGER NOT NULL,
  token_method TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS call_block (
  call_id      TEXT NOT NULL REFERENCES call(id),
  block_id     TEXT NOT NULL REFERENCES block(content_hash),
  position     INTEGER NOT NULL,
  label        TEXT NOT NULL,
  label_source TEXT NOT NULL,
  PRIMARY KEY (call_id, position)
);
`;

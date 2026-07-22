"""The `.ctrace` SQLite schema. One file = one run. `SCHEMA_VERSION` is written
into every run row so an old/foreign file can be rejected with a clear message
instead of crashing a reader."""

SCHEMA_VERSION = 1

DDL = """
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
"""

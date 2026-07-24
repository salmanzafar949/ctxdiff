# The `.ctrace` format — cross-SDK contract

A `.ctrace` file is the shareable capture bundle produced by every ctxdiff SDK.
It is a plain **SQLite** database holding exactly **one run**, its **calls**, and
the content-addressed **blocks** those calls are built from. Any ctxdiff SDK
(Python today, JavaScript/TypeScript as of the JS capture core) writes this
format, and any ctxdiff reader — including the Python `ctxdiff view` /
`diff` / `tokens` / `cache` CLIs — opens it, regardless of which SDK produced it.

This document is the normative contract. Both SDKs implement it and a
cross-language conformance test asserts a JS-written file opens byte-compatibly
in the Python reader.

## Schema version

`SCHEMA_VERSION = 2`, written into `run.schema_version` on every file.

- Readers **accept** any version `1..SCHEMA_VERSION` and never migrate on open
  (a debugger must not rewrite the evidence it inspects). A v1 file's missing
  attribution columns surface as `NULL`.
- A file whose version is **newer** than the reader supports is rejected with a
  clear error ("upgrade ctxdiff to read this file").

Version history:

- **v1** — `run` + `call` + `block` + `call_block`, no per-call attribution.
- **v2** — `call` gains three nullable `TEXT` columns: `agent`, `step`,
  `provider`, so multi-agent runs attribute each call to the agent that made
  it, the sticky step label active at the time, and the provider it went
  through.

## Tables (DDL)

```sql
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
```

`PRAGMA foreign_keys = ON` is set by writers and readers. `id`s are UUID4 hex
(32 chars, no dashes). JSON columns (`params`, `usage`, `models`) are ordinary
`JSON.stringify` / `json.dumps` output — their internal key order is **not**
significant (readers parse them back), unlike block `text` (below).

## Block identity — the hashing contract

A block's identity is its `content_hash`, so equal content (same role + kind +
text) is stored once and referenced by many calls. Identity does **not** depend
on the label, because the same text can be labeled differently per call.

### `normalizeText(text)`

- If `text` is a **string**, return it verbatim (preserve wire truth).
- Otherwise serialize it as JSON with **sorted keys** and **no ASCII escaping**,
  byte-identical to Python `json.dumps(text, sort_keys=True, ensure_ascii=False)`:
  - object keys sorted recursively; separators are `", "` and `": "` (with the
    spaces CPython's json emits by default);
  - arrays keep order;
  - non-ASCII characters are left literal (`café`, `😀`), not `\u`-escaped;
  - control chars, quotes and backslashes escape identically in both languages.

  > **Cross-language edge:** an integer-valued float in a JSON-Schema numeric
  > keyword (e.g. `default: 3.0` inside a tool schema) normalizes differently —
  > Python `json.dumps(3.0)` → `"3.0"` vs JS `JSON.stringify(3.0)` → `"3"`,
  > because JS has no runtime int/float distinction — so the same tool schema
  > authored in both SDKs would hash differently. This affects only a
  > cross-language **diff** of the same app captured in both SDKs; it does NOT
  > affect JS→Python reads (readers store and return hashes verbatim, never
  > re-hash), and within a single language dedup is fully consistent.

### `contentHash(role, kind, text)`

```
sha256( role + "\x00" + kind + "\x00" + normalizeText(text) )  -> lowercase hex
```

The `\x00` (NUL) separator cannot appear in role/kind, so `("a","bc")` and
`("ab","c")` never collide.

**Golden vector:** `contentHash("user", "message", "hi")` ==
`4e6c4093072114cd3ec3641653e12f750391cded3515bf460ccd07162c647685`.

## Vocabularies

**Block `role`:** `system` | `user` | `assistant` | `tool` (raw provider role
passed through; unknown roles are preserved, never rejected).

**Block `kind`:** `message` | `content_part` | `tool_schema`.

**`token_method`:** `tiktoken` (exact, OpenAI via `o200k_base`) | `estimate`
(`max(1, ceil(len/4))` for non-empty text; `0` for empty). Empty text is always
0 tokens but keeps the provider's method label.

**`label` / `label_source`:** a developer `tag()` substring match wins first →
`label_source = "tagged"`; else `kind == "tool_schema"` → label `tool_schema`;
else role maps `{system→system, tool→tool_output, user→user,
assistant→history}` (fallback: raw role) → `label_source = "heuristic"`.

## Writer semantics

- **Dedup:** `INSERT OR IGNORE` on `block` by `content_hash` — first writer of a
  hash wins, repeats are no-ops. `call_block` records each membership with its
  `position` and label.
- **Per-run model roll-up:** each recorded call appends its `params.model` (or
  `params.modelId` for Bedrock's Converse shape) to `run.models` the first time
  it is seen, preserving first-seen order and deduping repeats. `run.models`
  starts empty and is backfilled from real calls (never seeded with a blank).
- **One call = one transaction** (all-or-nothing).
- `started_at` is an ISO-8601 string supplied by the tracer; `ctxdiff_version`
  is the writing SDK's package version.

## SDKs

- **Python** — `ctxdiff.store.ctrace.CTrace` (repo root `src/`).
- **JavaScript/TypeScript** — `ctxdiff` npm package (`js/`), storage via
  Node's built-in `node:sqlite`.

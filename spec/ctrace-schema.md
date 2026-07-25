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

The **image block representation** (below) was added *within* v2 and did **not**
bump the version. It introduces no column and no DDL change: it is a new value
for the existing `block.kind` column plus a text convention for the existing
`block.text` column. Both directions therefore keep working unchanged — a file
written before it opens exactly as it did, and a file containing image blocks
opens in an older ctxdiff, which simply renders the descriptor it finds in
`text` and labels it by role. A version bump would have rejected new files in
every already-released reader, for no gain.

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

  > **Degenerate `text: null` edge:** a content part with an explicit null text
  > value (e.g. `{"type": "text", "text": null}`) — which no real provider SDK
  > emits — is handled differently by the two SDKs. Python keeps the value as
  > `None`, which normalizes to `"null"` but then fails the `block.text NOT NULL`
  > insert, so the whole call is dropped fail-open (nothing recorded). JS coerces
  > the null to `""` at extraction, so the call IS recorded with an empty-text
  > block. The JS choice is deliberate: `block.text` is typed `string`, and
  > silently losing a call to a degenerate input is worse than an empty block.
  > Only reachable via hand-built malformed payloads.

### `contentHash(role, kind, text)`

```
sha256( role + "\x00" + kind + "\x00" + normalizeText(text) )  -> lowercase hex
```

The `\x00` (NUL) separator cannot appear in role/kind, so `("a","bc")` and
`("ab","c")` never collide.

**Golden vector:** `contentHash("user", "message", "hi")` ==
`4e6c4093072114cd3ec3641653e12f750391cded3515bf460ccd07162c647685`.

## Image blocks

An image content part is **not** stored as its bytes. It becomes a block with
`kind = "image"` whose `text` is a short descriptor, whose identity is a digest
of the image bytes, and whose `token_count` is the provider's published
vision-token formula.

This is the one place where a block's `text` is a stand-in rather than the
content itself, and it exists because the alternative was actively wrong: a
`data:image/png;base64,…` part serialized into `text` put a 100 KB blob in the
file and reported it to the tokenizer as prose — tens of thousands of phantom
tokens for an image that really costs a few hundred, in exactly the
vision/computer-use traces that most need accurate attribution.

### The four fields

| field | value |
| --- | --- |
| `kind` | `"image"` |
| `text` | `[image <W>×<H> · ~<N> tok]`, e.g. `[image 1024×768 · ~765 tok]` |
| `content_hash` | `contentHash(role, "image", hashInput)` — see below |
| `token_count` | the provider's vision estimate (`0` when unknowable) |
| `token_method` | always `"estimate"` |

`×` is U+00D7 MULTIPLICATION SIGN and `·` is U+00B7 MIDDLE DOT. No format or
MIME type appears in the descriptor: the block answers "what does this cost my
context window", not "what file is this".

**Degradation.** The descriptor states exactly what is known and no more:

| known | text | `token_count` |
| --- | --- | --- |
| size and cost | `[image 1024×768 · ~765 tok]` | the estimate |
| cost only | `[image · ~85 tok]` | the estimate |
| neither | `[image]` | `0` |

`0` rather than a guess: a fabricated number would be indistinguishable from a
measured one in every view, whereas a zero shows up as a gap between the call's
block total and the provider's reported `usage`.

### `hashInput` — identity is the pixels, plus what changes their cost

```
<payload>[";detail=" + detail][";part=" + stableJson(remainder)]
```

where `<payload>` is the first of these that applies:

```
data present  ->  "image:sha256:" + sha256(image bytes)   (lowercase hex)
url only      ->  "image:url:"    + url
reference only->  "image:ref:"    + file id or file uri
nothing       ->  "image:unknown"
```

That string is passed to `contentHash(role, kind, hashInput)` in place of the
text. Two consequences follow, and both are intended:

- **The same picture is one block** — across turns, across sessions, and across
  providers and wrappers. An OpenAI data URI, an Anthropic base64 source, a
  Gemini `inline_data` and a Bedrock `source.bytes` carrying the same bytes all
  hash identically, which is what lets a diff say "this screenshot has been in
  context for twelve turns".
- **Two different images of the same size do not collide**, even though they
  render the identical descriptor. If identity were the text, a changed
  screenshot would silently read as unchanged.

**The two suffixes.** A block's identity is the *request*, not only the bytes
inside it, so the terms that change what the payload costs or does are appended
to the payload term. Both are absent for every documented shape, which is what
keeps the cross-provider dedup above intact.

| suffix | when | why it is identity |
| --- | --- | --- |
| `;detail=<detail>` | the part carried OpenAI's fidelity hint | it changes the cost nine-fold (85 at `"low"` vs 765 at `"high"` for one 1024×768 screenshot) and changes the descriptor with it. Without it the standard computer-use pattern — the same screenshot at `"low"` in history and `"high"` for the current turn — collapses into one block, `INSERT OR IGNORE` keeps whichever count was written first, and the diff calls a 9× cost change "unchanged" |
| `;part=<stable JSON>` | the part carried keys ctxdiff does not interpret | chiefly Anthropic's `cache_control`, which rides as a *sibling* of the image payload. A cache breakpoint moving on or off an image is exactly what the cache profiler exists to report, so it must move the hash — as it does for a text part |

The remainder is computed as "everything the shape reader did not consume":
the payload keys (`data` / `url` / `bytes` / `file_id` / `file_uri` /
`s3Location`), the discriminators (`type`, `format`), `detail`, and the media
type are all already represented in the block, and every other key is folded in
verbatim. Nested carriers (`image_url`, `source`, `inline_data`, `file_data`,
`image`) are descended into and dropped when they empty out; nothing else is,
so `cache_control: {"type": "ephemeral"}` keeps its own `type`. Values that
cannot be serialized identically by both SDKs (raw bytes under an unrecognized
key) are omitted rather than risking a cross-language hash divergence.

### `token_count` is not identity — the first writer wins

As with every other block, `token_count` is **not** part of identity. A block
deduped across two providers therefore keeps whichever provider's count was
written first (`INSERT OR IGNORE`) — the same pre-existing property that a text
block has when it appears under both an exact and an estimating provider.

For an image this has one extra visible consequence, because the count is also
**embedded in `text`**: the same screenshot sent to OpenAI and then to Anthropic
is one block reading `[image 1024×768 · ~765 tok]`, so the Anthropic turn
renders OpenAI's 765 where Anthropic's own formula says 1049. That is a
property of content-addressed storage, not of the descriptor: dropping the
number from `text` would move the same first-writer count into the
`token_count` column without making it any more per-call, while costing the
back-compatible promise that an older reader can render the descriptor
verbatim. It is documented here instead, and asserted by
`tests/test_images.py::test_a_cross_provider_dedup_keeps_the_first_writers_count_and_text`
and its JS twin.

Both SDKs mark every image `token_method = "estimate"`, so any call containing
one is reported as approximate and reconciled against the provider's own
`usage` — which is where a cross-provider dedup shows up as a delta.

### Bytes are never fetched

A remote `http(s)` URL and a provider-side file id are recorded as references
and degrade to `[image]`. ctxdiff does **not** issue a request to learn their
dimensions. Fetching would make a local debugging tool a network client, leak
the trace subject's URLs, and make a trace's numbers depend on whether the host
was online. Both SDKs assert this with a test that fails on any attempted socket
connection.

### Dimensions

Read from the image's own header — PNG `IHDR`, JPEG `SOFn` (after walking the
marker chain), the GIF logical screen descriptor, and WebP `VP8 ` / `VP8L` /
`VP8X`. Header-only, so no image library is a dependency of either SDK. Any
other format, a truncated header or a zero dimension reads as unknown and
degrades as above.

Headers are *read*, never verified against the pixels, so both SDKs bound what
they will believe:

- **A side above 65535 reads as unknown.** PNG's `IHDR` is a pair of uint32s
  and WebP's `VP8X` canvas is 24-bit, so a truncated download or a hostile file
  can declare `4294967295 × 4294967295` in a structurally valid header.
  Believing it produced a single block estimated at 8,068,951,256,159,688
  tokens, which rounds every other block in the run to 0.0% and makes the run
  total meaningless. 65535 is exactly what JPEG's and GIF's own size fields can
  express; past that it is a broken header, not a picture. The one guard fixes
  both the number and the descriptor, since both derive from the sniffed size.
- **The JPEG marker walk stops after 1 MiB.** A well-formed JPEG hops segment
  to segment and reaches its frame header in a handful of steps; a payload that
  merely *begins* `FF D8` desynchronizes and degrades to byte-at-a-time resync
  over the whole buffer. Capture runs synchronously on the host application's
  thread, so an unbounded walk over a corrupt multi-MB screenshot stalls the
  agent being traced — which the fail-open guarantee does not permit. 1 MiB sits
  far above the largest legitimate run of pre-frame metadata (a full 64 KiB EXIF
  segment plus a chained ICC profile); past it the image reads as unknown.

### Token estimates

Each provider's own published cost model, applied to the sniffed dimensions.
All arithmetic is integer in both SDKs (floor division and ceiling division, not
floats) so the two languages cannot round apart.

| provider | formula |
| --- | --- |
| `openai` (and any unrecognized provider, incl. Azure and OpenAI-compatible endpoints) | `detail: "low"` → flat **85**. Otherwise fit into 2048×2048, then scale the shortest side to 768, then **85 + 170 × ⌈w/512⌉⌈h/512⌉**. A missing or `"auto"` detail is treated as high. |
| `anthropic`, `bedrock` | scale the longest edge to ≤1568, then **⌈w×h / 750⌉** (minimum 1) |
| `gemini` | **258** when both sides ≤384; otherwise tile side = clamp(⌊min(w,h)×2/3⌋, 256, 768) and **258 × ⌈w/tile⌉⌈h/tile⌉** |

These are estimates and are labeled as such everywhere: `token_method` is
`"estimate"`, so any call containing an image is reported as approximate, and
the `~` in the descriptor repeats the claim where a human reads it. A vision
cost is never reported as an exact tiktoken count.

### Which parts become image blocks

| provider | shape |
| --- | --- |
| OpenAI Chat Completions | `{"type": "image_url", "image_url": {"url": …, "detail": …}}` (also a bare string `image_url`, and `file_id`) |
| OpenAI Responses | `{"type": "input_image", "image_url": "<string>", "detail": …}` or `{"type": "input_image", "file_id": …}` |
| Anthropic Messages | `{"type": "image", "source": {…}}` with a `base64`, `url` or `file` source |
| Gemini | `inline_data` / `inlineData` with an `image/*` MIME type; `file_data` / `fileData` with an `image/*` MIME type |
| Bedrock Converse | `{"image": {"format": …, "source": {"bytes": …}}}`, or an `s3Location` reference |

Anything else is untouched and keeps the previous behavior byte for byte —
including a non-image `inline_data` (audio, video, PDF), an OpenAI `input_audio`
or `file` part, and any payload whose image data cannot be decoded at all. A
part is only ever *promoted* to an image block; nothing is dropped.

### Reading a file written before this existed

Nothing is migrated. A pre-existing `.ctrace` holds its images as
`content_part` blocks whose text is the JSON-serialized part, counted by
`tiktoken`, and continues to open, render and diff exactly as it always did —
same hash, same kind, same text, same token method. Conversely a file
containing `image` blocks opens in an older ctxdiff: the reader sees an
unfamiliar `kind` in a column that was always free-form text, renders the
descriptor from `text`, and labels it by role — which is the identical label
this SDK stores, since `basic_label` has no image branch.

## Vocabularies

**Block `role`:** `system` | `user` | `assistant` | `tool` (raw provider role
passed through; unknown roles are preserved, never rejected).

**Block `kind`:** `message` | `content_part` | `tool_schema` | `image`
(see [Image blocks](#image-blocks); unknown kinds are preserved, never rejected).

**`token_method`:** `tiktoken` (exact, OpenAI via `o200k_base`) | `estimate`
(`max(1, ceil(len/4))` for non-empty text; `0` for empty). Empty text is always
0 tokens but keeps the provider's method label. An **image** block is always
`estimate` — its count is a vision-token estimate, never a tokenization (see
below). The vocabulary is deliberately NOT extended with a third value:
`estimate` is what every already-released reader tests for when deciding to
print its "approximate" marker, so a new spelling would make an older ctxdiff
render a vision estimate as if it were exact.

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

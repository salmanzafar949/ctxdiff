# The cross-SDK golden corpus

This directory is the mechanism that keeps one of ctxdiff's headline claims true
permanently: **a trace renders to the same numbers in the Python SDK and the JS
SDK.**

## Why it exists

The two SDKs count tokens with two *different* libraries — [`tiktoken`][tt]
(Rust/Python) and [`gpt-tokenizer`][gt] (pure JS) — which independently
reimplement the same `o200k_base` BPE table and ship on independent release
cadences.

The `.ctrace` format itself is safe either way: a block's identity is
`sha256(role ‖ kind ‖ text)` and **token counts are not part of the hash** (see
`ctxdiff.models.content_hash`), so a trace written by one SDK always opens in
the other. But every *rendered number* is downstream of a token count —
`ctxdiff tokens`, the cache profiler's re-billed totals, the dashboard's
percentages. The day the two libraries disagree about one encoding, those
numbers diverge.

The existing conformance suites (`js/test/conformance.test.ts`,
`js/test/analyze-conformance.test.ts`) compare the two SDKs **against each
other**, live, at whatever versions happen to be installed. That is a good
check, and it has two blind spots this corpus closes:

1. **A simultaneous drift passes.** If both libraries adopt the same new merge
   in the same week, the SDKs still agree with each other — and every published
   number silently changes.
2. **On a machine with only one SDK, they don't run.** The JS conformance tests
   need a Python venv; without one they used to skip.

A committed golden cannot drift with the tools. If either SDK's numbers move,
the check fails. If **both** move, it *still* fails — until someone
deliberately regenerates and reviews the diff.

[tt]: https://github.com/openai/tiktoken
[gt]: https://github.com/niieani/gpt-tokenizer

## Layout

```
spec/golden/
├── manifest.json      # THE shared contract: pins, timezone, fixtures, cases
├── corpus/*.json      # language-neutral scenario fixtures (not .ctrace files)
├── expected/
│   ├── cli/*.txt      # expected CLI stdout, FULL TEXT
│   └── html/*.json    # expected HTML dashboard, {sha256, bytes}
├── harness.py         # the Python half (used by tests/test_golden.py)
├── regenerate.py      # the ONE regenerator
└── run-regenerate.mjs # `npm run golden:regen` shim into regenerate.py
```

The JS half lives at `js/test/helpers/golden.ts` and `js/test/golden.test.ts`.
Both halves read **the same** `manifest.json`, the same `corpus/`, and the same
`expected/` — there is no second copy of the case list anywhere.

## Why the fixtures are JSON scenarios, not committed `.ctrace` files

A committed binary `.ctrace` carries token counts *baked in* by whoever
generated it. Reading one back would never run a tokenizer, and the check would
be vacuous — it would prove only that SQLite can read a file. So each SDK
rebuilds every fixture from the JSON scenario using **its own** `content_hash` /
`count_tokens` / `basic_label`. The tokenizer genuinely runs at check time,
which is the entire point.

The rows are written with plain SQL rather than through `CTrace.record_call`,
because `record_call` mints `uuid4` ids and stamps a wall clock — and both reach
the rendered output (`ctxdiff sessions` prints a short session id and a
local-time column), so a fixture built that way could never have a fixed
expectation. The writer path is not left untested: `js/test/conformance.test.ts`
drives it across the language boundary. This corpus deliberately targets the
**read** side (analyze → render), which is where tokenizer drift surfaces.

## Full text for the CLI, a hash for the HTML — and why

**CLI stdout is stored as full text.** Each file is a few hundred bytes, and its
diff *is* the review: when a number moves you see exactly which number, in which
turn, by how much. That is the artifact a reviewer needs in order to say "yes,
that change was supposed to move the token count for the emoji block."

**The HTML dashboard is stored as `{sha256, bytes}`.** Each export is 60–200 KB
of inlined template plus a JSON island; committing five of them verbatim would
add roughly a megabyte of *unreviewable* diff to every regeneration, and nobody
can meaningfully read a minified template diff anyway. The byte length sits next
to the hash because a hash alone says only "something changed" — a size delta
immediately separates "one number moved" from "the template was rewritten." When
an HTML hash breaks, the CLI goldens beside it usually say why in plain text.

## What the corpus covers

| fixture | what it is there to catch |
| --- | --- |
| `special-tokens` | Literal control-token spellings quoted as ordinary content — `<|endoftext|>`, `<|endofprompt|>`, the `<|im_*|>` and `<|fim_*|>` family, a near-miss and an unterminated one — inline, in prose and inside a fenced block. This is the case both tokenizers refuse to encode unless the disallowed-special guard is switched off, and the case that used to latch Python's encoder for the whole process. It is deliberately **first** in the manifest's fixture list, so every other fixture is tokenized after it in the same process: if a per-text refusal ever leaks back into the encoder, the "no silent estimate fallback" assertion fails on every fixture below. |
| `unicode` | Emoji ZWJ families, skin-tone modifiers, flag sequences, CJK/Hangul/Devanagari/Arabic/Hebrew, combining vs. precomposed accents, invisible bidi/zero-width marks, astral math alphanumerics — the likeliest place for two BPE implementations to disagree, and (as it turned out) the place where a UTF-16-vs-code-point truncation bug in the JS CLI was caught. |
| `code-and-schemas` | Fenced Python/SQL/TypeScript blocks and two long JSON tool schemas, one registered but never invoked. Dense punctuation and indentation runs are the merge-table region that moves most between releases; the unused schema puts the bloat detector's token cost and percentage on the compared surface. |
| `multiagent-project` | Two sessions in one project file, two agents each: the `sessions` listing (labels, local-time rendering), the `agents` rollup across sessions, per-agent attribution, cross-session and cross-agent diffs, a developer-tagged `rag` block, and a live clock in the system prompt so the cache profiler reports a real prefix break with attribution and a fix hint. |
| `multipart-content` | Multimodal `content_part` blocks around an image: an audio part, an opaque file id, a nested tool-call argument string and a `function_call_output`. The image itself is now an `image` block (its base64 is neither stored nor tokenized), and the SAME image is sent on both turns, so the diff must report it unchanged. The `</script>` sequences inside the block text also put the exporter's escaping path under the HTML hash. |
| `image-blocks` | The image block representation end to end: every provider shape (OpenAI `image_url` data URI and remote URL at both `detail` levels, OpenAI Responses `input_image` and `file_id`, Anthropic base64/url/file sources, Gemini `inline_data` and `file_data`), all four sniffable formats, both degradations (unknown format, un-fetched remote URL), the three providers' published vision formulas, dedup of a re-sent screenshot, and a SECOND image of the same size whose different bytes must keep it a separate block. The `auditor` agent puts that second image at the SAME position in the next turn of the SAME agent — two different screenshots rendering the identical descriptor — which pins how the cache profiler must explain an image break (the two block digests; a character offset into a stand-in descriptor explains nothing). A non-image `inline_data` sits alongside to prove only image MIME types leave the stable-JSON path. |
| `round-numbers` | Whole-number floats. Python's `json.dumps` writes `100.0`; JS's `JSON.stringify` writes `100`, which is why the JS viewer needs the `PyFloat` shim. Turn 1 has a single block (exactly `100.0%`) and turn 3's provider usage equals its block total (an exact `Δ +0`), so a regression in the float-spelling shim shows up as a one-character diff. It is also where the **1-decimal rounding tie** is pinned: `--context-window 400` makes turn 1 exactly `13/400 = 3.25%`, an `odd/4` value that is exactly representable, so CPython's round-half-to-EVEN prints `3.2%` where JS's `toFixed(1)` (half away from zero) prints `3.3%`. |
| `tagged-eviction` | A block the developer `tracer.tag()`ed that entered a context and left it for good — and the four ways one only *looks* like it did. Session 1 carries the counter-examples (a block that comes back, a block whose text was edited in place, heuristic history churning every turn); session 2 interleaves two agents so a hand-off cannot read as a loss and then loses the block for real on the researcher's own timeline, chip and all. **Session 3 is the shape capture actually produces**: `tags` on the FIRST call only, because `tracer.tag()` is next-call-only and a block tagged once is `tagged` on exactly one call and heuristic thereafter. Sessions 1 and 2 set `tags` on every call, which is not what a recorder writes — and that is precisely what once hid a bug that made the whole feature silent on its own advertised example ("tagged at turn 1 … evicted at turn 5"). A detector that decides taggedness from the older side of the evicting pair reports nothing for session 3 and exits 0. |
| `astral-estimate` | Astral-plane text on the three ESTIMATE providers (bedrock, anthropic, gemini) — the half of the corpus `unicode` could not cover, because `unicode` is openai and openai is the only provider counted EXACTLY. Emoji, ZWJ families, regional-indicator and tag-sequence flags, skin-tone modifiers, math alphanumerics, CJK ext B/G, and astral text inside a tool schema, sized so the `/4` divide (not the `max(1, …)` floor) decides every count. It exists because a real divergence walked past the whole corpus: Python's `len()` counts CODE POINTS and JS's `text.length` counts UTF-16 CODE UNITS, so every astral character cost double in the JS SDK and a Converse system block of `Répondez en français 🇫🇷` rendered 7 tokens there and 6 here. Nothing was hashed differently — only every rendered number. Reverting that one-line fix now fails eight cases in this fixture. |

Six commands are captured across those fixtures — `diff` (within-session,
cross-session, cross-agent), `tokens` (run, per-turn, per-agent, per-session,
with and without a `--context-window`), `cache` (project-wide and per-agent),
`check` (passing and failing assertions, with their exit codes), `sessions`,
`agents` — plus the HTML export of every fixture.

## Regenerating

Run this whenever a change is **supposed** to move a rendered number: a new
analyzer, a reworded report line, a deliberate tokenizer re-pin.

```bash
python spec/golden/regenerate.py          # rewrite, then verify the JS SDK agrees
python spec/golden/regenerate.py --check  # verify only; write nothing
cd js && npm run golden:regen             # the same thing from the JS toolchain
```

`regenerate.py` writes what the **Python** SDK renders and then runs the JS
golden suite against the freshly written files. It refuses to exit 0 if the two
SDKs disagree — so a regenerated golden is only blessed once **both** produce
it. `--no-js` exists for a machine with no Node, and says so loudly rather than
passing silently.

Then **read the diff**. A golden regeneration is a claim that the new numbers
are correct; the diff is the evidence for it, and it belongs in the pull
request.

## Re-pinning a tokenizer

The pins live in `pyproject.toml` (`tiktoken==…`) and `js/package.json`
(`"gpt-tokenizer": "…"`, no caret), and are mirrored in `manifest.json` so both
test suites can *assert* the environment matches rather than assume it.

1. Bump **both** libraries together — re-pinning one alone is what drift *is*.
2. Update `tokenizers.*.pinned_version` in `manifest.json`.
3. Run `python spec/golden/regenerate.py`.
4. Review the diff. **Empty** means the new releases still agree with the old
   ones on every string in the corpus — the happy path, and worth saying so in
   the PR. **Non-empty** means a tokenizer changed its mind about real text:
   find out which encoding moved before merging.

## Known cross-SDK divergences

**None currently.**

### Regenerated: images stopped being counted as prose

`multipart-content` originally froze the WRONG behavior. Its base64 `data:` image
URI was JSON-serialized into the block text, so the blob was stored in the
`.ctrace` and counted by `tiktoken` as ordinary prose — the fixture's `why` even
said so approvingly ("long base64 runs stress the tokenizer"). That is precisely
the bug the image block representation fixes, so the fixture and its goldens were
regenerated together with the change, and `image-blocks` was added to cover the
representation properly. The resulting numbers are the honest ones: the image is
costed by OpenAI's published vision formula rather than by tokenizing its
encoding, and the turn is now correctly reported as approximate. Recorded here rather than hidden either way, because a
corpus that quietly omits the cases it fails is worse than no corpus.

### Fixed: a literal `<|endoftext|>` no longer latches the Python tokenizer

This section used to record a live divergence. It is kept as the account of what
was wrong, because the fixture that now covers it only makes sense against it.

Both libraries refused to encode a literal special-token spelling, and both SDKs
fell back to the character estimate *for that block*. They then diverged on what
happened next: Python latched the failure into a module-level
`_ENCODER_UNAVAILABLE` sentinel, so **every subsequent openai count in the
process** became an estimate, while JS fell back only for the offending text.
One user message quoting `<|endoftext|>` — a prompt-injection writeup, a
tokenizer tutorial, a pasted model card — therefore made a Python-captured trace
report estimates for everything after it, rendered as ordinary numbers, while a
JS-captured trace of the same conversation reported exact counts.

A second, quieter divergence sat underneath it: the two libraries do not agree
on *which* literals are special. tiktoken's `o200k_base` reserves only
`<|endoftext|>` and `<|endofprompt|>`; gpt-tokenizer's guard rejects the whole
`<|…|>` family. So `<|im_start|>` was an exact count in Python and an estimate
in JS, with no error on either side.

**The fix** (both SDKs, one commit):

1. Encode with the guard switched off — `disallowed_special=()` in Python,
   `disallowedSpecial: new Set()` in JS — so the literal is counted as the plain
   text it is. That is the truthful number: the OpenAI API escapes those
   spellings rather than honouring them, so the model received characters too.
   `a <|endoftext|> b` is now 9 tokens, `tiktoken`, in both SDKs.
2. Scope Python's latch to encoder **construction** failure — tiktoken missing,
   encoding unknown, download blocked, the network case the sentinel was always
   for. A text that will not encode now falls back for itself alone, marked
   `token_method='estimate'`, and leaves the encoder live for every other block.

`<|endoftext|>` consequently moved **into** the shared corpus as the
`special-tokens` fixture, and the pin became a convergence check:
`tests/test_golden.py::test_convergence_a_special_token_no_longer_poisons_the_python_encoder`,
with `::test_the_special_token_fixture_does_not_poison_the_others` rebuilding the
whole corpus in reverse fixture order to prove the result is order-independent.
The cross-language `(count, method)` comparison lives in
`js/test/conformance.test.ts`.

## Adding a case

Add the argv to `cli_cases` in `manifest.json` (using the `{trace}`
placeholder), run `python spec/golden/regenerate.py`, and commit the new
expectation alongside it. Both suites pick it up automatically — the Python side
parametrizes over the manifest, the JS side iterates it — so there is nothing
else to edit. A case that exits non-zero is rejected by the regenerator rather
than frozen into the expectations.

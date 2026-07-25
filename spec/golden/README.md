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
| `unicode` | Emoji ZWJ families, skin-tone modifiers, flag sequences, CJK/Hangul/Devanagari/Arabic/Hebrew, combining vs. precomposed accents, invisible bidi/zero-width marks, astral math alphanumerics — the likeliest place for two BPE implementations to disagree, and (as it turned out) the place where a UTF-16-vs-code-point truncation bug in the JS CLI was caught. |
| `code-and-schemas` | Fenced Python/SQL/TypeScript blocks and two long JSON tool schemas, one registered but never invoked. Dense punctuation and indentation runs are the merge-table region that moves most between releases; the unused schema puts the bloat detector's token cost and percentage on the compared surface. |
| `multiagent-project` | Two sessions in one project file, two agents each: the `sessions` listing (labels, local-time rendering), the `agents` rollup across sessions, per-agent attribution, cross-session and cross-agent diffs, a developer-tagged `rag` block, and a live clock in the system prompt so the cache profiler reports a real prefix break with attribution and a fix hint. |
| `multipart-content` | Vision/multimodal `content_part` blocks: a base64 `data:` image URI, an audio part, an opaque file id, a nested tool-call argument string and a `function_call_output`. Long base64 runs stress the tokenizer; the `</script>` sequences inside the block text also put the exporter's escaping path under the HTML hash. |
| `round-numbers` | Whole-number floats. Python's `json.dumps` writes `100.0`; JS's `JSON.stringify` writes `100`, which is why the JS viewer needs the `PyFloat` shim. Turn 1 has a single block (exactly `100.0%`) and turn 3's provider usage equals its block total (an exact `Δ +0`), so a regression in the float-spelling shim shows up as a one-character diff. |

Five commands are captured across those fixtures — `diff` (within-session,
cross-session, cross-agent), `tokens` (run, per-turn, per-agent, per-session),
`cache` (project-wide and per-agent), `sessions`, `agents` — plus the HTML
export of every fixture.

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

Recorded here rather than hidden, because a corpus that quietly omits the cases
it fails is worse than no corpus.

**A literal `<|endoftext|>` latches the Python tokenizer for the whole
process.** Both libraries refuse to encode a literal special-token spelling, and
both SDKs correctly fall back to the character estimate *for that block*. They
then diverge on what happens next: Python latches the failure into a
module-level sentinel, so every subsequent openai count in the process is an
estimate, while JS falls back only for the offending text. One user message
quoting `<|endoftext|>` therefore makes a Python-captured trace report estimates
for everything after it while a JS-captured trace of the same conversation
reports exact counts.

It is pinned by `tests/test_golden.py::test_known_divergence_a_special_token_poisons_the_python_encoder`
and kept **out** of the shared corpus for two stated reasons: a shared golden
must be reproducible by both SDKs and this case by definition is not, and
Python's latch is process-wide, so a fixture containing it would turn every
later fixture's counts into estimates and make the suite order-dependent.
Scoping the latch to encoder *construction* failure — the network-download case
its docstring actually justifies — would fix it and let the case move into the
corpus. That is a behavior change and belongs in its own reviewed commit.

## Adding a case

Add the argv to `cli_cases` in `manifest.json` (using the `{trace}`
placeholder), run `python spec/golden/regenerate.py`, and commit the new
expectation alongside it. Both suites pick it up automatically — the Python side
parametrizes over the manifest, the JS side iterates it — so there is nothing
else to edit. A case that exits non-zero is rejected by the regenerator rather
than frozen into the expectations.

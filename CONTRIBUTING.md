# Contributing to ctxdiff

Thanks for your interest in `ctxdiff` — a local-first context-window debugger for LLM agents. This guide covers how to set up, the conventions the codebase follows, and how to get a change merged.

By participating, you agree to keep the project welcoming and respectful. Assume good faith, keep discussion technical, and be kind.

---

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction (ideally a small `.ctrace` or a snippet that wraps a client).
- **Request a feature** — describe the debugging problem you're trying to solve, not just the API you imagined. Context matters.
- **Add a provider adapter** — see [Adding a provider](#adding-a-provider-adapter).
- **Improve docs** — the README and docstrings are part of the product.
- **Pick up a roadmap milestone** — M2–M5 (CLI diff, token heatmap, cache profiler, web viewer) are open for design discussion first.

For anything larger than a bug fix, **open an issue to discuss the approach before writing code** — it saves everyone a round trip.

---

## Development setup

`ctxdiff` targets **Python ≥ 3.10** and has one runtime dependency (`tiktoken`).

```bash
git clone https://github.com/salmanzafar949/ctxdiff
cd ctxdiff

python -m venv venv && source venv/bin/activate

pip install -e ".[dev]"      # ctxdiff + pytest
pytest                       # unit suite — should be all green
```

To work on provider integrations, install the optional eval extra, which pulls in the real SDKs:

```bash
pip install -e ".[eval]"     # + openai, anthropic, langchain, respx
pytest tests/eval            # real-SDK integration tests (HTTP stubbed, no network, no keys)
```

The eval suite drives the **real** `openai` / `anthropic` / `langchain` SDKs with their HTTP transport stubbed by `respx`, so it needs no API keys and makes no network calls. It skips cleanly if the `eval` extra isn't installed, so the default `pytest` run never requires those heavy dependencies.

---

## Project layout

```
src/ctxdiff/
├── trace.py           # public entry point: trace.init / wrap / tag; the client proxy
├── models.py          # Block / RawBlock / CallBlock; content hashing; labeling
├── tokenize/          # token counting (tiktoken exact + estimate fallback)
├── store/             # SQLite .ctrace schema + read/write
└── capture/           # provider adapters (openai, anthropic) + fail-open recorder
tests/                 # unit tests (mirrors src/)
tests/eval/            # real-SDK integration tests (opt-in via the eval extra)
```

Each module has one clear responsibility. Capture is deliberately dumb — it records what was sent, verbatim — and all interpretation (labeling, and the future diff/token/cache analyzers) lives downstream in pure functions over the store.

---

## How we work

`ctxdiff` is built **test-first**, in small, independently reviewable steps. Please follow the same rhythm:

1. **Write the failing test first** (TDD). A change without a test that would fail without it is unlikely to be merged.
2. **Make it pass** with the minimal implementation.
3. **Keep the suite green** — run `pytest` (and `pytest tests/eval` if you touched capture or a provider) before pushing.
4. **Commit in focused units** with clear messages.

### Non-negotiable invariants

These are the project's core guarantees. A change that weakens one will be sent back:

- **Fail-open, always.** No error originating in `ctxdiff` may ever reach the host application's call path. The only exception that propagates is the host's own LLM error, re-raised unchanged. If you add a capture path, wrap it so it cannot throw into the caller — and add a test that forces it to fail and asserts the host call still succeeds.
- **Local-first.** No network calls, no telemetry, no external services in library code. Token counting must degrade to an estimate rather than reach the network.
- **Wire-level truth.** Adapters record what was actually sent; they don't editorialize. Interpretation is a separate, re-runnable layer.
- **Honest numbers.** Estimated token counts are always labeled `token_method="estimate"`, never presented as exact.
- **Redaction before disk.** Any new field that stores payload text must pass through the redaction hook before being written.

### Code style

- **Every function and method gets a docstring** stating *what it does and how it works* — the mechanism, not just a restatement of the name. Comment non-obvious logic inline. The code should read as self-documenting.
- Use type hints. `from __future__ import annotations` at the top of modules.
- Names describe *what* a thing does, not *how* it's implemented.
- Prefer small, focused files over large ones. If a file you're editing has grown unwieldy, a scoped split is welcome; unrelated refactors are not.
- Keep dependencies minimal — `tiktoken` is the only runtime dependency, and adding another needs a strong justification.

### Commits & pull requests

- Use clear, conventional-style commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Keep a PR focused on one thing. Describe *what problem it solves* and *how you verified it* (include the test output).
- Make sure the full suite passes and new behavior is covered by tests.
- PRs are reviewed for spec compliance (does it do what it should, nothing more) and code quality (is it well-built).

---

## Adding a provider adapter

An adapter is the only provider-aware code in `ctxdiff`. To add one (e.g. a new SDK):

1. Create `src/ctxdiff/capture/<provider>.py` with a class implementing the `Adapter` protocol from `capture/base.py`:
   - `provider: str` — the module-root name used for detection.
   - `create_path: tuple[str, ...]` — the attribute path from the client to its completion method (e.g. `("chat", "completions", "create")`).
   - `extract_blocks(kwargs) -> list[RawBlock]` — flatten the request into ordered blocks (tool schemas, then messages), preserving send order.
   - `extract_params(kwargs) -> dict` — the request minus block-content keys.
   - `extract_usage(response) -> dict | None` — provider-reported token usage.
2. Register it in `_ADAPTERS` in `trace.py`.
3. Add unit tests in `tests/` and, ideally, a real-SDK integration test in `tests/eval/` using `respx` to stub the HTTP layer (see the existing eval tests for the pattern).

Adapters must not import the provider SDK at module load — duck-type the request kwargs and response object so `ctxdiff` stays dependency-light.

---

## Reporting security-sensitive issues

Because `ctxdiff` captures the most sensitive data in an AI stack, please report anything with security implications (e.g. a path that could leak un-redacted payloads) privately to the maintainer rather than in a public issue.

---

## License

By contributing, you agree that your contributions are licensed under the project's [Apache-2.0](LICENSE) license.

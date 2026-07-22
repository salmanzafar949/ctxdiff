<!-- Thanks for contributing to ctxdiff! Please keep this PR focused on one thing. -->

## What & why

<!-- What does this change, and what problem does it solve? -->

## How it was verified

<!-- Paste the relevant test output. New behavior should be covered by a test. -->

```
$ pytest
```

## Checklist

- [ ] Tests added/updated and the full suite passes (`pytest`; `pytest tests/eval` if capture or a provider was touched)
- [ ] New/changed functions have what/how docstrings
- [ ] The core invariants still hold: **fail-open**, **local-first** (no network in library code), **wire-level truth**, **honest token counts**, **redaction before disk**
- [ ] Docs updated if user-facing behavior changed
- [ ] Commit messages are conventional-style and focused

<!-- For larger changes, please open an issue to discuss the approach first. -->

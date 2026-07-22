# Security Policy

`ctxdiff` captures the most sensitive data in an AI stack — system prompts, retrieved documents, tool arguments — and writes it to local `.ctrace` files. We take issues that could expose that data seriously.

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report privately through one of:

1. **GitHub private vulnerability reporting** — go to the [Security tab](https://github.com/salmanzafar949/ctxdiff/security) → **Report a vulnerability**. (Preferred.)
2. **Email** the maintainer at `salmanzafar949@gmail.com` with the details.

Please include a description, a reproduction if possible, and the affected version. We'll acknowledge your report as soon as we can and keep you updated on the fix.

## Scope

Examples of in-scope concerns:

- A path where payload text is written to disk **without** passing through the redaction hook.
- A code path that makes a network call from library code (violating the local-first guarantee).
- A way for `ctxdiff` capture to crash or alter the host application (violating the fail-open guarantee).

## Supported versions

`ctxdiff` is in early development (pre-1.0). Fixes land on `main`; please test against the latest `main` before reporting.

#!/usr/bin/env python3
"""Regenerate js/src/viewer/template.ts from the Python SDK's dashboard template.

The JS viewer renders byte-identically to the Python viewer by embedding the
EXACT same HTML/CSS/JS page. Rather than hand-copy that 700-line string (and risk
drift), this script lifts `ctxdiff.viewer.template._PAGE` verbatim and writes it
into a TypeScript module as a source-safe string literal (`json.dumps` with
ensure_ascii=True — a JSON string literal is a valid JS string literal, and its
runtime value is byte-for-byte `_PAGE`). Run from the repo root:

    PYTHONPATH=src python js/scripts/gen-template.py

Re-run whenever the Python template changes; commit the regenerated template.ts.
"""
import json
import os
import sys

# Resolve the repo root (two levels up from this file: js/scripts/ -> repo).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from ctxdiff.viewer.template import _PAGE  # noqa: E402

HEADER = '''/**
 * The dashboard page as one self-contained HTML/CSS/JS string, LIFTED VERBATIM
 * from the Python SDK's `ctxdiff.viewer.template` (its `_PAGE` constant) so both
 * SDKs render byte-identically. Everything is inline — no external stylesheet,
 * script, font, image, or URL of any kind — and the runtime reads the embedded
 * JSON island with `.textContent` and renders all block text with `.textContent`
 * (never `.innerHTML`), so untrusted trace data can never become live markup.
 *
 * This file is GENERATED from the Python template (see scripts/gen-template.py);
 * the PAGE string below is an exact byte-for-byte copy. `renderPage` fills the
 * two markers `__CTXDIFF_TITLE__` (already HTML-escaped) and `__CTXDIFF_DATA__`
 * (already `</`-escaped), title first so nothing in the data can shadow it.
 */

/** The complete dashboard document, with two unfilled markers. Byte-identical
 * to the Python SDK's `_PAGE`. */
export const PAGE = '''

FOOTER = ''';

/**
 * Fill the page's two markers and return the complete HTML document.
 * `projectTitle` must already be HTML-escaped (it lands in `<title>`);
 * `dataJson` must already be `</`-escaped for the JSON island. Title is
 * substituted before data so nothing in the data can shadow the title marker.
 * Mirrors Python `render_page`.
 *
 * SECURITY: the replacements use FUNCTION replacers, not string replacers.
 * `String.prototype.replace(str, replacement)` interprets `$`-patterns in a
 * STRING replacement (`$'` = the tail after the match, `$\\``, `$&`, `$$`), and
 * both arguments here are attacker-influenced — a `$'` in block text or a
 * project title would otherwise expand to the page tail (which contains a real
 * `</script>`), closing the JSON island early and rendering following markup
 * live. A function replacer returns its value verbatim with NO `$` expansion,
 * which also restores byte-parity with Python's `str.replace` (no expansion).
 */
export function renderPage(projectTitle: string, dataJson: string): string {
  return PAGE.replace("__CTXDIFF_TITLE__", () => projectTitle).replace(
    "__CTXDIFF_DATA__",
    () => dataJson,
  );
}
'''

out = os.path.join(REPO, "js", "src", "viewer", "template.ts")
with open(out, "w", encoding="utf-8") as f:
    f.write(HEADER + json.dumps(_PAGE, ensure_ascii=True) + FOOTER)
print(f"wrote {out} ({len(_PAGE)} template chars)")

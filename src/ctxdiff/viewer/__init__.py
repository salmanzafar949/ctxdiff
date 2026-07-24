"""The viewer package: turn a `.ctrace` into a self-contained HTML dashboard.

Public surface is two names — `build_payload` (pure: trace -> JSON-ready dict)
and `export_html` (writes the standalone `.html`). The page template lives in
`template.py` as plain Python string constants (no data files) so packaging
stays trivial."""
from ctxdiff.viewer.export import build_payload, export_html

__all__ = ["build_payload", "export_html"]

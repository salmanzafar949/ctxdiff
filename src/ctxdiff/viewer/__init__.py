"""The viewer package: turn a trace into a self-contained HTML dashboard.

Public surface is three names — `build_payload` (pure: store -> JSON-ready
dict), `export_html` (opens a `.ctrace` by path and writes the standalone
`.html`) and `export_store` (the same, from an already-open store handle, so a
trace living in Postgres/MySQL exports identically). The page template lives in
`template.py` as plain Python string constants (no data files) so packaging
stays trivial."""
from ctxdiff.viewer.export import build_payload, export_html, export_store

__all__ = ["build_payload", "export_html", "export_store"]

"""The viewer package: turn a trace into a self-contained HTML dashboard.

Public surface is four names — `build_payload` (pure: store -> the JSON-ready
dict of ONE session's detail view), `build_project_payload` (that plus the
project index the three-level dashboard navigates: all agents -> an agent's
sessions -> one session's detail), `export_html` (opens a `.ctrace` by path and
writes the standalone `.html`) and `export_store` (the same, from an
already-open store handle, so a trace living in Postgres/MySQL exports
identically). The page template lives in `template.py` as plain Python string
constants (no data files) so packaging stays trivial."""
from ctxdiff.viewer.export import (
    build_payload,
    build_project_payload,
    export_html,
    export_store,
)

__all__ = ["build_payload", "build_project_payload", "export_html", "export_store"]

"""The SQLite backend — ctxdiff's zero-config default, and the reference
implementation of `StoreBackend`.

This module is a thin, connection-less FACTORY around `CTrace` (which is the
actual `Store` implementation, unchanged): it exists so the local-first path
travels through exactly the same seam as Postgres/MySQL, instead of `trace.py`
special-casing "no backend configured -> reach for the SQLite class". Nothing
about the `.ctrace` file, its schema, or its behavior changes — a user who never
calls `configure()` gets byte-identical files to before."""
from __future__ import annotations

import os

from ctxdiff.store.ctrace import CTrace


class SQLiteStore:
    """Points ctxdiff at a local `.ctrace` file (or a directory of them).

    `path` resolution, in the order a user expects:
    - `None` (the default) — one file per project in the CURRENT directory:
      `./<project>.ctrace`, exactly what `trace.init(project)` has always done.
    - an existing DIRECTORY — `<dir>/<project>.ctrace`, so
      `CTXDIFF_STORE=~/traces` keeps every project's DB in one place while
      preserving the one-file-per-project model.
    - anything else — that exact file, for every project (an explicit
      `trace.init(project, path=...)` or `CTXDIFF_STORE=./my.ctrace`).

    Constructing this touches no disk: the file is created/opened only when
    `open_session()`/`open_reader()` is called, matching the `StoreBackend`
    contract that a backend is inert until used."""

    def __init__(self, path: str | None = None):
        """Record where traces should live. `path` may be None (per-project file
        in the cwd), a directory, or a file — see the class docstring; nothing is
        resolved or created until a session/reader is opened, since the project
        name (needed for the per-project default) isn't known yet."""
        self.path = path

    def path_for(self, project: str) -> str:
        """Resolve the concrete `.ctrace` file this backend uses for `project`.
        How: applies the three rules in the class docstring — `None` and an
        existing directory both expand to `<dir>/<project>.ctrace` (cwd for
        None), anything else is returned verbatim. Exposed publicly (rather than
        being private to `open_session`) because `Tracer.path` reports it to the
        user and existing tests assert on it."""
        if self.path is None:
            return f"{project}.ctrace"
        if os.path.isdir(self.path):
            return os.path.join(self.path, f"{project}.ctrace")
        return self.path

    def open_session(self, project: str, provider: str, model: str = "",
                     started_at: str = "") -> CTrace:
        """Start a NEW session in this project's `.ctrace`, creating the file if
        absent. A straight delegation to `CTrace.open_or_create_session` — which
        already does the schema-if-not-exists, the append-a-run-row, the WAL/
        busy_timeout write configuration and the bounded locked-retry — so the
        backend seam adds exactly zero behavior to the local path."""
        return CTrace.open_or_create_session(
            self.path_for(project), project=project, provider=provider,
            model=model, started_at=started_at)

    def open_reader(self) -> CTrace:
        """Open this backend's file read-side, bound to its NEWEST session.
        Requires a concrete file: with `path=None` or a directory there is no
        single file to read (the project name is a write-time input only), so
        this raises a message pointing at `--run`, which is how the CLI already
        selects a file in that case."""
        if self.path is None or os.path.isdir(self.path):
            raise ValueError(
                "ctxdiff: SQLiteStore has no single file to read "
                f"(path={self.path!r}); pass an explicit .ctrace path")
        return CTrace.open(self.path)

"""Scaffold-level smoke checks, including the INSTALLED console script.

The console-script test is the Python counterpart of `js/test/packaging.test.ts`,
added after ctxdiff 0.2.0 shipped an npm CLI that did nothing: its entry-point
guard string-matched `process.argv[1].endsWith("cli.js")`, and npm installs
`bin` as a SYMLINK (`node_modules/.bin/ctxdiff`), so the guard was false and
`main()` never ran for any user. Every JS test passed, because they all ran the
BUILT file at its real path rather than the INSTALLED shim.

Python cannot have that bug: `[project.scripts] ctxdiff = "ctxdiff.cli:main"`
makes the installer generate a wrapper that IMPORTS `main` and calls it, so
nothing about the invocation path is inspected. The `if __name__ ==
"__main__"` guards in `cli/main.py` and `__main__.py` compare Python's own
module identity — not a filename — and only cover `python -m ctxdiff`. The
tests below pin that end to end for one cent of runtime: they run the real
generated script and the real `-m` entry, not `main()` in-process.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_package_imports():
    """The package imports and exposes a version string — proves the scaffold is installable."""
    import ctxdiff
    assert isinstance(ctxdiff.__version__, str)


def _console_script() -> Path | None:
    """Locate the `ctxdiff` console script generated for the running interpreter.

    Looks in the interpreter's own script directory (``venv/bin``, or
    ``Scripts`` on Windows) rather than on PATH, so a DIFFERENT ctxdiff that
    happens to come earlier on PATH can never be what gets tested.
    """
    bindir = Path(sys.executable).parent
    for name in ("ctxdiff.exe", "ctxdiff"):
        candidate = bindir / name
        if candidate.exists():
            return candidate
    return None


def test_installed_console_script_runs():
    """The INSTALLED `ctxdiff` entry point runs and prints help — not a silent no-op.

    Asserts non-empty stdout FIRST, because "exits 0 having printed nothing" is
    the exact shape of the npm regression this test exists to rule out here.
    """
    script = _console_script()
    if script is None:
        pytest.skip(
            "no `ctxdiff` console script next to sys.executable — "
            "install the package (`pip install -e .`) to exercise the installed entry point"
        )
    proc = subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert proc.stdout.strip() != "", f"installed console script printed nothing (stderr: {proc.stderr!r})"
    assert "usage: ctxdiff" in proc.stdout
    assert proc.returncode == 0  # argparse's own --help


def test_python_dash_m_entry_point_runs():
    """`python -m ctxdiff --help` delegates to the same CLI (the MCP-config path)."""
    proc = subprocess.run(
        [sys.executable, "-m", "ctxdiff", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert "usage: ctxdiff" in proc.stdout
    assert proc.returncode == 0

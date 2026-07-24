"""ctxdiff's CLI package. Re-exports `main` so the console-script entry point
declared in pyproject.toml (`ctxdiff = "ctxdiff.cli:main"`) resolves directly
against the package without callers needing to know about the `main`
submodule."""
from ctxdiff.cli.main import main

__all__ = ["main"]

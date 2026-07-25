"""`python -m ctxdiff ...` — the same CLI as the `ctxdiff` console script.

It exists because of MCP. An MCP client config names an executable and its
arguments, and that config has to work on a machine whose PATH the user did not
arrange: a virtualenv the editor never activated, a `pipx` install, a `uv` tool
directory. `python -m ctxdiff mcp --runs-dir ...` names the interpreter
explicitly, so the server that starts is the one from the environment ctxdiff is
actually installed in — which is exactly the failure mode "command not found:
ctxdiff" in an editor's MCP log is hardest to diagnose from.

Kept to a delegation and nothing else, so the two entry points can never
diverge."""
from __future__ import annotations

import sys

from ctxdiff.cli import main

if __name__ == "__main__":
    sys.exit(main())

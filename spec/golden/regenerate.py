#!/usr/bin/env python3
"""Rewrite ctxdiff's cross-SDK golden expectations.

Run it whenever a change is SUPPOSED to move a rendered number — a new
analyzer, a reworded report line, a deliberate tokenizer re-pin — so the change
lands as a reviewable diff of `spec/golden/expected/` instead of a manual chore
or, worse, a weakened assertion.

    python spec/golden/regenerate.py            # rewrite, then verify JS agrees
    python spec/golden/regenerate.py --check    # verify only; write nothing
    python spec/golden/regenerate.py --no-js    # skip the JS cross-check

From the JS side the same thing is `npm run golden:regen` in `js/`, which
shells out to this script.

The JS cross-check is the point of the second half. Regeneration writes what
the PYTHON SDK renders; if that were the end of it, a Python-only drift would
be quietly promoted to "expected". So after writing, this script runs the JS
golden suite against the freshly written files and refuses to exit 0 if the two
SDKs disagree — the regenerated goldens are only blessed once both produce
them. `--no-js` exists for the case where Node genuinely is not available, and
it says so loudly rather than passing silently.

It also reports the tokenizer versions it regenerated under, because that is
the fact a reviewer needs in order to judge the diff: the same diff means
something completely different under a re-pin than under a code change.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(GOLDEN_DIR))

# Import both the harness (a sibling file) and the ctxdiff package (from the
# working tree, not whatever happens to be pip-installed) — regenerating must
# reflect the source you are about to commit.
sys.path.insert(0, GOLDEN_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import harness  # noqa: E402  (path set up immediately above)


def _js_package_version(package: str) -> str:
    """The version of an npm package as installed under `js/node_modules`, or a
    marker string when it is not installed. Read straight from the package's
    own `package.json` rather than by shelling out to npm, so it works offline
    and costs nothing."""
    import json

    path = os.path.join(REPO_ROOT, "js", "node_modules", package, "package.json")
    if not os.path.exists(path):
        return "(not installed)"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("version", "(unknown)")


def regenerate(check_only: bool) -> int:
    """Build every fixture, render every case, and either write the results or
    compare them to what is committed. Returns the number of expectations that
    differ (0 when everything already matches)."""
    manifest = harness.load_manifest()
    differing = 0

    with tempfile.TemporaryDirectory(prefix="ctxdiff-golden-") as work:
        with harness.fixed_environment(manifest):
            traces = harness.build_all(manifest, work)

            for case in manifest["cli_cases"]:
                text = harness.render_cli_case(case, traces)
                if check_only:
                    if harness.read_cli_golden(case["name"]) != text:
                        print(f"  DIFFERS  cli/{case['name']}.txt")
                        differing += 1
                else:
                    harness.write_cli_golden(case["name"], text)

            for case in manifest["html_cases"]:
                value = harness.render_html_case(case, traces, work)
                if check_only:
                    if harness.read_html_golden(case["name"]) != value:
                        print(f"  DIFFERS  html/{case['name']}.json")
                        differing += 1
                else:
                    harness.write_html_golden(case["name"], value)

    return differing


def run_js_check() -> int:
    """Run the JS golden suite against the goldens just written, returning its
    exit code (or a non-zero marker when it could not be run at all).

    Deliberately NOT silent on absence: a missing `npm`, a missing
    `node_modules` or a failing suite all return non-zero, because "the JS side
    could not confirm these numbers" and "the JS side disagrees" are the same
    thing from the point of view of trusting a regenerated golden."""
    js_dir = os.path.join(REPO_ROOT, "js")
    if shutil.which("npm") is None:
        print("cross-check: npm not found — cannot confirm the JS SDK agrees.",
              file=sys.stderr)
        return 2
    if not os.path.isdir(os.path.join(js_dir, "node_modules")):
        print("cross-check: js/node_modules missing — run `npm ci` in js/ first.",
              file=sys.stderr)
        return 2
    print("cross-checking the regenerated goldens against the JS SDK…")
    proc = subprocess.run(
        ["npm", "test", "--silent", "--", "test/golden.test.ts"],
        cwd=js_dir,
    )
    return proc.returncode


def main(argv: list[str]) -> int:
    """Parse the flags, regenerate (or check), cross-check the JS side, and
    report the tokenizer versions the result was produced under."""
    parser = argparse.ArgumentParser(
        prog="regenerate.py",
        description="rewrite ctxdiff's cross-SDK golden expectations")
    parser.add_argument("--check", action="store_true",
                        help="compare only; write nothing (exit 1 on any difference)")
    parser.add_argument("--no-js", action="store_true",
                        help="skip the JS cross-check (say so in the output)")
    args = parser.parse_args(argv)

    manifest = harness.load_manifest()
    print(harness.summary())
    print(f"tokenizers: tiktoken {harness.installed_tokenizer_version()} "
          f"(pinned {manifest['tokenizers']['python']['pinned_version']}), "
          f"gpt-tokenizer {_js_package_version('gpt-tokenizer')} "
          f"(pinned {manifest['tokenizers']['javascript']['pinned_version']})")

    differing = regenerate(check_only=args.check)
    if args.check:
        if differing:
            print(f"{differing} expectation(s) differ — run without --check to "
                  f"rewrite them, then review the diff.", file=sys.stderr)
            return 1
        print("all expectations match.")
    else:
        print(f"wrote {len(manifest['cli_cases'])} CLI + "
              f"{len(manifest['html_cases'])} HTML expectations to "
              f"spec/golden/expected/")

    if args.no_js:
        print("JS cross-check SKIPPED (--no-js): these goldens have been "
              "confirmed by the Python SDK only.", file=sys.stderr)
        return 0

    code = run_js_check()
    if code != 0:
        print("the JS SDK does NOT reproduce these goldens — do not commit "
              "them until the disagreement is understood.", file=sys.stderr)
        return code
    print("both SDKs reproduce these goldens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

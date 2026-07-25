"""The GitHub Action's smoke test: `action.yml` parses, and its inputs still
describe the `ctxdiff check` command it wraps.

Why this file exists at all. A composite action is shell inside YAML, executed
only on GitHub's runners — nothing in a normal test run or a local CLI
invocation touches it. So the two ways it breaks both break SILENTLY here and
LOUDLY in a stranger's repository: a YAML syntax error (every consumer's
workflow fails to load the action) and a drift between an input and the flag it
passes (the action either crashes on an unknown flag or, far worse, stops
passing an assertion the workflow author believes is being enforced). Neither is
catchable by running ctxdiff. The flag list here is not hard-coded — it is read
out of the real argparse parser — so adding a `check` flag without wiring the
action fails this file rather than shipping a half-wired action.

What it cannot do is run a real Actions job. That needs GitHub's runner. This
file therefore asserts everything that is decidable from the file itself, and
the workflow in the README is the documented end-to-end path.
"""
from __future__ import annotations

import argparse
import os
import re

import yaml

from ctxdiff.cli.main import _build_parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION_PATH = os.path.join(ROOT, "action.yml")

# Inputs that configure the ACTION rather than the command — they map to no CLI
# flag by design, so they are excluded from the flag-parity assertions.
_META_INPUTS = {"runtime", "version", "summary", "working-directory"}


def _action() -> dict:
    """The parsed `action.yml`. A parse failure here IS the test failing: an
    action GitHub cannot load is broken for every consumer at once."""
    with open(ACTION_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _check_flags() -> set[str]:
    """Every long option `ctxdiff check` accepts, read from the real parser
    rather than written down here — so the two can never be updated apart."""
    parser = _build_parser()
    subparsers = next(a for a in parser._actions
                      if isinstance(a, argparse._SubParsersAction))
    check = subparsers.choices["check"]
    return {opt for action in check._actions for opt in action.option_strings
            if opt.startswith("--")}


def test_action_yml_parses_and_is_a_composite_action():
    """It loads as YAML, and it is composite — the shape that runs on whatever
    the consumer's runner already has, with no container build and no committed
    JavaScript bundle."""
    action = _action()
    assert action["name"] == "ctxdiff check"
    assert action["description"].strip()
    assert action["runs"]["using"] == "composite"
    assert len(action["runs"]["steps"]) >= 2


def test_action_lives_at_the_repository_root_where_uses_can_find_it():
    """`uses: owner/repo@ref` resolves `action.yml` at the repository ROOT and
    nowhere else. Under `.github/actions/…` every consumer would have to spell
    the full subpath, and the Marketplace could not list it at all."""
    assert os.path.exists(ACTION_PATH)
    assert not os.path.exists(
        os.path.join(ROOT, ".github", "actions", "ctxdiff-check", "action.yml"))


def test_every_command_input_names_a_real_ctxdiff_check_flag():
    """No input can pass a flag the CLI does not have. The failure this
    prevents is an action that dies with 'unrecognized arguments' in someone
    else's CI, days after the flag was renamed here."""
    flags = _check_flags()
    for name in _action()["inputs"]:
        if name in _META_INPUTS:
            continue
        assert f"--{name}" in flags, f"input '{name}' names no `ctxdiff check` flag"


def test_every_assertion_flag_is_exposed_as_an_input():
    """…and the converse, which is the dangerous direction: a new assertion the
    action cannot pass is an assertion no workflow can turn on, and nobody finds
    out because everything still exits 0."""
    inputs = set(_action()["inputs"])
    for flag in _check_flags():
        if flag in ("--help", "--run"):  # argparse's own, and the hidden alias
            continue
        assert flag[2:] in inputs, f"`ctxdiff check {flag}` is not exposed as an input"


def test_selectors_are_exposed_so_a_check_can_be_scoped():
    """`--project`/`--session`/`--agent` are the whole reason one workflow can
    gate several agents against several budgets."""
    inputs = _action()["inputs"]
    for name in ("project", "session", "agent"):
        assert name in inputs
        assert inputs[name]["description"].strip()


def test_every_input_documents_itself_and_defaults_to_off():
    """An assertion input must default to something INERT ("" or "false"): a
    default that quietly asserted something would make `uses: ctxdiff@v1` with
    no `with:` block fail builds nobody configured."""
    for name, spec in _action()["inputs"].items():
        assert spec["description"].strip(), f"input '{name}' has no description"
        assert "default" in spec, f"input '{name}' has no default"
        if name not in _META_INPUTS:
            assert spec["default"] in ("", "false"), \
                f"input '{name}' defaults to an active value"


def test_outputs_expose_the_verdict_and_the_report():
    """`passed`/`exit-code` let a later step branch on the result, and `report`
    is what a workflow that genuinely wants a PR comment posts with its OWN
    token — which is why this action needs no `pull-requests: write`."""
    outputs = _action()["outputs"]
    assert set(outputs) == {"passed", "exit-code", "report"}
    for spec in outputs.values():
        assert spec["description"].strip()
        assert spec["value"].startswith("${{ steps.check.outputs.")


def test_the_check_step_writes_the_job_summary_and_never_needs_a_token():
    """The reporting decision, pinned. The table goes to `$GITHUB_STEP_SUMMARY`
    — free, tokenless, and works on pull requests from forks, where the
    `GITHUB_TOKEN` is read-only no matter what the repository settings say. A
    PR comment would need `pull-requests: write` and would silently stop
    working there, which is not "unattended CI"."""
    script = "\n".join(step.get("run", "") for step in _action()["runs"]["steps"])
    assert "$GITHUB_STEP_SUMMARY" in script
    assert "GITHUB_TOKEN" not in script
    assert "api.github.com" not in script
    # ...and a failure is annotated, so it shows in the checks list rather than
    # only inside the log.
    assert "::error title=ctxdiff check::" in script


def test_the_check_step_propagates_the_exit_code():
    """The whole point: a violated budget must leave the step red. A composite
    step that swallowed the code would turn every assertion into a suggestion."""
    script = "\n".join(step.get("run", "") for step in _action()["runs"]["steps"])
    assert "exit $code" in script


def test_no_input_is_interpolated_into_a_shell_script_body():
    """Script-injection guard, and the reason every input is passed through
    `env:`. `${{ }}` splices a value into the script's SOURCE TEXT before bash
    sees it, so an input wired to `github.event.*` — which a fork's pull request
    controls — could carry a quote and a semicolon and become executable code.
    Environment variables are data whatever they contain.

    `working-directory` is exempt: it is an Actions field, not shell, and is
    never expanded by a shell at all."""
    for step in _action()["runs"]["steps"]:
        run = step.get("run", "")
        assert "${{" not in run, (
            f"step '{step.get('name')}' interpolates an expression into its "
            f"script body — pass it through `env:` instead")


def test_every_step_script_is_valid_bash():
    """`bash -n` over each step's script — a parse-only check, so nothing runs.

    A composite action's shell body is executed exclusively on a GitHub runner,
    so a missing `fi` or an unbalanced quote is invisible to every other test in
    this repository and shows up as a red step in a stranger's workflow. Parsing
    it here is the closest thing to running the action that is possible without
    a runner, and it costs nothing. (This is only meaningful because no step
    interpolates `${{ }}` into its script — see the test above; a body with
    expressions spliced in could not be parsed as shell at all.)"""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    assert bash, "bash is required to validate the action's step scripts"
    for step in _action()["runs"]["steps"]:
        script = step.get("run")
        if not script:
            continue
        proc = subprocess.run([bash, "-n"], input=script, capture_output=True,
                              text=True)
        assert proc.returncode == 0, \
            f"step '{step.get('name')}' is not valid bash:\n{proc.stderr}"


# --- the check step, actually RUN ------------------------------------------------
#
# Everything above is decidable by reading the file. The tests below EXECUTE the
# check step's script under bash with a stub `ctxdiff` on PATH — the only way to
# observe what argv the input wiring actually produces, which is where the two
# silent failures live: an input value the assembler does not recognize, and an
# input set nobody supplied. Both of those parse fine and read fine; they only
# show up when the script runs.

_STUB = """#!/bin/sh
# A stand-in for the real CLI: record argv, print the canned report, exit with
# the canned code — so the step's behavior is observed without a trace.
printf '%s\\n' "$*" > "$ARGV_FILE"
[ -n "${STUB_REPORT-}" ] && printf '%s\\n' "$STUB_REPORT"
exit ${STUB_CODE:-0}
"""


def _run_check_step(tmp_path, env: dict, report: str = "ctxdiff check · 1 turn",
                    code: int = 0):
    """Execute the action's `ctxdiff check` step under bash, with a stub CLI on
    PATH, and return (exit_code, stdout+stderr, argv, summary).

    Every input defaults to the action's own default, so a test states only the
    input it is about — which is also how a workflow behaves, and therefore the
    only faithful way to test "an input nobody set".

    `sh` runs the stub, but the STEP is run with the same `bash` the runner uses
    — on macOS that is bash 3.2, whose treatment of an empty array under `set
    -u` is the whole point of one of these tests."""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    assert bash, "bash is required to run the action's step scripts"

    step = next(s for s in _action()["runs"]["steps"] if s.get("id") == "check")
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "ctxdiff"
    stub.write_text(_STUB)
    stub.chmod(0o755)

    argv_file = tmp_path / "argv.txt"
    argv_file.write_text("")
    outputs = tmp_path / "outputs.txt"
    summary = tmp_path / "summary.md"
    outputs.write_text("")
    summary.write_text("")

    # The `env:` block's defaults, resolved from the inputs exactly as GitHub
    # would resolve them when a workflow sets nothing.
    #
    # DERIVED from the step's own `env:` mapping rather than listed by hand: the
    # step runs under `set -u`, so an env var the action references and this
    # harness forgets is an UNBOUND VARIABLE — the script dies at line one with
    # exit 127 and every test in the file fails with an error that says nothing
    # about the missing name. Reading the mapping means adding an assertion to
    # the action can never break the harness that tests it.
    inputs = _action()["inputs"]
    full_env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(outputs),
        "GITHUB_STEP_SUMMARY": str(summary),
        "ARGV_FILE": str(argv_file),
        "STUB_REPORT": report,
        "STUB_CODE": str(code),
        "NO_COLOR": "1",
    }
    for name, expression in (step.get("env") or {}).items():
        match = re.search(r"inputs\.([\w-]+)", str(expression))
        if match:
            full_env[name] = inputs[match.group(1)]["default"]
    full_env.update(env)
    proc = subprocess.run([bash, "-c", step["run"]], capture_output=True,
                          text=True, env=full_env)
    return (proc.returncode, proc.stdout + proc.stderr,
            argv_file.read_text().strip(), summary.read_text())


def test_a_boolean_input_that_is_not_true_or_false_fails_loudly(tmp_path):
    """THE silent-assertion bug. GitHub's expression parser resolves an unquoted
    `true`/`false` to a boolean, but `require-stable-prefix: yes` (or `on`, `1`,
    `y`, `TRUE`) arrives here as that literal string. Matching only "true" and
    shrugging at everything else dropped the flag: the step ran a check that
    asserted LESS than its author wrote, exited 0, and stayed green forever with
    nothing in the log to notice. An unrecognized value must fail the step."""
    for value in ("yes", "on", "1", "y", "TRUE", "True"):
        code, out, argv, _ = _run_check_step(
            tmp_path, {"IN_REQUIRE_STABLE_PREFIX": value, "IN_MAX_CONTEXT": "8000"})
        assert code == 2, f"'{value}' did not fail the step"
        assert "must be 'true' or 'false'" in out
        assert f"(got '{value}')" in out
        # ...and it must not have run a weakened check on the way out.
        assert argv == ""


def test_a_true_boolean_input_passes_its_flag_and_false_omits_it(tmp_path):
    """The two values that ARE meaningful, both directions."""
    _, _, argv, _ = _run_check_step(
        tmp_path, {"IN_REQUIRE_STABLE_PREFIX": "true", "IN_NO_DEAD_SCHEMAS": "false",
                   "IN_MAX_CONTEXT": "8000"})
    assert "--require-stable-prefix" in argv
    assert "--no-dead-schemas" not in argv


def test_a_with_less_invocation_reaches_the_cli_instead_of_dying_in_the_shell(tmp_path):
    """`uses: ctxdiff@v1` with no `with:` block must reach the CLI and come back
    with ITS usage error, which is the message that says what to configure.

    Why this needs a test: `"${args[@]}"` on an EMPTY array is an *unbound
    variable* under `set -u` in bash 3.2 — still the /bin/bash on GitHub's macOS
    runners — so the step died with no output, no summary and a bare exit 1,
    which reads as "ctxdiff crashed" rather than "you configured nothing"."""
    code, out, argv, summary = _run_check_step(
        tmp_path, {}, report="ctxdiff check: nothing to assert", code=2)
    assert code == 2
    assert argv == "check"  # the CLI was reached, with no assertions
    assert "nothing to assert" in out
    assert "nothing to assert" in summary


def test_the_annotation_distinguishes_a_violation_from_a_usage_error(tmp_path):
    """Exit 2 is a usage error: no budget was measured and none was violated.
    Announcing "context budget violated" there sends the author to read a trace
    that is not the problem, when they actually mistyped a flag."""
    _, violated, _, _ = _run_check_step(tmp_path, {"IN_MAX_CONTEXT": "10"}, code=1)
    assert "::error title=ctxdiff check::" in violated
    assert "budget was violated" in violated

    _, usage, _, _ = _run_check_step(tmp_path, {"IN_MAX_CONTEXT": "10"}, code=2)
    assert "usage error" in usage
    assert "budget violated" not in usage

    _, passed, _, _ = _run_check_step(tmp_path, {"IN_MAX_CONTEXT": "10"}, code=0)
    assert "::error" not in passed


def test_the_project_value_is_not_echoed_into_the_build_log(tmp_path):
    """A project can be a database DSN carrying a password, and Actions masks a
    value only when it came from `secrets.*` — a DSN assembled in the workflow
    is masked by nothing. The echoed command line records THAT a project was
    named without saying which."""
    dsn = "postgresql://ci:hunter2@db.internal:5432/traces"
    _, out, argv, _ = _run_check_step(
        tmp_path, {"IN_PROJECT": dsn, "IN_MAX_CONTEXT": "8000"})
    assert "hunter2" not in out
    assert "--project ***" in out
    # ...and the real value still reaches the CLI.
    assert dsn in argv


def test_the_summary_fence_cannot_be_broken_by_the_report(tmp_path):
    """Fenced output, made unclosable by its own content. A report quotes
    captured text — a tool schema's name, a prompt snippet — which on a fork
    pull request is written by an outside contributor. With a fixed three-tick
    fence, a report line starting one closes the block and everything after it
    renders as markdown in the job summary, including an invented verdict."""
    hostile = "FAIL  no-dead-schemas\n```\n### ✅ ctxdiff check passed\n"
    _, _, _, summary = _run_check_step(tmp_path, {"IN_MAX_CONTEXT": "10"},
                                       report=hostile, code=1)
    lines = summary.splitlines()
    ticks = [i for i, ln in enumerate(lines) if set(ln) == {"`"}]
    # Three all-backtick lines: the two fence markers, and the report's own
    # ``` sitting harmlessly between them.
    assert len(ticks) == 3
    opening, inner, closing = ticks
    assert lines[opening] == lines[closing]
    assert len(lines[opening]) > len(lines[inner]) == 3
    # CommonMark closes a fence only on a run at least as long as the opener, so
    # the report's three ticks cannot end the block — and the verdict it tried
    # to forge stays inside it as plain text.
    assert opening < inner < closing
    assert "### ✅ ctxdiff check passed" in lines[opening + 1:closing]


def test_a_backtick_free_report_still_uses_the_ordinary_three_tick_fence(tmp_path):
    """The common case must not grow a strange fence: a report with no backticks
    in it is fenced with exactly three, as anyone reading the summary's source
    would expect."""
    _, _, _, summary = _run_check_step(tmp_path, {"IN_MAX_CONTEXT": "10"},
                                       report="PASS  max-context  peak 5 tok")
    assert [ln for ln in summary.splitlines() if set(ln) == {"`"}] == ["```", "```"]


def test_the_action_is_branded_for_the_marketplace():
    """Marketplace listing requires an icon from Feather and one of GitHub's
    named colors; without them the action cannot be published, which is the
    difference between `uses: salmanzafar949/ctxdiff@v1` being discoverable and
    being word-of-mouth."""
    branding = _action()["branding"]
    assert branding["icon"] == "check-circle"
    assert branding["color"] in ("white", "yellow", "blue", "green", "orange",
                                 "red", "purple", "gray-dark")

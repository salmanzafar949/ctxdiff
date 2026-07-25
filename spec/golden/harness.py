"""The PYTHON half of ctxdiff's cross-SDK golden harness.

What this is for. ctxdiff ships two SDKs that make one promise: a number
rendered by the Python CLI and the same number rendered by the JS CLI are
byte-identical. But the two count tokens with two DIFFERENT libraries —
`tiktoken` (Rust/Python) and `gpt-tokenizer` (pure JS) — that reimplement the
same `o200k_base` BPE table on independent release cadences. Block
content-hashes are structurally tokenizer-independent (token counts are not
part of the hashed tuple — see `ctxdiff.models.content_hash`), so a `.ctrace`
stays portable no matter what; but every RENDERED number — `ctxdiff tokens`,
the cache profiler's re-billed totals, the HTML dashboard's percentages — is
downstream of a token count and would diverge the day the two libraries
disagree on one encoding. The conformance suites compare the two SDKs against
EACH OTHER at whatever versions happen to be installed, so a simultaneous drift
(or a drift on a machine where only one SDK runs) is invisible there. This
harness closes that: a committed, human-readable expectation that BOTH SDKs
must reproduce.

How it works, and why it rebuilds rather than reads a committed `.ctrace`.
The fixtures under `corpus/` are language-neutral JSON scenarios, not SQLite
files. Each SDK materializes them with ITS OWN `content_hash`, `count_tokens`
and `basic_label`, so the tokenizer is genuinely exercised at check time. A
committed binary `.ctrace` would carry token counts baked in by whoever
generated it and the tokenizer would never run at all — the check would be
vacuous.

Why the rows are written with plain SQL instead of `CTrace.record_call`. The
goldens must be byte-stable, and `record_call`/`create` mint `uuid4` run and
call ids and stamp a wall clock. Those ids and timestamps reach the rendered
output (`sessions` prints a short session id and a local-time column), so a
fixture built through the writer path could never have a fixed expectation.
The scenario files therefore carry explicit ids and `started_at` values and the
builder writes them verbatim. The writer path is not left untested — it is what
`js/test/conformance.test.ts` exercises across the language boundary; this
harness deliberately targets the READ side (analyze → render), which is where
tokenizer drift surfaces.

Everything the harness needs from the manifest — the timezone, the pinned
tokenizer versions, the `ctxdiff_version` literal, the fixtures and the cases —
lives in `manifest.json`, which the JS harness reads too. There is no second
copy of the case list.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
EXPECTED_DIR = os.path.join(GOLDEN_DIR, "expected")
CLI_DIR = os.path.join(EXPECTED_DIR, "cli")
HTML_DIR = os.path.join(EXPECTED_DIR, "html")


def load_manifest() -> dict:
    """Read `manifest.json` — the single shared description of the corpus, the
    cases and the pins. Read fresh on every call (it is tiny) so a test that
    edits a temp copy of the tree cannot be served a stale cache."""
    with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus(manifest: dict, fixture_id: str) -> dict:
    """Read one fixture scenario by id, resolving its path through the
    manifest so the manifest stays the only place a fixture is named."""
    entry = next(f for f in manifest["fixtures"] if f["id"] == fixture_id)
    with open(os.path.join(GOLDEN_DIR, entry["corpus"]), encoding="utf-8") as fh:
        return json.load(fh)


# --- building a fixture -------------------------------------------------------


def build_ctrace(manifest: dict, fixture_id: str, out_dir: str) -> str:
    """Materialize one corpus fixture into a real `.ctrace` under `out_dir` and
    return its path.

    The blocks are hashed, counted and labeled by the SDK's OWN public
    functions — `content_hash`, `count_tokens`, `basic_label` — exactly as
    `capture.recorder` does when a live call is captured, so the golden is a
    genuine test of those three and of the tokenizer behind `count_tokens`.
    Only the row plumbing (ids, `started_at`, insertion) is done by hand, and
    only so the result is byte-stable; see the module docstring.

    The file is named from the manifest (`unicode.ctrace`, …) because the name
    is USER-VISIBLE: `ctxdiff sessions` labels each row with the trace's
    basename, so a temp-file name would leak into the compared output."""
    # Imported here rather than at module import so this file can be read (and
    # `load_manifest` used) without the ctxdiff package on sys.path — the
    # regeneration script arranges that itself.
    from ctxdiff.models import basic_label, content_hash
    from ctxdiff.store.schema import DDL, SCHEMA_VERSION
    from ctxdiff.tokenize.counter import count_tokens

    entry = next(f for f in manifest["fixtures"] if f["id"] == fixture_id)
    corpus = load_corpus(manifest, fixture_id)
    path = os.path.join(out_dir, entry["ctrace"])
    version = manifest["ctxdiff_version"]

    conn = sqlite3.connect(path)
    try:
        conn.executescript(DDL)
        with conn:
            for session in corpus["sessions"]:
                conn.execute(
                    "INSERT INTO run VALUES (?,?,?,?,?,?,?)",
                    (session["run_id"], session["project"], session["started_at"],
                     session["provider"], json.dumps(session["models"]),
                     version, SCHEMA_VERSION),
                )
                for call in session["calls"]:
                    conn.execute(
                        "INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (call["call_id"], session["run_id"], call["seq"],
                         json.dumps(call["params"]),
                         json.dumps(call["usage"]) if call.get("usage") is not None else None,
                         call.get("latency_ms"), call.get("error"),
                         call.get("agent"), call.get("step"), call.get("provider")),
                    )
                    # `tags` mirrors what `tracer.tag(label, needle)` registers:
                    # a (label, needle) pair whose needle, when found inside a
                    # block, overrides the role-based label. Passing it through
                    # `basic_label` keeps the tagged path on the golden surface.
                    tagged = [tuple(t) for t in call.get("tags") or []]
                    provider = call.get("provider") or session["provider"]
                    for position, (role, kind, text) in enumerate(call["blocks"]):
                        token_count, token_method = count_tokens(text, provider)
                        chash = content_hash(role, kind, text)
                        label, label_source = basic_label(role, kind, text, tagged)
                        conn.execute(
                            "INSERT OR IGNORE INTO block VALUES (?,?,?,?,?,?)",
                            (chash, role, kind, text, token_count, token_method),
                        )
                        conn.execute(
                            "INSERT INTO call_block VALUES (?,?,?,?,?)",
                            (call["call_id"], chash, position, label, label_source),
                        )
    finally:
        conn.close()
    return path


def build_all(manifest: dict, out_dir: str) -> dict[str, str]:
    """Build every fixture into `out_dir`, returning {fixture id -> path}."""
    return {f["id"]: build_ctrace(manifest, f["id"], out_dir)
            for f in manifest["fixtures"]}


# --- running the cases --------------------------------------------------------


@contextlib.contextmanager
def fixed_environment(manifest: dict):
    """Pin the two ambient inputs that would otherwise make the goldens depend
    on the machine running them, and restore both afterwards.

    `TZ`, because `ctxdiff sessions` renders each session's start time in the
    VIEWER's local zone (see `cli.select.format_local`) — unpinned, the same
    trace prints a different column in Dubai and in London. `time.tzset()` is
    required for the change to take: CPython caches the zone at first use.

    `NO_COLOR`, because the renderers emit ANSI escapes on a TTY and the golden
    is meant to be a reviewable plain-text diff.
    """
    saved = {k: os.environ.get(k) for k in ("TZ", "NO_COLOR")}
    os.environ["TZ"] = manifest["tz"]
    if manifest.get("no_color", True):
        os.environ["NO_COLOR"] = "1"
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if hasattr(time, "tzset"):
            time.tzset()


@dataclass(frozen=True)
class CliResult:
    """One CLI case's captured result: the exit code and both streams."""
    code: int
    out: str
    err: str


def run_cli(argv: list[str]) -> CliResult:
    """Run the ctxdiff CLI IN PROCESS with stdout/stderr captured.

    In-process rather than spawned: it is ~30x faster across the case matrix and
    it exercises the very `cli.main` the package installs. The one thing a
    subprocess would give for free — a clean module state — is not needed here
    because every case is a read-only analysis over a file passed by path."""
    from ctxdiff.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return CliResult(code=int(code or 0), out=out.getvalue(), err=err.getvalue())


def case_argv(case: dict, traces: dict[str, str]) -> list[str]:
    """Resolve a manifest case's argv, substituting `{trace}` with the built
    fixture's path. Kept as a placeholder rather than a literal so the manifest
    stays machine-independent and both SDKs interpret it the same way."""
    return [arg.replace("{trace}", traces[case["fixture"]]) for arg in case["argv"]]


def render_cli_case(case: dict, traces: dict[str, str]) -> str:
    """Produce one CLI case's stdout, failing loudly on a non-zero exit.

    Every committed case is a SUCCESS path, so a non-zero exit means the case
    is malformed (or the CLI regressed) — surfacing that as an exception rather
    than silently golden-ing an error message is what stops a broken case from
    being frozen into the expectations by the next regeneration."""
    result = run_cli(case_argv(case, traces))
    if result.code != 0:
        raise RuntimeError(
            f"golden case {case['name']!r} exited {result.code}\n"
            f"argv: {case_argv(case, traces)}\nstderr:\n{result.err}")
    return result.out


def render_html_case(case: dict, traces: dict[str, str], work_dir: str) -> dict:
    """Export one fixture's HTML dashboard and return its committed shape:
    `{"sha256": ..., "bytes": ...}`.

    A hash and a size rather than the full document, deliberately. The dashboard
    is a single self-contained file of ~100-200 KB per fixture, nearly all of it
    an inlined template plus a JSON island; committing five of them verbatim
    would add roughly a megabyte of unreviewable diff to every regeneration, and
    a reviewer cannot meaningfully read a minified template diff anyway. The
    byte length sits next to the hash because a hash alone says only "something
    changed" — a size delta immediately separates "one number moved" from "the
    template was rewritten". The CLI goldens are stored as FULL TEXT for the
    opposite reason: they are small, and their diffs are the human-readable
    record of exactly which number moved."""
    from ctxdiff.viewer import export_html

    out_path = os.path.join(work_dir, f"{case['name']}.html")
    export_html(traces[case["fixture"]], out_path)
    with open(out_path, "rb") as fh:
        data = fh.read()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


# --- reading and writing the expectations -------------------------------------


def cli_golden_path(name: str) -> str:
    """Where one CLI case's expected stdout lives."""
    return os.path.join(CLI_DIR, f"{name}.txt")


def html_golden_path(name: str) -> str:
    """Where one HTML case's expected {sha256, bytes} lives."""
    return os.path.join(HTML_DIR, f"{name}.json")


def read_cli_golden(name: str) -> str:
    """The committed stdout for a CLI case. A missing file is an ERROR, never
    an empty string: 'the expectation does not exist' must fail the check
    rather than quietly compare against nothing."""
    path = cli_golden_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no golden for CLI case {name!r} at {path} — run "
            f"`python spec/golden/regenerate.py` and review the diff")
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def read_html_golden(name: str) -> dict:
    """The committed {sha256, bytes} for an HTML case; missing is an error for
    the same reason as `read_cli_golden`."""
    path = html_golden_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no golden for HTML case {name!r} at {path} — run "
            f"`python spec/golden/regenerate.py` and review the diff")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_cli_golden(name: str, text: str) -> None:
    """Write one CLI expectation. `newline=""` keeps the bytes exactly as the
    renderer produced them, so a Windows checkout cannot silently rewrite the
    line endings the JS harness will compare against."""
    os.makedirs(CLI_DIR, exist_ok=True)
    with open(cli_golden_path(name), "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def write_html_golden(name: str, value: dict) -> None:
    """Write one HTML expectation as pretty JSON with a trailing newline, so it
    reads as a normal text file in a diff."""
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(html_golden_path(name), "w", encoding="utf-8", newline="") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --- the pins ------------------------------------------------------------------


def installed_tokenizer_version() -> str:
    """The tiktoken version actually importable in this interpreter, used by
    `tests/test_golden.py` to assert the environment matches the manifest pin.
    Reported via `importlib.metadata` (the INSTALLED distribution) rather than
    `tiktoken.__version__`, because the pin in `pyproject.toml` constrains the
    distribution and that is what a drifting resolver would change."""
    from importlib.metadata import version

    return version("tiktoken")


def summary() -> str:
    """A one-line description of the corpus, for the regeneration script's
    output and for anyone printing the harness from a REPL."""
    manifest = load_manifest()
    return (f"{len(manifest['fixtures'])} fixtures, "
            f"{len(manifest['cli_cases'])} CLI cases, "
            f"{len(manifest['html_cases'])} HTML cases")


if __name__ == "__main__":  # pragma: no cover — convenience only
    print(summary(), file=sys.stderr)

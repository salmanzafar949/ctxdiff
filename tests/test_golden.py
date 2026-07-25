"""CROSS-SDK GOLDEN CHECK (Python side).

Every fixture in `spec/golden/corpus/` is rebuilt with THIS SDK's tokenizer,
hasher and labeler, rendered through THIS SDK's CLI and viewer, and compared to
the committed expectations under `spec/golden/expected/` — the very same files
`js/test/golden.test.ts` compares against on the JS side.

The property this defends, which nothing else in the suite defends. ctxdiff's
two SDKs count tokens with two independent libraries — `tiktoken` here,
`gpt-tokenizer` in JS — that reimplement the same `o200k_base` BPE table on
independent release cadences. A `.ctrace` is safe either way (token counts are
not part of a block's content hash), but every RENDERED number is downstream of
a token count. The existing conformance suites compare the two SDKs to EACH
OTHER, which means they pass happily if both drift the same way and degrade to
nothing on a machine where only one SDK is installed. A committed golden cannot
drift with the tools: if either SDK's numbers move this fails, and if BOTH move
it still fails until someone runs `python spec/golden/regenerate.py` and reviews
the diff.

Nothing here skips. A golden check that cannot run has not proved anything, and
a green tick that means "we didn't look" is exactly the failure mode this whole
mechanism exists to remove — so a missing expectation, an unpinned tokenizer or
an unreadable corpus is an ERROR.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest

# The harness and the regeneration script live beside the corpus they describe,
# under `spec/golden/`, so there is ONE copy of the case list that both SDKs
# read. That directory is not a package (it is language-neutral data plus two
# scripts), so it is put on the path here rather than imported by dotted name.
_GOLDEN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spec", "golden")
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)

import harness  # noqa: E402  (path set up immediately above)
import regenerate  # noqa: E402

MANIFEST = harness.load_manifest()


@pytest.fixture(scope="module", autouse=True)
def _fixed_environment():
    """Pin TZ and NO_COLOR for every test in this module.

    TZ because `ctxdiff sessions` renders each session's start time in the
    VIEWER's local zone, so an unpinned zone makes the same trace print a
    different column in Dubai and in London; NO_COLOR because the renderers emit
    ANSI escapes on a TTY and the goldens are meant to be reviewable plain text.
    Restored afterwards so the rest of the suite is unaffected."""
    with harness.fixed_environment(MANIFEST):
        yield


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    """Build every corpus fixture once for the module.

    Module-scoped because building is the expensive half (it tokenizes the whole
    corpus) and the fixtures are read-only for every case below."""
    out = tmp_path_factory.mktemp("golden-traces")
    return harness.build_all(MANIFEST, str(out))


# --- the pin -------------------------------------------------------------------


def test_installed_tiktoken_matches_the_manifest_pin():
    """The environment really has the tokenizer the goldens were produced under.

    A FAILURE, never a skip. The numbers below were frozen under a known
    tokenizer; comparing them under an unknown one would either pass by luck or
    fail with a misleading message about the CLI. Naming the real cause here is
    what makes a resolver drift diagnosable from one line of CI output."""
    assert harness.installed_tokenizer_version() == \
        MANIFEST["tokenizers"]["python"]["pinned_version"]


def test_the_pin_in_pyproject_is_exact_and_matches_the_manifest():
    """`pyproject.toml` pins tiktoken to `==` the manifest's version.

    Guards the mechanism rather than the numbers: a well-meaning `>=` relaxation
    would leave every assertion in this file passing today and silently reopen
    the drift window tomorrow. Read as text — the point is the OPERATOR, which a
    parsed requirement object would normalize away."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        pyproject = fh.read()
    pinned = MANIFEST["tokenizers"]["python"]["pinned_version"]
    assert f'dependencies = ["tiktoken=={pinned}"]' in pyproject


def test_the_pin_in_js_package_json_is_exact_and_matches_the_manifest():
    """`js/package.json` pins gpt-tokenizer to an EXACT version (no caret).

    Checked from the Python suite deliberately: the Python CI job does not run
    Node, so without this assertion a caret creeping back into the JS manifest
    would only be caught by the JS job — and the two halves of one pin have to
    move together or the pin means nothing."""
    import json

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "js", "package.json"), encoding="utf-8") as fh:
        pkg = json.load(fh)
    pinned = MANIFEST["tokenizers"]["javascript"]["pinned_version"]
    assert pkg["dependencies"]["gpt-tokenizer"] == pinned


def test_fixtures_really_run_the_exact_tokenizer(traces):
    """No block was silently counted by the character heuristic.

    This guards against the most dangerous way a golden suite goes vacuous: if
    `count_tokens` began falling back to the estimate for every block, the
    goldens would still be internally consistent and every comparison below
    would still pass — while measuring nothing at all about the tokenizer. Every
    openai-provider block must carry `token_method == 'tiktoken'`."""
    from ctxdiff.store.ctrace import CTrace

    exact = 0
    for path in traces.values():
        ct = CTrace.open(path)
        try:
            for session in ct.list_sessions():
                for call in ct.get_calls(session.id):
                    for cb in ct.get_call_blocks(call.id):
                        if (call.provider or session.provider) == "openai":
                            assert cb.block.token_method == "tiktoken"
                            exact += 1
        finally:
            ct.close()
    assert exact > 50


# --- the expectations ----------------------------------------------------------


@pytest.mark.parametrize("case", MANIFEST["cli_cases"], ids=lambda c: c["name"])
def test_cli_golden(case, traces):
    """One CLI case's stdout is byte-identical to its committed expectation."""
    assert harness.render_cli_case(case, traces) == harness.read_cli_golden(case["name"])


@pytest.mark.parametrize("case", MANIFEST["html_cases"], ids=lambda c: c["name"])
def test_html_golden(case, traces, tmp_path):
    """One fixture's exported dashboard hashes to its committed {sha256, bytes}."""
    rendered = harness.render_html_case(case, traces, str(tmp_path))
    assert rendered == harness.read_html_golden(case["name"])


def test_no_orphaned_expectations():
    """Every committed expectation is claimed by a case in the manifest.

    Without this, deleting a case from the manifest would leave its expectation
    file behind, still committed, still looking authoritative, and never
    compared against anything again — a stale number masquerading as a checked
    one."""
    expected_cli = {f"{c['name']}.txt" for c in MANIFEST["cli_cases"]}
    expected_html = {f"{c['name']}.json" for c in MANIFEST["html_cases"]}
    assert set(os.listdir(harness.CLI_DIR)) == expected_cli
    assert set(os.listdir(harness.HTML_DIR)) == expected_html


# --- the meta-tests: proving the harness is not vacuous -------------------------


def _clone_expectations(tmp_path, monkeypatch):
    """Point the harness at a WRITABLE copy of the committed expectations and
    return its path. Lets the meta-tests corrupt an expectation without ever
    touching the real files (a meta-test that dirtied the working tree would be
    its own kind of hazard)."""
    clone = tmp_path / "expected"
    shutil.copytree(harness.EXPECTED_DIR, clone)
    monkeypatch.setattr(harness, "EXPECTED_DIR", str(clone))
    monkeypatch.setattr(harness, "CLI_DIR", str(clone / "cli"))
    monkeypatch.setattr(harness, "HTML_DIR", str(clone / "html"))
    return clone


def test_harness_catches_a_mutated_cli_expectation(tmp_path, monkeypatch):
    """THE META-TEST. Corrupt one digit of one committed number in a temp copy
    and assert the check REJECTS it.

    A golden suite that compares against an empty string, a missing file or its
    own fresh output is worse than no suite: it is a green tick that means
    nothing. This drives the very code path CI runs — `regenerate.py --check` —
    so what is proved is that the CI gate fails, not merely that two strings in
    this test differ."""
    clone = _clone_expectations(tmp_path, monkeypatch)
    target = clone / "cli" / "round-numbers.tokens.txt"
    original = target.read_text(encoding="utf-8")

    # Sanity first: unmutated, the check must be clean. Otherwise a later
    # "differs" result would prove nothing about the mutation.
    assert regenerate.regenerate(check_only=True) == 0

    mutated = original.replace("100.0%", "111.1%", 1)
    assert mutated != original, "the fixture golden no longer contains the mutated token"
    target.write_text(mutated, encoding="utf-8")

    assert regenerate.regenerate(check_only=True) == 1


def test_harness_catches_a_mutated_html_expectation(tmp_path, monkeypatch):
    """The same proof for the HTML side, where the expectation is a hash rather
    than full text — a hash comparison is the easiest kind to accidentally write
    as a no-op, so it gets its own mutation."""
    clone = _clone_expectations(tmp_path, monkeypatch)
    target = clone / "html" / "unicode.json"
    value = harness.read_html_golden("unicode")
    value["sha256"] = "0" * 64
    target.write_text(__import__("json").dumps(value), encoding="utf-8")

    assert regenerate.regenerate(check_only=True) == 1


def test_a_missing_expectation_fails_loudly(tmp_path, monkeypatch):
    """A deleted expectation raises instead of comparing against nothing.

    The other way a golden suite dies quietly: `read_golden` returning "" for a
    file that isn't there, so the case passes for anything that renders
    empty — or, worse, is skipped."""
    clone = _clone_expectations(tmp_path, monkeypatch)
    (clone / "cli" / "unicode.tokens.txt").unlink()
    with pytest.raises(FileNotFoundError, match="no golden for CLI case"):
        harness.read_cli_golden("unicode.tokens")
    with pytest.raises(FileNotFoundError, match="no golden for HTML case"):
        harness.read_html_golden("never-generated")


def test_a_failing_case_is_never_frozen_into_the_goldens(traces):
    """A case that exits non-zero raises instead of golden-ing its error text.

    Without this, a typo'd argv would regenerate into a committed expectation
    containing a usage message, and the case would then "pass" forever while
    testing nothing but argparse."""
    broken = {"name": "synthetic", "fixture": next(iter(traces)),
              "argv": ["diff", "--turn", "1", "--project", "{trace}"]}
    with pytest.raises(RuntimeError, match="exited 2"):
        harness.render_cli_case(broken, traces)


# --- a known cross-SDK divergence, pinned rather than hidden --------------------


def test_known_divergence_a_special_token_poisons_the_python_encoder(traces):
    """PINS A REAL, CURRENT DISAGREEMENT between the two SDKs, so it cannot
    change without someone noticing.

    Text containing a literal special-token spelling (`<|endoftext|>`) makes
    BOTH tokenizers raise — tiktoken's `disallowed_special` guard and
    gpt-tokenizer's equivalent — and both SDKs correctly fall back to the
    character estimate for that block. They then diverge on what happens NEXT:

      * Python latches the failure into the module-level `_ENCODER_UNAVAILABLE`
        sentinel, so EVERY subsequent openai count in the process is an
        estimate;
      * JS falls back only for the offending text and keeps counting exactly.

    So one user message quoting `<|endoftext|>` — a tokenizer tutorial, a
    prompt-injection writeup, a training-data sample — makes a Python-captured
    trace report estimates for everything after it while a JS-captured trace of
    the same conversation reports exact counts. That is a genuine parity gap.

    It is pinned HERE and kept OUT of the shared corpus for two reasons, both
    stated so the choice is reviewable rather than silent. First, a shared
    golden must be reproducible by both SDKs, and this case is by definition not
    — freezing it would mean committing two contradictory expectations and
    losing the "same goldens" guarantee for everything else. Second, Python's
    latch is PROCESS-WIDE: a corpus fixture containing this text would turn
    every later fixture's counts into estimates, making the whole suite
    order-dependent. Fixing the latch (scoping it to encoder CONSTRUCTION
    failure, which is the network-download case its docstring actually
    justifies, rather than to any encode error) would let this move into the
    corpus; that is a behavior change and belongs in its own reviewed commit."""
    from ctxdiff.tokenize import counter

    saved = counter._ENCODER
    try:
        counter._ENCODER = None  # a clean process, as a fresh import would be
        assert counter.count_tokens("hello world", "openai") == (2, "tiktoken")

        # The offending block itself: both SDKs agree here — estimate, 5 tokens
        # for these 17 characters (ceil(17/4)).
        assert counter.count_tokens("a <|endoftext|> b", "openai") == (5, "estimate")

        # ...and here they do NOT. Python has latched; JS would still say
        # (2, 'tiktoken'). Asserted as the CURRENT TRUTH, not as desired
        # behavior — if this line starts failing, the latch was fixed and the
        # two SDKs now agree, which is good news that should be reviewed, not a
        # regression.
        assert counter.count_tokens("hello world", "openai") == (3, "estimate")
    finally:
        counter._ENCODER = saved

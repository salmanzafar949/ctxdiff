"""How an MCP result is shaped, capped, fenced and redacted.

This module holds the three properties that make the MCP surface safe to point
at a debugging agent, and NOTHING about ctxdiff's analyses — it is pure string
and dict work, importable with the `mcp` extra absent.

**1. Token discipline.** ctxdiff exists to keep context windows small; an MCP
tool that answered a question by pasting a 50 KB retrieved document into the
caller's context would be self-refuting. So every result is compact JSON —
content-hash prefixes, labels, token counts, structure — with captured text
limited to a short `preview` (added/evicted blocks) or the CHANGED hunk alone
(modified blocks). Whole block text is never returned by default; it is fetched
deliberately, and paged, via `ctxdiff_block`. `fit()` then enforces a hard
`MAX_RESULT_BYTES` cap on the ENCODED result and sets `truncated: true` so the
agent is told, in-band, that it is looking at a subset.

**2. The consent boundary.** MCP hands whatever we return to the client's model,
which for most users is a cloud model — the exact opposite of ctxdiff's
"nothing leaves your machine" default. `TextPolicy(redact=True)` (the server's
`--redact` flag) is a never-return-raw-text mode: labels, hashes, token counts
and structure still flow, every captured string is withheld, and that includes
`ctxdiff_block`, whose entire job is text. Redaction is applied twice on
purpose — at each call site (`TextPolicy.preview`/`.hunk`/`.body` return None)
and again as a final recursive `scrub()` over the assembled payload — so
forgetting it in one new field cannot leak.

**3. The injection boundary.** A `.ctrace` is a recording of attacker-influenced
strings: end-user messages, retrieved documents, tool output. Returning them to
an agent is a prompt-injection vector, so every captured string is stripped of
ANSI escapes (a terminal-injection vector in its own right) and wrapped in a
`<captured-untrusted-input>` fence. Any literal fence tag INSIDE the captured
text is defanged first, so content cannot close the wrapper early and speak
outside it."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass

# --- limits ------------------------------------------------------------------

# Hard cap on one tool result, measured on the UTF-8 ENCODED JSON (what actually
# lands in the agent's context), not on character count — JSON escaping can
# double the size of quote-heavy text, so a character budget would not hold.
MAX_RESULT_BYTES = 15_000

# Captured-text budgets. A preview exists to make a block RECOGNIZABLE ("is this
# the RAG chunk I think it is?"), not readable; a hunk shows the change and
# nothing else. Both are deliberately small — `ctxdiff_block` is how you read.
PREVIEW_CHARS = 160
HUNK_CHARS = 500

# `ctxdiff_block`'s paging window: the default slice, and the ceiling a caller
# may ask for. The ceiling sits below MAX_RESULT_BYTES so a full-size slice of
# plain text still leaves room for the metadata around it; a slice of
# escape-heavy text is shrunk further by the caller's own fitting loop.
BLOCK_CHARS_DEFAULT = 4_000
BLOCK_CHARS_MAX = 12_000

# How much of a 64-hex content hash is quoted in results. Long enough to be
# unambiguous within one run, short enough not to cost a line per block —
# `ctxdiff_block` accepts this prefix (or any other) straight back.
HASH_PREFIX_CHARS = 12

# --- the untrusted-input fence -------------------------------------------------

FENCE_OPEN = "<captured-untrusted-input>"
FENCE_CLOSE = "</captured-untrusted-input>"

# Both fence tags, in either polarity, however they are cased — what has to be
# defanged inside captured text so the content cannot close its own wrapper.
_FENCE_TAG_RE = re.compile(r"<(/?)(captured-untrusted-input)", re.IGNORECASE)

# ANSI/VT escape sequences: CSI (ESC [ ... final byte) and the two-character
# forms. Captured text can carry these verbatim — a tool output that painted a
# terminal, or a deliberate escape — and passing them through would let recorded
# content move the cursor or recolor the agent's transcript.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)")

# The note attached to a redacted result, so the agent understands the missing
# text is a POLICY, not an error, and stops asking for it.
REDACTED_NOTE = ("this ctxdiff MCP server runs with --redact: captured text is "
                 "never returned, only labels, content hashes, token counts and "
                 "structure")

# Keys whose values are (or may contain) captured text. The call sites already
# withhold these under --redact; `scrub()` re-checks the assembled payload so a
# field added later without thinking cannot leak.
#
# `unused_tools` is here because a tool's NAME is captured text too, however
# much it reads as configuration: it is a JSON `"name"` lifted out of a recorded
# tool_schema block, i.e. whatever the traced application put on the wire. It
# was the one field in this surface that reached the model unfenced AND survived
# --redact, so it gets both enforcements like everything else.
_TEXT_KEYS = frozenset({"text", "preview", "hunk", "detail", "snippet",
                        "unused_tools"})

# Lists `fit()` may shorten to get under the cap, longest-first. Everything
# scalar (totals, counts, verdicts) survives every level of truncation, so a
# capped result still answers the question numerically even when it stops
# enumerating.
_SHRINKABLE_KEYS = frozenset({"changes", "turns", "breaks", "runs",
                              "tagged_evictions", "unused_tools", "by_label",
                              "turns_present", "top_changes"})

# `ctxdiff_explain` names its per-turn findings with this suffix
# (`tagged_evictions_at_this_turn`, `cache.breaks_at_this_turn`). Matched by
# SUFFIX rather than listed, because forgetting to add one here is not a
# cosmetic miss: a list the shrinker cannot trim is a list that can hold the
# result over the cap with no way down, which is how `fit()` once span forever
# on a turn that evicted 45 tagged blocks.
_SHRINKABLE_SUFFIX = "_at_this_turn"


def _is_shrinkable(key) -> bool:
    """Whether `fit()` may drop items from the list under `key` — one of the
    known enumerable keys, or any `*_at_this_turn` finding list."""
    return isinstance(key, str) and (key in _SHRINKABLE_KEYS
                                     or key.endswith(_SHRINKABLE_SUFFIX))


def strip_ansi(text: str) -> str:
    """Remove ANSI/VT escape sequences from captured text. Applied to every
    string that leaves this module: the MCP result is rendered into a terminal
    transcript by most clients, and a recorded tool output containing a cursor
    or color escape would otherwise repaint it."""
    return _ANSI_RE.sub("", text)


def defang_fence(text: str) -> str:
    """Neutralize any literal `<captured-untrusted-input>` tag occurring INSIDE
    captured text by escaping its opening angle bracket.

    Without this the fence is worthless against the exact attacker who matters:
    a document that contains the closing tag would end the fence early and put
    the rest of its content outside it, where an agent reads it as ours. The
    replacement is visible (`&lt;/captured-untrusted-input`) rather than silent,
    so a reader can see the content tried."""
    return _FENCE_TAG_RE.sub(r"&lt;\1\2", text)


def fence(text: str) -> str:
    """Wrap captured text in the untrusted-input fence, ANSI-stripped and with
    any nested fence tag defanged. The server's instructions tell the model that
    everything inside these markers is DATA, never instructions; this is the
    marker."""
    return FENCE_OPEN + defang_fence(strip_ansi(text)) + FENCE_CLOSE


def flatten(text: str, limit: int) -> str:
    """Collapse text to one line (runs of whitespace -> single spaces) and hard
    cut it at `limit` characters with an ellipsis. Same shape as the CLI's
    snippets: previews are for recognition, and a multi-line preview would cost
    a dozen lines of the agent's context to say what one line says."""
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def short_hash(content_hash: str) -> str:
    """The hash prefix quoted in results — the handle `ctxdiff_block` takes."""
    return content_hash[:HASH_PREFIX_CHARS]


# --- the redaction policy ------------------------------------------------------


@dataclass(frozen=True)
class TextPolicy:
    """Whether this server may return captured text at all — the `--redact`
    flag, passed down to every payload builder instead of read from a global, so
    a test can construct both modes side by side and no builder can forget to
    consult it.

    Each accessor returns None when redacting, and payload builders drop None
    values (`compact`), so a redacted result simply has no text FIELDS rather
    than fields holding empty strings — an agent should see that the data is
    absent, not that the block was empty."""

    redact: bool = False

    def preview(self, text: str, limit: int = PREVIEW_CHARS) -> str | None:
        """A one-line, fenced, recognition-sized excerpt of a captured block."""
        if self.redact:
            return None
        return fence(flatten(text, limit))

    def hunk(self, text: str, limit: int = HUNK_CHARS) -> str | None:
        """A fenced changed-hunk (the `-`/`+` lines of a modified block). Larger
        budget than a preview because the hunk IS the answer to "what changed",
        but still bounded — a rewritten system prompt is read via
        `ctxdiff_block`, not pasted into a diff result."""
        if self.redact:
            return None
        return fence(text[:limit] + ("…" if len(text) > limit else ""))

    def body(self, text: str) -> str | None:
        """A fenced verbatim slice of block text, newlines intact — the one
        place ctxdiff returns content as written, and therefore the one place
        `--redact` most has to reach."""
        if self.redact:
            return None
        return fence(strip_ansi(text))


def compact(payload: dict) -> dict:
    """Drop None-valued keys from a payload dict, recursively through nested
    dicts and lists. Lets every builder write `"preview": policy.preview(...)`
    unconditionally: under `--redact` the key simply does not appear, and a
    genuinely absent number (no provider usage, no previous turn) disappears the
    same way instead of costing a `null` in the agent's context."""
    out: dict = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            out[key] = compact(value)
        elif isinstance(value, list):
            out[key] = [compact(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def scrub(payload: dict, policy: TextPolicy) -> dict:
    """Second, independent enforcement of `--redact`: walk the ASSEMBLED payload
    and delete every key that carries captured text, adding the standing
    `redacted` note once at the top level.

    Deliberately redundant with `TextPolicy`. The first enforcement is a
    discipline every call site must remember; this one is a property of the
    result itself, so a field added next year without reading this docstring
    still cannot leak text out of a redacted server. A no-op when not
    redacting."""
    if not policy.redact:
        return payload
    scrubbed = _scrub_value(payload)
    scrubbed["redacted"] = True
    scrubbed["redacted_note"] = REDACTED_NOTE
    return scrubbed


def _scrub_value(value):
    """Recursive half of `scrub`: rebuild dicts without their text keys, and
    recurse through lists, leaving every other value untouched."""
    if isinstance(value, dict):
        return {k: _scrub_value(v) for k, v in value.items()
                if k not in _TEXT_KEYS}
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    return value


# --- encoding and the hard cap -------------------------------------------------


def encode(payload: dict) -> str:
    """Serialize a payload to the exact string the agent receives: compact JSON
    (no indentation, no spaces after separators — this is a context-efficiency
    tool spending its own customer's context) with non-ASCII left as-is, since
    escaping every accented character would inflate real prompts for nothing."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def size(text: str) -> int:
    """Encoded byte length — the quantity `MAX_RESULT_BYTES` is measured in."""
    return len(text.encode("utf-8"))


def fit(payload: dict, cap: int = MAX_RESULT_BYTES) -> str:
    """Encode `payload`, shrinking it until it fits under `cap`, and mark it
    `truncated` when anything was dropped.

    Why a cap at all: a single block can be a 50 KB retrieved document and a
    long run can hold hundreds of them, so an uncapped result would blow out the
    context window of the very agent that came here for help with its context
    window. The cap is enforced on the encoded bytes, after assembly, so no
    builder has to predict JSON escaping.

    Shrink order, weakest-value-first:

    1. **Trim the longest enumerable list** (changes, turns, breaks, runs...).
       An omitted twelfth diff entry costs the agent less than a missing total,
       and `omitted` tells it how many it is not seeing.
    2. **Halve the longest captured string.** Only `ctxdiff_block` can carry one
       big enough to matter; the diff/token tools have already bounded theirs.
    3. **Drop captured text entirely**, keeping the structure.
    4. Failing all that, fall back to the SCALAR SKELETON (`_last_resort`):
       every list and every captured string gone, every number kept.

    Every step reports honestly whether it made the payload smaller, and every
    step that can run at all strictly shrinks it — which is what makes the loop
    terminate. It once did not: a short fenced string was "halved" into an
    identical string forever while the encoded size never moved, and a server
    stuck in here answers no further tool call for the rest of the session.

    The `truncated` marker is set BEFORE the shrink loop, not after, so the keys
    it adds are themselves inside the budget."""
    encoded = encode(payload)
    if size(encoded) <= cap:
        return encoded

    payload = copy.deepcopy(payload)  # never mutate the builder's dict
    payload["truncated"] = True
    payload["truncated_note"] = (
        f"result capped at {cap} bytes to protect your context window — some "
        "items or text were dropped; call ctxdiff_block(run, content_hash) for "
        "a block's full text, or narrow the request (one turn, one agent)")
    encoded = encode(payload)

    while size(encoded) > cap and _shrink_once(payload):
        encoded = encode(payload)

    if size(encoded) > cap:
        return _last_resort(payload, cap)
    return encoded


def _shrink_once(payload: dict) -> bool:
    """Perform ONE shrink step on `payload` in place, returning False when
    nothing is left to shrink. Steps are ordered by what a reader can most
    afford to lose (see `fit`)."""
    return (_trim_longest_list(payload)
            or _halve_longest_text(payload)
            or _drop_all_text(payload))


# The scalar skeleton's own note. Deliberately different from
# `truncated_note`: at this point there are no items left to fetch one by one,
# so pointing at ctxdiff_block would be advice the agent cannot act on.
LAST_RESORT_NOTE = (
    "this result was too large to enumerate at all, so only its totals and "
    "counts are returned — every list and every captured excerpt was dropped. "
    "Narrow the request: one turn, or one agent.")

# The handful of keys the very last fallback keeps. They identify WHICH
# question was being answered, which is the minimum an agent needs to ask a
# narrower one, and they are all short by construction (a run handle is a
# filename this server itself minted; the rest are integers).
_IDENTITY_KEYS = ("run", "session", "project", "agent", "turn", "turn_a",
                  "turn_b")

# Keys the scalar skeleton must not drop while it is shedding weight — the
# identifiers above, the in-band explanation of what happened, and the counter
# `_drop_largest_optional_key` keeps (which must be exempt from its own sweep,
# or deleting it and re-adding it would be a step that never terminates).
_LAST_RESORT_KEEP = (frozenset(_IDENTITY_KEYS)
                     | {"truncated", "truncated_note", "omitted_keys"})

# How long a string may be in the scalar skeleton. Not captured text — hints,
# verdicts, `fix_hint`, and developer-set names like a project or an agent —
# but "not captured" is not the same as "short", and the whole point of this
# path is that what it returns fits.
_LAST_RESORT_STRING_CHARS = 200


def _last_resort(payload: dict, cap: int) -> str:
    """Reduce `payload` to its SCALARS and return that, still under `cap`.

    Reached when the ordinary shrink steps have run out of things they know how
    to shrink and the result is somehow still too big — an unbounded dict of
    per-agent totals, say, which is neither a list nor captured text.

    What this must not do is what it used to: return a bare "result too large"
    with no run, no turn, no totals and no counts. `fit`'s whole contract is
    that the numbers survive every level of truncation — they ARE the answer,
    and an agent handed 150 bytes of apology has been told nothing at all and
    has no way to ask better. So lists and captured text go, dicts of numbers
    stay, and only if THAT still does not fit do keys start being dropped —
    largest first, identifiers last."""
    skeleton = _scalars_only(payload)
    skeleton["truncated"] = True
    skeleton["truncated_note"] = LAST_RESORT_NOTE
    encoded = encode(skeleton)
    while size(encoded) > cap and _drop_largest_optional_key(skeleton):
        encoded = encode(skeleton)
    return encoded


def _scalars_only(payload: dict) -> dict:
    """A copy of `payload` holding only its numbers, booleans and strings:
    lists dropped wholesale, captured-text keys dropped, nested dicts kept when
    they still have something scalar in them (which is how `counts`, `totals`
    and `provider_usage` survive), and every string clipped — a project or agent
    name is the developer's, not a recording, but it is still whatever length
    they chose."""
    out: dict = {}
    for key, value in payload.items():
        if key in _TEXT_KEYS or isinstance(value, list):
            continue
        if isinstance(value, dict):
            nested = _scalars_only(value)
            if nested:
                out[key] = nested
        elif isinstance(value, str):
            out[key] = flatten(value, _LAST_RESORT_STRING_CHARS)
        else:
            out[key] = value
    return out


def _drop_largest_optional_key(skeleton: dict) -> bool:
    """Delete the top-level key whose ENCODED value is biggest, skipping the
    identifiers and the truncation note, and return whether anything went.

    The guaranteed-terminating floor under `_last_resort`: there are finitely
    many keys, each pass removes one, and what is left when they are gone is a
    handful of short identifiers plus a constant note — small by construction,
    so the recursion cannot run away."""
    candidates = [k for k in skeleton if k not in _LAST_RESORT_KEEP]
    if not candidates:
        return False
    largest = max(candidates, key=lambda k: size(encode({k: skeleton[k]})))
    del skeleton[largest]
    skeleton["omitted_keys"] = skeleton.get("omitted_keys", 0) + 1
    return True


def _walk(value):
    """Yield every (container, key, value) triple reachable in a payload, so the
    shrink steps can find the longest list / longest string without each of them
    re-implementing the traversal. Dict items are snapshotted before yielding, so
    a caller may delete the key it was just handed."""
    if isinstance(value, dict):
        for k, v in list(value.items()):
            yield value, k, v
            yield from _walk(v)
    elif isinstance(value, list):
        for v in list(value):
            yield from _walk(v)


def _trim_longest_list(payload: dict) -> bool:
    """Drop the LAST element of the longest shrinkable list (see
    `_SHRINKABLE_KEYS`) and count it in `omitted`. Last rather than first
    because every list here is ordered most-relevant-first: turns ascend to the
    one you asked about, changes are position-ordered, runs are newest-first."""
    best_container = best_key = None
    best_len = 0
    for container, key, value in _walk(payload):
        if _is_shrinkable(key) and isinstance(value, list) and len(value) > best_len:
            best_container, best_key, best_len = container, key, len(value)
    if best_container is None or best_len == 0:
        return False
    best_container[best_key].pop()
    payload["omitted"] = payload.get("omitted", 0) + 1
    return True


def _halve_longest_text(payload: dict) -> bool:
    """Halve the longest captured-text string in the payload, keeping it fenced
    if it was fenced, and return whether the value ACTUALLY GOT SHORTER.

    That return value is load-bearing, not bookkeeping. `_shrink_once` is an
    or-chain: a step claiming progress it did not make stops the chain there, so
    `_drop_all_text` is never reached and `fit`'s loop spins on an unchanged
    payload — forever, on the process that owns the client's stdio session. The
    old floor test compared the FENCED length (53 characters of markers) against
    a budget it then applied to the INNER text, so a short fenced value was
    rewritten to itself and reported as progress every single time.

    Halving rather than trimming keeps the step count logarithmic; comparing
    lengths afterwards is what makes "no smaller than this" observable instead
    of guessed at."""
    best_container = best_key = None
    best_len = 0
    for container, key, value in _walk(payload):
        if key in _TEXT_KEYS and isinstance(value, str) and len(value) > best_len:
            best_container, best_key, best_len = container, key, len(value)
    if best_container is None:
        return False
    current = best_container[best_key]
    fenced = current.startswith(FENCE_OPEN) and current.endswith(FENCE_CLOSE)
    inner = (current[len(FENCE_OPEN):-len(FENCE_CLOSE)] if fenced else current)
    halved = inner[:len(inner) // 2] + "…"
    rewritten = FENCE_OPEN + halved + FENCE_CLOSE if fenced else halved
    if len(rewritten) >= len(current):
        return False  # already at its floor — let the next shrink step run
    best_container[best_key] = rewritten
    return True


def _drop_all_text(payload: dict) -> bool:
    """Last resort before giving up: remove every captured-text key, keeping the
    structure and every number. Returns False once there is none left."""
    dropped = False
    for container, key, value in _walk(payload):
        if key in _TEXT_KEYS and isinstance(value, str):
            del container[key]
            dropped = True
    return dropped

"""Configure once, then everything just works.

Two ways to point ctxdiff at a database instead of a local file:

    import ctxdiff
    from ctxdiff import PostgresStore
    ctxdiff.configure(store=PostgresStore(dsn="postgresql://user@host/db"))

...or, with no code change at all, the `CTXDIFF_STORE` environment variable:

    CTXDIFF_STORE=postgresql://user@host/db   # Postgres
    CTXDIFF_STORE=mysql://user@host/db        # MySQL
    CTXDIFF_STORE=sqlite:///abs/path.ctrace   # SQLite, explicit
    CTXDIFF_STORE=~/traces                    # a directory of .ctrace files

Every subsequent `trace.init(project)` uses whatever is configured. Resolution
order is explicit-beats-ambient: an explicit `trace.init(store=...)` argument,
then `configure()`, then `CTXDIFF_STORE`, then — when NOTHING is configured —
`None`, which callers read as "the zero-config default": a local
`./<project>.ctrace`, byte-identical to ctxdiff's behavior before backends
existed. Local-first is the default and stays the default.

The env var is read at resolution time (not import time) so a test or a
subprocess can set it after ctxdiff is already imported."""
from __future__ import annotations

import os
from urllib.parse import unquote, urlparse

from ctxdiff.store.base import StoreBackend

ENV_VAR = "CTXDIFF_STORE"

# Every backend name a scheme can carry. A value with NO `://` that is EXACTLY
# one of these is a typo for the URL form, never a filename — see `from_dsn`,
# where treating `CTXDIFF_STORE=postgres` as a path would silently write a local
# SQLite file called "postgres" while the user believed they had a database.
_BACKEND_NAMES = frozenset({"postgres", "postgresql", "mysql", "mariadb",
                            "sqlite", "sqlite3"})

# The process-wide default set by `configure()`. None means "not configured",
# which is deliberately distinct from `SQLiteStore()`: it lets `resolve()` fall
# through to the env var before defaulting, and lets `configure(store=None)`
# reset ctxdiff back to the zero-config local path.
_configured: StoreBackend | None = None


def configure(store: StoreBackend | None = None) -> None:
    """Set (or clear) the process-wide default backend. `configure(store=X)`
    makes every later `trace.init(project)` open its session in X;
    `configure(store=None)` clears it, restoring the zero-config local
    `.ctrace` default. Deliberately a plain module-level global rather than
    thread/context state: it is a once-at-startup deployment choice, and a
    backend is immutable and connection-less so sharing one across threads is
    safe (each `open_session()` makes its own connection)."""
    global _configured
    _configured = store


def configured() -> StoreBackend | None:
    """Return whatever `configure()` last set (None if never set / cleared) —
    without consulting the environment. Used by tests and by `resolve()`."""
    return _configured


def resolve(store: StoreBackend | None = None) -> StoreBackend | None:
    """Resolve the backend to use, explicit-beats-ambient: an explicitly passed
    `store`, else the `configure()`d default, else `CTXDIFF_STORE` parsed via
    `from_dsn`, else None ("nothing configured — use the zero-config local
    default"). May raise ValueError for an unparseable `CTXDIFF_STORE`; callers
    on the capture path resolve inside their fail-open guard so a typo'd env var
    degrades capture with a warning instead of breaking the host."""
    if store is not None:
        return store
    if _configured is not None:
        return _configured
    dsn = os.environ.get(ENV_VAR, "").strip()
    if dsn:
        return from_dsn(dsn)
    return None


def from_dsn(dsn: str) -> StoreBackend:
    """Build a backend from a `CTXDIFF_STORE`-style string.

    How: the URL scheme picks the adapter — `postgres`/`postgresql` (plus
    `+driver` suffixes like `postgresql+psycopg`, which are tolerated and
    ignored so a SQLAlchemy-shaped URL pasted from an existing app just works),
    `mysql`, and `sqlite`/`sqlite3`. A string with NO recognizable scheme is
    treated as a filesystem path (that is what a bare `~/traces` or
    `./my.ctrace` obviously means) — UNLESS it is exactly a backend NAME
    (`postgres`, `mysql`, `sqlite`, ...), which is a typo for the URL form and
    is refused: writing a local SQLite file literally named "postgres" for a
    user who asked for Postgres is the one lie this module exists to prevent.
    A Windows drive letter (`C:\\...`) is explicitly not mistaken for a scheme.
    An unknown scheme raises ValueError naming what IS supported, rather than
    silently falling back to a local file the user never asked for.

    The adapter modules are imported HERE, lazily, so `ctxdiff.store.config`
    stays importable with no driver installed; the drivers themselves are
    imported later still (only on connect) — see `postgres.py`/`mysql.py`."""
    scheme = dsn.split("://", 1)[0].lower() if "://" in dsn else ""
    base = scheme.split("+", 1)[0]  # tolerate SQLAlchemy-style 'mysql+pymysql'

    if base in ("postgres", "postgresql"):
        from ctxdiff.store.postgres import PostgresStore
        # psycopg understands both spellings natively, but a '+driver' suffix
        # is SQLAlchemy-only syntax that libpq rejects — normalize it away.
        return PostgresStore(dsn=_strip_driver_suffix(dsn, base))
    if base in ("mysql", "mariadb"):
        from ctxdiff.store.mysql import MySQLStore
        return MySQLStore(dsn=_strip_driver_suffix(dsn, base))
    if base in ("sqlite", "sqlite3"):
        from ctxdiff.store.sqlite import SQLiteStore
        return SQLiteStore(path=_sqlite_path(dsn))
    if base:
        raise ValueError(
            f"ctxdiff: unsupported {ENV_VAR} scheme '{base}://' — expected "
            "postgresql://, mysql://, sqlite:// or a filesystem path")
    # No scheme, but the whole value IS a backend name: the user meant the URL
    # form and dropped the `://` (or their shell ate it). Refuse loudly — the
    # alternative is a local `./postgres` SQLite file, silently, for a user who
    # asked for a database and would be told `tracer.path` is "postgres".
    if dsn.strip().lower() in _BACKEND_NAMES:
        raise ValueError(
            f"ctxdiff: {ENV_VAR}='{dsn}' looks like a backend name, not a "
            f"location — it is missing '://'. Use e.g. "
            f"{dsn.strip().lower()}://user:password@host:port/database "
            "(or pass a filesystem path for a local .ctrace)")
    # A plain path. `~` is expanded here so the common `CTXDIFF_STORE=~/traces`
    # (which a shell may or may not expand) works.
    from ctxdiff.store.sqlite import SQLiteStore
    return SQLiteStore(path=os.path.expanduser(dsn))


def _strip_driver_suffix(dsn: str, base: str) -> str:
    """Rewrite `postgresql+psycopg://...` to `postgresql://...` (and the MySQL
    equivalent), leaving a plain DSN untouched. Why: users paste DSNs out of
    existing SQLAlchemy config, where the `+driver` part names the Python
    driver — meaningless (and rejected) at the libpq/PyMySQL level."""
    head, sep, rest = dsn.partition("://")
    if "+" in head:
        return f"{base}{sep}{rest}"
    return dsn


def _sqlite_path(dsn: str) -> str:
    """Turn a `sqlite://` URL into a filesystem path, accepting every spelling
    people actually write:

    - `sqlite:///var/traces/a.ctrace` — three slashes, the plain URL reading of
      an absolute path;
    - `sqlite:////var/traces/a.ctrace` — four slashes, which is how SQLAlchemy
      spells the SAME absolute path (it treats the first `/` as a separator).
      Both land on `/var/traces/a.ctrace`, because a leading run of slashes is
      collapsed — `//var` is never a different place from `/var`;
    - `sqlite://rel/dir` — two slashes, where the first segment parses as a
      netloc and is rejoined onto the path;
    - `sqlite://~/traces` — `~` is expanded.

    Percent-escapes are decoded, so a path with a space or `#` survives."""
    parsed = urlparse(dsn)
    # netloc is non-empty only for the two-slash relative spelling; rejoin it
    # with the path so 'sqlite://rel/dir' -> 'rel/dir'.
    raw = (parsed.netloc + parsed.path) if parsed.netloc else parsed.path
    if raw.startswith("//"):
        raw = "/" + raw.lstrip("/")
    return os.path.expanduser(unquote(raw))

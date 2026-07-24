"""Configure once, then everything just works — and when nothing is configured,
nothing changes.

These tests pin the resolution rules `trace.init()` and the CLI both follow:
an explicit argument, then `configure()`, then `CTXDIFF_STORE`, then the
zero-config local `./<project>.ctrace` default. The last one is the guardrail:
ctxdiff is local-first, and a user who configures nothing must keep getting
byte-identical behavior to before backends existed."""
from __future__ import annotations

import logging
import os

import pytest

import ctxdiff
from ctxdiff import trace
from ctxdiff.store import config as store_config
from ctxdiff.store.mysql import MySQLStore
from ctxdiff.store.postgres import PostgresStore
from ctxdiff.store.sql import SQLStore
from ctxdiff.store.sqlite import SQLiteStore
from ctxdiff.cli.main import main as cli_main

from tests import fakedb

PG_DSN = "postgresql://u:p@localhost:5432/ctxdiff"


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


class _FakeCompletions:
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()


class _FakeOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeChat()


@pytest.fixture(autouse=True)
def _clean_configuration(monkeypatch):
    """Every test starts from "nothing configured" and leaves nothing behind:
    the process-wide default is a module global, so a leaked `configure()` would
    silently redirect unrelated tests' traces into a stub database."""
    monkeypatch.delenv(store_config.ENV_VAR, raising=False)
    store_config.configure(None)
    yield
    store_config.configure(None)


def _call(wrapped, text="hi"):
    """Drive one recorded completion through a wrapped fake client."""
    return wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": text}])


# --- the zero-config default --------------------------------------------------


def test_nothing_configured_writes_the_local_project_ctrace(tmp_path, monkeypatch):
    """With no `configure()` and no env var, `trace.init(project)` still writes
    `./<project>.ctrace` — the local-first default is untouched."""
    monkeypatch.chdir(tmp_path)
    t = trace.init("myproj")
    assert os.path.basename(t.path) == "myproj.ctrace"
    t.wrap(_FakeOpenAI())
    t.close()
    assert (tmp_path / "myproj.ctrace").exists()


def test_configure_none_restores_the_local_default(tmp_path, monkeypatch):
    """`configure(store=None)` clears a previously configured database, so
    later runs go back to a local file."""
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))
    ctxdiff.configure(store=None)
    t = trace.init("back-to-local")
    t.wrap(_FakeOpenAI())
    t.close()
    assert (tmp_path / "back-to-local.ctrace").exists()


# --- configure() --------------------------------------------------------------


def test_configure_applies_to_every_later_init(tmp_path, monkeypatch):
    """Configure ONCE at startup: every later `trace.init(project)` opens its
    session in that store, with no per-call arguments and nothing written
    locally."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    for project in ("alpha", "beta"):
        t = trace.init(project)
        _call(t.wrap(_FakeOpenAI()), project)
        t.close()

    assert list(tmp_path.glob("*.ctrace")) == []      # nothing landed on disk
    reader = PostgresStore(dsn=PG_DSN).open_reader()
    try:
        assert [s.project for s in reader.list_sessions()] == ["alpha", "beta"]
        assert reader.get_run().project == "beta"     # newest session
    finally:
        reader.close()


def test_configured_store_records_calls_it_can_read_back(tmp_path, monkeypatch):
    """The configured store is LIVE: a wrapped client's calls land in it and
    read back with their blocks, models and attribution intact."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    t = trace.init("live")
    wrapped = t.wrap(_FakeOpenAI(), agent="planner")
    _call(wrapped, "first question")
    _call(wrapped, "second question")
    t.close()

    reader = PostgresStore(dsn=PG_DSN).open_reader()
    try:
        assert isinstance(reader, SQLStore)
        calls = reader.get_calls()
        assert [c.seq for c in calls] == [1, 2]
        assert [c.agent for c in calls] == ["planner", "planner"]
        assert reader.get_run().models == ["gpt-4o"]
        texts = [cb.block.text for cb in reader.get_call_blocks(calls[0].id)]
        assert "first question" in texts
    finally:
        reader.close()


def test_explicit_store_argument_beats_configure(tmp_path, monkeypatch):
    """`trace.init(store=...)` overrides the process-wide default — explicit
    beats ambient."""
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))
    t = trace.init("explicit", store=SQLiteStore(path=str(tmp_path / "x.ctrace")))
    t.wrap(_FakeOpenAI())
    t.close()
    assert (tmp_path / "x.ctrace").exists()


def test_explicit_path_argument_beats_a_configured_database(tmp_path, monkeypatch):
    """A `path=` argument names a FILE, so it wins over a configured database.
    A caller who passes a path and gets a row in someone's Postgres would
    rightly call that a bug."""
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))
    target = str(tmp_path / "explicit.ctrace")
    t = trace.init("proj", path=target)
    assert t.path == target
    t.wrap(_FakeOpenAI())
    t.close()
    assert os.path.exists(target)


# --- CTXDIFF_STORE ------------------------------------------------------------


@pytest.mark.parametrize("dsn,expected", [
    ("postgresql://u@h/db", PostgresStore),
    ("postgres://u@h/db", PostgresStore),
    ("postgresql+psycopg://u@h/db", PostgresStore),
    ("mysql://u@h/db", MySQLStore),
    ("mysql+pymysql://u@h/db", MySQLStore),
    ("mariadb://u@h/db", MySQLStore),
    ("sqlite:///tmp/x.ctrace", SQLiteStore),
    ("./local.ctrace", SQLiteStore),
    ("/var/traces", SQLiteStore),
])
def test_env_var_scheme_selects_the_backend(dsn, expected, monkeypatch):
    """Every spelling a user might reasonably put in `CTXDIFF_STORE` resolves
    to the right backend — including SQLAlchemy-style `+driver` suffixes pasted
    from an existing app's config, and a bare filesystem path."""
    monkeypatch.setenv(store_config.ENV_VAR, dsn)
    assert isinstance(store_config.resolve(), expected)


def test_env_var_strips_the_sqlalchemy_driver_suffix(monkeypatch):
    """The `+driver` part is SQLAlchemy syntax that libpq/PyMySQL reject, so it
    is normalized away rather than passed to the driver."""
    monkeypatch.setenv(store_config.ENV_VAR, "postgresql+psycopg://u@h/db")
    assert store_config.resolve().dsn == "postgresql://u@h/db"


def test_env_var_sqlite_url_forms_resolve_to_paths(monkeypatch):
    """`sqlite:///abs`, SQLAlchemy's four-slash `sqlite:////abs`, `sqlite://rel`
    and `~` all become real paths."""
    monkeypatch.setenv(store_config.ENV_VAR, "sqlite:///var/traces/a.ctrace")
    assert store_config.resolve().path == "/var/traces/a.ctrace"
    monkeypatch.setenv(store_config.ENV_VAR, "sqlite:////var/traces/a.ctrace")
    assert store_config.resolve().path == "/var/traces/a.ctrace"
    monkeypatch.setenv(store_config.ENV_VAR, "sqlite://rel/dir")
    assert store_config.resolve().path == "rel/dir"
    monkeypatch.setenv(store_config.ENV_VAR, "~/traces")
    assert store_config.resolve().path == os.path.expanduser("~/traces")


def test_env_var_directory_keeps_one_file_per_project(tmp_path, monkeypatch):
    """`CTXDIFF_STORE=<a directory>` keeps every project's DB in one place
    while preserving the one-file-per-project model."""
    traces = tmp_path / "traces"
    traces.mkdir()
    monkeypatch.setenv(store_config.ENV_VAR, str(traces))
    t = trace.init("myproj")
    assert t.path == str(traces / "myproj.ctrace")
    t.wrap(_FakeOpenAI())
    t.close()
    assert (traces / "myproj.ctrace").exists()


def test_env_var_file_is_used_for_the_session(tmp_path, monkeypatch):
    """`CTXDIFF_STORE=<a file>` sends every project's sessions to that one
    file."""
    target = tmp_path / "everything.ctrace"
    monkeypatch.setenv(store_config.ENV_VAR, str(target))
    t = trace.init("proj")
    assert t.path == str(target)
    t.wrap(_FakeOpenAI())
    t.close()
    assert target.exists()


def test_env_var_routes_a_run_into_the_database(tmp_path, monkeypatch):
    """The env-var shortcut is a real path to capture: no code change, and the
    run lands in the database rather than on disk."""
    fakedb.install(monkeypatch, "pymysql", str(tmp_path / "my.sqlite"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(store_config.ENV_VAR, "mysql://u:p@localhost/ctxdiff")

    t = trace.init("env-run")
    _call(t.wrap(_FakeOpenAI()))
    t.close()

    assert list(tmp_path.glob("*.ctrace")) == []
    reader = MySQLStore(dsn="mysql://u:p@localhost/ctxdiff").open_reader()
    try:
        assert reader.get_run().project == "env-run"
        assert len(reader.get_calls()) == 1
    finally:
        reader.close()


def test_configure_beats_the_env_var(tmp_path, monkeypatch):
    """Code beats environment: `configure()` wins over `CTXDIFF_STORE`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(store_config.ENV_VAR, "postgresql://u@h/db")
    ctxdiff.configure(store=SQLiteStore(path=str(tmp_path / "chosen.ctrace")))
    t = trace.init("proj")
    t.wrap(_FakeOpenAI())
    t.close()
    assert (tmp_path / "chosen.ctrace").exists()


@pytest.mark.parametrize("name", ["postgres", "postgresql", "mysql", "mariadb",
                                  "sqlite", "sqlite3", "POSTGRES", "MySQL"])
def test_a_bare_backend_name_is_refused_not_written_to_a_local_file(name):
    """`CTXDIFF_STORE=postgres` (no `://`) is a TYPO, not a filename. Treating
    it as a path is the exact lie the design forbids: ctxdiff would silently
    write a local SQLite file literally named "postgres" while the user
    believed their traces were going to a database. A bare backend NAME is
    therefore rejected the same way an unknown scheme is."""
    with pytest.raises(ValueError, match="looks like a backend name"):
        store_config.from_dsn(name)


def test_a_bare_backend_name_degrades_capture_and_writes_no_file(
        tmp_path, monkeypatch, caplog):
    """...and end to end: `CTXDIFF_STORE=postgres` degrades capture with one
    warning (a tracing problem never breaks the host) and leaves NO file behind
    — least of all one named "postgres"."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(store_config.ENV_VAR, "postgres")
    caplog.set_level(logging.WARNING, logger="ctxdiff")

    t = trace.init("proj")                    # must not raise
    assert t.path is None                     # never reported as a local path
    client = _FakeOpenAI()
    wrapped = t.wrap(client)
    assert isinstance(_call(wrapped), _Resp)
    t.close()

    assert len(client.chat.completions.calls) == 1
    assert not (tmp_path / "postgres").exists()
    assert list(tmp_path.iterdir()) == []
    assert len([r for r in caplog.records if "capture degraded" in r.message]) == 1


def test_unknown_scheme_is_rejected_with_a_helpful_message(monkeypatch):
    """An unsupported scheme names what IS supported instead of silently
    falling back to a local file the user never asked for."""
    monkeypatch.setenv(store_config.ENV_VAR, "redis://localhost:6379/0")
    with pytest.raises(ValueError, match="postgresql://, mysql://, sqlite://"):
        store_config.resolve()


def test_a_bad_env_var_degrades_capture_instead_of_breaking_the_host(
        tmp_path, monkeypatch, caplog):
    """A typo'd `CTXDIFF_STORE` is still a TRACING problem, and tracing problems
    never take down the traced program: `init()` does not raise, the host's
    calls run untouched, one warning is logged, and no surprise file appears."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(store_config.ENV_VAR, "redis://localhost:6379/0")
    caplog.set_level(logging.WARNING, logger="ctxdiff")

    t = trace.init("proj")                    # must not raise
    client = _FakeOpenAI()
    wrapped = t.wrap(client)
    assert isinstance(_call(wrapped), _Resp)
    assert isinstance(_call(wrapped), _Resp)
    t.close()

    assert len(client.chat.completions.calls) == 2
    assert len([r for r in caplog.records if "capture degraded" in r.message]) == 1
    assert list(tmp_path.glob("*.ctrace")) == []


# --- the read side ------------------------------------------------------------


def test_cli_reads_from_the_configured_database(tmp_path, monkeypatch, capsys):
    """The analyzers/CLI read from the configured store too — `ctxdiff tokens`
    in a directory with no `.ctrace` at all analyzes the newest session in the
    database."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    t = trace.init("dashboards")
    _call(t.wrap(_FakeOpenAI()), "how many tokens is this")
    t.close()

    assert cli_main(["tokens"]) == 0
    assert "turn 1" in capsys.readouterr().out.lower()


def test_cli_export_from_the_configured_database(tmp_path, monkeypatch, capsys):
    """`ctxdiff export` works against a database too, defaulting the dashboard
    to `./<project>.html` since there is no trace filename to borrow one
    from."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    t = trace.init("dash")
    _call(t.wrap(_FakeOpenAI()))
    t.close()

    assert cli_main(["export"]) == 0
    out = capsys.readouterr().out.strip()
    assert os.path.basename(out) == "dash.html"
    assert "ctxdiff — dash" in open(out, encoding="utf-8").read()


def test_cli_reads_the_env_var_ctrace_file(tmp_path, monkeypatch, capsys):
    """`CTXDIFF_STORE=<a file>` is honoured on the READ side too: the CLI
    analyzes that file even when the working directory holds other traces."""
    target = str(tmp_path / "configured.ctrace")
    t = trace.init("configured-proj", path=target)
    _call(t.wrap(_FakeOpenAI()), "the configured one")
    t.close()

    decoy = tmp_path / "decoy"
    decoy.mkdir()
    other = trace.init("decoy-proj", path=str(decoy / "decoy.ctrace"))
    _call(other.wrap(_FakeOpenAI()), "not this one")
    other.close()

    monkeypatch.chdir(decoy)
    monkeypatch.setenv(store_config.ENV_VAR, target)
    out = str(tmp_path / "report.html")
    assert cli_main(["export", "--out", out]) == 0
    page = open(out, encoding="utf-8").read()
    assert "the configured one" in page          # the configured file's call
    assert "not this one" not in page            # not the cwd's trace


def test_cli_run_flag_still_beats_a_configured_database(tmp_path, monkeypatch,
                                                       capsys):
    """`--run PATH` names a file, so it wins over a configured database — the
    read-side mirror of `path=` on the write side."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    local = str(tmp_path / "local.ctrace")
    t = trace.init("local-proj", path=local)
    _call(t.wrap(_FakeOpenAI()))
    t.close()

    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))
    assert cli_main(["tokens", "--run", local]) == 0
    assert "turn 1" in capsys.readouterr().out.lower()


def test_cli_runs_lists_sessions_from_the_configured_store(tmp_path, monkeypatch,
                                                           capsys):
    """`ctxdiff runs` follows the configured backend like every other read
    command. Globbing `*.ctrace` in the cwd would print "no .ctrace files"
    while `diff`/`tokens`/`view` all read the database — the one command that
    lies about where a user's traces are."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    for project in ("alpha", "beta"):
        t = trace.init(project)
        _call(t.wrap(_FakeOpenAI()), f"{project} question")
        t.close()

    assert cli_main(["runs"]) == 0
    out = capsys.readouterr().out
    assert "project=alpha" in out
    assert "project=beta" in out
    assert "turns=1" in out
    assert "no .ctrace files" not in out


def test_cli_runs_on_an_empty_configured_store_says_so(tmp_path, monkeypatch,
                                                       capsys):
    """A configured store with nothing in it reports THAT, rather than the
    cwd-flavoured "no .ctrace files in the current directory" — the user asked
    for a database, so the answer must be about the database."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    assert cli_main(["runs"]) == 0
    assert "no sessions" in capsys.readouterr().out


def test_cli_error_message_is_not_double_prefixed(tmp_path, monkeypatch, capsys):
    """Errors ctxdiff raises already carry the `ctxdiff: ` prefix, so the CLI
    must not add a second one — "ctxdiff: ctxdiff: no sessions recorded" reads
    like a bug in the tool."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))

    assert cli_main(["tokens"]) == 1
    err = capsys.readouterr().err
    assert "no sessions recorded" in err
    assert "ctxdiff: ctxdiff:" not in err
    assert err.count("ctxdiff:") == 1


def test_cli_reports_a_dead_configured_database_instead_of_crashing(
        tmp_path, monkeypatch, capsys):
    """A read command against an unreachable database exits 1 with the error,
    never a traceback."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"),
                   fail_connect=OSError("connection refused"))
    monkeypatch.chdir(tmp_path)
    ctxdiff.configure(store=PostgresStore(dsn=PG_DSN))
    assert cli_main(["tokens"]) == 1
    assert "connection refused" in capsys.readouterr().err


def test_cli_reports_a_bad_env_var_instead_of_crashing(tmp_path, monkeypatch,
                                                       capsys):
    """An unparseable `CTXDIFF_STORE` is reported by the CLI as an error with
    the supported schemes — the read side's mirror of the capture side's
    fail-open degradation."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(store_config.ENV_VAR, "redis://localhost:6379/0")
    assert cli_main(["tokens"]) == 1
    assert "postgresql://, mysql://, sqlite://" in capsys.readouterr().err


def test_top_level_exports_are_the_configured_api():
    """The documented one-liner (`ctxdiff.configure(store=PostgresStore(...))`)
    resolves against the top-level package, not a deep module path."""
    assert ctxdiff.configure is store_config.configure
    assert ctxdiff.PostgresStore is PostgresStore
    assert ctxdiff.MySQLStore is MySQLStore
    assert ctxdiff.SQLiteStore is SQLiteStore

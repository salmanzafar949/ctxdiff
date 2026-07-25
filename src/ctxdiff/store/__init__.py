"""Storage backends. `Store`/`StoreBackend` are the protocols every backend
implements (see `base.py`); `CTrace`/`SQLiteStore` are the local-first default,
`PostgresStore`/`MySQLStore` point ctxdiff at a database you already run.

Nothing here imports a database driver at module level — `psycopg`/`pymysql`
are imported only when a connection is actually opened, so ctxdiff's core stays
dependency-light (tiktoken only) and a missing extra becomes a clear install
hint instead of an import crash."""
from ctxdiff.store.base import (
    Call,
    EmptyStoreError,
    Run,
    Session,
    Store,
    StoreBackend,
    parse_started_at,
)
from ctxdiff.store.config import ENV_VAR, configure, configured, from_dsn, resolve
from ctxdiff.store.ctrace import CTrace
from ctxdiff.store.mysql import MySQLStore
from ctxdiff.store.postgres import PostgresStore
from ctxdiff.store.sqlite import SQLiteStore

__all__ = [
    "Call", "Run", "Session", "Store", "StoreBackend", "EmptyStoreError",
    "parse_started_at",
    "CTrace", "SQLiteStore", "PostgresStore", "MySQLStore",
    "configure", "configured", "resolve", "from_dsn", "ENV_VAR",
]

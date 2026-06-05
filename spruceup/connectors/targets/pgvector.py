import asyncio
import typing

import psycopg
import psycopg_pool

from ..base import TargetConnector
from ...models import ChunkWrapper


_PY_TO_PG: dict[type, str] = {
    str:   "TEXT",
    int:   "INTEGER",
    float: "DOUBLE PRECISION",
    bytes: "BYTEA",
    bool:  "BOOLEAN",
}

_POOL_MAX_SIZE = 5


def _py_to_pg_type(tp, embedding_dimensions: int) -> str:
    origin = typing.get_origin(tp)
    if origin is list:
        args = typing.get_args(tp)
        if args == (float,):
            return f"vector({embedding_dimensions})"
        inner = _py_to_pg_type(args[0], embedding_dimensions) if args else "TEXT"
        return f"{inner}[]"
    return _PY_TO_PG.get(tp, "TEXT")


class PgVectorTarget(TargetConnector):
    def __init__(self, connstr: str, table: str, schema: type) -> None:
        self.connstr = connstr
        self.table = table
        self._schema = schema
        self._pool: psycopg_pool.AsyncConnectionPool | None = None
        self._pool_lock = asyncio.Lock()

    @property
    def display_name(self) -> str:
        return self.table

    @property
    def schema(self) -> type:
        return self._schema

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        hints = typing.get_type_hints(self._schema)
        user_col_defs = [
            f"{col} {_py_to_pg_type(tp, embedding_dimensions)}"
            for col, tp in hints.items()
        ]
        col_defs = ["id TEXT PRIMARY KEY"] + user_col_defs
        with psycopg.connect(self.connstr) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ({', '.join(col_defs)})"
            )

    async def _get_pool(self) -> psycopg_pool.AsyncConnectionPool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                pool = psycopg_pool.AsyncConnectionPool(
                    self.connstr,
                    min_size=1,
                    max_size=_POOL_MAX_SIZE,
                    open=False,
                )
                await pool.open()
                self._pool = pool
        return self._pool

    async def sync(self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            if upserts:
                hints = typing.get_type_hints(self._schema)
                col_names = list(hints.keys())
                all_cols = ["id"] + col_names
                placeholders = ", ".join(["%s"] * len(all_cols))
                sql = (
                    f"INSERT INTO {self.table} ({', '.join(all_cols)}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT (id) DO NOTHING"
                )
                rows = [
                    [f"{file_id}:{chunk.user_chunk_object_hash.hex()}"] + [getattr(chunk.user_chunk, col) for col in col_names]
                    for chunk in upserts
                ]
                async with conn.cursor() as cur:
                    await cur.executemany(sql, rows)

            if deletes:
                placeholders = ", ".join(["%s"] * len(deletes))
                await conn.execute(
                    f"DELETE FROM {self.table} WHERE id IN ({placeholders})",
                    [f"{file_id}:{h.hex()}" for h in deletes],
                )

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

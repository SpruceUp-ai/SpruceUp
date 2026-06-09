import asyncio
import typing
from typing import LiteralString, cast

import psycopg
import psycopg_pool
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import TupleRow

from ..base import TargetConnector
from ...models import ChunkWrapper
from ...utils.schema import schema_hints


_PY_TO_PG: dict[type, LiteralString] = {
    str:   "TEXT",
    int:   "INTEGER",
    float: "DOUBLE PRECISION",
    bytes: "BYTEA",
    bool:  "BOOLEAN",
}

_POOL_MAX_SIZE = 5

_Pool = psycopg_pool.AsyncConnectionPool[psycopg.AsyncConnection[TupleRow]]


def _py_to_pg_type(tp) -> LiteralString:
    origin = typing.get_origin(tp)
    if origin is list:
        args = typing.get_args(tp)
        inner = _py_to_pg_type(args[0]) if args else "TEXT"
        return f"{inner}[]"
    return _PY_TO_PG.get(tp, "TEXT")


class PgVectorTarget(TargetConnector):
    def __init__(self, connstr: str, table: str, schema: type, vector_column: str) -> None:
        super().__init__(schema, vector_column)
        self.connstr = connstr
        self.table = table
        self._pool: _Pool | None = None
        self._pool_lock = asyncio.Lock()

    @property
    def display_name(self) -> str:
        return self.table

    def identity(self) -> str:
        info = conninfo_to_dict(self.connstr)
        return (
            f"pgvector:{info.get('host')}:{info.get('port')}:"
            f"{info.get('dbname')}:{self.table}"
        )

    def ensure_table_exists(self, embedding_dimensions: int, recreate: bool = False) -> None:
        col_defs: list[sql.Composable] = [sql.SQL("id TEXT PRIMARY KEY")]
        for col, tp in schema_hints(self._schema).items():
            if col == self._vector_column:
                dim = cast(LiteralString, str(int(embedding_dimensions)))
                col_defs.append(sql.SQL("{} vector({})").format(sql.Identifier(col), sql.SQL(dim)))
            else:
                col_defs.append(sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(_py_to_pg_type(tp))))
        table = sql.Identifier(self.table)
        with psycopg.connect(self.connstr) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if recreate:
                conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
            conn.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                    table, sql.SQL(", ").join(col_defs)
                )
            )

    async def _get_pool(self) -> _Pool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                pool: _Pool = psycopg_pool.AsyncConnectionPool(
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
                insert_sql = sql.SQL(
                    "INSERT INTO {table} ({cols}) VALUES ({ph}) "
                    "ON CONFLICT (id) DO UPDATE SET {updates}"
                ).format(
                    table=sql.Identifier(self.table),
                    cols=sql.SQL(", ").join(sql.Identifier(c) for c in all_cols),
                    ph=sql.SQL(", ").join([sql.Placeholder()] * len(all_cols)),
                    updates=sql.SQL(", ").join(
                        sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in col_names
                    ),
                )
                rows = [
                    [f"{file_id}:{chunk.user_chunk_object_hash.hex()}"] + [getattr(chunk.user_chunk, col) for col in col_names]
                    for chunk in upserts
                ]
                async with conn.cursor() as cur:
                    await cur.executemany(insert_sql, rows)

            if deletes:
                delete_sql = sql.SQL("DELETE FROM {table} WHERE id IN ({ph})").format(
                    table=sql.Identifier(self.table),
                    ph=sql.SQL(", ").join([sql.Placeholder()] * len(deletes)),
                )
                await conn.execute(
                    delete_sql, [f"{file_id}:{h.hex()}" for h in deletes]
                )

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

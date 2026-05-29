import typing

import psycopg

from ..base import TargetConnector
from ...models import ChunkWrapper


_PY_TO_PG: dict[type, str] = {
    str:   "TEXT",
    int:   "INTEGER",
    float: "DOUBLE PRECISION",
    bytes: "BYTEA",
    bool:  "BOOLEAN",
}


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
    def __init__(self, connstr: str, table: str, schema: type, primary_key: str) -> None:
        self.connstr = connstr
        self.table = table
        self.schema = schema
        self.primary_key = primary_key

    @property
    def display_name(self) -> str:
        return self.table

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        hints = typing.get_type_hints(self.schema)
        col_defs = [
            f"{col} {_py_to_pg_type(tp, embedding_dimensions)}{' PRIMARY KEY' if col == self.primary_key else ''}"
            for col, tp in hints.items()
        ]
        with psycopg.connect(self.connstr) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ({', '.join(col_defs)})"
            )

    async def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        async with await psycopg.AsyncConnection.connect(self.connstr) as conn:
            if upserts:
                hints = typing.get_type_hints(self.schema)
                col_names = list(hints.keys())
                placeholders = ", ".join(["%s"] * len(col_names))
                update_set = ", ".join(
                    f"{col} = EXCLUDED.{col}" for col in col_names if col != self.primary_key
                )
                sql = (
                    f"INSERT INTO {self.table} ({', '.join(col_names)}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT ({self.primary_key}) DO UPDATE SET {update_set}"
                )
                rows = [
                    [getattr(chunk.user_chunk, col) for col in col_names]
                    for chunk in upserts
                ]
                async with conn.cursor() as cur:
                    await cur.executemany(sql, rows)

            if deletes:
                placeholders = ", ".join(["%s"] * len(deletes))
                await conn.execute(
                    f"DELETE FROM {self.table} WHERE {self.primary_key} IN ({placeholders})",
                    deletes,
                )

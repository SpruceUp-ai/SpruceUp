import typing
from dataclasses import dataclass

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


def _py_to_pg_type(tp) -> str:
    origin = typing.get_origin(tp)
    if origin is list:
        args = typing.get_args(tp)
        if args == (float,):
            return "vector(1536)"
        inner = _py_to_pg_type(args[0]) if args else "TEXT"
        return f"{inner}[]"
    return _PY_TO_PG.get(tp, "TEXT")


@dataclass
class PgVectorTarget(TargetConnector):
    connstr: str
    table: str
    schema: type
    primary_key: str

    @property
    def display_name(self) -> str:
        return self.table

    def ensure_table_exists(self) -> None:
        hints = typing.get_type_hints(self.schema)
        col_defs = [
            f"{col} {_py_to_pg_type(tp)}{' PRIMARY KEY' if col == self.primary_key else ''}"
            for col, tp in hints.items()
        ]
        with psycopg.connect(self.connstr) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ({', '.join(col_defs)})"
            )

    def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        with psycopg.connect(self.connstr) as conn:
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
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)

            if deletes:
                placeholders = ", ".join(["%s"] * len(deletes))
                conn.execute(
                    f"DELETE FROM {self.table} WHERE {self.primary_key} IN ({placeholders})",
                    deletes,
                )

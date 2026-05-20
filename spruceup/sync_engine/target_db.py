import typing

import psycopg

from ..models import ChunkWrapper, TargetTableConfig

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


def ensure_table_exists(conn: psycopg.Connection, config: TargetTableConfig) -> None:
    hints = typing.get_type_hints(config.schema_class)
    col_defs = []
    for col_name, col_type in hints.items():
        pg_type = _py_to_pg_type(col_type)
        suffix = " PRIMARY KEY" if col_name == config.primary_key else ""
        col_defs.append(f"{col_name} {pg_type}{suffix}")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {config.table_name} ({', '.join(col_defs)})"
    )


def upsert_chunks(conn: psycopg.Connection, chunks: list[ChunkWrapper], config: TargetTableConfig) -> None:
    if not chunks:
        return
    hints = typing.get_type_hints(config.schema_class)
    col_names = list(hints.keys())
    placeholders = ", ".join(["%s"] * len(col_names))
    update_set = ", ".join(
        f"{col} = EXCLUDED.{col}" for col in col_names if col != config.primary_key
    )
    sql = (
        f"INSERT INTO {config.table_name} ({', '.join(col_names)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({config.primary_key}) DO UPDATE SET {update_set}"
    )
    rows = [[getattr(chunk.user_chunk, col) for col in col_names] for chunk in chunks]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)


def delete_chunks(conn: psycopg.Connection, chunk_ids: list, config: TargetTableConfig) -> None:
    if not chunk_ids:
        return
    placeholders = ", ".join(["%s"] * len(chunk_ids))
    conn.execute(
        f"DELETE FROM {config.table_name} WHERE {config.primary_key} IN ({placeholders})",
        chunk_ids,
    )

"""Reset load-test state between runs.

Removes the load-test SQLite manifest and (optionally) drops the load-test
Postgres table. Only touches load-test artifacts — never the project's real
spruceup_manifest.db or production tables — so each run starts from a clean,
comparable state.

Example:
    python reset.py                          # drop the manifest only
    python reset.py --drop-pg --pg-table loadtest_chunks
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/tmp/loadtest_manifest.db")
    ap.add_argument("--drop-pg", action="store_true")
    ap.add_argument("--pg-table", default="loadtest_chunks")
    args = ap.parse_args()

    for suffix in ("", "-wal", "-shm", "-journal"):
        path = args.manifest + suffix
        if os.path.exists(path):
            os.remove(path)
            print(f"removed {path}")

    if args.drop_pg:
        import dotenv
        import psycopg

        dotenv.load_dotenv()
        if not args.pg_table.replace("_", "").isalnum():
            raise SystemExit(f"refusing unsafe table name: {args.pg_table!r}")
        with psycopg.connect(os.getenv("PG_CONNSTR")) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {args.pg_table}")
        print(f"dropped pg table {args.pg_table}")


if __name__ == "__main__":
    main()

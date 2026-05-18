import sqlite3


def init_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS files
            (id INTEGER PRIMARY KEY, inode INTEGER, file_path TEXT, hash_value BLOB);
        """
    )
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS transform_hashes
            (func_name TEXT PRIMARY KEY, source_hash BLOB);
        """
    )
    con.commit()
    con.close()

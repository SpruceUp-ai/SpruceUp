import sqlite3


def init_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS files
            (id BLOB PRIMARY KEY, file_content_hash BLOB, file_identifier INT, file_mtime FLOAT, file_extension TEXT);
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

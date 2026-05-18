import sqlite3


def init_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS data_sources (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id             BLOB PRIMARY KEY,
            file_path      TEXT NOT NULL,
            inode          INTEGER,
            transform_hash BLOB,
            content_hash   BLOB,
            mtime          REAL,
            data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
            file_type      TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id                    BLOB PRIMARY KEY,
            file_id               BLOB REFERENCES files(id) ON DELETE CASCADE,
            user_chunk_object_hash BLOB,
            user_chunk_object     BLOB
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS transform_hashes (
            func_name   TEXT PRIMARY KEY,
            source_hash BLOB
        )
        """
    )
    con.execute(
        "INSERT OR IGNORE INTO data_sources (id, source_type) VALUES (1, 'local')"
    )
    con.commit()
    con.close()

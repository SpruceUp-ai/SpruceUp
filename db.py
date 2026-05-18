import sqlite3


def init_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS data_sources (
                id serial,
                name varchar(50) NOT NULL
            );
        """
    )
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS files (
                id BLOB PRIMARY KEY,
                file_transform_hash BLOB,
                file_content_hash BLOB,
                file_identifier_inode INT,
                file_mtime FLOAT,
                file_extension TEXT,
                data_source_id INT REFERENCES data_sources(id) ON DELETE CASCADE
            );
        """
    )
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS chunks(
                id BLOB PRIMARY KEY,
                file_id BLOB REFERENCES files(id) ON DELETE CASCADE,
                chunk_transform_hash BLOB,
                user_chunk_object_hash BLOB,
                user_chunk_object BLOB
            );
        """
    )
    con.execute(
        """
            CREATE TABLE IF NOT EXISTS transform_hashes(
                func_name TEXT PRIMARY KEY,
                source_hash BLOB
            );
        """
    )
    con.execute(
        """
            INSERT INTO data_sources VALUES ('local');
        """
    )

    con.commit()
    con.close()

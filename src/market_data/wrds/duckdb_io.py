from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from market_data.wrds.schemas import SCHEMA_BY_TABLE, create_all_tables_sql, table_columns


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class DuckDBManager:
    """DuckDB helper for WRDS ingestion and canonical materialization."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self, *, read_only: bool = False) -> Iterator[Any]:
        import duckdb

        try:
            conn = duckdb.connect(str(self.db_path), read_only=read_only)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Conflicting lock" in message:
                mode = "read-only" if read_only else "read-write"
                raise RuntimeError(
                    f"DuckDB is locked by another writer and cannot be opened in {mode} mode: {self.db_path}"
                ) from exc
            raise
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create all WRDS source and canonical tables if missing."""

        with self.connect(read_only=False) as conn:
            for sql in create_all_tables_sql():
                conn.execute(sql)
            for table_name in SCHEMA_BY_TABLE:
                self._migrate_missing_columns(conn, table_name)
            self._migrate_table_primary_key(conn, "wrds_compustat_quarterly")
            self._migrate_table_primary_key(conn, "financials_quarterly_canonical")

    def overwrite_with_query(self, table_name: str, select_sql: str) -> int:
        """Replace a table's contents with the rows returned by a SELECT."""

        with self.connect(read_only=False) as conn:
            conn.execute(f'DELETE FROM {_quote(table_name)}')
            columns = ", ".join(_quote(column) for column in table_columns(table_name))
            conn.execute(
                f"""
                INSERT INTO {_quote(table_name)} ({columns})
                SELECT {columns}
                FROM ({select_sql}) AS src
                """
            )
            return int(conn.execute(f"SELECT COUNT(*) FROM {_quote(table_name)}").fetchone()[0])

    def execute(self, sql: str, params: Any | None = None) -> None:
        """Execute a SQL statement without returning rows."""

        with self.connect(read_only=False) as conn:
            if params:
                conn.execute(sql, params)
            else:
                conn.execute(sql)

    @staticmethod
    def _split_table_name(table_name: str) -> tuple[str, str]:
        raw = str(table_name).strip()
        if "." not in raw:
            return "main", raw
        schema_name, relation_name = raw.split(".", 1)
        return schema_name.strip('"'), relation_name.strip('"')

    @staticmethod
    def _attach_databases(conn: Any, attachments: dict[str, Path] | None) -> None:
        if not attachments:
            return
        for alias, path in attachments.items():
            alias_name = str(alias).strip()
            db_path = str(Path(path).expanduser()).replace("'", "''")
            conn.execute(f"ATTACH '{db_path}' AS {_quote(alias_name)}")

    def fetch_df(
        self,
        sql: str,
        params: Any | None = None,
        *,
        attachments: dict[str, Path] | None = None,
    ) -> pd.DataFrame:
        """Execute a SELECT and return a pandas DataFrame."""

        with self.connect(read_only=True) as conn:
            self._attach_databases(conn, attachments)
            if params:
                return conn.execute(sql, params).df()
            return conn.execute(sql).df()

    def fetch_one(
        self,
        sql: str,
        params: Any | None = None,
        *,
        attachments: dict[str, Path] | None = None,
    ) -> tuple[Any, ...] | None:
        """Execute a SELECT and return a single row."""

        with self.connect(read_only=True) as conn:
            self._attach_databases(conn, attachments)
            if params:
                return conn.execute(sql, params).fetchone()
            return conn.execute(sql).fetchone()

    def merge_dataframe(self, table_name: str, frame: pd.DataFrame, key_columns: tuple[str, ...]) -> int:
        """Upsert a pandas DataFrame into DuckDB using MERGE."""

        prepared = self._prepare_frame(table_name, frame)
        if prepared.empty:
            return 0

        target_columns = list(table_columns(table_name))
        view_name = f"stage_{table_name}"
        on_clause = " AND ".join(f'tgt.{_quote(key)} = src.{_quote(key)}' for key in key_columns)
        update_columns = [column for column in target_columns if column not in key_columns]
        update_clause = ", ".join(f'{_quote(column)} = src.{_quote(column)}' for column in update_columns)
        insert_columns = ", ".join(_quote(column) for column in target_columns)
        insert_values = ", ".join(f"src.{_quote(column)}" for column in target_columns)
        merge_sql = f"""
            MERGE INTO {_quote(table_name)} AS tgt
            USING {view_name} AS src
            ON {on_clause}
            WHEN MATCHED THEN UPDATE SET {update_clause}
            WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})
        """

        with self.connect(read_only=False) as conn:
            conn.register(view_name, prepared)
            try:
                conn.execute(merge_sql)
            finally:
                conn.unregister(view_name)
        return int(len(prepared))

    def replace_dataframe(self, table_name: str, frame: pd.DataFrame) -> int:
        """Replace a table completely with the provided DataFrame."""

        prepared = self._prepare_frame(table_name, frame)
        with self.connect(read_only=False) as conn:
            conn.execute(f'DELETE FROM {_quote(table_name)}')
            if prepared.empty:
                return 0
            view_name = f"stage_{table_name}"
            conn.register(view_name, prepared)
            try:
                columns = ", ".join(_quote(column) for column in table_columns(table_name))
                conn.execute(f'INSERT INTO {_quote(table_name)} ({columns}) SELECT {columns} FROM {view_name}')
            finally:
                conn.unregister(view_name)
            return int(len(prepared))

    def checkpoint_latest_end(self, dataset_name: str) -> date | None:
        """Return the latest completed chunk_end for a dataset."""

        row = self.fetch_one(
            """
            SELECT MAX(chunk_end)
            FROM wrds_ingest_checkpoints
            WHERE dataset_name = ? AND status = 'completed'
            """,
            [dataset_name],
        )
        if row is None:
            return None
        value = row[0]
        if isinstance(value, datetime):
            return value.date()
        return value

    def checkpoint_completed_keys(self, dataset_name: str) -> set[str]:
        """Return all completed chunk keys for a dataset."""

        df = self.fetch_df(
            """
            SELECT chunk_key
            FROM wrds_ingest_checkpoints
            WHERE dataset_name = ? AND status = 'completed'
            """,
            [dataset_name],
        )
        return set(df["chunk_key"].astype(str).tolist()) if not df.empty else set()

    def summarize_table(
        self,
        table_name: str,
        *,
        date_column: str | None = None,
        key_columns: tuple[str, ...] = (),
        distinct_columns: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Return a compact quality summary for a table."""

        result: dict[str, Any] = {"table": table_name}
        count_row = self.fetch_one(f"SELECT COUNT(*) FROM {_quote(table_name)}")
        result["row_count"] = int(count_row[0]) if count_row else 0
        if result["row_count"] == 0:
            return result

        if date_column:
            date_sql = f"SELECT MIN({_quote(date_column)}), MAX({_quote(date_column)}) FROM {_quote(table_name)}"
            min_date, max_date = self.fetch_one(date_sql)
            result["min_date"] = min_date.isoformat() if hasattr(min_date, "isoformat") else None
            result["max_date"] = max_date.isoformat() if hasattr(max_date, "isoformat") else None

        null_ratios: dict[str, float] = {}
        with self.connect(read_only=True) as conn:
            for column in key_columns:
                value = conn.execute(
                    f"""
                    SELECT SUM(CASE WHEN {_quote(column)} IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
                    FROM {_quote(table_name)}
                    """
                ).fetchone()[0]
                null_ratios[column] = float(value or 0.0)
            result["null_ratios"] = null_ratios

            distincts: dict[str, int] = {}
            for column in distinct_columns:
                value = conn.execute(f"SELECT COUNT(DISTINCT {_quote(column)}) FROM {_quote(table_name)}").fetchone()[0]
                distincts[column] = int(value or 0)
            result["distinct_counts"] = distincts
        return result

    def table_exists(self, table_name: str, *, attachments: dict[str, Path] | None = None) -> bool:
        """Return True when a table exists in the main DuckDB schema."""

        schema_name, relation_name = self._split_table_name(table_name)
        row = self.fetch_one(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
              AND (table_schema = ? OR table_catalog = ?)
            """,
            [relation_name, schema_name, schema_name],
            attachments=attachments,
        )
        return bool(row and row[0])

    def schema_info(self, table_name: str, *, attachments: dict[str, Path] | None = None) -> list[dict[str, Any]]:
        """Return schema metadata for a DuckDB table."""

        schema_name, relation_name = self._split_table_name(table_name)
        df = self.fetch_df(
            """
            SELECT
                column_name AS name,
                data_type AS type,
                CASE WHEN is_nullable = 'NO' THEN 1 ELSE 0 END AS notnull,
                0 AS pk
            FROM information_schema.columns
            WHERE table_name = ?
              AND (table_schema = ? OR table_catalog = ?)
            ORDER BY ordinal_position
            """,
            [relation_name, schema_name, schema_name],
            attachments=attachments,
        )
        if df.empty:
            return []
        rows: list[dict[str, Any]] = []
        for record in df.to_dict(orient="records"):
            rows.append(
                {
                    "column_name": record.get("name"),
                    "column_type": record.get("type"),
                    "not_null": bool(record.get("notnull")),
                    "primary_key": bool(record.get("pk")),
                }
            )
        return rows

    def duplicate_key_rows(self, table_name: str, key_columns: tuple[str, ...]) -> int:
        """Return the number of rows that violate the logical primary key."""

        if not key_columns:
            return 0
        group_sql = ", ".join(_quote(column) for column in key_columns)
        row = self.fetch_one(
            f"""
            SELECT COALESCE(SUM(dup_rows), 0)
            FROM (
                SELECT COUNT(*) - 1 AS dup_rows
                FROM {_quote(table_name)}
                GROUP BY {group_sql}
                HAVING COUNT(*) > 1
            ) dupes
            """
        )
        return int(row[0] or 0) if row else 0

    def inspect_rows(self, table_name: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return a small sample of table rows."""

        df = self.fetch_df(f"SELECT * FROM {_quote(table_name)} LIMIT {int(limit)}")
        return df.to_dict(orient="records")

    def table_profile(
        self,
        table_name: str,
        *,
        date_column: str | None = None,
        key_columns: tuple[str, ...] = (),
        distinct_columns: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Return a richer validation profile for a table."""

        profile = self.summarize_table(
            table_name,
            date_column=date_column,
            key_columns=key_columns,
            distinct_columns=distinct_columns,
        )
        profile["duplicate_primary_key_rows"] = self.duplicate_key_rows(table_name, key_columns)
        profile["schema"] = self.schema_info(table_name)
        return profile

    def upsert_run(self, payload: dict[str, Any]) -> None:
        """Write or update a wrds_ingest_runs row."""

        frame = pd.DataFrame([payload])
        self.merge_dataframe("wrds_ingest_runs", frame, ("run_id",))

    def upsert_checkpoint(self, payload: dict[str, Any]) -> None:
        """Write or update a wrds_ingest_checkpoints row."""

        frame = pd.DataFrame([payload])
        self.merge_dataframe("wrds_ingest_checkpoints", frame, ("dataset_name", "chunk_key"))

    def _prepare_frame(self, table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Align a DataFrame to an explicit table schema."""

        if table_name not in SCHEMA_BY_TABLE:
            raise KeyError(f"Unknown WRDS table: {table_name}")

        prepared = frame.copy()
        target_columns = list(table_columns(table_name))
        for column in target_columns:
            if column not in prepared.columns:
                prepared[column] = None

        prepared = prepared[target_columns]
        table_def = SCHEMA_BY_TABLE[table_name]
        for column in table_def.columns:
            if column.duckdb_type == "DATE":
                prepared[column.name] = pd.to_datetime(prepared[column.name], errors="coerce").dt.date
            elif column.duckdb_type == "TIMESTAMPTZ":
                prepared[column.name] = pd.to_datetime(prepared[column.name], errors="coerce", utc=True)
            elif column.duckdb_type in {"DOUBLE"}:
                prepared[column.name] = pd.to_numeric(prepared[column.name], errors="coerce")
            elif column.duckdb_type in {"INTEGER", "BIGINT"}:
                prepared[column.name] = pd.to_numeric(prepared[column.name], errors="coerce")
            elif column.duckdb_type == "BOOLEAN":
                prepared[column.name] = prepared[column.name].astype("boolean")
            else:
                prepared[column.name] = prepared[column.name].map(self._stringify)
        return prepared

    @staticmethod
    def _stringify(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    def _migrate_table_primary_key(self, conn: Any, table_name: str) -> None:
        expected = SCHEMA_BY_TABLE[table_name]
        info = conn.execute(f"PRAGMA table_info({_quote(table_name)})").fetchall()
        actual_pk = tuple(row[1] for row in info if int(row[5] or 0) > 0)
        if actual_pk == expected.primary_key:
            return

        temp_name = f"{table_name}__migrated"
        conn.execute(f'DROP TABLE IF EXISTS {_quote(temp_name)}')
        conn.execute(self._create_table_sql_for_name(temp_name, expected))
        columns = ", ".join(_quote(column.name) for column in expected.columns)
        conn.execute(f'INSERT INTO {_quote(temp_name)} ({columns}) SELECT {columns} FROM {_quote(table_name)}')
        conn.execute(f'DROP TABLE {_quote(table_name)}')
        conn.execute(f'ALTER TABLE {_quote(temp_name)} RENAME TO {_quote(table_name)}')

    def _migrate_missing_columns(self, conn: Any, table_name: str) -> None:
        expected = SCHEMA_BY_TABLE[table_name]
        info = conn.execute(f"PRAGMA table_info({_quote(table_name)})").fetchall()
        actual_columns = {str(row[1]) for row in info}
        for column in expected.columns:
            if column.name in actual_columns:
                continue
            conn.execute(f'ALTER TABLE {_quote(table_name)} ADD COLUMN {_quote(column.name)} {column.duckdb_type}')

    @staticmethod
    def _create_table_sql_for_name(table_name: str, table_def: Any) -> str:
        parts: list[str] = []
        for column in table_def.columns:
            segment = f'{_quote(column.name)} {column.duckdb_type}'
            if column.not_null:
                segment += " NOT NULL"
            parts.append(segment)
        if table_def.primary_key:
            joined = ", ".join(_quote(name) for name in table_def.primary_key)
            parts.append(f"PRIMARY KEY ({joined})")
        columns_sql = ",\n                ".join(parts)
        return f"""
            CREATE TABLE {_quote(table_name)} (
                {columns_sql}
            )
        """

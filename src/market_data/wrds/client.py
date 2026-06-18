from __future__ import annotations

import getpass
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.wrds.config import WRDSSettings


@dataclass(frozen=True)
class WRDSQueryResult:
    """A typed wrapper for fetched WRDS chunk data."""

    rows: pd.DataFrame
    attempts: int


@dataclass(frozen=True)
class WRDSExecutionError(RuntimeError):
    """Safe wrapper for WRDS SQL execution failures."""

    attempts: int
    exception_class: str
    exception_message: str

    def __str__(self) -> str:
        return f"{self.exception_class}: {self.exception_message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "exception_class": self.exception_class,
            "exception_message": self.exception_message,
        }


@dataclass(frozen=True)
class WRDSCredentialStatus:
    """Safe credential availability summary without leaking secret values."""

    username_available: bool
    password_available: bool
    pgpass_available: bool
    interactive: bool
    username_source: str
    password_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "username_available": self.username_available,
            "password_available": self.password_available,
            "pgpass_available": self.pgpass_available,
            "interactive": self.interactive,
            "username_source": self.username_source,
            "password_source": self.password_source,
        }


class WRDSClient:
    """Thin retrying wrapper around wrds.Connection."""

    def __init__(self, settings: WRDSSettings) -> None:
        self.settings = settings
        self._connection: Any | None = None

    def __enter__(self) -> "WRDSClient":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def open(self) -> None:
        """Open a WRDS connection if it is not already open."""

        if self._connection is not None:
            return
        try:
            import wrds
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("python package `wrds` is required for WRDS ingestion") from exc
        self._connection = wrds.Connection(**self._resolve_connection_kwargs())

    def close(self) -> None:
        """Close the WRDS connection."""

        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None

    def raw_sql(self, sql: str, params: dict[str, Any] | None = None) -> WRDSQueryResult:
        """Execute a WRDS SQL query with simple retry logic."""

        self.open()
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                rows = self._connection.raw_sql(sql, params=params or {})
                if rows is None:
                    rows = pd.DataFrame()
                return WRDSQueryResult(rows=rows, attempts=attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(self.settings.backoff_seconds * attempt)
        if last_error is None:
            raise WRDSExecutionError(
                attempts=self.settings.max_retries,
                exception_class="RuntimeError",
                exception_message="WRDS query failed with no exception details",
            )
        raise WRDSExecutionError(
            attempts=self.settings.max_retries,
            exception_class=last_error.__class__.__name__,
            exception_message=str(last_error),
        ) from last_error

    def connection_mode_summary(self) -> dict[str, Any]:
        """Return safe metadata about how WRDS credentials will be resolved."""

        return self._credential_status().to_dict()

    def probe(self, *, live: bool = False) -> dict[str, Any]:
        """Return safe connection diagnostics and optionally run a tiny live query."""

        summary = {"credential_status": self.connection_mode_summary()}
        if not live:
            return summary
        result = self.raw_sql("SELECT CURRENT_DATE AS server_date, CURRENT_USER AS server_user")
        summary["live_probe"] = {
            "attempts": result.attempts,
            "row_count": int(len(result.rows)),
            "columns": result.rows.columns.tolist(),
        }
        if not result.rows.empty:
            row = result.rows.iloc[0]
            summary["live_probe"]["server_date"] = str(row.get("server_date"))
            summary["live_probe"]["server_user_present"] = bool(row.get("server_user"))
        return summary

    def preflight_credentials(self) -> dict[str, Any]:
        """Raise immediately if non-interactive credential requirements are not satisfied."""

        self._resolve_connection_kwargs()
        return self.connection_mode_summary()

    def list_relations(self, libraries: list[str], *, preview_limit: int = 20) -> dict[str, Any]:
        """List accessible WRDS libraries and a preview of tables in each library."""

        self.open()
        available_libraries = sorted(self._connection.list_libraries())
        requested = [library.strip() for library in libraries if library.strip()]
        if not requested:
            requested = available_libraries

        libraries_payload: dict[str, Any] = {}
        for library in requested:
            if library not in available_libraries:
                libraries_payload[library] = {
                    "available": False,
                    "table_count": 0,
                    "table_preview": [],
                }
                continue
            tables = sorted(self._connection.list_tables(library=library))
            libraries_payload[library] = {
                "available": True,
                "table_count": len(tables),
                "table_preview": tables[: max(1, int(preview_limit))],
            }
        return {
            "library_count": len(available_libraries),
            "libraries": libraries_payload,
        }

    def describe_relation(self, library: str, table: str) -> dict[str, Any]:
        """Describe a WRDS table schema."""

        self.open()
        frame = self._connection.describe_table(library=library, table=table)
        if frame is None:
            frame = pd.DataFrame()
        return {
            "library": library,
            "table": table,
            "columns": frame.to_dict(orient="records"),
        }

    def _resolve_connection_kwargs(self) -> dict[str, Any]:
        status = self._credential_status()
        pgpass_entry = self._pgpass_entry()
        username = self.settings.wrds_username or os.getenv("PGUSER") or (pgpass_entry.get("username") if pgpass_entry else None)
        password = self.settings.wrds_password or (pgpass_entry.get("password") if pgpass_entry else None)

        if not username:
            raise RuntimeError(
                "WRDS username not found. Export WRDS_USERNAME (or WRDS_USER / PGUSER), or configure ~/.pgpass."
            )

        if not password and not status.pgpass_available:
            if status.interactive and self.settings.allow_interactive_password_prompt:
                password = getpass.getpass("WRDS password: ")
            else:
                raise RuntimeError(
                    "WRDS password not found. Export WRDS_PASSWORD (or PGPASSWORD), or configure ~/.pgpass."
                )

        kwargs: dict[str, Any] = {
            "wrds_username": username,
            "autoconnect": True,
            "verbose": False,
        }
        if password:
            kwargs["wrds_password"] = password
        return kwargs

    def _credential_status(self) -> WRDSCredentialStatus:
        pgpass_entry = self._pgpass_entry()
        username = self.settings.wrds_username or os.getenv("PGUSER") or (pgpass_entry.get("username") if pgpass_entry else None)
        password = self.settings.wrds_password or (pgpass_entry.get("password") if pgpass_entry else None)
        pgpass_path = Path(os.getenv("PGPASSFILE", Path.home() / ".pgpass"))
        interactive = bool(sys.stdin.isatty() and sys.stderr.isatty())
        if self.settings.wrds_username or os.getenv("PGUSER"):
            username_source = "environment"
        elif pgpass_entry and pgpass_entry.get("username"):
            username_source = "pgpass"
        else:
            username_source = "missing"
        if self.settings.wrds_password or os.getenv("PGPASSWORD"):
            password_source = "environment"
        elif pgpass_entry and pgpass_entry.get("password"):
            password_source = "pgpass"
        elif interactive and self.settings.allow_interactive_password_prompt:
            password_source = "interactive_getpass"
        else:
            password_source = "missing"
        return WRDSCredentialStatus(
            username_available=bool(username),
            password_available=bool(password),
            pgpass_available=pgpass_path.exists(),
            interactive=interactive,
            username_source=username_source,
            password_source=password_source,
        )

    def _pgpass_entry(self) -> dict[str, str] | None:
        pgpass_path = Path(os.getenv("PGPASSFILE", Path.home() / ".pgpass"))
        if not pgpass_path.exists():
            return None
        try:
            lines = pgpass_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:  # noqa: BLE001
            return None
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 4)
            if len(parts) != 5:
                continue
            host, port, database, username, password = parts
            if host not in {"*", "wrds-pgdata.wharton.upenn.edu"}:
                continue
            if database not in {"*", "wrds"}:
                continue
            return {
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
            }
        return None

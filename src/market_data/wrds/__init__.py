"""WRDS-first ingestion subsystem for the local DuckDB research database."""

from market_data.wrds.cli import add_wrds_subparser, run_wrds_command

__all__ = ["add_wrds_subparser", "run_wrds_command"]

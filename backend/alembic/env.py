"""Migrations run against CHAVRUTA_DATABASE_URL (or the URL passed in the Alembic config)."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from chavruta.config import Settings
from chavruta.db import Base

url = context.config.get_main_option("sqlalchemy.url") or Settings().database_url
with create_engine(url).connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

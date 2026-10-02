import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path
import app.db.models  # noqa: F401 - import all models so metadata is populated
from app.core.config import settings
from app.db.base import Base

from sqlalchemy import pool

from alembic import context

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# set sqlalchemy.url from settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# add your model's MetaData object here
target_metadata = Base.metadata

# The database is shared with the Build backend, whose own Alembic chain uses
# the default `alembic_version` table. Tracking our revisions there would make
# each service see the other's revision ids and refuse to migrate.
VERSION_TABLE = "ai_service_alembic_version"


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online():
    from sqlalchemy.ext.asyncio import create_async_engine

    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

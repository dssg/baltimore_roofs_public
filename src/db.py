import logging
import os
import random
from typing import Any, Sequence

from dotenv import find_dotenv, load_dotenv
import psycopg2
import psycopg2.extras
import sqlalchemy

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


def connection_string() -> str:
    """Get the PostgreSQL connection string from the environment.

    Returns:
        The PostgreSQL connection string
    """
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}"
        f"/{os.environ['PGDATABASE']}"
    )


def maybe_drop_existing_schemas(prefix):
    confirm_word = random.choice(["Y", "Yes", "OK", "Confirm"])
    schemas = "\n    ".join(fetch_schemas_with_prefix(prefix))
    if len(schemas) == 0:
        return
    choice = input(
        f"""The following schemas starting with "{prefix}" already exist:
    {schemas}

Enter "{confirm_word}" to drop: """
    )
    if choice.strip() == confirm_word:
        run(f"DROP SCHEMA IF EXISTS {prefix}_model_prep CASCADE")
        run(f"DROP SCHEMA IF EXISTS {prefix}_model_predictions CASCADE")
        run(f"DROP SCHEMA IF EXISTS {prefix}_model_results CASCADE")


def fetch_schemas_with_prefix(prefix):
    results = run_query(
        f"""
        SELECT schema_name FROM information_schema.schemata
        WHERE
        catalog_name = 'baltimore-roofs'
        AND schema_name LIKE '{prefix}_%';
        """
    )
    return [r["schema_name"] for r in results]


def engine() -> sqlalchemy.engine.Engine:
    """Return a SQLAlchemy engine connected to the database."""
    return sqlalchemy.create_engine(connection_string())


def run_query(
    query: str, params: tuple[Any, ...] = None
) -> list[psycopg2.extras.DictRow]:
    """Get results from the database.

    This creates a new database connection for every query, which should be fine for
    small things, but bad for large things.

    Args:
        query: The SQL query to run
        params: The set of parameters for the query

    Returns:
        A list of rows
    """
    conn = psycopg2.connect(
        connection_string(), cursor_factory=psycopg2.extras.DictCursor
    )
    cur = conn.cursor()

    logger.debug("Running query: %s", cur.mogrify(query, params))
    cur.execute(query, params)
    results = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return results


def run(query: str, params: tuple[Any, ...] = None) -> None:
    """Run a query with no return.

    Args:
        query (str): The SQL query to run
        params (tuple[Any, ...], optional): The set of params for the query.
            Defaults to None.
    """
    conn = psycopg2.connect(
        connection_string(), cursor_factory=psycopg2.extras.DictCursor
    )
    cur = conn.cursor()

    logger.debug("Running query: %s", cur.mogrify(query, params))
    cur.execute(query, params)
    conn.commit()
    cur.close()
    conn.close()


def batch_insert(query: str, values: Sequence[tuple[Any, ...]]) -> None:
    """Insert lots of data.

    Args:
        query (str): SQL query with "VALUES %s"
        values (list[tuple[Any, ...]]): All the values to insert
    """
    conn = psycopg2.connect(connection_string())
    cur = conn.cursor()

    logger.debug("Inserting %s values", len(values))
    psycopg2.extras.execute_values(cur, query, values)
    conn.commit()
    cur.close()
    conn.close()

import click
import logging

import ohio.ext.pandas  # noqa: F401
import pandas as pd
from sqlalchemy import create_engine

from src import db


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("schema", type=str)
@click.argument("table", type=str)
def load_data(input_filepath, schema, table):
    """Load data from Excel or CSV into the database.

    This function assumes you have all the requisite database credentials
    as environment variables already.

    Args:
        input_filepath (str): The filepath of the sheet to import
        schema (str): The name of the schema to put the table in
        table (str): The name of the table to put the data in
    """

    logger = logging.getLogger(__name__)
    engine = create_engine(db.connection_string())
    if ".xls" in input_filepath:
        df = pd.read_excel(input_filepath)
    else:
        df = pd.read_csv(input_filepath)

    logger.info("Importing %s rows from the input file", df.shape[0])
    df.pg_copy_to(table, engine, schema=schema)


if __name__ == "__main__":
    load_data()

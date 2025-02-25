import pandas as pd

from .clickhouse_globals import DATABASES, client


def coerce_uppercase_tablename(table: str) -> str:
    """
    Given a table name in the format
    "[database].[table]" or "[table]",
    coerces and returns the table portion to uppercase.
    ex. series.aapl becomes series.AAPL
    """
    *database, table = table.split(".")
    table = table.upper()

    if len(database) > 1:
        raise ValueError(f"Table name {table} is invalid.")

    if database == []:
        return table
    return f"{database[0]}.{table}"


def list_tables(database: str = None) -> list[str]:
    """
    Lists all time series tables in the database,
    within a certain database if database is provided

    Optional
    :param database: The specific database to get tables for

    :return: Table names
    """
    if database is not None:
        assert database in DATABASES
        # Each row is info about one table
        # The first entry in each is the table database, second is the table name
        tables_unformatted = client.command(
                    f"""
                    SELECT table_schema, table_name FROM information_schema.tables
                    WHERE table_schema = '{database}'
                    """
        )
        return ".".join(tables_unformatted).split("\n")

    if database is None:
        tables_unformatted = client.command(
                """
                    SELECT table_schema, table_name FROM information_schema.tables
                    WHERE table_schema != 'INFORMATION_SCHEMA' AND
                    table_schema != 'information_schema' AND
                    table_schema != 'system'
                    """
        )
        return ".".join(tables_unformatted).split("\n")


def table_columns(table: str) -> list[str]:
    """
    Gets the column names of the table

    :param table: Table name

    :return: Table column names
    """
    tablecols = " ".join(client.command(f"DESCRIBE TABLE {table}")).split("\n")
    return list(map(lambda c: c.strip(), tablecols))


def get_timespan(table: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> list[tuple]:
    """
    Gets all time series entries within a certain timespan
    Start and end date inclusive

    :param table: The table to get time series data from

    Optional
    :param start: First date to get data for
                  Defaults to earliest date in the table
    :param end: Last date to get data for
                Defaults to most recent date in the table

    :return: All matching timeseries entries
    """
    table = coerce_uppercase_tablename(table)

    if start is None and end is None:
        data = client.command(f"SELECT * FROM {table}")
    elif end is None:
        data = client.command(f"SELECT * FROM {table} WHERE date >= '{start}'")
    elif start is None:
        data = client.command(f"SELECT * FROM {table} WHERE date <= '{end}'")
    else:
        data = client.command(f"SELECT * FROM {table} WHERE date >= '{start}' AND date <= '{end}'")

    # List strings are tab-separated; make them newline-separated
    # and convert each resulting row into a tuple
    return list(map(lambda r: tuple(r.split(", ")), ", ".join(data).split("\n")))

import pandas as pd

from .env import DATABASES, client, OLDEST_ENTRY_DATE


def coerce_uppercase_tablename(table: str) -> str:
    """
    Given a table name in the format
    "[database].[table]" or "[table]",
    replaces spaces with underscores, and
    coerces and returns the table portion to uppercase.
    ex. series.aapl becomes series.AAPL
        "spy us equity" becomes SPY_US_EQUITY

    :param table: Tablename with or without the leading database schema

    :return: Tablename with the name uppercased and spaces replaced
    """
    table = table.replace(" ", "_")
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


def upsert_entries(table: str, rows: list[tuple] | pd.DataFrame, *, ch_client=None):
    """
    Inserts or updates entries in the table

    :param table: The table name to update data for, case insensitive
    :param rows: Entries to update the table with
                 If rows is list[tuple], the first element of each tuple must be the date
                 If rows is DataFrame, column labels must match table column names
                 and the date column must be convertible to pd.Timestamp

    Optional
    :param ch_client: Alternate Clickhouse client to use for insertion
    """
    if ch_client is None:
        ch_client = client

    table = coerce_uppercase_tablename(table)

    if isinstance(rows, pd.DataFrame):
        rows["date"] = pd.to_datetime(rows["date"])  # Type coercion
        # Omit all entries before OLDEST_ENTRY_DATE
        rows = rows[rows["date"] >= OLDEST_ENTRY_DATE]
        ch_client.insert_df(table, rows)

    else:
        # Omit all entries before OLDEST_ENTRY_DATE
        rows = list(filter(lambda r: r[0] >= OLDEST_ENTRY_DATE, rows))
        ch_client.command(f"INSERT INTO {table} (*) VALUES", rows)

    # Deletes duplicate values
    ch_client.command(f"OPTIMIZE TABLE {table}")


def get_recent_entry(table: str) -> pd.Timestamp:
    """Gets the most recent entry for a table.
    If the table has no entries, defaults to the earliest date possible

    :param table: Table to query for most recent entry

    :return: Timestamp of the most recent entry
    """
    return pd.Timestamp(client.command(f"SELECT max(date) FROM {table}"))

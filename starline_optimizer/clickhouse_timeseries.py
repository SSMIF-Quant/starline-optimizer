# Utils for updating/reading clickhouse timeseries in the series database

import yfinance as yf
import numpy as np
from .clickhouse_globals import client
from .clickhouse import coerce_uppercase_tablename, get_recent_entry, upsert_entries


def create_series_table(ticker: str):
    """
    Creates a timeseries table in the database
    The table will be under "series" database
    Table name will be coerced into uppercase
    If multiple rows have the same date field, the row that was inserted
    most recently takes priority as the "true" value

    :param table: The name of the timeseries ticker
    """
    ticker = ticker.upper()
    client.command("CREATE DATABASE IF NOT EXISTS series COMMENT 'timeseries data'")
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS series.{ticker} (
            date DateTime64(3) NOT NULL,
            price Float64 NOT NULL,
            volume UInt32 NOT NULL,
            _updated_at Datetime NOT NULL MATERIALIZED NOW()
        )
        ENGINE = ReplacingMergeTree(_updated_at)
        ORDER BY date
        """
    )


def update_timeseries(table: str):
    """Adds additional entries into the database for the table if not all entries are up to date.
    Uses yfinance to obtain new data.

    :param table: The table to update
    """
    table = coerce_uppercase_tablename(table)
    _, ticker = table.split(".")
    create_series_table(ticker)  # If the table doesn't exist beforehand
    start_date = get_recent_entry(table)

    # TODO we can do better than yfinance
    dataraw = yf.Tickers(ticker).download(start=start_date)
    if dataraw is None:
        raise RuntimeError(f"Failed to download yfinance data for ticker {ticker}")

    # DataFrame manip to get dataframes with date, price, volume columns for each ticker
    data = dataraw.swaplevel(axis=1)[ticker][["Close", "Volume"]]
    data.reset_index(inplace=True)
    data.columns = ["date", "price", "volume"]
    data.ffill(inplace=True)  # Fill None values
    data["volume"] = np.uint32(data["volume"])  # Coerce volume from int to unsigned int
    upsert_entries(table, data)

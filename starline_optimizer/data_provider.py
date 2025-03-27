import pandas as pd
import cvxportfolio as cvx

from .clickhouse import get_timespan
from .clickhouse_timeseries import update_timeseries

# TODO Returns forecast dataframe from Clickhouse

# past_returns, current_returns, past_volumes, current_volumes, current_prices
type DataInstance = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]


def _tuples_to_df(data: list[tuple[pd.Timestamp, float, int]]) -> pd.DataFrame:
    """
    Converts a list of tuples that represent rows in the "series" database into a DataFrame

    :param data: List of database rows:
                 the first element is the timestamp
                 the second element is the price
                 the third element is the volume

    :return: A dataframe with timestamp index and price and volume columns
    """
    date, price, volume = zip(*data)
    date = pd.to_datetime(date)
    df = pd.DataFrame({"price": price, "volume": volume}, index=date)
    return df


class DataProvider(cvx.data.MarketData):
    """Serves market data for the optimization engine."""

    tickers: list[str]
    __prices: pd.DataFrame
    __return: pd.DataFrame
    __volume: pd.DataFrame

    def __init__(self, tickers: list[str]):
        """Initializes DataProvider with price and volume data.
        Both DataFrames must have pd.Timestamp indexes and columns with the price data.

        :param tickers: List of tickers for this DataProvider instance
        """
        for t in tickers:
            update_timeseries(f"series.{t}")
        
        # Get data for each ticker and ensure unique indices
        data = []
        for t in tickers:
            df = _tuples_to_df(get_timespan(f"series.{t}"))
            df.index = pd.to_datetime(df.index)
            # Convert price to float and volume to int immediately
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df[~df.index.duplicated(keep='last')]  # Remove duplicates
            data.append(df)
        
        # Create separate price and volume series
        prices = []
        volumes = []
        for d, t in zip(data, tickers):
            price_series = d[["price"]].rename(columns={"price": t})
            volume_series = d[["volume"]].rename(columns={"volume": t})
            prices.append(price_series)
            volumes.append(volume_series)

        # Concatenate with outer join to preserve all dates
        prices_df = pd.concat(prices, axis=1, join='outer')
        volumes_df = pd.concat(volumes, axis=1, join='outer')

        # Forward fill missing values
        prices_df = prices_df.ffill()
        volumes_df = volumes_df.fillna(0)

        self.tickers = tickers
        self.__prices = prices_df
        self.__return = prices_df.pct_change().fillna(0)
        self.__return["USDOLLAR"] = 0.04/252  # Daily risk-free rate
        self.__volume = volumes_df.astype(int)

    def serve(self, t: pd.Timestamp) -> DataInstance:
        """Serve data for policy and simulator at time t.

        :param t: Trading time. It must be included in the timestamps returned
                  by self.trading_calendar.

        :return: past_returns, current_returns, past_volumes, current_volumes, current_prices
        """
        date_pos = self.__prices.index.get_loc(t)

        if not isinstance(date_pos, int):
            raise pd.errors.DataError(f"Price data for DataProvider has duplicate timestamps {t}.")

        past_returns = self.__return.iloc[: date_pos - 1]
        curr_returns = self.__return.iloc[date_pos]
        past_volumes = self.__volume.iloc[: date_pos - 1]
        curr_volumes = self.__volume.iloc[date_pos]
        curr_prices = self.__prices.iloc[date_pos]

        return (past_returns, curr_returns, past_volumes, curr_volumes, curr_prices)

    def trading_calendar(
        self,
        start_time: None | pd.Timestamp = None,
        end_time: None | pd.Timestamp = None,
        include_end: bool = True,
    ) -> pd.Index:
        """Get trading calendar between times.

        :param start_time: Initial time of the trading calendar. Always inclusive
                           if present. If None, use the first available time.
        :param end_time: Final time of the trading calendar. If None,
                         use the last available time.
        :param include_end: Include end time.

        :return: Trading calendar.
        """
        calendar = self.__prices.index
        start_date_pos = 0 if start_time is None else calendar.get_loc(start_time)

        if end_time is not None:
            end_date_pos = calendar.get_loc(end_time)
            if not isinstance(end_date_pos, int):
                raise pd.errors.DataError("Price data for DataProvider has duplicate timestamps.")
            if not include_end:
                end_date_pos -= 1
        else:
            end_date_pos = None

        return calendar[start_date_pos:end_date_pos]

    @property
    def periods_per_year(self) -> int:
        return 252  # TODO quarterly, monthly, etc

    @property
    def full_universe(self) -> pd.Index:
        """Full universe, which might not be available for trading.

        :return: Full universe.
        """
        return self.__prices.columns
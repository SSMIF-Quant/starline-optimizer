import os

import pandas as pd

from clickhouse_connect import get_client, common

# This should always be set before creating a client
common.set_setting("autogenerate_session_id", False)

__REQUIRED_ENV_VARS = [
    "CLICKHOUSE_HOST",
    "CLICKHOUSE_PORT",
    "CLICKHOUSE_USER",
    "CLICKHOUSE_PASSWORD",
    "CLICKHOUSE_DATABASE",
]
for key in __REQUIRED_ENV_VARS:
    if key not in os.environ:
        raise OSError(f"Required environment variable {key} is missing.")

DB_SETTINGS = {
    "host": os.environ["CLICKHOUSE_HOST"],
    "port": os.environ["CLICKHOUSE_PORT"],
    "user": os.environ["CLICKHOUSE_USER"],
    "password": os.environ["CLICKHOUSE_PASSWORD"],
    "database": os.environ["CLICKHOUSE_DATABASE"],
}

client = get_client(**DB_SETTINGS)

# Omit all database entries before 1/1/2000
OLDEST_ENTRY_DATE = pd.Timestamp(year=2000, month=1, day=1)

# Running tally of the database names we have
DATABASES = ["series", "fred"]

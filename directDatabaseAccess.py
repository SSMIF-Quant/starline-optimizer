import os
from dotenv import load_dotenv
import clickhouse_connect
import csv

# Load environment variables from .env file
load_dotenv()

# Retrieve database credentials from .env
CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST')
CLICKHOUSE_PORT = os.getenv('CLICKHOUSE_PORT')  # Default port is 9000
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER')
CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')

# Connect to ClickHouse
client = clickhouse_connect.get_client(
    host=CLICKHOUSE_HOST,
    port=int(CLICKHOUSE_PORT),
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD
)

# List of tickers to query
tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# Directory to save CSV files
output_directory = "/Users/tksohan/starline-optimizer/ticker_data"
os.makedirs(output_directory, exist_ok=True)

# Query data for each ticker and save to CSV
for ticker in tickers:
    query = f"SELECT * FROM series.{ticker}"
    result = client.query(query).result_rows

    # Define the output file path
    output_file = os.path.join(output_directory, f"{ticker}.csv")

    # Write the result to a CSV file
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(client.query(query).column_names)
        # Write rows
        writer.writerows(result)

    print(f"Data for {ticker} saved to {output_file}")
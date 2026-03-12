import yfinance as yf
import pandas as pd
from db import engine
from sqlalchemy import text

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "SPY", "QQQ", "DIA", "IWM",
    "XLF", "XLE", "XLI",
    "TLT", "IEF", "LQD",
    "GLD", "SLV", "USO",
    "XLP", "XLU"
]

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def load_assets():
    print("Loading asset metadata...")

    insert_stmt = text("""
        INSERT INTO assets (symbol)
        VALUES (:symbol)
        ON CONFLICT (symbol) DO NOTHING;
    """)

    with engine.begin() as conn:
        for ticker in TICKERS:
            conn.execute(insert_stmt, {"symbol": ticker})


def load_prices():
    print("Downloading market data...")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        auto_adjust=True,
        progress=True
    )

    for ticker in TICKERS:
        print(f"Processing {ticker}...")

        df = data[ticker].reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        df["adj_close"] = df["close"]
        df["daily_return"] = df["close"].pct_change()
        df = df.dropna()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT id FROM assets WHERE symbol = :symbol"),
                {"symbol": ticker}
            ).fetchone()

        if result is None:
            raise Exception(f"Asset not found for {ticker}")

        asset_id = result[0]
        df["asset_id"] = asset_id

        delete_prices = text("""
            DELETE FROM price_data WHERE asset_id = :aid
        """)

        delete_returns = text("""
            DELETE FROM returns WHERE asset_id = :aid
        """)

        with engine.begin() as conn:
            conn.execute(delete_prices, {"aid": asset_id})
            conn.execute(delete_returns, {"aid": asset_id})

        price_insert = df[[
            "asset_id",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume"
        ]]

        price_insert.to_sql(
            "price_data",
            engine,
            if_exists="append",
            index=False
        )

        return_insert = df[[
            "asset_id",
            "date",
            "daily_return"
        ]]

        return_insert.to_sql(
            "returns",
            engine,
            if_exists="append",
            index=False
        )

    print("Data load complete.")


if __name__ == "__main__":
    load_assets()
    load_prices()
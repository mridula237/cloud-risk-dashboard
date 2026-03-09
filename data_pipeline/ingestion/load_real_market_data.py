import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text

# -----------------------------
# DATABASE CONFIG
# -----------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost/risk_platform"
)

engine = create_engine(DATABASE_URL)

# -----------------------------
# ASSETS (20 diversified tickers)
# -----------------------------

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


# -----------------------------
# LOAD ASSETS
# -----------------------------

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


# -----------------------------
# LOAD PRICE DATA
# -----------------------------

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

        # auto_adjust=True → close is already adjusted
        df["adj_close"] = df["close"]

        df["daily_return"] = df["close"].pct_change()

        df = df.dropna()

        # -------------------------
        # GET ASSET ID
        # -------------------------

        asset_query = text("""
            SELECT asset_id FROM assets WHERE symbol = :symbol
        """)

        asset_id_df = pd.read_sql(
            asset_query,
            engine,
            params={"symbol": ticker}
        )

        asset_id = asset_id_df["asset_id"].iloc[0]

        df["asset_id"] = asset_id

        # -------------------------
        # CLEAN EXISTING DATA
        # -------------------------

        delete_prices = text("""
            DELETE FROM price_data WHERE asset_id = :aid
        """)

        delete_returns = text("""
            DELETE FROM returns WHERE asset_id = :aid
        """)

        with engine.begin() as conn:
            conn.execute(delete_prices, {"aid": asset_id})
            conn.execute(delete_returns, {"aid": asset_id})

        # -------------------------
        # INSERT PRICE DATA
        # -------------------------

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

        # -------------------------
        # INSERT RETURNS
        # -------------------------

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


# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":

    load_assets()
    load_prices()
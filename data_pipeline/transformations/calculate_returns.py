import os
import pandas as pd
from db import engine
from sqlalchemy import text

# -----------------------------
# CALCULATE RETURNS
# -----------------------------

def calculate_returns():

    query = """
        SELECT asset_id, date, adj_close
        FROM price_data
        ORDER BY asset_id, date;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        print("No price data found.")
        return

    df["daily_return"] = (
        df.groupby("asset_id")["adj_close"]
        .pct_change()
    )

    returns = df[["asset_id", "date", "daily_return"]].dropna()

    # Clear old returns before inserting new
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE returns;"))

    returns.to_sql(
        "returns",
        engine,
        if_exists="append",
        index=False
    )

    print("Returns calculated and inserted successfully.")


if __name__ == "__main__":
    calculate_returns()
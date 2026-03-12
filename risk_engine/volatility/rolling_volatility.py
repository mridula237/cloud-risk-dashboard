import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

import os
from db import engine

def calculate_rolling_volatility(window=30):
    query = """
        SELECT asset_id, date, daily_return
        FROM returns
        ORDER BY asset_id, date;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        print("No returns data found.")
        return None

    df["rolling_vol"] = (
        df.groupby("asset_id")["daily_return"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["annualized_vol"] = df["rolling_vol"] * np.sqrt(252)

    result = df[["asset_id", "date", "annualized_vol"]].dropna()

    return result


if __name__ == "__main__":
    vol_df = calculate_rolling_volatility(window=30)

    if vol_df is not None:
        print(vol_df.head())
        print("\nTotal rows:", len(vol_df))
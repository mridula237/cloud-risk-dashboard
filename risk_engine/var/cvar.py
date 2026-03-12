import pandas as pd
import numpy as np
from db import engine



def calculate_cvar(confidence=0.95):
    """
    Calculates Conditional VaR (Expected Shortfall)
    """

    query = """
        SELECT asset_id, date, daily_return
        FROM returns
        ORDER BY asset_id, date;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        print("No returns data found.")
        return None

    results = []

    for asset_id, group in df.groupby("asset_id"):

        var_threshold = np.percentile(
            group["daily_return"],
            (1 - confidence) * 100
        )

        cvar_value = group[
            group["daily_return"] <= var_threshold
        ]["daily_return"].mean()

        results.append({
            "asset_id": asset_id,
            "confidence": confidence,
            "historical_var": var_threshold,
            "cvar": cvar_value
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = calculate_cvar(0.95)

    if df is not None:
        print(df)
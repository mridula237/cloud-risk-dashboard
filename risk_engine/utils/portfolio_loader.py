import os
import pandas as pd
from db import engine
# -----------------------------
# DATABASE CONFIG
# -----------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@postgres-db:5432/riskdb"
)



# -----------------------------
# PORTFOLIO RETURNS
# -----------------------------

def get_portfolio_returns(portfolio_id: int):

    weights_query = """
        SELECT asset_id, weight
        FROM positions
        WHERE portfolio_id = %(pid)s;
    """

    weights_df = pd.read_sql(weights_query, engine, params={"pid": portfolio_id})

    if weights_df.empty:
        return None

    asset_ids = tuple(weights_df["asset_id"].tolist())

    returns_query = f"""
        SELECT asset_id, date, daily_return
        FROM returns
        WHERE asset_id IN {asset_ids}
        ORDER BY date;
    """

    returns_df = pd.read_sql(returns_query, engine)

    if returns_df.empty:
        return None

    returns_pivot = returns_df.pivot(
        index="date",
        columns="asset_id",
        values="daily_return"
    )

    weights = weights_df.set_index("asset_id")["weight"]

    returns_pivot = returns_pivot[weights.index]

    portfolio_returns = returns_pivot.dot(weights)

    return portfolio_returns.dropna()
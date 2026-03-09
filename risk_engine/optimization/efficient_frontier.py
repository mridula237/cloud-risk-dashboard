import numpy as np
import pandas as pd

from risk_engine.utils.portfolio_loader import get_portfolio_returns
from sqlalchemy import create_engine

engine = create_engine("postgresql://localhost/risk_platform")


def compute_efficient_frontier(portfolio_id: int, num_portfolios=5000):

    # -----------------------------
    # Get asset weights
    # -----------------------------

    weights_query = """
        SELECT asset_id, weight
        FROM positions
        WHERE portfolio_id = %(pid)s;
    """

    weights_df = pd.read_sql(
        weights_query,
        engine,
        params={"pid": portfolio_id}
    )

    if weights_df.empty:
        return None

    asset_ids = tuple(weights_df["asset_id"].tolist())

    # -----------------------------
    # Get asset returns
    # -----------------------------

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

    returns_pivot = returns_pivot.dropna()

    # -----------------------------
    # Expected returns & covariance
    # -----------------------------

    mean_returns = returns_pivot.mean()
    cov_matrix = returns_pivot.cov()

    num_assets = len(mean_returns)

    results = {
        "returns": [],
        "risk": [],
        "weights": []
    }

    # -----------------------------
    # Monte Carlo portfolios
    # -----------------------------

    for _ in range(num_portfolios):

        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)

        portfolio_variance = np.dot(
            weights.T,
            np.dot(cov_matrix, weights)
        )

        portfolio_std = np.sqrt(portfolio_variance)

        results["returns"].append(float(portfolio_return))
        results["risk"].append(float(portfolio_std))
        results["weights"].append(weights.tolist())

    return results
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("postgresql://localhost/risk_platform")


def calculate_portfolio_var(portfolio_id=1, confidence=0.95):

    # Load portfolio weights first
    weights_query = f"""
        SELECT asset_id, weight
        FROM positions
        WHERE portfolio_id = {portfolio_id};
    """

    weights_df = pd.read_sql(weights_query, engine)

    if weights_df.empty:
        print("No portfolio weights found.")
        return None

    asset_ids = tuple(weights_df["asset_id"].tolist())

    # Load only relevant returns
    returns_query = f"""
        SELECT asset_id, date, daily_return
        FROM returns
        WHERE asset_id IN {asset_ids}
        ORDER BY date;
    """

    returns_df = pd.read_sql(returns_query, engine)

    if returns_df.empty:
        print("No returns found.")
        return None

    # Pivot to matrix
    returns_pivot = returns_df.pivot(
        index="date",
        columns="asset_id",
        values="daily_return"
    )

    weights = weights_df.set_index("asset_id")["weight"]

    returns_pivot = returns_pivot[weights.index]

    # Portfolio returns
    portfolio_returns = returns_pivot.dot(weights)

    portfolio_returns = portfolio_returns.dropna()

    # Historical VaR
    var_value = np.percentile(
        portfolio_returns,
        (1 - confidence) * 100
    )

    cvar_value = portfolio_returns[
        portfolio_returns <= var_value
    ].mean()

    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Rolling volatility (30-day)
    rolling_vol = portfolio_returns.rolling(30).std()

    # Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1

    # -----------------------------------------------------
    # ADDITIONAL PORTFOLIO METRICS (NEW CODE)
    # -----------------------------------------------------

    mean_return = portfolio_returns.mean()

    volatility = portfolio_returns.std()

    max_drawdown = drawdown.min()

    # Sharpe ratio (assume risk-free rate = 0)
    if volatility != 0:
        sharpe_ratio = mean_return / volatility
    else:
        sharpe_ratio = 0

    # -----------------------------------------------------

    # Clean NaNs for JSON safety
    portfolio_returns = portfolio_returns.fillna(0)
    cumulative_returns = cumulative_returns.fillna(0)
    rolling_vol = rolling_vol.fillna(0)
    drawdown = drawdown.fillna(0)

    return {
        "portfolio_id": portfolio_id,
        "confidence": confidence,

        # Existing metrics
        "portfolio_var": float(var_value),
        "portfolio_cvar": float(cvar_value),

        # NEW METRICS
        "mean_return": float(mean_return),
        "volatility": float(volatility),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),

        # Time series data
        "timeseries": portfolio_returns.to_dict(),
        "cumulative": cumulative_returns.to_dict(),
        "rolling_volatility": rolling_vol.to_dict(),
        "drawdown": drawdown.to_dict()
    }


if __name__ == "__main__":
    result = calculate_portfolio_var(1, 0.95)
    print(result)
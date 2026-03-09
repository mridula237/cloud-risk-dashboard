import numpy as np
from risk_engine.utils.portfolio_loader import get_portfolio_returns


def calculate_historical_var(portfolio_id=1, confidence=0.95):

    portfolio_returns = get_portfolio_returns(portfolio_id)

    if portfolio_returns is None or portfolio_returns.empty:
        return None

    var_value = np.percentile(
        portfolio_returns,
        (1 - confidence) * 100
    )

    cvar_value = portfolio_returns[
        portfolio_returns <= var_value
    ].mean()

    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_vol = portfolio_returns.rolling(30).std()
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1

    return {
        "portfolio_id": portfolio_id,
        "confidence": confidence,
        "model": "historical",
        "portfolio_var": float(var_value),
        "portfolio_cvar": float(cvar_value),
        "timeseries": portfolio_returns.fillna(0).to_dict(),
        "cumulative": cumulative_returns.fillna(0).to_dict(),
        "rolling_volatility": rolling_vol.fillna(0).to_dict(),
        "drawdown": drawdown.fillna(0).to_dict()
    }
import numpy as np
from scipy.stats import norm
from risk_engine.utils.portfolio_loader import get_portfolio_returns


def calculate_parametric_var(portfolio_id=1, confidence=0.95):

    portfolio_returns = get_portfolio_returns(portfolio_id)

    if portfolio_returns is None or portfolio_returns.empty:
        return None

    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    alpha = 1 - confidence
    z = norm.ppf(alpha)

    var_value = mu + z * sigma
    cvar_value = mu - sigma * (norm.pdf(z) / alpha)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_vol = portfolio_returns.rolling(30).std()
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1

    return {
        "portfolio_id": portfolio_id,
        "confidence": confidence,
        "model": "parametric",
        "mu": float(mu),
        "sigma": float(sigma),
        "portfolio_var": float(var_value),
        "portfolio_cvar": float(cvar_value),
        "timeseries": portfolio_returns.fillna(0).to_dict(),
        "cumulative": cumulative_returns.fillna(0).to_dict(),
        "rolling_volatility": rolling_vol.fillna(0).to_dict(),
        "drawdown": drawdown.fillna(0).to_dict()
    }
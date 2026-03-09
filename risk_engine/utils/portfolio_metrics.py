import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Compute Sharpe Ratio for portfolio returns
    """

    mean_return = returns.mean()
    volatility = returns.std()

    sharpe = (mean_return - risk_free_rate / 252) / volatility

    return float(sharpe)


def max_drawdown(returns):
    """
    Compute Maximum Drawdown
    """

    cumulative = (1 + returns).cumprod()

    peak = cumulative.cummax()

    drawdown = (cumulative - peak) / peak

    return float(drawdown.min())
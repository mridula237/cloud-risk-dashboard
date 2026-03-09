import numpy as np
from risk_engine.utils.portfolio_loader import get_portfolio_returns

SCENARIOS = {
    "market_crash": -0.35,
    "covid_crash": -0.20,
    "tech_crash": -0.25,
    "bond_crash": -0.15
}


def run_stress_test(portfolio_id: int):

    returns = get_portfolio_returns(portfolio_id)

    if returns is None:
        return None

    portfolio_vol = np.std(returns)

    results = {}

    for scenario, shock in SCENARIOS.items():
        results[scenario] = float(shock * portfolio_vol)

    return results
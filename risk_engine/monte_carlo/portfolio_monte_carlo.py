import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Database connection
import os
from db import engine

def monte_carlo_portfolio(portfolio_id=1, simulations=10000, confidence=0.95):
    """
    Monte Carlo simulation for portfolio VaR & CVaR.
    """

    # -------------------------
    # 1. Load Returns Data
    # -------------------------
    returns_query = """
        SELECT asset_id, date, daily_return
        FROM returns
        ORDER BY date;
    """
    returns_df = pd.read_sql(returns_query, engine)

    # -------------------------
    # 2. Load Portfolio Weights
    # -------------------------
    weights_query = f"""
        SELECT asset_id, weight
        FROM positions
        WHERE portfolio_id = {portfolio_id};
    """
    weights_df = pd.read_sql(weights_query, engine)

    if returns_df.empty or weights_df.empty:
        print("Missing data.")
        return None

    # -------------------------
    # 3. Pivot Returns
    # -------------------------
    returns_pivot = returns_df.pivot(
        index="date",
        columns="asset_id",
        values="daily_return"
    )

    # Align weights
    weights = weights_df.set_index("asset_id")["weight"]

    # Keep only assets in portfolio
    returns_pivot = returns_pivot[weights.index]

    # Drop missing rows
    returns_pivot = returns_pivot.dropna()

    # -------------------------
    # 4. Estimate Mean & Covariance
    # -------------------------
    mean_vector = returns_pivot.mean()
    cov_matrix = returns_pivot.cov()

    # -------------------------
    # 5. Monte Carlo Simulation
    # -------------------------
    simulated_returns = np.random.multivariate_normal(
        mean_vector,
        cov_matrix,
        simulations
    )

    portfolio_simulated = simulated_returns.dot(weights.values)

    # -------------------------
    # 6. Compute Risk Metrics
    # -------------------------
    var_simulated = np.percentile(
        portfolio_simulated,
        (1 - confidence) * 100
    )

    cvar_simulated = portfolio_simulated[
        portfolio_simulated <= var_simulated
    ].mean()

    result = {
    "portfolio_id": portfolio_id,
    "simulations": simulations,
    "confidence": confidence,
    "monte_carlo_var": float(var_simulated),
    "monte_carlo_cvar": float(cvar_simulated),
    "simulation_returns": portfolio_simulated.tolist()
}
    # -------------------------
    # 7. Store Risk Run
    # -------------------------
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO risk_runs
                (portfolio_id, model_type, confidence, simulations, var_value, cvar_value)
                VALUES
                (:portfolio_id, 'monte_carlo', :confidence, :simulations, :var_value, :cvar_value);
            """),
            {
                "portfolio_id": portfolio_id,
                "confidence": confidence,
                "simulations": simulations,
                "var_value": float(var_simulated),
                "cvar_value": float(cvar_simulated)
            }
        )

    return result


if __name__ == "__main__":
    output = monte_carlo_portfolio(portfolio_id=1)
    print(output)
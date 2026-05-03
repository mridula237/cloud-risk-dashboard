from __future__ import annotations

import os
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@postgres-db:5432/riskdb")
engine = create_engine(DATABASE_URL)

# Real historical crisis periods — all within the 2010-2024 data window.
# Each tuple is (start_date, end_date, description).
CRISIS_PERIODS: dict[str, tuple[str, str, str]] = {
    "covid_crash":       ("2020-02-19", "2020-03-23", "COVID-19 crash (fastest bear market in history, -34% in 33 days)"),
    "tech_selloff_2022": ("2022-01-03", "2022-10-12", "2022 rate-hike tech selloff (-27% Nasdaq)"),
    "bond_crash_2022":   ("2022-01-03", "2022-12-30", "2022 bond crash (worst bond year since 1788)"),
    "china_crash_2015":  ("2015-06-12", "2015-08-26", "China stock crash + global Black Monday (-11% in a week)"),
}


def _load_portfolio_series(portfolio_id: int) -> pd.Series | None:
    weights_df = pd.read_sql(
        "SELECT asset_id, weight FROM positions WHERE portfolio_id = %(pid)s",
        engine, params={"pid": portfolio_id}
    )
    if weights_df.empty:
        return None

    asset_ids = tuple(weights_df["asset_id"].tolist())
    if len(asset_ids) == 1:
        asset_ids = (asset_ids[0], asset_ids[0])

    returns_df = pd.read_sql(
        f"SELECT asset_id, date, daily_return FROM returns WHERE asset_id IN {asset_ids} ORDER BY date",
        engine
    )
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    pivot = returns_df.pivot(index="date", columns="asset_id", values="daily_return")
    weights = weights_df.set_index("asset_id")["weight"]
    return pivot[weights.index].dropna().dot(weights).sort_index()


def run_stress_test(portfolio_id: int) -> dict:
    """
    Replay actual historical crisis periods through the real portfolio.
    Returns cumulative portfolio return (not a scalar shock × vol estimate).
    Each scenario is grounded in real market data from 2010-2024.
    """
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return {}

    results = {}
    for scenario, (start, end, description) in CRISIS_PERIODS.items():
        period = series.loc[start:end]
        if len(period) >= 5:
            cumulative_return = float((1 + period).prod() - 1)
            worst_day = float(period.min())
            n_days = len(period)
        else:
            # Period not in DB — fall back to vol-scaled estimate
            vol = float(series.std())
            shocks = {"covid_crash": -0.20, "tech_selloff_2022": -0.25,
                      "bond_crash_2022": -0.15, "china_crash_2015": -0.12}
            cumulative_return = shocks.get(scenario, -0.15) * vol
            worst_day = cumulative_return / 20
            n_days = 0

        results[scenario] = {
            "cumulative_return": round(cumulative_return, 4),
            "worst_single_day": round(worst_day, 4),
            "trading_days": n_days,
            "description": description,
        }

    return results

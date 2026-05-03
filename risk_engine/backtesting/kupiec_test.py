from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@postgres-db:5432/riskdb")
engine = create_engine(DATABASE_URL)


def _load_portfolio_series(portfolio_id: int) -> pd.Series:
    weights_df = pd.read_sql(
        "SELECT asset_id, weight FROM positions WHERE portfolio_id = %(pid)s",
        engine, params={"pid": portfolio_id}
    )
    if weights_df.empty:
        raise ValueError(f"No positions found for portfolio_id={portfolio_id}")

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


def kupiec_pof_test(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.

    Validates whether the historical VaR model produces violations at the
    theoretically expected rate. A 95% VaR should be breached exactly 5% of days.

    H₀: model is correct  → p-value > 0.05 = PASS
    H₁: model is wrong    → p-value < 0.05 = FAIL
    """
    returns = _load_portfolio_series(portfolio_id).dropna()

    p = 1.0 - confidence  # expected failure rate (e.g. 0.05 for 95% VaR)

    var_series = returns.rolling(window).quantile(p)
    aligned = pd.DataFrame({"ret": returns, "var": var_series}).dropna()

    T = len(aligned)
    x = int((aligned["ret"] < aligned["var"]).sum())
    actual_rate = x / T if T > 0 else 0.0

    # Kupiec LR statistic
    if x == 0:
        lr = -2.0 * T * np.log(1.0 - p)
    elif x == T:
        lr = -2.0 * T * np.log(p)
    else:
        lr = -2.0 * (
            np.log((1.0 - p) ** (T - x) * p ** x)
            - np.log((1.0 - actual_rate) ** (T - x) * actual_rate ** x)
        )

    p_value = float(1.0 - chi2.cdf(lr, df=1))
    passed = p_value > 0.05

    # Christoffersen independence test (checks whether violations cluster)
    ind_stat, ind_pval = _christoffersen_independence(aligned["ret"].values, aligned["var"].values)

    return {
        "portfolio_id": portfolio_id,
        "confidence": confidence,
        "window": window,
        "n_observations": T,
        "expected_violations": int(round(p * T)),
        "actual_violations": x,
        "expected_rate": round(p, 4),
        "actual_rate": round(actual_rate, 4),
        "kupiec_lr_statistic": round(lr, 4),
        "kupiec_p_value": round(p_value, 4),
        "kupiec_passed": passed,
        "independence_lr_statistic": round(ind_stat, 4) if ind_stat is not None else None,
        "independence_p_value": round(ind_pval, 4) if ind_pval is not None else None,
        "interpretation": (
            "VaR model is statistically accurate" if passed
            else "VaR model is mis-specified — violations occur at unexpected rate"
        ),
    }


def _christoffersen_independence(returns: np.ndarray, var: np.ndarray) -> tuple[float | None, float | None]:
    """
    Christoffersen (1998) test for violation independence.
    Tests whether VaR breaches cluster in time (which they shouldn't in a good model).
    """
    hits = (returns < var).astype(int)
    n = len(hits)
    if n < 4:
        return None, None

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = hits[i - 1], hits[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Unconditional violation probability
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-9)
    pi01 = n01 / (n00 + n01 + 1e-9)
    pi11 = n11 / (n10 + n11 + 1e-9)

    eps = 1e-9
    pi01 = np.clip(pi01, eps, 1 - eps)
    pi11 = np.clip(pi11, eps, 1 - eps)
    pi   = np.clip(pi,   eps, 1 - eps)

    lr = -2.0 * (
        (n00 + n10) * np.log(1.0 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1.0 - pi01) - n01 * np.log(pi01)
        - n10 * np.log(1.0 - pi11) - n11 * np.log(pi11)
    )

    p_val = float(1.0 - chi2.cdf(lr, df=1))
    return float(lr), p_val

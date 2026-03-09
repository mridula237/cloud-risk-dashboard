import numpy as np
from scipy.stats import norm, chi2
from risk_engine.utils.portfolio_loader import get_portfolio_returns


# -------------------------------------------------------
# Compute VaR for different models
# -------------------------------------------------------
def compute_var(train_returns, model, confidence):

    alpha = 1 - confidence

    if model == "historical":
        return np.percentile(train_returns, alpha * 100)

    elif model == "parametric":
        mu = train_returns.mean()
        sigma = train_returns.std(ddof=1)
        z = norm.ppf(alpha)
        return mu + z * sigma

    elif model == "monte-carlo":
        mu = train_returns.mean()
        sigma = train_returns.std(ddof=1)
        sims = np.random.normal(mu, sigma, 20000)
        return np.percentile(sims, alpha * 100)

    else:
        raise ValueError("Model must be: historical, parametric, or monte-carlo")


# -------------------------------------------------------
# Kupiec Proportion of Failures Test
# -------------------------------------------------------
def kupiec_test(violations_count, total_obs, confidence):

    p = 1 - confidence
    x = violations_count
    n = total_obs

    if x == 0 or x == n:
        return {
            "lr_statistic": None,
            "p_value": None,
            "reject_model_95": None
        }

    pi = x / n

    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - pi)) +
        x * np.log(p / pi)
    )

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "reject_model_95": bool(lr > 3.84)
    }


# -------------------------------------------------------
# Rolling Window VaR Backtest
# -------------------------------------------------------
def backtest_var(portfolio_id: int,
                 model: str = "historical",
                 confidence: float = 0.95,
                 window: int = 252):

    r = get_portfolio_returns(portfolio_id)

    if r is None or len(r) < window + 10:
        return None

    r = r.sort_index()

    var_series = {}
    violations = {}

    for i in range(window, len(r)):
        train = r.iloc[i-window:i]
        test_ret = r.iloc[i]

        var_t = compute_var(train, model, confidence)

        date = str(r.index[i])
        var_series[date] = float(var_t)
        violations[date] = bool(test_ret < var_t)

    violations_count = sum(violations.values())
    total_days = len(violations)

    violation_rate = violations_count / total_days
    expected_rate = 1 - confidence

    kupiec_result = kupiec_test(
        violations_count=violations_count,
        total_obs=total_days,
        confidence=confidence
    )

    return {
        "portfolio_id": portfolio_id,
        "model": model,
        "confidence": confidence,
        "window": window,
        "violation_rate": float(violation_rate),
        "expected_violation_rate": float(expected_rate),
        "deviation": float(violation_rate - expected_rate),
        "violations_count": int(violations_count),
        "total_test_days": int(total_days),
        "kupiec_test": kupiec_result,
        "var_series": var_series,
        "violations": violations
    }
import os
import math
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine

from risk_engine.utils.correlation_matrix import get_correlation_matrix
from risk_engine.stress_testing.portfolio_stress import run_stress_test
from risk_engine.backtesting.kupiec_test import kupiec_pof_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@postgres-db:5432/riskdb")
engine = create_engine(DATABASE_URL)

app = FastAPI(title="Cloud Risk Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(x):
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    if isinstance(x, np.floating):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/")
def root():
    return {"message": "Risk Platform API running"}


# -------------------------------------------------------
# Shared loader: portfolio daily returns series
# -------------------------------------------------------
def _load_portfolio_series(portfolio_id: int) -> pd.Series | None:
    query = """
        SELECT r.date, r.asset_id, r.daily_return, p.weight
        FROM returns r
        JOIN positions p ON p.asset_id = r.asset_id
        WHERE p.portfolio_id = %(pid)s
        ORDER BY r.date
    """
    df = pd.read_sql(query, engine, params={"pid": portfolio_id})
    if df.empty:
        return None

    pivot = df.pivot_table(index="date", columns="asset_id", values="daily_return")
    weights = df.drop_duplicates("asset_id").set_index("asset_id")["weight"]
    pivot = pivot[weights.index].dropna()
    return pivot.dot(weights).dropna()


# -------------------------------------------------------
# Portfolios list
# -------------------------------------------------------
@app.get("/portfolios")
def list_portfolios():
    df = pd.read_sql(
        "SELECT DISTINCT portfolio_id FROM positions ORDER BY portfolio_id",
        engine
    )
    return [{"portfolio_id": int(r)} for r in df["portfolio_id"]]


# -------------------------------------------------------
# Returns
# -------------------------------------------------------
@app.get("/portfolio/returns/{portfolio_id}")
def portfolio_returns(portfolio_id: int):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []
    return _json_safe([{"date": str(d), "return": float(v)} for d, v in series.items()])


# -------------------------------------------------------
# Volatility
# -------------------------------------------------------
@app.get("/portfolio/volatility/{portfolio_id}")
def portfolio_volatility(portfolio_id: int, window: int = 30):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []
    vol = series.rolling(window).std().dropna()
    return _json_safe([{"date": str(d), "volatility": float(v)} for d, v in vol.items()])


# -------------------------------------------------------
# Monte Carlo distribution histogram
# -------------------------------------------------------
@app.get("/portfolio/monte_carlo/{portfolio_id}")
def portfolio_monte_carlo(portfolio_id: int, n: int = 2000, bucket_size: float = 0.001):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []

    mu = float(series.mean())
    sigma = float(series.std())
    sims = np.random.normal(mu, sigma, n)

    bins: dict[float, int] = {}
    for v in sims:
        key = math.floor(v / bucket_size) * bucket_size
        bins[key] = bins.get(key, 0) + 1

    return _json_safe([{"bucket": float(k), "count": int(c)} for k, c in sorted(bins.items())])


# -------------------------------------------------------
# Drawdown
# -------------------------------------------------------
@app.get("/portfolio/drawdown/{portfolio_id}")
def portfolio_drawdown(portfolio_id: int):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []

    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    dd = ((cumulative - peak) / peak).fillna(0.0)
    return _json_safe([{"date": str(d), "drawdown": float(v)} for d, v in dd.items()])


# -------------------------------------------------------
# Efficient Frontier
# -------------------------------------------------------
@app.get("/portfolio/efficient_frontier/{portfolio_id}")
def efficient_frontier(portfolio_id: int, points: int = 1500):
    query = """
        SELECT r.date, r.asset_id, r.daily_return
        FROM returns r
        JOIN positions p ON p.asset_id = r.asset_id
        WHERE p.portfolio_id = %(pid)s
        ORDER BY r.date
    """
    df = pd.read_sql(query, engine, params={"pid": portfolio_id})
    if df.empty or df["asset_id"].nunique() < 2:
        return {"risk": [], "returns": []}

    pivot = df.pivot_table(index="date", columns="asset_id", values="daily_return").dropna()
    mean_returns = pivot.mean()
    cov_matrix = pivot.cov()
    num_assets = len(mean_returns)

    risks, rets = [], []
    for _ in range(points):
        w = np.random.random(num_assets)
        w /= w.sum()
        risks.append(float(np.sqrt(w @ cov_matrix.values @ w)))
        rets.append(float(w @ mean_returns.values))

    return _json_safe({"risk": risks, "returns": rets})


# -------------------------------------------------------
# Correlation matrix
# -------------------------------------------------------
@app.get("/portfolio/correlation")
def correlation_matrix():
    return get_correlation_matrix()


# -------------------------------------------------------
# Portfolio allocation
# -------------------------------------------------------
@app.get("/portfolio/{portfolio_id}/allocation")
def get_portfolio_allocation(portfolio_id: int):
    query = """
        SELECT a.symbol, p.weight
        FROM positions p
        JOIN assets a ON p.asset_id = a.asset_id
        WHERE p.portfolio_id = %(pid)s
    """
    df = pd.read_sql(query, engine, params={"pid": portfolio_id})
    return df.to_dict(orient="records")


# -------------------------------------------------------
# Stress test — real historical scenario replay
# -------------------------------------------------------
@app.get("/portfolio/stress/{portfolio_id}")
def portfolio_stress(portfolio_id: int):
    raw = run_stress_test(portfolio_id)
    # Flatten to [{scenario, loss, worst_single_day, trading_days, description}]
    # so the frontend bar chart still works (loss = cumulative_return)
    out = []
    for scenario, data in raw.items():
        if isinstance(data, dict):
            out.append({
                "scenario": scenario,
                "loss": data.get("cumulative_return"),
                "worst_single_day": data.get("worst_single_day"),
                "trading_days": data.get("trading_days"),
                "description": data.get("description", ""),
            })
        else:
            out.append({"scenario": scenario, "loss": data})
    return _json_safe(out)


# -------------------------------------------------------
# Kupiec VaR backtest
# -------------------------------------------------------
@app.get("/portfolio/backtest/var/{portfolio_id}")
def var_backtest(portfolio_id: int, confidence: float = 0.95, window: int = 252):
    return _json_safe(kupiec_pof_test(portfolio_id, confidence=confidence, window=window))


# -------------------------------------------------------
# Monte Carlo path simulation (uses real portfolio stats)
# -------------------------------------------------------
@app.post("/simulate")
def simulate_portfolio(data: dict):
    investment = float(data.get("investment", 10000))
    days = int(data.get("days", 252))
    simulations = int(data.get("simulations", 200))
    portfolio_id = int(data.get("portfolio_id", 1))

    series = _load_portfolio_series(portfolio_id)
    if series is not None and len(series) > 30:
        mu = float(series.mean()) * 252       # annualise
        sigma = float(series.std()) * math.sqrt(252)
    else:
        mu, sigma = 0.08, 0.20                # sensible fallback

    paths = []
    for _ in range(simulations):
        price = investment
        path = []
        for day in range(days):
            shock = np.random.normal(mu / 252, sigma / math.sqrt(252))
            price *= 1 + shock
            path.append({"day": day + 1, "value": round(price, 2)})
        paths.append(path)

    return {"paths": paths}

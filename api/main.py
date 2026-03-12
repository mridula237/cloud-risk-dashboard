import os
import math
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from db import engine
from risk_engine.utils.correlation_matrix import get_correlation_matrix
from fastapi.middleware.cors import CORSMiddleware

from risk_engine.stress_testing.portfolio_stress import run_stress_test
from risk_engine.utils.correlation_matrix import get_correlation_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/risk_platform")

app = FastAPI(title="Cloud Risk Platform API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer,)):
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

    # keep only assets in this portfolio
    pivot = pivot[weights.index].dropna()
    series = pivot.dot(weights)

    return series.dropna()

@app.get("/portfolio/correlation")
def correlation_matrix():
    return get_correlation_matrix()
# -------------------------------------------------------
# ✅ REQUIRED by your frontend: /portfolio/returns/{id}
# returns: [{date, return}]
# -------------------------------------------------------
@app.get("/portfolio/returns/{portfolio_id}")
def portfolio_returns(portfolio_id: int):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []

    out = [{"date": str(d), "return": float(v)} for d, v in series.items()]
    return _json_safe(out)

# -------------------------------------------------------
# ✅ REQUIRED by your frontend: /portfolio/volatility/{id}
# returns: [{date, volatility}]
# -------------------------------------------------------
@app.get("/portfolio/volatility/{portfolio_id}")
def portfolio_volatility(portfolio_id: int, window: int = 30):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []

    vol = series.rolling(window).std().dropna()
    out = [{"date": str(d), "volatility": float(v)} for d, v in vol.items()]
    return _json_safe(out)

@app.get("/allocation")
def portfolio_allocation():
    import pandas as pd
    from sqlalchemy import create_engine
    
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT a.symbol, p.weight
    FROM positions p
    JOIN assets a ON p.asset_id = a.asset_id
    WHERE p.portfolio_id = 1
    """

    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")

# -------------------------------------------------------
# ✅ REQUIRED by your frontend: /portfolio/monte_carlo/{id}
# returns histogram: [{bucket, count}]
# -------------------------------------------------------
@app.get("/portfolio/monte_carlo/{portfolio_id}")
def portfolio_monte_carlo(portfolio_id: int, n: int = 2000, bucket_size: float = 0.001):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []

    mu = float(series.mean())
    sigma = float(series.std())
    sims = np.random.normal(mu, sigma, n)

    bins = {}
    for v in sims:
        key = math.floor(v / bucket_size) * bucket_size
        bins[key] = bins.get(key, 0) + 1

    out = [{"bucket": float(k), "count": int(c)} for k, c in sorted(bins.items())]
    return _json_safe(out)

# -------------------------------------------------------
# ✅ REQUIRED by your frontend: /portfolio/drawdown/{id}
# returns: [{date, drawdown}]
# -------------------------------------------------------
@app.get("/portfolio/drawdown/{portfolio_id}")
def portfolio_drawdown(portfolio_id: int):
    series = _load_portfolio_series(portfolio_id)
    if series is None:
        return []

    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    dd = ((cumulative - peak) / peak).fillna(0.0)

    out = [{"date": str(d), "drawdown": float(v)} for d, v in dd.items()]
    return _json_safe(out)

# -------------------------------------------------------
# ✅ REQUIRED by your frontend: /portfolio/efficient_frontier/{id}
# returns: {risk: [...], returns: [...]}
# -------------------------------------------------------
@app.get("/portfolio/efficient_frontier/{portfolio_id}")
def efficient_frontier(portfolio_id: int, points: int = 1500):
    # Need asset-level returns matrix (not portfolio series)
    query = """
        SELECT r.date, r.asset_id, r.daily_return
        FROM returns r
        JOIN positions p ON p.asset_id = r.asset_id
        WHERE p.portfolio_id = %(pid)s
        ORDER BY r.date
    """
    df = pd.read_sql(query, engine, params={"pid": portfolio_id})
    if df.empty:
        return {"risk": [], "returns": []}

    pivot = df.pivot_table(index="date", columns="asset_id", values="daily_return").dropna()
    if pivot.shape[1] < 2:
        return {"risk": [], "returns": []}

    mean_returns = pivot.mean()          # Series of assets
    cov_matrix = pivot.cov()             # DataFrame covariance matrix

    num_assets = len(mean_returns)

    risks = []
    rets = []

    for _ in range(points):
        w = np.random.random(num_assets)
        w = w / np.sum(w)

        port_ret = float(np.dot(w, mean_returns.values))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))

        risks.append(port_vol)
        rets.append(port_ret)

    return _json_safe({"risk": risks, "returns": rets})

# -------------------------------------------------------
# Optional: stress + correlation (you had these)
# -------------------------------------------------------
@app.get("/portfolio/stress/{portfolio_id}")
def portfolio_stress(portfolio_id: int):
    return run_stress_test(portfolio_id)

@app.get("/portfolio/correlation")
def correlation_matrix():
    return get_correlation_matrix()

@app.get("/portfolio/{portfolio_id}/allocation")
def get_portfolio_allocation(portfolio_id: int):

    query = f"""
        SELECT asset_id as asset, weight
        FROM positions
        WHERE portfolio_id = {portfolio_id}
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return []

    return df.to_dict(orient="records")

@app.post("/simulate")
def simulate_portfolio(data: dict):
    investment = float(data.get("investment", 10000))
    days = int(data.get("days", 252))
    simulations = int(data.get("simulations", 200))

    mu = 0.08
    sigma = 0.20

    paths = []

    for _ in range(simulations):
        price = investment
        series = []

        for day in range(days):
            shock = np.random.normal(mu / 252, sigma / np.sqrt(252))
            price = price * (1 + shock)
            series.append({
                "day": day + 1,
                "value": round(price, 2)
            })

        paths.append(series)

    return {
        "paths": paths
    }
# Cloud Risk Analytics Platform

A full-stack, cloud-deployed portfolio risk engine built with FastAPI, React, PostgreSQL, and Docker — hosted on AWS EC2.

## Live Demo
**http://98.84.27.24**

---

## What It Does

Ingests 15 years of real market data (2010–2024) from Yahoo Finance across 20 assets, computes institutional-grade portfolio risk metrics, and runs a production ML pipeline that predicts Value at Risk breaches — all visualized in an interactive dashboard.

---

## Risk Metrics

- **Daily Returns** — weighted portfolio return series across 15 years of market data
- **Rolling Volatility** — 30-day annualized volatility trend
- **EWMA Volatility** — RiskMetrics λ=0.94 conditional volatility (same model used by JPMorgan) — captures volatility clustering that rolling std misses
- **Sharpe Ratio** — risk-adjusted return metric computed client-side
- **Maximum Drawdown** — worst peak-to-trough decline over the full period
- **Value at Risk (VaR)** — 95% confidence loss threshold (historical, parametric, and Monte Carlo)
- **CVaR / Expected Shortfall** — average tail loss beyond the VaR threshold

---

## Models & Simulations

- **Monte Carlo Simulation** — 200+ simulated portfolio paths using real portfolio mean and volatility, with confidence bands (5th, 25th, 50th, 75th, 95th percentile)
- **Efficient Frontier** — 1,500-point random weight optimization identifying the maximum Sharpe ratio portfolio across 20 assets
- **Historical Stress Testing** — replays the actual portfolio through 4 real crisis periods using true daily returns from the database (not scalar shocks)
- **Return Distribution** — histogram of simulated daily returns with VaR reference line
- **Kupiec VaR Backtest** — statistical test validating whether the VaR model produces violations at the theoretically correct rate

---

## Historical Stress Test Scenarios

Rather than applying a scalar shock multiplied by volatility, each scenario replays the actual portfolio returns through the real crisis dates — preserving the true correlation structure between all 20 assets during those periods.

| Scenario | Period | Event |
|---|---|---|
| `covid_crash` | Feb 19 – Mar 23, 2020 | Fastest bear market in history (–34% in 33 days) |
| `tech_selloff_2022` | Jan – Oct 2022 | Fed rate-hike selloff (–27% Nasdaq) |
| `bond_crash_2022` | Jan – Dec 2022 | Worst bond year since 1788 |
| `china_crash_2015` | Jun – Aug 2015 | China circuit breakers + global Black Monday |

---

## Machine Learning — VaR Violation Predictor

A production-grade ML pipeline that predicts whether a Value at Risk breach will occur within the next 5 trading days.

### Problem Setup
```
label(t) = 1  if  min(returns[t+1 : t+5]) < rolling_252d_VaR(t)
         = 0  otherwise

Positive rate: ~5.4%  (severe class imbalance)
Train: 2,815 samples | Test: 704 samples (strict temporal split)
```

### Feature Engineering — 15 Features

| Feature | Description |
|---|---|
| `return` | Daily portfolio return |
| `vol_10`, `vol_20`, `vol_60` | Rolling standard deviation (10/20/60-day windows) |
| `mean_5`, `mean_20` | Rolling mean return (trend signal) |
| `momentum_5`, `momentum_10` | Rolling return sum (momentum persistence) |
| `drawdown` | Cumulative drawdown from peak |
| `skew_20`, `kurt_20` | Rolling skewness and kurtosis (distribution shape) |
| `ewma_vol` | **RiskMetrics EWMA volatility** — λ=0.94, captures volatility clustering |
| `vix` | **VIX level** — market fear gauge |
| `vix_chg` | **VIX 1-day change** — fear acceleration |
| `vix_zscore` | **VIX 20-day z-score** — is fear abnormally elevated? |

### Class Imbalance — Three-Layer Solution

Only 5.4% of days are VaR violations. A naive classifier predicting "no breach" every day scores 94.6% accuracy while being completely useless.

1. **Class weighting** — LogReg `{0:1, 1:6}`, XGBoost dynamic `scale_pos_weight = negatives/positives`
2. **SMOTE oversampling** — generates synthetic minority samples in training only; minority class boosted from 5.4% → 23% of training set; test set stays real and untouched
3. **Threshold optimization** — searches 91 decision thresholds (0.05 → 0.95), selects the one maximizing F1-score

### Model Performance

| Model | ROC-AUC | Balanced Accuracy | MCC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.571 | 0.580 | 0.101 | 11.4% | 32.6% | 0.169 |
| XGBoost | 0.603 | 0.563 | 0.119 | 16.7% | 18.6% | 0.176 |
| LogReg + SMOTE | 0.562 | 0.584 | 0.108 | 11.9% | 32.6% | 0.174 |
| **XGBoost + SMOTE** | 0.578 | **0.573** | 0.070 | 7.8% | **62.8%** | 0.139 |

> XGBoost + SMOTE is the preferred production configuration: recall increases from 18.6% → 62.8%, catching 27 of 43 real breaches vs. only 8 without SMOTE. For a risk system, missing a breach is more costly than a false alarm.

> Balanced Accuracy and MCC are used as primary metrics — raw accuracy is misleading on imbalanced datasets.

### Validation & Monitoring

- **Walk-forward validation** — expanding-window time-series cross-validation; each fold trains on all past data and tests on the next 21 days; no data leakage
- **Threshold optimization** — F1-maximizing search across 91 decision boundary candidates
- **Feature importance** — XGBoost gain-based importance; top features: `drawdown`, `mean_20`, `ewma_vol`, `vix_zscore`
- **Model drift detection (PSI)** — Population Stability Index monitors feature distribution shifts; PSI > 0.25 triggers retraining
- **Model run logging** — every training run persisted to PostgreSQL with ROC-AUC, balanced accuracy, MCC, precision, recall, F1

### Kupiec VaR Backtesting

Statistical validation that the VaR model is correctly calibrated:

```
H₀: model is correct — violations occur at the expected 5% rate
LR = -2 × log(L₀ / L₁)  ~  χ²(1)
p-value > 0.05 → PASS  (VaR is statistically accurate)
p-value < 0.05 → FAIL  (VaR is mis-specified)
```

Also includes the **Christoffersen independence test** — checks whether violations cluster in time. A good VaR model should have independent violations, not clustered ones.

---

## Portfolio

- **20-asset portfolio** spanning equities, ETFs, bonds, commodities, and sector funds
- **Portfolio selector** — switch between multiple portfolios; all charts and metrics reload dynamically
- **Optimized weights** via mean-variance optimization (maximum Sharpe ratio)
- **Asset Correlation Heatmap** — full 20×20 Pearson correlation matrix with color-coded intensity
- **Portfolio Allocation Pie Chart** — visual breakdown of optimized weights

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, Recharts, Vite |
| **Backend** | FastAPI, Python 3.12, Uvicorn |
| **Database** | PostgreSQL 15, SQLAlchemy ORM |
| **Data Ingestion** | yfinance, pandas |
| **Risk Engine** | NumPy, SciPy, scikit-learn, statsmodels |
| **ML Models** | XGBoost, Logistic Regression, walk-forward CV |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Infrastructure** | AWS EC2 (t3.micro), Docker, Docker Compose |
| **Web Server** | Nginx (reverse proxy + SPA routing) |

---

## Architecture

```
Browser
   │
   ▼
Nginx (port 80)          ← serves React SPA, proxies /api/*
   │
   ▼
FastAPI (port 8000)      ← risk engine, ML pipeline, DB queries
   │
   ▼
PostgreSQL (port 5432)   ← 15 years of market data, model run history
```

All three services run as Docker containers orchestrated with Docker Compose on a single AWS EC2 t3.micro instance.

---

## Project Structure

```
cloud-risk-dashboard/
├── api/
│   └── main.py                          # FastAPI routes (18 endpoints)
├── data_pipeline/
│   ├── ingestion/
│   │   └── load_real_market_data.py     # yfinance → PostgreSQL
│   └── transformations/
│       └── calculate_returns.py         # pct_change() daily returns
├── risk_engine/
│   ├── backtesting/
│   │   └── kupiec_test.py               # Kupiec POF + Christoffersen tests
│   ├── ml/
│   │   └── violation_model.py           # VaR violation ML pipeline (SMOTE, walk-forward, PSI)
│   ├── monte_carlo/
│   │   └── portfolio_monte_carlo.py
│   ├── optimization/
│   │   └── efficient_frontier.py        # mean-variance optimization
│   ├── stress_testing/
│   │   └── portfolio_stress.py          # historical scenario replay
│   ├── utils/
│   │   ├── correlation_matrix.py
│   │   └── portfolio_metrics.py
│   └── var/
│       ├── historical_var.py
│       ├── parametric_var.py
│       └── cvar.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Dashboard UI (portfolio selector, Kupiec card)
│   │   ├── api.js                       # API client
│   │   └── PortfolioSimulator.jsx
│   ├── Dockerfile
│   └── nginx.conf
├── artifacts/                           # Persisted trained model bundles (.joblib)
├── Dockerfile                           # API image (Python 3.12-slim)
├── docker-compose.yml
└── requirements.txt
```

---

## Assets Tracked

| Category | Tickers |
|---|---|
| **Large Cap Equities** | AAPL, MSFT, NVDA, AMZN, GOOGL |
| **Broad Market ETFs** | SPY, QQQ, DIA, IWM |
| **Sector ETFs** | XLF, XLE, XLI, XLP, XLU |
| **Bonds** | TLT, IEF, LQD |
| **Commodities** | GLD, SLV, USO |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/portfolios` | List all available portfolios |
| GET | `/portfolio/returns/{id}` | Daily weighted portfolio returns |
| GET | `/portfolio/volatility/{id}` | 30-day rolling volatility series |
| GET | `/portfolio/drawdown/{id}` | Peak-to-trough drawdown series |
| GET | `/portfolio/monte_carlo/{id}` | Return distribution histogram |
| GET | `/portfolio/efficient_frontier/{id}` | 1,500-point efficient frontier |
| GET | `/portfolio/{id}/allocation` | Portfolio weights by asset |
| GET | `/portfolio/correlation` | 20×20 asset correlation matrix |
| GET | `/portfolio/stress/{id}` | Historical crisis scenario returns |
| GET | `/portfolio/backtest/var/{id}` | Kupiec VaR backtest + Christoffersen test |
| POST | `/simulate` | Monte Carlo path simulation (uses real portfolio μ/σ) |
| POST | `/ml/train/violation/{id}` | Train VaR violation classifier (SMOTE + walk-forward) |
| GET | `/ml/predict/violation/{id}` | Predict next breach probability |
| GET | `/ml/walkforward/{id}` | Walk-forward validation results by fold |
| GET | `/ml/feature-importance/{id}` | Feature importance ranking |
| GET | `/ml/drift/{id}` | PSI model drift report |
| GET | `/ml/compare/{id}` | LogReg vs XGBoost head-to-head comparison |

---

## Running Locally

### Prerequisites
- Docker & Docker Compose

### Steps

```bash
# Clone the repo
git clone https://github.com/mridula237/cloud-risk-dashboard.git
cd cloud-risk-dashboard

# Start all services (PostgreSQL + FastAPI + Nginx/React)
docker compose up --build -d

# Load 15 years of market data from Yahoo Finance
docker exec risk-platform-api python data_pipeline/ingestion/load_real_market_data.py

# Calculate daily returns
docker exec risk-platform-api python data_pipeline/transformations/calculate_returns.py

# Open dashboard
open http://localhost
```

---

## Dashboard Tabs

| Tab | What You See |
|---|---|
| **Overview** | Daily Returns chart, Portfolio Value Growth from $10K, 30-Day Volatility trend, 6 KPI cards |
| **Simulation** | Interactive Monte Carlo paths, Confidence bands (5/25/50/75/95th pct), VaR, Probability of loss |
| **Risk** | Kupiec backtest result, Return distribution, Drawdown chart, Historical stress scenarios |
| **Portfolio** | 20×20 Correlation heatmap, Allocation pie chart, Efficient frontier scatter |

---

## Skills Demonstrated

- **Data Engineering** — automated ingestion pipeline from Yahoo Finance into PostgreSQL; 15 years × 20 assets × OHLCV
- **Quantitative Finance** — VaR (historical + parametric + Monte Carlo), CVaR, EWMA volatility, Sharpe ratio, drawdown, efficient frontier, Kupiec backtesting
- **Machine Learning** — XGBoost + logistic regression classifiers; SMOTE for class imbalance; walk-forward time-series CV; threshold optimization; PSI drift detection; balanced accuracy + MCC evaluation
- **Backend Development** — RESTful API with FastAPI; SQLAlchemy ORM; NaN/Inf-safe JSON serialization
- **Frontend Development** — React 19 dashboard; Recharts visualizations; portfolio selector with reactive data loading
- **Cloud & DevOps** — Docker containerization; AWS EC2 deployment; Nginx reverse proxy + SPA routing; Docker Compose orchestration

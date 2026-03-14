# Cloud Risk Analytics Platform

A full-stack, cloud-deployed portfolio risk engine built with FastAPI, React, PostgreSQL, and Docker — hosted on AWS EC2.

## Live Demo
**http://98.84.27.24**

---

## What It Does

Ingests real market data from Yahoo Finance and computes a suite of portfolio risk metrics across 20 assets, visualized in an interactive dashboard.

### Risk Metrics
- **Daily Returns** — weighted portfolio return series (2010–2024)
- **Rolling Volatility** — 30 day annualized volatility trend
- **Sharpe Ratio** — risk-adjusted return metric
- **Maximum Drawdown** — worst peak-to-trough decline
- **Value at Risk (VaR)** — 95% confidence loss threshold
- **CVaR / Expected Shortfall** — tail risk beyond VaR

### Models & Simulations
- **Monte Carlo Simulation** — 200+ simulated portfolio paths with confidence bands (5th, 25th, 50th, 75th, 95th percentile)
- **Efficient Frontier** — 1,500-point Monte Carlo optimization identifying the maximum Sharpe ratio portfolio
- **Stress Testing** — scenario analysis across historical market shocks
- **Return Distribution** — histogram of simulated daily returns

### Machine Learning — VaR Violation Predictor
A production-grade ML pipeline that predicts whether a Value at Risk breach will occur within the next N trading days:

- **Two classifiers** — Logistic Regression and XGBoost, benchmarked head-to-head
- **Feature engineering** — 11 risk features including rolling volatility (10/20/60-day), momentum, drawdown, skewness, and kurtosis
- **Walk-forward validation** — expanding-window time-series cross-validation (no data leakage)
- **Threshold optimization** — F1-maximizing decision threshold search across 91 candidates
- **Feature importance** — identifies which risk signals drive violation predictions
- **Model drift detection** — Population Stability Index (PSI) monitors feature distribution shifts over time
- **Model run logging** — all training runs logged to PostgreSQL with ROC-AUC, precision, recall, F1

### Portfolio
- **20-asset portfolio** spanning equities, ETFs, bonds, commodities, and sector funds
- **Optimized weights** via mean-variance optimization (maximum Sharpe ratio)
- **Asset Correlation Heatmap** — full 20×20 correlation matrix with color-coded intensity
- **Portfolio Allocation Pie Chart** — visual breakdown of optimized weights

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React, Recharts, Vite |
| **Backend** | FastAPI, Python 3.12, Uvicorn |
| **Database** | PostgreSQL 15 |
| **Data Ingestion** | yfinance, pandas, SQLAlchemy |
| **Risk Engine** | NumPy, SciPy, scikit-learn, statsmodels |
| **ML Models** | Logistic Regression, XGBoost, walk-forward CV |
| **Infrastructure** | AWS EC2 (t3.micro), Docker, Docker Compose |
| **Web Server** | Nginx (reverse proxy) |

---

## Architecture

```
Browser
   │
   ▼
Nginx (port 80)
   │  /api/* → proxy
   ▼
FastAPI (port 8000)
   │
   ▼
PostgreSQL (port 5432)
```

All three services run as Docker containers orchestrated with Docker Compose on a single AWS EC2 instance.

---

## Project Structure

```
cloud-risk-dashboard/
├── api/
│   └── main.py                        # FastAPI routes
├── data_pipeline/
│   ├── ingestion/
│   │   └── load_real_market_data.py   # yfinance data loader
│   └── transformations/
│       └── calculate_returns.py       # return calculations
├── risk_engine/
│   ├── ml/
│   │   └── violation_model.py         # VaR violation ML pipeline
│   ├── monte_carlo/
│   │   └── portfolio_monte_carlo.py
│   ├── optimization/
│   │   └── efficient_frontier.py      # max Sharpe ratio optimizer
│   ├── stress_testing/
│   │   └── portfolio_stress.py
│   ├── utils/
│   │   ├── correlation_matrix.py
│   │   └── portfolio_metrics.py
│   └── var/
│       ├── historical_var.py
│       ├── parametric_var.py
│       └── cvar.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main dashboard UI
│   │   ├── api.js                     # API client
│   │   └── PortfolioSimulator.jsx
│   ├── Dockerfile
│   └── nginx.conf
├── Dockerfile                         # API Dockerfile
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
| GET | `/portfolio/returns/{id}` | Daily portfolio returns |
| GET | `/portfolio/volatility/{id}` | 30-day rolling volatility |
| GET | `/portfolio/drawdown/{id}` | Drawdown series |
| GET | `/portfolio/monte_carlo/{id}` | Return distribution histogram |
| GET | `/portfolio/efficient_frontier/{id}` | Efficient frontier points |
| GET | `/portfolio/{id}/allocation` | Portfolio weights by asset |
| GET | `/portfolio/correlation` | Asset correlation matrix |
| GET | `/portfolio/stress/{id}` | Stress test scenarios |
| POST | `/simulate` | Monte Carlo path simulation |
| POST | `/ml/train/violation/{id}` | Train VaR violation classifier |
| GET | `/ml/predict/violation/{id}` | Predict next violation probability |
| GET | `/ml/walkforward/{id}` | Walk-forward validation results |
| GET | `/ml/feature-importance/{id}` | Feature importance ranking |
| GET | `/ml/drift/{id}` | Model drift report (PSI) |
| GET | `/ml/compare/{id}` | Compare LogReg vs XGBoost |

---

## Running Locally

### Prerequisites
- Docker & Docker Compose

### Steps

```bash
# Clone the repo
git clone https://github.com/yourusername/cloud-risk-dashboard.git
cd cloud-risk-dashboard

# Start all services
docker compose up --build -d

# Load market data (runs yfinance download)
docker exec risk-platform-api python data_pipeline/ingestion/load_real_market_data.py

# Calculate returns
docker exec risk-platform-api python data_pipeline/transformations/calculate_returns.py

# Open dashboard
open http://localhost
```

---

## Dashboard Tabs

| Tab | Charts |
|---|---|
| **Overview** | Daily Returns, Portfolio Value Growth, 30-Day Volatility |
| **Simulation** | Monte Carlo Paths, Confidence Bands, Risk Metrics |
| **Risk** | Return Distribution, Drawdown, Stress Test Scenarios |
| **Portfolio** | Correlation Heatmap, Allocation Pie Chart, Efficient Frontier |

---

## Skills Demonstrated

- **Data Engineering** — automated ingestion pipeline from Yahoo Finance into PostgreSQL
- **Quantitative Finance** — VaR, CVaR, Sharpe ratio, drawdown, beta, stress testing
- **Machine Learning** — XGBoost & logistic regression classifiers, walk-forward time-series CV, threshold optimization, drift detection via PSI
- **Backend Development** — RESTful API design with FastAPI, SQLAlchemy ORM
- **Frontend Development** — interactive React dashboard with Recharts visualizations
- **Cloud & DevOps** — Docker containerization, AWS EC2 deployment, Nginx reverse proxy

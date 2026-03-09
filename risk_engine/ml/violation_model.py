from __future__ import annotations

import os
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import joblib


# ============================================================
# Database
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/risk_platform")
engine = create_engine(DATABASE_URL)


# ============================================================
# Model Storage
# ============================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _model_path(portfolio_id: int, model_type: str) -> Path:
    return MODEL_DIR / f"violation_{portfolio_id}_{model_type}.joblib"


# ============================================================
# JSON Safety
# ============================================================

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert NaN/Inf to None so FastAPI JSONResponse never crashes.
    Also converts numpy/pandas scalars/timestamps safely.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        return _safe_float(obj)

    # numpy scalar
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass

    # pandas Timestamp/date-like
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    # last resort
    try:
        v = float(obj)
        return _safe_float(v)
    except Exception:
        return str(obj)


# ============================================================
# Load Portfolio Returns
# ============================================================

def _load_portfolio_returns(portfolio_id: int) -> pd.Series:
    weights_query = """
        SELECT asset_id, weight
        FROM positions
        WHERE portfolio_id = %(pid)s;
    """
    weights_df = pd.read_sql(weights_query, engine, params={"pid": portfolio_id})
    if weights_df.empty:
        raise ValueError("No positions found for portfolio_id.")

    asset_ids = tuple(weights_df["asset_id"].tolist())
    # Ensure tuple formatting works for IN clause
    if len(asset_ids) == 1:
        asset_ids = (asset_ids[0], asset_ids[0])

    returns_query = f"""
        SELECT asset_id, date, daily_return
        FROM returns
        WHERE asset_id IN {asset_ids}
        ORDER BY date;
    """
    returns_df = pd.read_sql(returns_query, engine)
    if returns_df.empty:
        raise ValueError("No returns found for portfolio assets.")

    returns_df["date"] = pd.to_datetime(returns_df["date"])

    pivot = returns_df.pivot(index="date", columns="asset_id", values="daily_return")
    weights = weights_df.set_index("asset_id")["weight"]

    # align + drop rows with missing asset returns
    pivot = pivot[weights.index].dropna()
    portfolio_returns = pivot.dot(weights).sort_index()

    return portfolio_returns


# ============================================================
# Feature Engineering (Horizon-aware label)
# ============================================================

def _build_dataset(
    portfolio_returns: pd.Series,
    confidence: float,
    window: int,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """
    Creates features on day t and label indicating whether a VaR violation
    happens within the next `horizon_days` days.

    label(t) = 1 if min(return[t+1 ... t+horizon_days]) < VaR(t)
    """
    r = portfolio_returns.copy().dropna()

    var_t = r.rolling(window).quantile(1 - confidence)

    # forward min over horizon
    future_min = r.shift(-1).rolling(horizon_days).min()
    label = (future_min < var_t).astype(int)

    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1

    df = pd.DataFrame({
        "return": r,
        "vol_10": r.rolling(10).std(),
        "vol_20": r.rolling(20).std(),
        "vol_60": r.rolling(60).std(),
        "mean_5": r.rolling(5).mean(),
        "mean_20": r.rolling(20).mean(),
        "momentum_5": r.rolling(5).sum(),
        "momentum_10": r.rolling(10).sum(),
        "drawdown": drawdown,
        "skew_20": r.rolling(20).skew(),
        "kurt_20": r.rolling(20).kurt(),
        "label": label,
    })

    df = df.dropna()
    return df


# ============================================================
# Model Factory
# ============================================================

def _make_model(model_type: str, y_train: np.ndarray):
    if model_type == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=4000,
                class_weight={0: 1, 1: 6},
            ))
        ])

    if model_type == "xgb":
        # Lazy import so your server still runs if xgboost isn't installed.
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError(
                "xgboost is not usable in this environment. "
                "If you're on Mac, install OpenMP: `brew install libomp`, "
                "then reinstall xgboost in your venv."
            ) from e

        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        scale = (neg / pos) if pos > 0 else 1.0

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale,
            eval_metric="logloss",
            n_jobs=4,
        )

    raise ValueError("model_type must be 'logreg' or 'xgb'.")


# ============================================================
# Metrics + Threshold
# ============================================================

def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)

    return _json_safe({
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": cm,
    })


def _best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t = 0.35
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_t)


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    # Pipeline + sklearn models
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback (rare)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise RuntimeError("Model does not support probability outputs.")


def _save_bundle(portfolio_id: int, model_type: str, bundle: Dict[str, Any]) -> None:
    joblib.dump(bundle, _model_path(portfolio_id, model_type))


def _load_bundle(portfolio_id: int, model_type: str) -> Dict[str, Any]:
    path = _model_path(portfolio_id, model_type)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found at {path}. Run /ml/train/violation/{portfolio_id}?model_type={model_type} first."
        )
    return joblib.load(path)


# ============================================================
# DB Logging + History
# ============================================================

def _log_model_run_to_db(
    portfolio_id: int,
    model_type: str,
    horizon_days: int,
    metrics: Dict[str, Any],
    threshold: float,
) -> None:
    try:
        q = text("""
            INSERT INTO ml_model_runs (
                portfolio_id, model_type, horizon_days,
                roc_auc, precision, recall, f1, threshold
            )
            VALUES (
                :portfolio_id, :model_type, :horizon_days,
                :roc_auc, :precision, :recall, :f1, :threshold
            )
        """)
        with engine.begin() as conn:
            conn.execute(q, {
                "portfolio_id": portfolio_id,
                "model_type": model_type,
                "horizon_days": horizon_days,
                "roc_auc": metrics.get("roc_auc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "threshold": threshold,
            })
    except Exception as e:
        # Don’t crash the API if logging fails
        print(f"Warning: failed to log model run: {e}")


def get_model_run_history(portfolio_id: int) -> List[Dict[str, Any]]:
    q = """
        SELECT *
        FROM ml_model_runs
        WHERE portfolio_id = %(pid)s
        ORDER BY created_at DESC;
    """
    df = pd.read_sql(q, engine, params={"pid": portfolio_id})
    return df.to_dict(orient="records")


# ============================================================
# PUBLIC API: TRAIN
# ============================================================

def train_violation_model(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
    model_type: str = "logreg",
    horizon_days: int = 5,
) -> Dict[str, Any]:
    """
    Trains model, finds best F1 threshold on held-out split, saves bundle, logs DB run.
    """
    pr = _load_portfolio_returns(portfolio_id)
    df = _build_dataset(pr, confidence, window, horizon_days=horizon_days)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = _make_model(model_type, y_train)
    model.fit(X_train, y_train)

    y_prob = _predict_proba(model, X_test)
    best_t = _best_threshold_f1(y_test, y_prob)
    metrics = _binary_metrics(y_test, y_prob, best_t)

    bundle = {
        "model": model,
        "features": feature_cols,
        "threshold": best_t,
        "horizon_days": horizon_days,
        "confidence": confidence,
        "window": window,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }
    _save_bundle(portfolio_id, model_type, bundle)

    _log_model_run_to_db(
        portfolio_id=portfolio_id,
        model_type=model_type,
        horizon_days=horizon_days,
        metrics=metrics,
        threshold=best_t,
    )

    return _json_safe({
        "portfolio_id": portfolio_id,
        "model_type": model_type,
        "confidence": confidence,
        "window": window,
        "horizon_days": horizon_days,
        "metrics": metrics,
    })


# ============================================================
# PUBLIC API: PREDICT
# ============================================================

def predict_next_day_violation(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
    model_type: str = "logreg",
    threshold: float | None = None,
    use_best_f1_threshold: bool = True,
) -> Dict[str, Any]:
    """
    IMPORTANT: This signature matches what your main.py is calling.
    - If threshold is given -> use it
    - Else if use_best_f1_threshold -> use saved threshold from trained bundle
    - Else -> use 0.5
    """
    bundle = _load_bundle(portfolio_id, model_type)
    model = bundle["model"]
    features = bundle["features"]
    saved_threshold = float(bundle.get("threshold", 0.35))
    horizon_days = int(bundle.get("horizon_days", 5))

    pr = _load_portfolio_returns(portfolio_id)
    df = _build_dataset(pr, confidence, window, horizon_days=horizon_days)

    X = df[features].values
    latest_X = X[-1:].copy()

    prob = float(_predict_proba(model, latest_X)[0])

    if threshold is not None:
        used_t = float(threshold)
    elif use_best_f1_threshold:
        used_t = saved_threshold
    else:
        used_t = 0.5

    return _json_safe({
        "portfolio_id": portfolio_id,
        "model_type": model_type,
        "violation_probability": prob,
        "threshold_used": used_t,
        "predicted_violation": bool(prob >= used_t),
        "horizon_days": horizon_days,
    })


# ============================================================
# PUBLIC API: WALK-FORWARD VALIDATION
# ============================================================

def walkforward_validation(
    portfolio_id: int,
    model_type: str = "logreg",
    confidence: float = 0.95,
    window: int = 252,
    horizon_days: int = 5,
    train_min_days: int = 756,
    step_days: int = 21,
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Expanding-window walk-forward:
    Train on [0:train_end) -> test on next step_days.
    """
    pr = _load_portfolio_returns(portfolio_id)
    df = _build_dataset(pr, confidence, window, horizon_days=horizon_days).sort_index()

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    dates = df.index.astype(str).tolist()

    n = len(df)
    folds = []
    series = []
    all_true = []
    all_prob = []

    train_end = train_min_days

    while train_end + step_days <= n:
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:train_end + step_days], y[train_end:train_end + step_days]
        test_dates = dates[train_end:train_end + step_days]

        model = _make_model(model_type, y_train)
        model.fit(X_train, y_train)

        y_prob = _predict_proba(model, X_test)
        fold_metrics = _binary_metrics(y_test, y_prob, threshold)

        folds.append(_json_safe({
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "metrics": fold_metrics,
        }))

        for d, yt, yp in zip(test_dates, y_test, y_prob):
            series.append(_json_safe({
                "date": d,
                "y_true": int(yt),
                "y_prob": float(yp),
                "y_pred": int(float(yp) >= threshold),
            }))

        all_true.extend(y_test.tolist())
        all_prob.extend(y_prob.tolist())

        train_end += step_days

    overall = _binary_metrics(np.array(all_true), np.array(all_prob), threshold)

    return _json_safe({
        "portfolio_id": portfolio_id,
        "model_type": model_type,
        "confidence": confidence,
        "window": window,
        "horizon_days": horizon_days,
        "overall": overall,
        "folds": folds,
        "series": series,
    })


# ============================================================
# PUBLIC API: FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
    model_type: str = "logreg",
) -> Dict[str, Any]:
    """
    Uses saved model if exists, else trains quickly (so endpoint always works).
    Returns a sorted list of feature importances.
    """
    try:
        bundle = _load_bundle(portfolio_id, model_type)
    except Exception:
        bundle = train_violation_model(
            portfolio_id=portfolio_id,
            confidence=confidence,
            window=window,
            model_type=model_type,
        )
        bundle = _load_bundle(portfolio_id, model_type)

    model = bundle["model"]
    features = bundle["features"]

    importances = None

    # Pipeline(LogReg) -> last step has coef_
    if model_type == "logreg":
        try:
            clf = model.named_steps["clf"]
            coef = np.asarray(clf.coef_).reshape(-1)
            importances = np.abs(coef)
        except Exception:
            importances = None

    elif model_type == "xgb":
        try:
            importances = np.asarray(model.feature_importances_)
        except Exception:
            importances = None

    if importances is None:
        return _json_safe({
            "portfolio_id": portfolio_id,
            "model_type": model_type,
            "feature_importance": [],
            "note": "Could not extract feature importance for this model.",
        })

    rows = [{"feature": f, "importance": float(v)} for f, v in zip(features, importances)]
    rows = sorted(rows, key=lambda x: x["importance"], reverse=True)

    return _json_safe({
        "portfolio_id": portfolio_id,
        "model_type": model_type,
        "feature_importance": rows,
    })


# ============================================================
# PUBLIC API: THRESHOLDS
# ============================================================

def get_thresholds(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
    model_type: str = "logreg",
    horizon_days: int = 5,
) -> Dict[str, Any]:
    """
    Recomputes best threshold on the same 80/20 split approach.
    Returns best threshold by F1 plus a small table of candidates.
    """
    pr = _load_portfolio_returns(portfolio_id)
    df = _build_dataset(pr, confidence, window, horizon_days=horizon_days)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = _make_model(model_type, y_train)
    model.fit(X_train, y_train)
    y_prob = _predict_proba(model, X_test)

    candidates = []
    best_t = 0.35
    best_f1 = -1.0

    for t in [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60]:
        m = _binary_metrics(y_test, y_prob, float(t))
        candidates.append({
            "threshold": float(t),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
        if (m["f1"] is not None) and (float(m["f1"]) > best_f1):
            best_f1 = float(m["f1"])
            best_t = float(t)

    # also do full search
    best_t_full = _best_threshold_f1(y_test, y_prob)

    return _json_safe({
        "portfolio_id": portfolio_id,
        "model_type": model_type,
        "horizon_days": horizon_days,
        "best_threshold_f1_full_search": best_t_full,
        "best_threshold_from_candidates": best_t,
        "candidates": candidates,
    })


# ============================================================
# PUBLIC API: DRIFT REPORT (PSI)
# ============================================================

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index for 1 feature.
    PSI ~ 0: no drift, >0.1 moderate, >0.25 major.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    psi = np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))
    return float(psi)


def drift_report(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
    recent_days: int = 120,
    model_type: str = "logreg",
    horizon_days: int = 5,
) -> Dict[str, Any]:
    """
    Compares feature distributions in:
    - baseline: older history (excluding recent_days)
    - recent: last recent_days
    """
    pr = _load_portfolio_returns(portfolio_id)
    df = _build_dataset(pr, confidence, window, horizon_days=horizon_days).sort_index()

    feature_cols = [c for c in df.columns if c != "label"]

    if len(df) <= recent_days + 50:
        return _json_safe({
            "portfolio_id": portfolio_id,
            "note": "Not enough history to compute drift_report reliably.",
            "psi": [],
        })

    baseline = df.iloc[:-recent_days]
    recent = df.iloc[-recent_days:]

    rows = []
    for col in feature_cols:
        psi_val = _psi(baseline[col].values, recent[col].values, bins=10)
        rows.append({"feature": col, "psi": psi_val})

    rows = sorted(rows, key=lambda x: x["psi"], reverse=True)

    overall = float(np.mean([r["psi"] for r in rows])) if rows else 0.0

    return _json_safe({
        "portfolio_id": portfolio_id,
        "model_type": model_type,
        "horizon_days": horizon_days,
        "recent_days": recent_days,
        "overall_psi_mean": overall,
        "psi": rows,
    })


# ============================================================
# PUBLIC API: COMPARE MODELS
# ============================================================

def compare_violation_models(
    portfolio_id: int,
    confidence: float = 0.95,
    window: int = 252,
    horizon_days: int = 5,
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Trains both logreg and xgb on the same split and compares metrics at a fixed threshold.
    """
    pr = _load_portfolio_returns(portfolio_id)
    df = _build_dataset(pr, confidence, window, horizon_days=horizon_days)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    out = {}

    for mt in ["logreg", "xgb"]:
        try:
            model = _make_model(mt, y_train)
            model.fit(X_train, y_train)
            y_prob = _predict_proba(model, X_test)
            out[mt] = _binary_metrics(y_test, y_prob, threshold)
        except Exception as e:
            out[mt] = {"error": str(e)}

    return _json_safe({
        "portfolio_id": portfolio_id,
        "confidence": confidence,
        "window": window,
        "horizon_days": horizon_days,
        "threshold": threshold,
        "results": out,
    })
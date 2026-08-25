from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size == 0:
        return {"r2": math.nan, "rmse": math.nan, "mae": math.nan, "mape": math.nan}
    denom = np.maximum(np.abs(y_true), 1e-30)
    return {
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else math.nan,
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / denom))),
    }


def pearson_corr(x, y) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) < 1e-30 or np.std(y) < 1e-30:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])

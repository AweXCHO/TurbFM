from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .metrics import regression_metrics


EPS = 1e-30


@dataclass
class SymbolicResult:
    name: str
    equation: str
    coefficients: Dict[str, float]
    metrics: Dict[str, float]
    backend: str


def _positive_frame(df: pd.DataFrame, features: Sequence[str], target: str) -> pd.DataFrame:
    cols = list(features) + [target]
    clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    mask = np.ones(len(clean), dtype=bool)
    for col in cols:
        mask &= clean[col].to_numpy(dtype=float) > 0
    return clean.loc[mask].copy()


def _format_powerlaw(target: str, intercept: float, feature_names: Sequence[str], coefs: Sequence[float]) -> str:
    const = math.exp(intercept)
    parts = [f"{const:.8g}"]
    for name, coef in zip(feature_names, coefs):
        parts.append(f"{name}^{coef:.6g}")
    return f"{target} = " + " * ".join(parts)


def _finite_positive_target_frame(
    df: pd.DataFrame,
    positive_features: Sequence[str],
    signed_features: Sequence[str],
    target: str,
) -> pd.DataFrame:
    cols = list(positive_features) + list(signed_features) + [target]
    clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    mask = clean[target].to_numpy(dtype=float) > 0
    for col in positive_features:
        mask &= clean[col].to_numpy(dtype=float) > 0
    return clean.loc[mask].copy()


def _format_mixed_loglaw(
    target: str,
    intercept: float,
    positive_features: Sequence[str],
    signed_features: Sequence[str],
    positive_coefs: Sequence[float],
    signed_coefs: Sequence[float],
) -> str:
    const = math.exp(intercept)
    parts = [f"{const:.8g}"]
    for name, coef in zip(positive_features, positive_coefs):
        parts.append(f"{name}^{coef:.6g}")
    exp_terms = [f"{coef:.6g}*{name}" for name, coef in zip(signed_features, signed_coefs)]
    if exp_terms:
        parts.append("exp(" + " + ".join(exp_terms) + ")")
    return f"{target} = " + " * ".join(parts)


def fit_powerlaw(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    name: str,
    out_dir: str | Path,
) -> SymbolicResult:
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _positive_frame(train_df, features, target)
    test = _positive_frame(test_df, features, target)
    if train.empty or test.empty:
        raise ValueError(f"{name}: no positive finite rows for {features} -> {target}")

    x_train = np.log(np.maximum(train[list(features)].to_numpy(dtype=float), EPS))
    y_train = np.log(np.maximum(train[target].to_numpy(dtype=float), EPS))
    x_test = np.log(np.maximum(test[list(features)].to_numpy(dtype=float), EPS))
    y_test_raw = test[target].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x_train, y_train)
    log_pred = model.predict(x_test)
    y_pred = np.exp(log_pred)
    metrics = regression_metrics(y_test_raw, y_pred)
    equation = _format_powerlaw(target, float(model.intercept_), features, model.coef_)

    coeffs = {"intercept_log": float(model.intercept_), "constant": float(math.exp(model.intercept_))}
    coeffs.update({f"exp_{feature}": float(coef) for feature, coef in zip(features, model.coef_)})

    (out_dir / "powerlaw_equation.txt").write_text(equation + "\n", encoding="utf-8")
    pd.DataFrame([coeffs]).to_csv(out_dir / "powerlaw_coefficients.csv", index=False)
    (out_dir / "powerlaw_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = test.copy()
    pred_df[f"{target}_pred"] = y_pred
    pred_df.to_csv(out_dir / "powerlaw_test_predictions.csv", index=False)

    return SymbolicResult(
        name=name,
        equation=equation,
        coefficients=coeffs,
        metrics=metrics,
        backend="powerlaw",
    )


def fit_mixed_loglaw(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    positive_features: Sequence[str],
    signed_features: Sequence[str],
    target: str,
    name: str,
    out_dir: str | Path,
) -> SymbolicResult:
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _finite_positive_target_frame(train_df, positive_features, signed_features, target)
    test = _finite_positive_target_frame(test_df, positive_features, signed_features, target)
    if train.empty or test.empty:
        raise ValueError(
            f"{name}: no finite rows for positive={positive_features}, signed={signed_features} -> {target}"
        )

    x_train_pos = np.log(np.maximum(train[list(positive_features)].to_numpy(dtype=float), EPS))
    x_test_pos = np.log(np.maximum(test[list(positive_features)].to_numpy(dtype=float), EPS))
    x_train_signed = train[list(signed_features)].to_numpy(dtype=float)
    x_test_signed = test[list(signed_features)].to_numpy(dtype=float)
    x_train = np.concatenate([x_train_pos, x_train_signed], axis=1)
    x_test = np.concatenate([x_test_pos, x_test_signed], axis=1)
    y_train = np.log(np.maximum(train[target].to_numpy(dtype=float), EPS))
    y_test_raw = test[target].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x_train, y_train)
    log_pred = model.predict(x_test)
    y_pred = np.exp(log_pred)
    metrics = regression_metrics(y_test_raw, y_pred)
    n_pos = len(positive_features)
    pos_coefs = model.coef_[:n_pos]
    signed_coefs = model.coef_[n_pos:]
    equation = _format_mixed_loglaw(
        target,
        float(model.intercept_),
        positive_features,
        signed_features,
        pos_coefs,
        signed_coefs,
    )

    coeffs = {"intercept_log": float(model.intercept_), "constant": float(math.exp(model.intercept_))}
    coeffs.update({f"exp_{feature}": float(coef) for feature, coef in zip(positive_features, pos_coefs)})
    coeffs.update({f"log_coef_{feature}": float(coef) for feature, coef in zip(signed_features, signed_coefs)})

    (out_dir / "mixed_loglaw_equation.txt").write_text(equation + "\n", encoding="utf-8")
    pd.DataFrame([coeffs]).to_csv(out_dir / "mixed_loglaw_coefficients.csv", index=False)
    (out_dir / "mixed_loglaw_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = test.copy()
    pred_df[f"{target}_pred"] = y_pred
    pred_df.to_csv(out_dir / "mixed_loglaw_test_predictions.csv", index=False)

    return SymbolicResult(
        name=name,
        equation=equation,
        coefficients=coeffs,
        metrics=metrics,
        backend="mixed_loglaw",
    )


def fit_pysr_logspace(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    name: str,
    out_dir: str | Path,
    niterations: int = 2000,
    random_state: int = 0,
) -> Optional[SymbolicResult]:
    try:
        from pysr import PySRRegressor
    except Exception:
        return None

    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _positive_frame(train_df, features, target)
    test = _positive_frame(test_df, features, target)
    if train.empty or test.empty:
        return None

    x_train = np.log(np.maximum(train[list(features)].to_numpy(dtype=float), EPS))
    y_train = np.log(np.maximum(train[target].to_numpy(dtype=float), EPS))
    x_test = np.log(np.maximum(test[list(features)].to_numpy(dtype=float), EPS))
    y_test_raw = test[target].to_numpy(dtype=float)

    model = PySRRegressor(
        niterations=niterations,
        populations=20,
        population_size=50,
        maxsize=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        model_selection="best",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(x_train, y_train, variable_names=[f"log_{f}" for f in features])
    log_pred = model.predict(x_test)
    y_pred = np.exp(log_pred)
    metrics = regression_metrics(y_test_raw, y_pred)
    equation = str(model.get_best()["equation"])

    (out_dir / "pysr_equation.txt").write_text(equation + "\n", encoding="utf-8")
    (out_dir / "pysr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.equations_.to_csv(out_dir / "pysr_equations.csv", index=False)

    return SymbolicResult(
        name=name,
        equation=equation,
        coefficients={},
        metrics=metrics,
        backend="pysr_logspace",
    )


def run_symbolic_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    run_pysr: bool = True,
    pysr_iterations: int = 2000,
    random_state: int = 0,
) -> List[SymbolicResult]:
    tasks = [
        ("oracle_aoa", ["J", "D_aperture", "alpha"], "AOA_var"),
        ("oracle_disp", ["AOA_var", "Delta_theta"], "disp_var"),
    ]
    if {"J", "D_aperture", "AOA_proxy"}.issubset(train_df.columns):
        tasks.append(("oracle_aoa_proxy", ["J", "D_aperture"], "AOA_proxy"))
    if {"AOA_var", "AOA_proxy"}.issubset(train_df.columns):
        tasks.append(("oracle_aoa_proxy_from_AOA", ["AOA_var"], "AOA_proxy"))
    mixed_tasks = []
    if {"hJ", "D_aperture", "AOA_var"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_check", ["hJ", "D_aperture"], "AOA_var"))
    if {"J_latent", "D_aperture", "AOA_var"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_check_Jlatent", ["J_latent", "D_aperture"], "AOA_var"))
    if {"hJ", "D_aperture", "AOA_proxy"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_proxy", ["hJ", "D_aperture"], "AOA_proxy"))
    if {"J_latent", "D_aperture", "AOA_proxy"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_proxy_Jlatent", ["J_latent", "D_aperture"], "AOA_proxy"))
    if {"AOA_latent", "AOA_proxy"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_proxy_AOAlatent", ["AOA_latent"], "AOA_proxy"))
    if {"hJ", "D_aperture", "AOA_proxy_pred"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_proxy_pred", ["hJ", "D_aperture"], "AOA_proxy_pred"))
    if {"J_latent", "D_aperture", "AOA_proxy_pred"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_proxy_pred_Jlatent", ["J_latent", "D_aperture"], "AOA_proxy_pred"))
    if {"AOA_latent", "AOA_proxy_pred"}.issubset(train_df.columns):
        tasks.append(("latent_aoa_proxy_pred_AOAlatent", ["AOA_latent"], "AOA_proxy_pred"))
    if {"AOA_latent", "Delta_theta", "disp_var"}.issubset(train_df.columns):
        tasks.append(("latent_disp_AOAlatent", ["AOA_latent", "Delta_theta"], "disp_var"))
    if {"AOA_latent", "h3", "h4", "Delta_theta", "disp_var"}.issubset(train_df.columns):
        mixed_tasks.append(
            ("latent_disp_AOA_extra", ["AOA_latent", "Delta_theta"], ["h3", "h4"], "disp_var")
        )
    if {"hAOA", "Delta_theta", "disp_var"}.issubset(train_df.columns):
        tasks.append(("latent_disp_hAOA", ["hAOA", "Delta_theta"], "disp_var"))
    if {"hJ", "hAOA", "D_aperture", "Delta_theta", "disp_var"}.issubset(train_df.columns):
        tasks.append(("latent_disp_full", ["hJ", "hAOA", "D_aperture", "Delta_theta"], "disp_var"))

    results: List[SymbolicResult] = []
    for name, features, target in tasks:
        result = fit_powerlaw(train_df, test_df, features, target, name, out_dir)
        results.append(result)
        if run_pysr:
            pysr_result = fit_pysr_logspace(
                train_df,
                test_df,
                features,
                target,
                name,
                out_dir,
                niterations=pysr_iterations,
                random_state=random_state,
            )
            if pysr_result is not None:
                results.append(pysr_result)
    for name, positive_features, signed_features, target in mixed_tasks:
        result = fit_mixed_loglaw(
            train_df,
            test_df,
            positive_features,
            signed_features,
            target,
            name,
            out_dir,
        )
        results.append(result)

    rows = []
    for result in results:
        row = {
            "name": result.name,
            "backend": result.backend,
            "equation": result.equation,
            **result.metrics,
            **result.coefficients,
        }
        rows.append(row)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "symbolic_results.csv", index=False)
    return results

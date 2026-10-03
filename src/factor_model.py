from __future__ import annotations

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.portfolio_construction import portfolio_returns
from src.utils import ASSET_CLASS_MAP, RANDOM_SEED, month_end_index


FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
EQUITY_ETFS = [ticker for ticker, asset_class in ASSET_CLASS_MAP.items() if asset_class == "Equity"]


def generate_fallback_factors(index: pd.DatetimeIndex, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    dates = month_end_index(index)
    means = np.array([0.0060, 0.0012, 0.0010, 0.0008, 0.0006, 0.0020])
    vols = np.array([0.0450, 0.0250, 0.0260, 0.0200, 0.0180, 0.0008])
    corr = np.array(
        [
            [1.00, 0.25, -0.10, -0.20, -0.15, 0.00],
            [0.25, 1.00, -0.20, -0.10, -0.05, 0.00],
            [-0.10, -0.20, 1.00, 0.25, 0.35, 0.00],
            [-0.20, -0.10, 0.25, 1.00, 0.30, 0.00],
            [-0.15, -0.05, 0.35, 0.30, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        ]
    )
    covariance = np.outer(vols, vols) * corr
    values = rng.multivariate_normal(means, covariance, size=len(dates))
    factors = pd.DataFrame(values, index=dates, columns=FACTOR_COLUMNS)
    factors["RF"] = factors["RF"].clip(lower=0.0, upper=0.006)
    factors.index.name = "date"
    return factors.round(6)


def fetch_or_generate_factors(processed_dir: Path, returns_index: pd.DatetimeIndex, mode: str = "synthetic") -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "fama_french_factors.csv"
    if mode == "market":
        from pandas_datareader import data as pdr

        start = pd.Timestamp(returns_index.min()).to_pydatetime()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            factors_raw = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[0]
        factors = factors_raw.copy() / 100.0
        factors.index = factors.index.to_timestamp("M")
        factors = factors.reindex(month_end_index(returns_index))[FACTOR_COLUMNS]
        if factors.isna().any().any() or not np.isfinite(factors.to_numpy()).all():
            raise ValueError("Market mode needs complete finite factor coverage; synthetic fallback is disabled.")
        source = "Kenneth French Data Library via pandas-datareader"
    elif mode == "synthetic":
        factors = generate_fallback_factors(returns_index)
        source = "Deterministic synthetic factor generator"
    else:
        raise ValueError("Factor mode must be synthetic or market.")
    factors.to_csv(path)
    provenance = {"mode": mode, "source": source,
                  "source_reference": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html" if mode == "market" else "src/factor_model.py::generate_fallback_factors",
                  "seed": RANDOM_SEED + 7 if mode == "synthetic" else None,
                  "factor_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "interpretation": "Synthetic diagnostics" if mode == "synthetic" else "Historical factor diagnostics"}
    (processed_dir / "factor_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    factors.attrs["provenance"] = provenance
    return factors


def run_factor_regression(
    profile_name: str,
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    aligned = pd.concat([portfolio_returns.rename("portfolio_return"), factors], axis=1).dropna()
    if len(aligned) < 12:
        raise ValueError("Not enough overlapping observations for factor regression.")

    y = aligned["portfolio_return"] - aligned["RF"]
    x = sm.add_constant(aligned[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    model = sm.OLS(y, x).fit()
    rows = []
    name_map = {"const": "Alpha", "Mkt-RF": "Mkt-RF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA"}
    for factor in ["const", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        rows.append(
            {
                "profile_name": profile_name,
                "factor": name_map[factor],
                "coefficient": float(model.params[factor]),
                "t_stat": float(model.tvalues[factor]),
                "p_value": float(model.pvalues[factor]),
                "r_squared": float(model.rsquared),
                "observations": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def build_factor_exposure_summary(
    recommended_returns: dict[str, pd.Series],
    factors: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for profile_name, returns in recommended_returns.items():
        frames.append(run_factor_regression(profile_name, returns, factors))
    return pd.concat(frames, ignore_index=True)


def normalize_equity_sleeve_weights(weights: pd.Series | dict[str, float]) -> pd.Series:
    """Normalize equity ETF weights so the equity sleeve sums to 100%."""
    series = pd.Series(weights, dtype=float)
    equity_weights = series.reindex(EQUITY_ETFS).fillna(0.0)
    sleeve_total = float(equity_weights.sum())
    if sleeve_total <= 0:
        raise ValueError("Portfolio has no equity sleeve to normalize.")
    return equity_weights / sleeve_total


def calculate_equity_sleeve_returns(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float],
) -> pd.Series:
    sleeve_weights = normalize_equity_sleeve_weights(weights)
    available = [ticker for ticker in sleeve_weights.index if ticker in returns.columns]
    sleeve_returns = portfolio_returns(returns[available], sleeve_weights.reindex(available))
    sleeve_returns.name = "Equity Sleeve"
    return sleeve_returns


def build_equity_sleeve_factor_summary(
    returns: pd.DataFrame,
    recommended_weights: dict[str, pd.Series],
    factors: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for profile_name, weights in recommended_weights.items():
        sleeve_returns = calculate_equity_sleeve_returns(returns, weights)
        regression = run_factor_regression(profile_name, sleeve_returns, factors)
        regression.insert(1, "analysis_scope", "Equity Sleeve")
        frames.append(regression)
    return pd.concat(frames, ignore_index=True)

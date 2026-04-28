from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RANDOM_SEED = 42

ETF_UNIVERSE: dict[str, str] = {
    "SPY": "U.S. Equity",
    "QQQ": "U.S. Growth Equity",
    "IWM": "U.S. Small Cap",
    "EFA": "Developed International Equity",
    "EEM": "Emerging Markets Equity",
    "AGG": "Core Bonds",
    "TLT": "Long Treasury Bonds",
    "SHY": "Short Treasury Bonds",
    "LQD": "Investment Grade Credit",
    "HYG": "High Yield Credit",
    "VNQ": "REITs",
    "GLD": "Gold",
    "DBC": "Commodities",
    "SGOV": "Cash / T-Bill Proxy",
}

ASSET_CLASS_MAP: dict[str, str] = {
    "SPY": "Equity",
    "QQQ": "Equity",
    "IWM": "Equity",
    "EFA": "Equity",
    "EEM": "Equity",
    "AGG": "Fixed Income",
    "TLT": "Fixed Income",
    "SHY": "Fixed Income",
    "LQD": "Fixed Income",
    "HYG": "Fixed Income",
    "VNQ": "Alternatives",
    "GLD": "Alternatives",
    "DBC": "Alternatives",
    "SGOV": "Cash",
}

SCENARIO_BUCKET_MAP: dict[str, str] = {
    "SPY": "Equity",
    "QQQ": "Equity",
    "IWM": "Equity",
    "EFA": "Equity",
    "EEM": "Emerging Markets",
    "AGG": "Fixed Income",
    "TLT": "Treasuries",
    "SHY": "Treasuries",
    "LQD": "Investment Grade Credit",
    "HYG": "High Yield Credit",
    "VNQ": "Alternatives",
    "GLD": "Gold",
    "DBC": "Commodities",
    "SGOV": "Cash",
}

ASSET_CLASS_ORDER = ["Equity", "Fixed Income", "Alternatives", "Cash"]


def ensure_directories(project_root: Path) -> dict[str, Path]:
    """Create the project folders used by the pipeline."""
    paths = {
        "data": project_root / "data",
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
        "output": project_root / "output",
        "charts": project_root / "charts",
        "src": project_root / "src",
        "tests": project_root / "tests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def month_end_index(index: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    return pd.DatetimeIndex(dates).to_period("M").to_timestamp("M")


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    if denominator is None or abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def format_weight_dict(weights: pd.Series) -> str:
    clean = weights[weights.abs() > 1e-6].sort_values(ascending=False)
    return ", ".join(f"{ticker}: {weight:.1%}" for ticker, weight in clean.items())


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def latest_complete_month_end(today: pd.Timestamp | None = None) -> pd.Timestamp:
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    first_of_month = today.replace(day=1)
    return first_of_month - pd.offsets.Day(1)


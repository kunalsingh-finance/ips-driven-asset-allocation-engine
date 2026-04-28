from __future__ import annotations

from pathlib import Path

import pandas as pd


PROFILE_COLUMNS = [
    "profile_name",
    "return_objective",
    "max_volatility_target",
    "max_drawdown_tolerance",
    "min_equity_allocation",
    "max_equity_allocation",
    "min_fixed_income_allocation",
    "max_fixed_income_allocation",
    "min_alternatives_allocation",
    "max_alternatives_allocation",
    "min_cash_allocation",
    "max_cash_allocation",
    "max_single_etf_weight",
    "rebalancing_threshold",
    "benchmark_definition",
]


BENCHMARK_DEFINITIONS: dict[str, dict[str, float]] = {
    "Capital Preservation": {"SPY": 0.30, "AGG": 0.60, "SGOV": 0.10},
    "Balanced Growth": {"SPY": 0.60, "AGG": 0.40},
    "Long-Term Endowment Growth": {
        "SPY": 0.45,
        "EFA": 0.15,
        "EEM": 0.10,
        "AGG": 0.15,
        "VNQ": 0.05,
        "GLD": 0.05,
        "SGOV": 0.05,
    },
}


def _benchmark_to_text(definition: dict[str, float]) -> str:
    return "; ".join(f"{ticker}:{weight:.4f}" for ticker, weight in definition.items())


def create_synthetic_ips_profiles(raw_dir: Path) -> pd.DataFrame:
    """Create and save synthetic IPS profiles used by the project."""
    profiles = [
        {
            "profile_name": "Capital Preservation",
            "return_objective": "Preserve capital, maintain liquidity, and earn a modest return above cash.",
            "max_volatility_target": 0.08,
            "max_drawdown_tolerance": 0.15,
            "min_equity_allocation": 0.15,
            "max_equity_allocation": 0.25,
            "min_fixed_income_allocation": 0.55,
            "max_fixed_income_allocation": 0.75,
            "min_alternatives_allocation": 0.00,
            "max_alternatives_allocation": 0.10,
            "min_cash_allocation": 0.10,
            "max_cash_allocation": 0.25,
            "max_single_etf_weight": 0.25,
            "rebalancing_threshold": 0.03,
            "benchmark_definition": _benchmark_to_text(BENCHMARK_DEFINITIONS["Capital Preservation"]),
        },
        {
            "profile_name": "Balanced Growth",
            "return_objective": "Compound capital with a balanced mix of growth assets and high-quality ballast.",
            "max_volatility_target": 0.13,
            "max_drawdown_tolerance": 0.25,
            "min_equity_allocation": 0.45,
            "max_equity_allocation": 0.70,
            "min_fixed_income_allocation": 0.20,
            "max_fixed_income_allocation": 0.45,
            "min_alternatives_allocation": 0.05,
            "max_alternatives_allocation": 0.20,
            "min_cash_allocation": 0.00,
            "max_cash_allocation": 0.10,
            "max_single_etf_weight": 0.30,
            "rebalancing_threshold": 0.05,
            "benchmark_definition": _benchmark_to_text(BENCHMARK_DEFINITIONS["Balanced Growth"]),
        },
        {
            "profile_name": "Long-Term Endowment Growth",
            "return_objective": "Maximize long-term real return while accepting meaningful interim volatility.",
            "max_volatility_target": 0.17,
            "max_drawdown_tolerance": 0.35,
            "min_equity_allocation": 0.50,
            "max_equity_allocation": 0.80,
            "min_fixed_income_allocation": 0.10,
            "max_fixed_income_allocation": 0.30,
            "min_alternatives_allocation": 0.10,
            "max_alternatives_allocation": 0.15,
            "min_cash_allocation": 0.00,
            "max_cash_allocation": 0.10,
            "max_single_etf_weight": 0.25,
            "rebalancing_threshold": 0.05,
            "benchmark_definition": _benchmark_to_text(BENCHMARK_DEFINITIONS["Long-Term Endowment Growth"]),
        },
    ]
    df = pd.DataFrame(profiles, columns=PROFILE_COLUMNS)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "ips_profiles.csv", index=False)
    return df


def load_ips_profiles(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "ips_profiles.csv"
    if not path.exists():
        return create_synthetic_ips_profiles(raw_dir)
    return pd.read_csv(path)


def parse_benchmark_definition(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in str(text).split(";"):
        if not part.strip():
            continue
        ticker, value = part.strip().split(":")
        weights[ticker.strip()] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Benchmark definition has no positive weights.")
    return {ticker: weight / total for ticker, weight in weights.items()}


def profile_to_constraints(profile: pd.Series | dict) -> dict[str, tuple[float, float]]:
    p = dict(profile)
    return {
        "Equity": (float(p["min_equity_allocation"]), float(p["max_equity_allocation"])),
        "Fixed Income": (
            float(p["min_fixed_income_allocation"]),
            float(p["max_fixed_income_allocation"]),
        ),
        "Alternatives": (
            float(p["min_alternatives_allocation"]),
            float(p["max_alternatives_allocation"]),
        ),
        "Cash": (float(p["min_cash_allocation"]), float(p["max_cash_allocation"])),
    }

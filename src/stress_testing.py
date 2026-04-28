from __future__ import annotations

import pandas as pd

from src.portfolio_construction import benchmark_weights
from src.utils import ASSET_CLASS_MAP, SCENARIO_BUCKET_MAP


SCENARIOS: dict[str, dict[str, float]] = {
    "Equity Bear Market": {
        "Equity": -0.25,
        "Fixed Income": 0.04,
        "Credit": -0.08,
        "Alternatives": -0.10,
        "Gold": 0.08,
        "Cash": 0.00,
    },
    "Rate Shock": {
        "Equity": -0.06,
        "Fixed Income": -0.12,
        "Credit": -0.07,
        "Alternatives": -0.05,
        "Gold": 0.02,
        "Cash": 0.00,
    },
    "Inflation Shock": {
        "Equity": -0.08,
        "Fixed Income": -0.07,
        "Credit": -0.05,
        "Alternatives": 0.06,
        "Gold": 0.08,
        "Commodities": 0.10,
        "Cash": 0.00,
    },
    "Credit Spread Widening": {
        "Equity": -0.10,
        "Treasuries": 0.03,
        "Investment Grade Credit": -0.06,
        "High Yield Credit": -0.12,
        "Alternatives": -0.07,
        "Cash": 0.00,
    },
    "Risk-On Rally": {
        "Equity": 0.15,
        "Fixed Income": -0.02,
        "Credit": 0.05,
        "Alternatives": 0.06,
        "Gold": -0.03,
        "Cash": 0.00,
    },
    "Dollar Liquidity Shock": {
        "Equity": -0.12,
        "Fixed Income": 0.02,
        "Credit": -0.10,
        "Emerging Markets": -0.18,
        "Gold": 0.05,
        "Cash": 0.00,
    },
}


def _scenario_return_for_ticker(ticker: str, scenario: dict[str, float]) -> float:
    bucket = SCENARIO_BUCKET_MAP[ticker]
    if bucket in scenario:
        return scenario[bucket]
    if bucket in {"Investment Grade Credit", "High Yield Credit"} and "Credit" in scenario:
        return scenario["Credit"]
    if bucket in {"Treasuries"} and "Fixed Income" in scenario:
        return scenario["Fixed Income"]
    if bucket == "Emerging Markets" and "Equity" in scenario:
        return scenario["Equity"]
    asset_class = ASSET_CLASS_MAP[ticker]
    return scenario.get(asset_class, 0.0)


def run_stress_tests(
    recommended_weights: dict[str, pd.Series],
    profiles: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    rows = []
    profile_lookup = {row["profile_name"]: row for _, row in profiles.iterrows()}
    for profile_name, weights in recommended_weights.items():
        profile = profile_lookup[profile_name]
        bench_weights = benchmark_weights(profile, tickers)
        for scenario_name, scenario in SCENARIOS.items():
            ticker_returns = pd.Series(
                {ticker: _scenario_return_for_ticker(ticker, scenario) for ticker in tickers}
            )
            portfolio_contrib = weights.reindex(tickers).fillna(0.0) * ticker_returns
            benchmark_contrib = bench_weights.reindex(tickers).fillna(0.0) * ticker_returns
            portfolio_return = float(portfolio_contrib.sum())
            benchmark_return = float(benchmark_contrib.sum())
            asset_class_contrib = portfolio_contrib.groupby(
                pd.Series({ticker: ASSET_CLASS_MAP[ticker] for ticker in tickers})
            ).sum()
            for ticker in tickers:
                rows.append(
                    {
                        "profile_name": profile_name,
                        "scenario_name": scenario_name,
                        "ticker": ticker,
                        "asset_class": ASSET_CLASS_MAP[ticker],
                        "scenario_return": float(ticker_returns[ticker]),
                        "portfolio_weight": float(weights.get(ticker, 0.0)),
                        "benchmark_weight": float(bench_weights.get(ticker, 0.0)),
                        "etf_contribution": float(portfolio_contrib[ticker]),
                        "benchmark_etf_contribution": float(benchmark_contrib[ticker]),
                        "asset_class_contribution": float(asset_class_contrib[ASSET_CLASS_MAP[ticker]]),
                        "portfolio_scenario_return": portfolio_return,
                        "benchmark_scenario_return": benchmark_return,
                        "relative_performance": portfolio_return - benchmark_return,
                    }
                )
    return pd.DataFrame(rows)


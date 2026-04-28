from __future__ import annotations

import pandas as pd

from src.factor_model import normalize_equity_sleeve_weights
from src.fetch_data import generate_fallback_monthly_returns
from src.governance import validate_recommended_constraints
from src.ips_profiles import create_synthetic_ips_profiles
from src.portfolio_construction import build_benchmark_returns, build_portfolio_candidates, portfolio_returns
from src.risk_analytics import build_rolling_risk_metrics, calculate_risk_contribution


def _balanced_growth_setup(tmp_path):
    returns = generate_fallback_monthly_returns(periods=72)
    profiles = create_synthetic_ips_profiles(tmp_path)
    profile = profiles[profiles["profile_name"] == "Balanced Growth"].iloc[0]
    benchmark = build_benchmark_returns(returns, profile)
    weights = build_portfolio_candidates(returns, profile, benchmark)["IPS Recommended Portfolio"]
    return returns, profile, benchmark, weights


def test_constraints_validation_passes_for_recommended_allocation(tmp_path) -> None:
    _, profile, _, weights = _balanced_growth_setup(tmp_path)

    validation = validate_recommended_constraints(profile, weights)

    assert set(validation["pass_fail"]) == {"PASS"}


def test_constraints_validation_detects_failure(tmp_path) -> None:
    _, profile, _, weights = _balanced_growth_setup(tmp_path)
    bad_weights = weights.copy()
    increase = 0.20 - float(bad_weights["HYG"])
    bad_weights["HYG"] = 0.20
    bad_weights["AGG"] -= increase

    validation = validate_recommended_constraints(profile, bad_weights)

    assert "FAIL" in set(validation["pass_fail"])


def test_risk_contribution_sums_to_approximately_100_percent(tmp_path) -> None:
    returns, _, _, weights = _balanced_growth_setup(tmp_path)

    contribution = calculate_risk_contribution(returns, weights)

    assert abs(float(contribution["percent_risk_contribution"].sum()) - 1.0) < 1e-6


def test_equity_sleeve_weights_normalize_to_100_percent(tmp_path) -> None:
    _, _, _, weights = _balanced_growth_setup(tmp_path)

    equity_weights = normalize_equity_sleeve_weights(weights)

    assert abs(float(equity_weights.sum()) - 1.0) < 1e-9
    assert set(equity_weights.index) == {"SPY", "QQQ", "IWM", "EFA", "EEM"}


def test_rolling_metrics_are_not_empty(tmp_path) -> None:
    returns, _, benchmark, weights = _balanced_growth_setup(tmp_path)
    portfolio = portfolio_returns(returns, weights)

    rolling = build_rolling_risk_metrics(portfolio, benchmark)

    assert not rolling.empty
    assert {"Recommended Portfolio", "Benchmark"} == set(rolling["series"])
    assert rolling[["rolling_12m_return", "rolling_12m_volatility", "rolling_max_drawdown"]].notna().all().all()

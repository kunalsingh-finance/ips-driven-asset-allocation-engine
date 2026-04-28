from __future__ import annotations

from src.fetch_data import generate_fallback_monthly_returns
from src.ips_profiles import create_synthetic_ips_profiles, profile_to_constraints
from src.portfolio_construction import (
    build_benchmark_returns,
    build_portfolio_candidates,
    class_allocation,
)


def test_optimizer_respects_core_constraints(tmp_path) -> None:
    returns = generate_fallback_monthly_returns(periods=72)
    profiles = create_synthetic_ips_profiles(tmp_path)
    profile = profiles[profiles["profile_name"] == "Balanced Growth"].iloc[0]
    benchmark = build_benchmark_returns(returns, profile)

    candidates = build_portfolio_candidates(returns, profile, benchmark)

    max_single = float(profile["max_single_etf_weight"])
    constraints = profile_to_constraints(profile)
    for weights in candidates.values():
        assert abs(float(weights.sum()) - 1.0) < 1e-6
        assert float(weights.min()) >= -1e-8
        assert float(weights.max()) <= max_single + 1e-6
        allocation = class_allocation(weights)
        for asset_class, (lower, upper) in constraints.items():
            assert allocation[asset_class] >= lower - 1e-5
            assert allocation[asset_class] <= upper + 1e-5


def test_balanced_growth_recommended_portfolio_has_core_exposures(tmp_path) -> None:
    returns = generate_fallback_monthly_returns(periods=72)
    profiles = create_synthetic_ips_profiles(tmp_path)
    profile = profiles[profiles["profile_name"] == "Balanced Growth"].iloc[0]
    benchmark = build_benchmark_returns(returns, profile)

    weights = build_portfolio_candidates(returns, profile, benchmark)["IPS Recommended Portfolio"]

    assert weights["SPY"] >= 0.15 - 1e-6
    assert weights["AGG"] >= 0.15 - 1e-6
    assert weights["SGOV"] >= 0.05 - 1e-6
    assert weights["HYG"] <= 0.10 + 1e-6
    assert weights["GLD"] <= 0.075 + 1e-6


def test_endowment_recommended_portfolio_limits_international_and_alternatives(tmp_path) -> None:
    returns = generate_fallback_monthly_returns(periods=72)
    profiles = create_synthetic_ips_profiles(tmp_path)
    profile = profiles[profiles["profile_name"] == "Long-Term Endowment Growth"].iloc[0]
    benchmark = build_benchmark_returns(returns, profile)

    weights = build_portfolio_candidates(returns, profile, benchmark)["IPS Recommended Portfolio"]

    assert weights[["EFA", "EEM"]].sum() <= 0.25 + 1e-6
    assert weights[["VNQ", "GLD", "DBC"]].sum() <= 0.15 + 1e-6
    assert weights["DBC"] >= 0.02 - 1e-6

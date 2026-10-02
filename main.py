from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd

from src.factor_model import build_equity_sleeve_factor_summary, build_factor_exposure_summary, fetch_or_generate_factors
from src.fetch_data import fetch_or_generate_monthly_returns
from src.generate_report import generate_charts, generate_excel_report, generate_investment_memo
from src.governance import build_etf_allocation_rationale, validate_recommended_constraints
from src.ips_profiles import create_synthetic_ips_profiles
from src.portfolio_construction import (
    build_benchmark_returns,
    build_portfolio_candidates,
    portfolio_returns,
    sample_feasible_portfolios,
    weights_to_frame,
)
from src.rebalancing import build_rebalancing_trades
from src.risk_analytics import build_risk_summary, build_rolling_risk_metrics, calculate_risk_contribution
from src.stress_testing import run_stress_tests
from src.utils import ensure_directories


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_PROFILE_NAME = "Balanced Growth"


def main() -> None:
    parser = argparse.ArgumentParser(description="IPS allocation demonstration with one explicit source mode")
    parser.add_argument("--data-mode", choices=["synthetic", "market"], default="synthetic")
    parser.add_argument("--end", default="2026-05-31", help="Fixed synthetic month-end or market download end date")
    args = parser.parse_args()
    paths = ensure_directories(PROJECT_ROOT)

    profiles = create_synthetic_ips_profiles(paths["raw"])
    monthly_returns = fetch_or_generate_monthly_returns(paths["processed"], paths["raw"], end=args.end, mode=args.data_mode)
    monthly_returns.index = pd.to_datetime(monthly_returns.index)

    benchmark_returns: dict[str, pd.Series] = {}
    all_weights: dict[str, dict[str, pd.Series]] = {}

    for _, profile in profiles.iterrows():
        profile_name = str(profile["profile_name"])
        benchmark = build_benchmark_returns(monthly_returns, profile)
        benchmark_returns[profile_name] = benchmark
        all_weights[profile_name] = build_portfolio_candidates(monthly_returns, profile, benchmark)

    weights_frame = weights_to_frame(all_weights)
    weights_frame.to_csv(paths["output"] / "portfolio_weights.csv", index=False)

    risk_summary = build_risk_summary(monthly_returns, all_weights, benchmark_returns)
    risk_summary.to_csv(paths["output"] / "portfolio_risk_summary.csv", index=False)

    recommended_weights = {
        profile_name: portfolios["IPS Recommended Portfolio"] for profile_name, portfolios in all_weights.items()
    }
    recommended_allocation = weights_to_frame(
        {profile_name: {"IPS Recommended Portfolio": weights} for profile_name, weights in recommended_weights.items()}
    )
    recommended_allocation.to_csv(paths["output"] / "recommended_allocation.csv", index=False)

    recommended_returns = {
        profile_name: portfolio_returns(monthly_returns, weights)
        for profile_name, weights in recommended_weights.items()
    }
    factors = fetch_or_generate_factors(paths["processed"], monthly_returns.index, mode=args.data_mode)
    provenance = {"mode": args.data_mode, "returns": monthly_returns.attrs["provenance"],
                  "factors": factors.attrs["provenance"],
                  "evaluation": "Same-sample allocation and factor diagnostics; no out-of-sample performance claim"}
    (paths["output"] / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    factor_summary = build_factor_exposure_summary(recommended_returns, factors)
    factor_summary.to_csv(paths["output"] / "factor_exposure_summary.csv", index=False)
    equity_factor_summary = build_equity_sleeve_factor_summary(monthly_returns, recommended_weights, factors)
    equity_factor_summary.to_csv(paths["output"] / "equity_sleeve_factor_summary.csv", index=False)

    stress_results = run_stress_tests(recommended_weights, profiles, list(monthly_returns.columns))
    stress_results.to_csv(paths["output"] / "stress_test_results.csv", index=False)

    rebalancing_trades = build_rebalancing_trades(recommended_weights, profiles)
    rebalancing_trades.to_csv(paths["output"] / "rebalancing_trades.csv", index=False)

    primary_profile = profiles[profiles["profile_name"] == PRIMARY_PROFILE_NAME]
    if primary_profile.empty:
        primary_profile = profiles.head(1)
    primary_profile = primary_profile.iloc[0]
    primary_name = str(primary_profile["profile_name"])

    efficient_frontier = sample_feasible_portfolios(monthly_returns, primary_profile)
    efficient_frontier.to_csv(paths["output"] / "efficient_frontier_samples.csv", index=False)
    primary_recommended_returns = recommended_returns[primary_name]
    primary_benchmark_returns = benchmark_returns[primary_name]
    risk_contribution = calculate_risk_contribution(monthly_returns, recommended_weights[primary_name])
    risk_contribution.to_csv(paths["output"] / "risk_contribution_summary.csv", index=False)
    rolling_risk_metrics = build_rolling_risk_metrics(primary_recommended_returns, primary_benchmark_returns)
    rolling_risk_metrics.to_csv(paths["output"] / "rolling_risk_metrics.csv", index=False)
    etf_rationale = build_etf_allocation_rationale(recommended_weights[primary_name])
    etf_rationale.to_csv(paths["output"] / "etf_allocation_rationale.csv", index=False)
    constraints_validation = validate_recommended_constraints(primary_profile, recommended_weights[primary_name])
    constraints_validation.to_csv(paths["output"] / "constraints_validation.csv", index=False)

    chart_paths = generate_charts(
        paths["charts"],
        primary_name,
        monthly_returns,
        all_weights[primary_name],
        benchmark_returns[primary_name],
        risk_summary,
        factor_summary,
        equity_factor_summary,
        risk_contribution,
        rolling_risk_metrics,
        stress_results,
        rebalancing_trades,
        efficient_frontier,
    )

    memo_path = generate_investment_memo(
        paths["output"] / "investment_memo.md",
        primary_profile,
        recommended_weights[primary_name],
        risk_summary,
        factor_summary,
        equity_factor_summary,
        risk_contribution,
        rolling_risk_metrics,
        etf_rationale,
        constraints_validation,
        stress_results,
        rebalancing_trades,
        provenance=provenance,
    )

    excel_report = generate_excel_report(
        paths["output"] / "investment_committee_report.xlsx",
        primary_name,
        profiles,
        all_weights,
        risk_summary,
        factor_summary,
        equity_factor_summary,
        risk_contribution,
        rolling_risk_metrics,
        etf_rationale,
        constraints_validation,
        stress_results,
        rebalancing_trades,
        efficient_frontier,
        chart_paths,
        provenance=provenance,
    )

    print(f"Pipeline complete. Source mode: {args.data_mode}; see output/data_provenance.json.")
    print(f"Monthly returns: {paths['processed'] / 'monthly_returns.csv'}")
    print(f"Portfolio risk summary: {paths['output'] / 'portfolio_risk_summary.csv'}")
    print(f"Factor exposure summary: {paths['output'] / 'factor_exposure_summary.csv'}")
    print(f"Equity sleeve factor summary: {paths['output'] / 'equity_sleeve_factor_summary.csv'}")
    print(f"Risk contribution summary: {paths['output'] / 'risk_contribution_summary.csv'}")
    print(f"Rolling risk metrics: {paths['output'] / 'rolling_risk_metrics.csv'}")
    print(f"ETF rationale: {paths['output'] / 'etf_allocation_rationale.csv'}")
    print(f"Constraints validation: {paths['output'] / 'constraints_validation.csv'}")
    print(f"Stress test results: {paths['output'] / 'stress_test_results.csv'}")
    print(f"Rebalancing trades: {paths['output'] / 'rebalancing_trades.csv'}")
    print(f"Excel report: {excel_report}")
    print(f"Investment memo: {memo_path}")
    print(f"Charts directory: {paths['charts']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import shutil
import tempfile

import numpy as np

import pandas as pd

from src.factor_model import FACTOR_COLUMNS, build_equity_sleeve_factor_summary, build_factor_exposure_summary, fetch_or_generate_factors
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
from src.publication import publication_attempt, publish_pack, verify_pack
from src.risk_analytics import build_risk_summary, build_rolling_risk_metrics, calculate_risk_contribution
from src.stress_testing import run_stress_tests
from src.utils import ETF_UNIVERSE, ensure_directories


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_PROFILE_NAME = "Balanced Growth"


def _build_pack(stage_root: Path, run_id: str, mode: str, end: str) -> None:
    """Generate everything in a private stage; providers never write to the live pack."""
    paths = ensure_directories(stage_root)

    profiles = create_synthetic_ips_profiles(paths["raw"])
    monthly_returns = fetch_or_generate_monthly_returns(paths["processed"], paths["raw"], end=end, mode=mode)
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    factors = fetch_or_generate_factors(paths["processed"], monthly_returns.index, mode=mode)
    requested_end = pd.Timestamp(end)
    complete_end = requested_end if requested_end.is_month_end else requested_end.to_period("M").start_time - pd.Timedelta(days=1)
    if (len(monthly_returns) < 36 or set(monthly_returns.columns) != set(ETF_UNIVERSE)
            or set(factors.columns) != set(FACTOR_COLUMNS)
            or monthly_returns.columns.has_duplicates or factors.columns.has_duplicates
            or monthly_returns.index.hasnans or monthly_returns.index.has_duplicates
            or not monthly_returns.index.is_monotonic_increasing
            or not monthly_returns.index.is_month_end.all()
            or monthly_returns.index.max() > complete_end
            or not np.isfinite(monthly_returns.to_numpy()).all()
            or not factors.index.equals(monthly_returns.index)
            or not np.isfinite(factors.to_numpy()).all()
            or monthly_returns.attrs.get("provenance", {}).get("mode") != mode
            or factors.attrs.get("provenance", {}).get("mode") != mode):
        raise ValueError("Returns and factors must be complete, aligned, finite inputs of the declared mode.")
    provenance = {"run_id": run_id, "mode": mode, "requested_end": end,
                  "start_date": str(monthly_returns.index.min().date()),
                  "end_date": str(monthly_returns.index.max().date()),
                  "complete_months": len(monthly_returns),
                  "returns": monthly_returns.attrs["provenance"],
                  "factors": factors.attrs["provenance"],
                  "evaluation": "Same-sample allocation and factor diagnostics; no out-of-sample performance claim"}
    (paths["output"] / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

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
        {name: Path("charts") / path.name for name, path in chart_paths.items()},
        provenance=provenance,
    )

    # Portable chart references and publication identifiers survive stage removal.
    with memo_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## Publication\n\nRun id: `{run_id}`. "
                     f"Input sample: {provenance['start_date']} to {provenance['end_date']} "
                     f"({provenance['complete_months']} complete months); requested end: {end}.\n")
def run_pipeline(project_root: Path, mode: str = "synthetic", end: str = "2026-05-31") -> dict:
    """Publish a complete pack or preserve the previous pack and record failure."""
    if mode not in {"synthetic", "market"}:
        raise ValueError("Data mode must be synthetic or market.")
    if not isinstance(end, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
        raise ValueError("End must be a valid calendar date in YYYY-MM-DD form.")
    timestamp = pd.Timestamp(end)
    if pd.isna(timestamp) or timestamp != timestamp.normalize():
        raise ValueError("End must be a valid calendar date in YYYY-MM-DD form.")
    project_root = Path(project_root).resolve()
    with publication_attempt(project_root, mode, end) as run_id:
        stage_root = Path(tempfile.mkdtemp(prefix=f".ips-stage-{run_id}-", dir=project_root))
        try:
            _build_pack(stage_root, run_id, mode, end)
            return publish_pack(stage_root, project_root, run_id, mode, end)
        finally:
            # This directory was created above, inside the resolved project root.
            # Cleanup cannot mask a provider/publication exception or invalidate SUCCESS.
            if stage_root.resolve().parent == project_root and not stage_root.is_symlink():
                shutil.rmtree(stage_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="IPS allocation demonstration with one explicit source mode")
    parser.add_argument("--data-mode", choices=["synthetic", "market"], default="synthetic")
    parser.add_argument("--end", default="2026-05-31", help="Fixed synthetic month-end or inclusive market download end date")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT, help="Root for generated data, reports, and charts")
    parser.add_argument("--verify-only", action="store_true", help="Verify the latest successful complete pack without fetching data")
    args = parser.parse_args()
    project_root = args.project_dir.resolve()
    try:
        manifest = verify_pack(project_root) if args.verify_only else run_pipeline(project_root, args.data_mode, args.end)
    except Exception as exc:
        raise SystemExit(f"IPS publication blocked: {exc}") from exc
    if args.verify_only:
        print(f"Verified IPS run {manifest['run_id']}; mode: {manifest['mode']}; {len(manifest['files'])} bound files.")
        return
    paths = {"processed": project_root / "data/processed", "output": project_root / "output", "charts": project_root / "charts"}
    memo_path = paths["output"] / "investment_memo.md"
    excel_report = paths["output"] / "investment_committee_report.xlsx"

    print(f"Pipeline complete. Source mode: {args.data_mode}; run: {manifest['run_id']}; verified complete pack.")
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

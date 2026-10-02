# IPS-Driven Multi-Asset Allocation & Rebalancing Engine

This project simulates an institutional investment management workflow for building, monitoring, and rebalancing a multi-asset ETF portfolio.

The engine converts an Investment Policy Statement (IPS) profile into portfolio constraints, constructs a recommended allocation, validates policy limits, compares the portfolio against a benchmark, analyzes risk contribution, runs stress tests, monitors drift, generates rebalancing trades, and produces an investment committee-style Excel report and memo.

The project is educational and does not represent financial advice, a live investment recommendation, or production investment management software.

## Screenshots

These screenshots illustrate an earlier workbook layout. The regenerated report
adds **Data Provenance** and **Source Summary** sheets; inspect those sheets and
`output/data_provenance.json` to establish the current source mode.

### Executive Summary

![Executive Summary](docs/screenshots/executive_summary.png)

### Recommended Allocation

![Recommended Allocation](docs/screenshots/recommended_allocation.png)

### Risk Contribution

![Risk Contribution](docs/screenshots/risk_contribution.png)

### Stress Test Results

![Stress Test Results](docs/screenshots/stress_test_results.png)

### Rebalancing Trades

![Rebalancing Trades](docs/screenshots/rebalancing_trades.png)

## What This Project Does

- Builds a Balanced Growth portfolio using IPS constraints
- Allocates across equity, fixed income, alternatives, and cash
- Validates allocation rules such as max ETF weight, SPY/AGG anchors, HYG cap, and asset-class ranges
- Calculates return, volatility, Sharpe ratio, Sortino ratio, drawdown, tracking error, information ratio, VaR, and CVaR
- Compares the portfolio against a 60/40 SPY/AGG benchmark
- Estimates ETF-level risk contribution using the covariance matrix
- Runs full-portfolio and equity-sleeve factor diagnostics
- Stress-tests the allocation under equity bear market, rate shock, inflation shock, credit-spread widening, and risk-on scenarios
- Monitors policy drift and generates rebalancing trade recommendations
- Produces an Excel investment committee report and investment memo

## Selected Results

The figures below are **fully synthetic same-sample demonstration results** from
the fixed seed 42, 132-month dataset ending May 31, 2026. They are neither realized
investment performance nor out-of-sample evidence. Synthetic factors use seed 49.

The final Balanced Growth model allocation is:

- Equity: 45.0%
- Fixed Income: 40.0%
- Alternatives: 8.6%
- Cash: 6.4%

Key portfolio metrics:

- Annualized return: 3.21%
- Annualized volatility: 9.13%
- Max drawdown: -15.63%
- Tracking error vs benchmark: 4.75%
- 95% historical VaR: -3.76%
- 95% historical CVaR: -5.04%

Risk contribution analysis identifies SPY, QQQ, and EFA as the largest estimated contributors to portfolio volatility. The rebalancing engine identifies 3 drift breaches and generates trade actions for SPY, AGG, and HYG.

## Investment Workflow

The project follows a portfolio analyst workflow:

1. Create synthetic IPS profiles.
2. Choose one explicit data mode: fully synthetic offline data or complete market data. No silent fallback mixes the modes.
3. Convert daily prices to month-end ETF returns.
4. Build IPS benchmark returns.
5. Construct constrained portfolio candidates.
6. Select an IPS Recommended Portfolio.
7. Validate IPS constraints.
8. Calculate risk and return metrics.
9. Estimate risk contribution from the covariance matrix.
10. Run full-portfolio and equity-sleeve factor diagnostics.
11. Apply market stress scenarios.
12. Generate rebalancing trades from synthetic drifted current weights.
13. Produce charts, CSV outputs, an Excel report, and a committee memo.

## ETF Universe

The allocation universe includes:

- SPY, QQQ, IWM, EFA, EEM for equity exposure
- AGG, TLT, SHY, LQD, HYG for fixed income and credit exposure
- VNQ, GLD, DBC for alternatives and real assets
- SGOV as a cash / T-bill proxy

## Methodology

Portfolio construction uses `scipy.optimize` with no shorting, full-investment constraints, IPS asset-class ranges, ETF-level minimums and caps, and concentration penalties.

The recommended allocation is not positioned as an attempt to beat a benchmark. It is positioned as an IPS-driven allocation that balances policy fit, diversification, downside awareness, benchmark tracking, and governance.

## Outputs

Key output files:

- `output/investment_committee_report.xlsx`
- `output/investment_memo.md`
- `output/recommended_allocation.csv`
- `output/portfolio_risk_summary.csv`
- `output/risk_contribution_summary.csv`
- `output/stress_test_results.csv`
- `output/rebalancing_trades.csv`
- `output/constraints_validation.csv`

Key charts:

- `charts/recommended_allocation.png`
- `charts/cumulative_performance_vs_benchmark.png`
- `charts/drawdown_comparison.png`
- `charts/risk_contribution.png`
- `charts/stress_test_results.png`
- `charts/policy_drift.png`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py --data-mode synthetic --end 2026-05-31
```

Run tests:

```bash
pytest -q
```

## Project Structure

```text
ips-driven-asset-allocation-engine/
├── README.md
├── requirements.txt
├── main.py
├── src/
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── output/
├── charts/
└── docs/
    └── screenshots/
```

## Data modes and provenance

The default `synthetic` mode makes no data requests. ETF and factor observations
are generated separately from fixed seeds, labeled in the memo/charts/workbook,
and exported with per-asset provenance and exact saved-input hashes. The fixed
end date supports repeatable demonstration output; use `--end` to change it.

`python main.py --data-mode market --end 2026-05-31` requires every ETF from
Yahoo Finance adjusted closes and complete Kenneth French factor coverage. Missing ETFs or factor data
stop the run, rather than introducing synthetic replacements. Incomplete common
return months are dropped and recorded; missing returns and benchmarks are not
filled with zero. `--end` is inclusive; a mid-month end excludes that partial
month from monthly diagnostics. Yahoo's adjusted prices account for distributions
and splits according to its vendor convention; these mutable source responses
are not a fund accounting or independently reconciled total-return series.
Stooq's previous download endpoint returned HTTP 404 during the October 2026
functional check, so market mode now uses the explicit Yahoo source above.

Inspect `output/data_provenance.json`, `data/processed/monthly_returns_provenance.json`
and `data/processed/factor_provenance.json`. IPS profiles and rebalancing drift are
synthetic in either mode. Allocations are estimated and evaluated on the same
sample; no chronological out-of-sample performance claim is made. A failed market
run does not validate artifacts from an earlier successful run.

## Limitations

- The declared synthetic or market sample may not represent future conditions.
- Optimizer outputs are sensitive to return samples, covariance estimates, and constraints.
- ETF proxies do not capture full fund due diligence, liquidity review, taxes, or account-level restrictions.
- Stress tests are deterministic approximations and should not be treated as comprehensive scenario analysis.
- This is an educational workflow, not financial advice or production investment software.

## Future Improvements

- Separate equity-sleeve and fixed-income-sleeve attribution
- Add transaction-lot-level rebalancing
- Add Black-Litterman expected return inputs
- Add regime-aware optimization
- Add tax-aware rebalancing
- Add liquidity scoring
- Add manager/fund due diligence layer

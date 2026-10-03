# Investment Committee Memo

## Data provenance

Source mode: **synthetic**. ETF source: Deterministic synthetic generator. Factor source: Deterministic synthetic factor generator. No market/synthetic asset substitution or zero return filling is applied. Same-sample allocation and factor diagnostics; no out-of-sample performance claim. Inspect data_provenance.json and the workbook's Data Provenance sheet for every asset.

## Executive Recommendation

This educational analysis recommends the **Balanced Growth** IPS allocation as a model portfolio for committee review. The recommendation is framed as an educational investment workflow, not financial advice, not a live investment recommendation, and not production investment software.

The allocation is designed to remain within IPS ranges, preserve core benchmark anchors, limit high-yield and single-ETF concentration, and maintain practical diversification across growth, defensive, real asset, and liquidity exposures.

## IPS Constraints

The IPS objective is: Compound capital with a balanced mix of growth assets and high-quality ballast.

- Maximum volatility target: 13.0%
- Maximum drawdown tolerance: 25.0%
- Maximum single ETF weight: 30.0%
- Rebalancing threshold: 5.0%
- Benchmark definition: SPY:0.6000; AGG:0.4000

The recommended allocation passed the IPS constraints validation.

## Recommended Allocation

Recommended ETF allocation:

AGG: 15.0%, SPY: 15.0%, EFA: 11.7%, QQQ: 9.9%, TLT: 9.6%, EEM: 7.2%, SGOV: 6.4%, LQD: 6.2%, GLD: 5.3%, SHY: 4.9%, HYG: 4.3%, VNQ: 3.3%, IWM: 1.2%

Asset-class allocation:

Equity: 45.0%, Fixed Income: 40.0%, Alternatives: 8.6%, Cash: 6.4%

## Allocation Rationale by Sleeve

**Equity sleeve:** The equity allocation combines core U.S. equity exposure through SPY, a measured growth tilt through QQQ, small-cap exposure through IWM, and non-U.S. diversification through EFA and EEM. The international and emerging market allocation is capped to avoid excessive concentration in one non-U.S. risk source.

**Fixed income sleeve:** AGG provides the core bond ballast. TLT adds duration exposure that may help in equity-led growth shocks, while SHY dampens rate sensitivity. LQD and HYG add credit income, with HYG capped because high yield can behave like equity during spread widening.

**Alternatives sleeve:** VNQ and GLD provide real asset and crisis-hedge exposure. DBC remains available as a commodity sleeve where policy allows, but the Balanced Growth model does not force a commodity allocation when the optimizer and constraints do not require it.

**Cash sleeve:** SGOV serves as the cash/T-bill proxy. This sleeve supports liquidity, reduces forced selling risk, and provides dry powder for rebalancing.

Sleeve weights are approximately Equity 45.0%, Fixed Income 40.0%, Alternatives 8.6%, and Cash 6.4%.

## Risk and Return Profile

- Annualized return: 3.21%
- Annualized volatility: 9.13%
- Sharpe ratio: 0.13
- Sortino ratio: 0.24
- Max drawdown: -15.63%
- Tracking error vs benchmark: 4.75%
- Information ratio: 0.56
- 95% historical VaR: -3.76%
- 95% historical CVaR: -5.04%

Rolling monitoring shows the latest 12-month portfolio return at 27.06%, volatility at 9.47%, and rolling drawdown at -2.81%. The latest benchmark readings are 22.70%, 10.01%, and -2.61%, respectively. This monitoring view is intended to show how risk changes over time rather than relying only on full-period averages.

## Benchmark Comparison

The recommended portfolio is compared against the IPS benchmark definition. Its beta to the benchmark is 0.81, with correlation of 0.88. Tracking error of 4.75% reflects active allocation choices across growth equity, international equity, duration, credit, alternatives, and cash. The benchmark remains the reference point, but the recommended allocation deliberately broadens the sources of risk and return.

## Risk Contribution Summary

Risk contribution analysis estimates which ETFs contribute most to total portfolio volatility using the covariance matrix and target weights. The largest estimated contributors are SPY (22.9%), QQQ (19.3%), EFA (19.2%). Risk contribution is reasonably distributed across multiple ETFs. This view is useful because a moderate weight can still contribute materially to risk if its volatility or covariance with the rest of the portfolio is high.

## Factor Exposure Interpretation

The factor regression shows limited explanatory power for the full multi-asset portfolio, which is expected because the portfolio includes bonds, credit, gold, cash, and alternatives. The full-portfolio regression R-squared is 0.02. These results should be treated as a diagnostic rather than a complete explanation of total portfolio returns.

## Equity-Sleeve Factor Analysis

The equity-sleeve regression normalizes only the equity ETF weights to 100% and runs the same Fama-French regression on that equity-only return stream. This is more appropriate for explaining equity return drivers because the regression is no longer diluted by bonds, cash, gold, and alternatives. The equity-sleeve regression R-squared is 0.01. A future version would extend this approach with separate fixed-income attribution for duration, curve, and credit-spread exposures.

## Stress-Test Summary

The weakest modeled scenario is **Equity Bear Market**, with an estimated portfolio return of -10.82% versus benchmark return of -13.40%. This is a realistic reminder that diversification can reduce, but not eliminate, broad market losses. The strongest modeled scenario is **Risk-On Rally**, with an estimated portfolio return of 6.73%. The least favorable relative scenario is **Risk-On Rally**, where the portfolio trails the benchmark by 1.47%.

## Rebalancing Actions

Synthetic current weights were generated to demonstrate policy monitoring. 3 ETFs breached the IPS drift threshold. The resulting trades move breached positions back to target weights while leaving smaller drifts untouched.

- SPY: Sell $6,000,000; drift 6.00%; target 15.00%; estimated cost $3,000
- AGG: Buy $5,500,000; drift -5.50%; target 15.00%; estimated cost $2,750
- HYG: Sell $5,500,000; drift 5.50%; target 4.33%; estimated cost $2,750

Estimated transaction costs are $8,500 on a $100,000,000 illustrative portfolio.

## Key Risks

- The declared source mode is a sample assumption; synthetic results are not historical market performance and neither mode predicts future outcomes.
- Optimization is sensitive to expected returns, covariance estimates, and the selected risk-free-rate assumption.
- ETF proxies simplify implementation and do not capture manager selection, taxes, liquidity tiers, or mandate-specific restrictions.
- Stress tests are deterministic approximations and should be expanded for real committee use.
- Factor models are less informative for full multi-asset portfolios than for isolated equity sleeves.

## Limitations

This project is an educational portfolio analytics workflow. It is not a substitute for fiduciary review, due diligence, legal review, tax analysis, or live investment governance.

## Future Improvements

- Separate equity-sleeve and fixed-income-sleeve attribution.
- Transaction-lot-level rebalancing.
- Black-Litterman expected return inputs.
- Regime-aware optimization.
- Tax-aware rebalancing.
- Liquidity scoring.
- Manager and fund due diligence layer.


## Publication

Run id: `6dcc8d9852ce4a228f2e6607227bf609`. Input sample: 2015-06-30 to 2026-05-31 (132 complete months); requested end: 2026-05-31.

from __future__ import annotations

import pandas as pd

from src.ips_profiles import profile_to_constraints
from src.portfolio_construction import ETF_WEIGHT_LIMITS, class_allocation
from src.utils import ASSET_CLASS_MAP


ETF_RATIONALE: dict[str, dict[str, str]] = {
    "SPY": {
        "portfolio_role": "Core U.S. equity exposure",
        "reason_for_inclusion": "Broad large-cap U.S. equity beta and primary growth anchor.",
        "key_risk": "Equity market drawdowns and valuation compression.",
    },
    "QQQ": {
        "portfolio_role": "Growth equity tilt",
        "reason_for_inclusion": "Adds exposure to growth-oriented and technology-heavy U.S. companies.",
        "key_risk": "Higher valuation sensitivity and sector concentration.",
    },
    "IWM": {
        "portfolio_role": "Small-cap exposure",
        "reason_for_inclusion": "Diversifies U.S. equity exposure beyond large-cap stocks.",
        "key_risk": "Higher cyclicality and liquidity sensitivity.",
    },
    "EFA": {
        "portfolio_role": "Developed international diversification",
        "reason_for_inclusion": "Broadens equity exposure outside the United States.",
        "key_risk": "Currency exposure and non-U.S. economic weakness.",
    },
    "EEM": {
        "portfolio_role": "Emerging market growth/risk exposure",
        "reason_for_inclusion": "Adds long-term emerging market growth potential within a capped sleeve.",
        "key_risk": "Political, currency, liquidity, and drawdown risk.",
    },
    "AGG": {
        "portfolio_role": "Core bond ballast",
        "reason_for_inclusion": "Provides diversified investment-grade bond exposure and portfolio ballast.",
        "key_risk": "Interest-rate risk and spread widening.",
    },
    "TLT": {
        "portfolio_role": "Duration / equity drawdown hedge",
        "reason_for_inclusion": "Adds long-duration Treasury exposure that may help in growth shocks.",
        "key_risk": "Rate shock and inflation sensitivity.",
    },
    "SHY": {
        "portfolio_role": "Short-duration stability",
        "reason_for_inclusion": "Reduces duration risk and supports liquidity within fixed income.",
        "key_risk": "Lower return potential if risk assets rally.",
    },
    "LQD": {
        "portfolio_role": "Investment-grade credit income",
        "reason_for_inclusion": "Adds credit income while remaining higher quality than high yield.",
        "key_risk": "Credit spread widening and rate sensitivity.",
    },
    "HYG": {
        "portfolio_role": "Capped high-yield credit exposure",
        "reason_for_inclusion": "Provides incremental income potential with an explicit cap.",
        "key_risk": "Can behave like equity during credit stress.",
    },
    "VNQ": {
        "portfolio_role": "Real estate / real asset exposure",
        "reason_for_inclusion": "Adds listed real estate exposure and potential inflation sensitivity.",
        "key_risk": "Rate sensitivity and real estate sector drawdowns.",
    },
    "GLD": {
        "portfolio_role": "Gold / inflation and crisis hedge",
        "reason_for_inclusion": "Diversifies against equity, currency, and inflation shocks.",
        "key_risk": "No income generation and potential real-rate sensitivity.",
    },
    "DBC": {
        "portfolio_role": "Commodity real asset exposure",
        "reason_for_inclusion": "Adds inflation-sensitive commodity exposure where the IPS permits it.",
        "key_risk": "Commodity volatility and roll-yield drag.",
    },
    "SGOV": {
        "portfolio_role": "Cash / liquidity / T-bill proxy",
        "reason_for_inclusion": "Provides liquidity, stability, and rebalancing flexibility.",
        "key_risk": "Reinvestment risk and opportunity cost in risk-on markets.",
    },
}


def build_etf_allocation_rationale(target_weights: pd.Series | dict[str, float]) -> pd.DataFrame:
    weights = pd.Series(target_weights, dtype=float)
    rows = []
    for ticker, target_weight in weights.items():
        rationale = ETF_RATIONALE.get(
            ticker,
            {
                "portfolio_role": "Portfolio exposure",
                "reason_for_inclusion": "Included as part of the model ETF universe.",
                "key_risk": "ETF-specific and market risk.",
            },
        )
        rows.append(
            {
                "ticker": ticker,
                "asset_class": ASSET_CLASS_MAP[ticker],
                "target_weight": float(target_weight),
                "portfolio_role": rationale["portfolio_role"],
                "reason_for_inclusion": rationale["reason_for_inclusion"],
                "key_risk": rationale["key_risk"],
            }
        )
    return pd.DataFrame(rows)


def _pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _add_row(
    rows: list[dict[str, object]],
    constraint_name: str,
    rule: str,
    actual_value: float,
    limit_value: str,
    condition: bool,
    comment: str,
) -> None:
    rows.append(
        {
            "constraint_name": constraint_name,
            "rule": rule,
            "actual_value": float(actual_value),
            "limit_value": limit_value,
            "pass_fail": _pass_fail(condition),
            "comment": comment,
        }
    )


def validate_recommended_constraints(
    profile: pd.Series | dict,
    target_weights: pd.Series | dict[str, float],
) -> pd.DataFrame:
    profile_dict = dict(profile)
    profile_name = str(profile_dict["profile_name"])
    weights = pd.Series(target_weights, dtype=float)
    rows: list[dict[str, object]] = []
    tolerance = 1e-6

    _add_row(
        rows,
        "Weights sum to 100%",
        "Total recommended weights must equal 100%.",
        weights.sum(),
        "100.00%",
        abs(float(weights.sum()) - 1.0) <= tolerance,
        "Confirms the allocation is fully invested.",
    )
    _add_row(
        rows,
        "No negative weights",
        "All ETF weights must be greater than or equal to 0%.",
        weights.min(),
        ">= 0.00%",
        float(weights.min()) >= -tolerance,
        "Confirms no short positions are used.",
    )
    max_single = float(profile_dict["max_single_etf_weight"])
    _add_row(
        rows,
        "Max single ETF weight not breached",
        "Each ETF must remain within the IPS single-position limit.",
        weights.max(),
        f"<= {max_single:.2%}",
        float(weights.max()) <= max_single + tolerance,
        "Limits concentration in any one ETF.",
    )

    allocations = class_allocation(weights)
    class_label = {
        "Equity": "equity",
        "Fixed Income": "fixed income",
        "Alternatives": "alternatives",
        "Cash": "cash",
    }
    for asset_class, (lower, upper) in profile_to_constraints(profile_dict).items():
        actual = allocations.get(asset_class, 0.0)
        _add_row(
            rows,
            f"{asset_class} allocation within IPS range",
            f"{asset_class} allocation must remain within the IPS range.",
            actual,
            f"{lower:.2%} - {upper:.2%}",
            actual >= lower - tolerance and actual <= upper + tolerance,
            f"Recommended {class_label[asset_class]} exposure is within policy.",
        )

    limits = ETF_WEIGHT_LIMITS.get(profile_name, {})
    named_checks = [
        ("SPY minimum satisfied", "SPY", "minimum"),
        ("AGG minimum satisfied", "AGG", "minimum"),
        ("HYG cap satisfied", "HYG", "maximum"),
        ("EFA cap satisfied", "EFA", "maximum"),
        ("TLT cap satisfied", "TLT", "maximum"),
        ("GLD cap satisfied", "GLD", "maximum"),
    ]
    for constraint_name, ticker, direction in named_checks:
        lower, upper = limits.get(ticker, (None, None))
        actual = float(weights.get(ticker, 0.0))
        if direction == "minimum":
            limit = 0.0 if lower is None else float(lower)
            condition = actual >= limit - tolerance
            rule = f"{ticker} weight must be at least {limit:.2%}."
            limit_text = f">= {limit:.2%}"
        else:
            limit = 1.0 if upper is None else float(upper)
            condition = actual <= limit + tolerance
            rule = f"{ticker} weight must not exceed {limit:.2%}."
            limit_text = f"<= {limit:.2%}"
        _add_row(
            rows,
            constraint_name,
            rule,
            actual,
            limit_text,
            condition,
            f"{ticker} policy test is {'satisfied' if condition else 'not satisfied'}.",
        )

    return pd.DataFrame(rows)

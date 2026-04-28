from __future__ import annotations

import pandas as pd

from src.rebalancing import create_rebalancing_trades
from src.rebalancing import generate_balanced_growth_demo_current_weights


def test_rebalance_drift_detection() -> None:
    target = pd.Series({"SPY": 0.60, "AGG": 0.40})
    current = pd.Series({"SPY": 0.52, "AGG": 0.48})

    trades = create_rebalancing_trades(
        profile_name="Balanced Growth",
        target_weights=target,
        current_weights=current,
        rebalance_threshold=0.05,
        portfolio_market_value=100_000_000,
    )

    spy = trades[trades["ticker"] == "SPY"].iloc[0]
    agg = trades[trades["ticker"] == "AGG"].iloc[0]
    assert bool(spy["drift_breach_flag"])
    assert spy["trade_direction"] == "Buy"
    assert bool(agg["drift_breach_flag"])
    assert agg["trade_direction"] == "Sell"


def test_balanced_growth_demo_creates_three_intentional_breaches() -> None:
    target = pd.Series({"SPY": 0.15, "AGG": 0.15, "HYG": 0.05, "QQQ": 0.15, "SGOV": 0.05, "EFA": 0.45})
    current = generate_balanced_growth_demo_current_weights(target)

    trades = create_rebalancing_trades(
        profile_name="Balanced Growth",
        target_weights=target,
        current_weights=current,
        rebalance_threshold=0.05,
        portfolio_market_value=100_000_000,
    )

    breached = trades[trades["drift_breach_flag"]]
    assert set(breached["ticker"]) == {"SPY", "AGG", "HYG"}
    assert trades.set_index("ticker").loc["SPY", "trade_direction"] == "Sell"
    assert trades.set_index("ticker").loc["AGG", "trade_direction"] == "Buy"
    assert trades.set_index("ticker").loc["HYG", "trade_direction"] == "Sell"


def test_transaction_cost_calculation() -> None:
    target = pd.Series({"SPY": 0.60, "AGG": 0.40})
    current = pd.Series({"SPY": 0.50, "AGG": 0.50})

    trades = create_rebalancing_trades(
        profile_name="Balanced Growth",
        target_weights=target,
        current_weights=current,
        rebalance_threshold=0.05,
        portfolio_market_value=100_000_000,
        transaction_cost_bps=5.0,
    )

    spy = trades[trades["ticker"] == "SPY"].iloc[0]
    assert round(float(spy["trade_amount"]), 2) == 10_000_000.00
    assert round(float(spy["estimated_transaction_cost"]), 2) == 5_000.00

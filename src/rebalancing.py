from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ASSET_CLASS_MAP, RANDOM_SEED


def generate_current_weights(
    target_weights: pd.Series,
    drift_scale: float = 0.035,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    drift = pd.Series(rng.normal(0.0, drift_scale, size=len(target_weights)), index=target_weights.index)
    current = (target_weights + drift).clip(lower=0.0)
    if current.sum() == 0:
        return target_weights.copy()
    current = current / current.sum()
    return current


def generate_balanced_growth_demo_current_weights(target_weights: pd.Series) -> pd.Series:
    """Create intentional drift breaches for the committee demo profile."""
    current = target_weights.copy()
    adjustments = {"SPY": 0.060, "AGG": -0.055, "HYG": 0.055}
    for ticker, adjustment in adjustments.items():
        if ticker in current.index:
            current[ticker] = max(0.0, current[ticker] + adjustment)

    protected = set(adjustments)
    difference = float(current.sum() - 1.0)
    if abs(difference) > 1e-12:
        candidates = [ticker for ticker in current.index if ticker not in protected]
        if difference > 0:
            room = current.loc[candidates].clip(lower=0.0)
            if room.sum() > 0:
                current.loc[candidates] -= room / room.sum() * difference
        else:
            room = (target_weights.loc[candidates] + 0.03 - current.loc[candidates]).clip(lower=0.0)
            if room.sum() > 0:
                current.loc[candidates] += room / room.sum() * abs(difference)

    current = current.clip(lower=0.0)
    return current / current.sum()


def create_rebalancing_trades(
    profile_name: str,
    target_weights: pd.Series,
    rebalance_threshold: float,
    current_weights: pd.Series | None = None,
    portfolio_market_value: float = 100_000_000.0,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    current = generate_current_weights(target_weights) if current_weights is None else current_weights.copy()
    current = current.reindex(target_weights.index).fillna(0.0)
    current = current / current.sum()
    target = target_weights / target_weights.sum()
    rows = []
    for ticker in target.index:
        drift = float(current[ticker] - target[ticker])
        breach = abs(drift) >= rebalance_threshold
        trade_amount = float((target[ticker] - current[ticker]) * portfolio_market_value) if breach else 0.0
        if trade_amount > 1:
            direction = "Buy"
        elif trade_amount < -1:
            direction = "Sell"
        else:
            direction = "Hold"
        post_trade_weight = float(target[ticker]) if breach else float(current[ticker])
        estimated_cost = abs(trade_amount) * transaction_cost_bps / 10_000.0
        rows.append(
            {
                "profile_name": profile_name,
                "ticker": ticker,
                "asset_class": ASSET_CLASS_MAP[ticker],
                "current_weight": float(current[ticker]),
                "target_weight": float(target[ticker]),
                "drift": drift,
                "drift_breach_flag": bool(breach),
                "trade_direction": direction,
                "trade_amount": trade_amount,
                "estimated_transaction_cost": estimated_cost,
                "post_trade_weight": post_trade_weight,
            }
        )
    return pd.DataFrame(rows)


def build_rebalancing_trades(
    recommended_weights: dict[str, pd.Series],
    profiles: pd.DataFrame,
    portfolio_market_value: float = 100_000_000.0,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    rows = []
    profile_lookup = {row["profile_name"]: row for _, row in profiles.iterrows()}
    for index, (profile_name, weights) in enumerate(recommended_weights.items()):
        profile = profile_lookup[profile_name]
        if profile_name == "Balanced Growth":
            current = generate_balanced_growth_demo_current_weights(weights)
        else:
            current = generate_current_weights(weights, seed=RANDOM_SEED + index)
        rows.append(
            create_rebalancing_trades(
                profile_name=profile_name,
                target_weights=weights,
                current_weights=current,
                rebalance_threshold=float(profile["rebalancing_threshold"]),
                portfolio_market_value=portfolio_market_value,
                transaction_cost_bps=transaction_cost_bps,
            )
        )
    return pd.concat(rows, ignore_index=True)

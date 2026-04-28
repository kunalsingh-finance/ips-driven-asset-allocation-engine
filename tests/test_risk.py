from __future__ import annotations

import pandas as pd

from src.risk_analytics import historical_var, max_drawdown, sharpe_ratio


def test_sharpe_ratio_calculation_is_positive_for_positive_series() -> None:
    returns = pd.Series([0.01, 0.015, 0.005, 0.012, 0.008, 0.018] * 4)

    assert sharpe_ratio(returns, risk_free_rate=0.0) > 0


def test_max_drawdown_calculation() -> None:
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])

    assert round(max_drawdown(returns), 6) == -0.244


def test_historical_var_calculation() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.02, 0.04])

    assert historical_var(returns, confidence=0.95) <= -0.05

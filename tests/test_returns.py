from __future__ import annotations

import pandas as pd

from src.fetch_data import calculate_monthly_returns
from src.portfolio_construction import portfolio_returns


def test_monthly_return_calculation() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-29"])
    prices = pd.DataFrame({"SPY": [95.0, 100.0, 104.0, 110.0, 99.0]}, index=dates)

    returns = calculate_monthly_returns(prices)

    assert returns.index[-1] == pd.Timestamp("2024-03-31")
    assert round(float(returns.loc["2024-02-29", "SPY"]), 6) == 0.10
    assert round(float(returns.loc["2024-03-31", "SPY"]), 6) == -0.10


def test_portfolio_return_calculation() -> None:
    returns = pd.DataFrame({"SPY": [0.10, -0.05], "AGG": [0.02, 0.01]})
    weights = {"SPY": 0.60, "AGG": 0.40}

    portfolio = portfolio_returns(returns, weights)

    assert round(float(portfolio.iloc[0]), 6) == 0.068
    assert round(float(portfolio.iloc[1]), 6) == -0.026


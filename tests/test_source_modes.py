import json
import numpy as np
import pandas as pd
import pytest

from src.fetch_data import calculate_monthly_returns, fetch_or_generate_monthly_returns
from src.factor_model import fetch_or_generate_factors
from src.utils import ETF_UNIVERSE
from src.portfolio_construction import portfolio_returns
from src.risk_analytics import calculate_risk_metrics


def test_synthetic_mode_is_offline_reproducible_and_per_asset_labeled(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic mode attempted market data")
    monkeypatch.setattr("src.fetch_data.fetch_daily_prices", forbidden)
    first = fetch_or_generate_monthly_returns(tmp_path / "p1", tmp_path / "r1")
    second = fetch_or_generate_monthly_returns(tmp_path / "p2", tmp_path / "r2")
    pd.testing.assert_frame_equal(first, second)
    manifest = json.loads((tmp_path / "p1/monthly_returns_provenance.json").read_text())
    assert manifest["mode"] == "synthetic"
    assert set(manifest["assets"]) == set(ETF_UNIVERSE)
    assert all(item["synthetic"] for item in manifest["assets"].values())
    assert manifest["monthly_returns_sha256"] == second.attrs["provenance"]["monthly_returns_sha256"]
    factors = fetch_or_generate_factors(tmp_path / "p1", first.index, mode="synthetic")
    assert factors.attrs["provenance"]["mode"] == "synthetic"


def test_market_mode_refuses_missing_assets_without_fallback(monkeypatch, tmp_path):
    prices = pd.DataFrame({"SPY": [100, 101]}, index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    monkeypatch.setattr("src.fetch_data.fetch_daily_prices", lambda **kwargs: prices)
    with pytest.raises(ValueError, match="requires every ETF"):
        fetch_or_generate_monthly_returns(tmp_path / "p", tmp_path / "r", mode="market")
    assert not (tmp_path / "p/monthly_returns.csv").exists()


def test_market_mode_drops_incomplete_months_and_records_them(monkeypatch, tmp_path):
    dates = pd.date_range("2020-01-31", periods=42, freq="ME")
    prices = pd.DataFrame({asset: 100 * 1.01 ** np.arange(len(dates)) for asset in ETF_UNIVERSE}, index=dates)
    prices.loc[dates[15], "AGG"] = np.nan
    monkeypatch.setattr("src.fetch_data.fetch_daily_prices", lambda **kwargs: prices)
    result = fetch_or_generate_monthly_returns(tmp_path / "p", tmp_path / "r", mode="market")
    assert dates[15] not in result.index and dates[16] not in result.index
    assert not result.isna().any().any()
    assert not result.eq(0).any().any()
    assert result.attrs["provenance"]["dropped_incomplete_months"] == [str(dates[i].date()) for i in (15, 16)]
    assert all(not item["synthetic"] for item in result.attrs["provenance"]["assets"].values())


def test_missing_month_does_not_become_zero_return():
    prices = pd.DataFrame({"SPY": [100, np.nan, 110]}, index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    assert calculate_monthly_returns(prices).empty


def test_portfolio_missing_inputs_cannot_be_zero_filled():
    with pytest.raises(ValueError, match="complete finite"):
        portfolio_returns(pd.DataFrame({"SPY": [0.1, np.nan]}), {"SPY": 1.0})
    with pytest.raises(ValueError, match="weighted assets"):
        portfolio_returns(pd.DataFrame({"SPY": [0.1]}), {"AGG": 1.0})


def test_missing_benchmark_month_cannot_be_zero_filled():
    returns = pd.DataFrame({"SPY": [0.1, 0.2]}, index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    benchmark = pd.Series([0.01], index=returns.index[:1])
    with pytest.raises(ValueError, match="Benchmark coverage"):
        calculate_risk_metrics("Demo", "Portfolio", returns, pd.Series({"SPY": 1.0}), benchmark)

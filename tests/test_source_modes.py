import json
import numpy as np
import pandas as pd
import pytest

from src.fetch_data import calculate_monthly_returns, fetch_daily_prices, fetch_or_generate_monthly_returns
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


@pytest.mark.parametrize("multi_index", [False, True])
def test_yahoo_market_download_uses_adjusted_prices_and_inclusive_end(monkeypatch, multi_index):
    index = pd.to_datetime(["2024-01-30", "2024-01-31"])
    raw = pd.DataFrame({"Adj Close": [100.0, 101.0], "Close": [100.0, 99.0]}, index=index)
    if multi_index:
        raw.columns = pd.MultiIndex.from_tuples([(field, "SPY") for field in raw.columns])
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setattr("yfinance.download", download)
    result = fetch_daily_prices(["SPY"], start="2024-01-01", end="2024-01-31")
    assert result["SPY"].tolist() == [100.0, 101.0]
    assert calls[0][1]["end"] == "2024-02-01"
    assert calls[0][1]["auto_adjust"] is False


def test_yahoo_unadjusted_close_cannot_silently_replace_adjusted_prices(monkeypatch):
    raw = pd.DataFrame({"Close": [100.0, 99.0]}, index=pd.date_range("2024-01-01", periods=2))
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="missing adjusted closes"):
        fetch_daily_prices(["SPY"], start="2024-01-01", end="2024-01-31")


def test_market_midmonth_end_excludes_partial_month_and_labels_yahoo(monkeypatch, tmp_path):
    dates = pd.date_range("2020-01-31", periods=42, freq="ME")
    prices = pd.DataFrame({asset: 100 * 1.01 ** np.arange(len(dates)) for asset in ETF_UNIVERSE}, index=dates)
    # The last available observation is in the middle of June, not a June month-end.
    prices.index = dates[:-1].append(pd.DatetimeIndex(["2023-06-15"]))
    monkeypatch.setattr("src.fetch_data.fetch_daily_prices", lambda **kwargs: prices)
    result = fetch_or_generate_monthly_returns(tmp_path / "p", tmp_path / "r", end="2023-06-15", mode="market")
    assert result.index.max() == pd.Timestamp("2023-05-31")
    provenance = result.attrs["provenance"]
    assert provenance["dropped_incomplete_months"] == ["2023-06-30"]
    assert provenance["assets"]["SGOV"]["source"] == "Yahoo Finance via yfinance"
    assert provenance["assets"]["SGOV"]["symbol"] == "SGOV"
    assert "Adj Close" in provenance["assets"]["SGOV"]["return_convention"]


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

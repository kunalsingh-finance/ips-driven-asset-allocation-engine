from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ASSET_CLASS_MAP, ETF_UNIVERSE, RANDOM_SEED, latest_complete_month_end


def calculate_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily prices to month-end returns."""
    if daily_prices.empty:
        raise ValueError("Daily price data is empty.")
    prices = daily_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().ffill()
    month_end_prices = prices.resample("ME").last()
    returns = month_end_prices.pct_change().dropna(how="all")
    return returns.dropna(axis=1, how="all")


def _fetch_stooq_close(ticker: str, start: str, end: str) -> pd.Series | None:
    try:
        from pandas_datareader import data as pdr

        symbol = f"{ticker}.US"
        frame = pdr.DataReader(symbol, "stooq", start=start, end=end)
        if frame.empty or "Close" not in frame:
            return None
        close = frame["Close"].sort_index()
        close.name = ticker
        return close
    except Exception:
        return None


def fetch_daily_prices(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch public ETF daily closes where available."""
    tickers = list(ETF_UNIVERSE) if tickers is None else tickers
    end = pd.Timestamp.today().strftime("%Y-%m-%d") if end is None else end
    closes = []
    for ticker in tickers:
        close = _fetch_stooq_close(ticker, start, end)
        if close is not None and close.notna().sum() > 250:
            closes.append(close)
    if not closes:
        return pd.DataFrame()
    prices = pd.concat(closes, axis=1).sort_index()
    return prices.dropna(how="all")


def generate_fallback_monthly_returns(
    periods: int = 132,
    end_date: pd.Timestamp | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate realistic sample ETF returns for offline, reproducible runs."""
    end_date = latest_complete_month_end() if end_date is None else pd.Timestamp(end_date)
    dates = pd.date_range(end=end_date, periods=periods, freq="ME")
    tickers = list(ETF_UNIVERSE)

    annual_return = {
        "SPY": 0.085,
        "QQQ": 0.105,
        "IWM": 0.080,
        "EFA": 0.065,
        "EEM": 0.070,
        "AGG": 0.030,
        "TLT": 0.035,
        "SHY": 0.020,
        "LQD": 0.040,
        "HYG": 0.055,
        "VNQ": 0.070,
        "GLD": 0.045,
        "DBC": 0.035,
        "SGOV": 0.025,
    }
    annual_volatility = {
        "SPY": 0.165,
        "QQQ": 0.210,
        "IWM": 0.220,
        "EFA": 0.180,
        "EEM": 0.240,
        "AGG": 0.055,
        "TLT": 0.150,
        "SHY": 0.018,
        "LQD": 0.080,
        "HYG": 0.115,
        "VNQ": 0.210,
        "GLD": 0.160,
        "DBC": 0.200,
        "SGOV": 0.006,
    }

    means = np.array([annual_return[t] / 12 for t in tickers])
    vols = np.array([annual_volatility[t] / np.sqrt(12) for t in tickers])
    corr = np.eye(len(tickers))
    for i, left in enumerate(tickers):
        for j, right in enumerate(tickers):
            if i == j:
                continue
            left_class = ASSET_CLASS_MAP[left]
            right_class = ASSET_CLASS_MAP[right]
            if left_class == right_class == "Equity":
                corr[i, j] = 0.78
            elif left_class == right_class == "Fixed Income":
                corr[i, j] = 0.55
            elif left_class == right_class == "Alternatives":
                corr[i, j] = 0.35
            elif "Cash" in {left_class, right_class}:
                corr[i, j] = 0.05
            elif {"Equity", "Fixed Income"} == {left_class, right_class}:
                corr[i, j] = -0.10
            elif {"Equity", "Alternatives"} == {left_class, right_class}:
                corr[i, j] = 0.45
            else:
                corr[i, j] = 0.15

    covariance = np.outer(vols, vols) * corr
    rng = np.random.default_rng(seed)
    simulated = rng.multivariate_normal(means, covariance, size=periods)
    returns = pd.DataFrame(simulated, index=dates, columns=tickers)

    shock_months = {
        int(periods * 0.30): {
            "SPY": -0.09,
            "QQQ": -0.11,
            "IWM": -0.12,
            "EFA": -0.10,
            "EEM": -0.13,
            "AGG": 0.02,
            "TLT": 0.06,
            "SHY": 0.003,
            "LQD": -0.015,
            "HYG": -0.045,
            "VNQ": -0.10,
            "GLD": 0.045,
            "DBC": -0.04,
            "SGOV": 0.002,
        },
        int(periods * 0.62): {
            "SPY": -0.14,
            "QQQ": -0.16,
            "IWM": -0.18,
            "EFA": -0.15,
            "EEM": -0.19,
            "AGG": 0.015,
            "TLT": 0.045,
            "SHY": 0.002,
            "LQD": -0.035,
            "HYG": -0.075,
            "VNQ": -0.16,
            "GLD": 0.06,
            "DBC": -0.09,
            "SGOV": 0.002,
        },
        int(periods * 0.82): {
            "SPY": 0.10,
            "QQQ": 0.13,
            "IWM": 0.12,
            "EFA": 0.09,
            "EEM": 0.11,
            "AGG": -0.015,
            "TLT": -0.055,
            "SHY": -0.002,
            "LQD": 0.02,
            "HYG": 0.045,
            "VNQ": 0.08,
            "GLD": -0.025,
            "DBC": 0.04,
            "SGOV": 0.002,
        },
    }
    for row, shock in shock_months.items():
        for ticker, value in shock.items():
            returns.iloc[row, returns.columns.get_loc(ticker)] += value

    return returns.clip(lower=-0.35, upper=0.35).round(6)


def fetch_or_generate_monthly_returns(
    processed_dir: Path,
    raw_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch live ETF returns, with a complete synthetic fallback."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    daily_prices = fetch_daily_prices(start=start, end=end)
    if not daily_prices.empty:
        daily_prices.to_csv(raw_dir / "daily_etf_prices.csv")

    try:
        live_returns = calculate_monthly_returns(daily_prices)
    except Exception:
        live_returns = pd.DataFrame()

    tickers = list(ETF_UNIVERSE)
    if live_returns.shape[1] >= 8 and len(live_returns) >= 36:
        live_returns = live_returns.reindex(columns=tickers)
        missing = [ticker for ticker in tickers if live_returns[ticker].isna().all()]
        if missing:
            fallback = generate_fallback_monthly_returns(
                periods=len(live_returns),
                end_date=live_returns.index.max(),
            )
            for ticker in missing:
                live_returns[ticker] = fallback[ticker].values
        monthly_returns = live_returns[tickers].fillna(0.0)
    else:
        monthly_returns = generate_fallback_monthly_returns()

    monthly_returns.index.name = "date"
    monthly_returns.to_csv(processed_dir / "monthly_returns.csv")
    return monthly_returns

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio_construction import portfolio_returns
from src.utils import ASSET_CLASS_MAP, safe_divide


def annualized_return(returns: pd.Series) -> float:
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() ** (12.0 / len(clean)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    clean = pd.Series(returns).dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(12.0))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    return safe_divide(
        annualized_return(returns) - risk_free_rate,
        annualized_volatility(returns),
        default=0.0,
    )


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    clean = pd.Series(returns).dropna()
    downside = clean[clean < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = float(downside.std(ddof=1) * np.sqrt(12.0))
    return safe_divide(annualized_return(clean) - risk_free_rate, downside_vol, default=0.0)


def max_drawdown(returns: pd.Series) -> float:
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return 0.0
    wealth = (1.0 + clean).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def tracking_error(portfolio: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return annualized_volatility(active)


def information_ratio(portfolio: pd.Series, benchmark: pd.Series) -> float:
    active_return = annualized_return(portfolio) - annualized_return(benchmark)
    return safe_divide(active_return, tracking_error(portfolio, benchmark), default=0.0)


def beta_vs_benchmark(portfolio: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    variance = float(aligned.iloc[:, 1].var(ddof=1))
    covariance = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]))
    return safe_divide(covariance, variance, default=0.0)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(1.0 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return 0.0
    var = historical_var(clean, confidence)
    tail = clean[clean <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def positive_month_percentage(returns: pd.Series) -> float:
    clean = pd.Series(returns).dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


def correlation_to_benchmark(portfolio: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(0.0 if pd.isna(corr) else corr)


def calculate_risk_metrics(
    profile_name: str,
    portfolio_name: str,
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float | str]:
    portfolio = portfolio_returns(returns, weights)
    benchmark = benchmark_returns.reindex(portfolio.index)
    if not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError("Benchmark coverage must be complete; missing months are not zero returns.")
    return {
        "profile_name": profile_name,
        "portfolio_name": portfolio_name,
        "annualized_return": annualized_return(portfolio),
        "annualized_volatility": annualized_volatility(portfolio),
        "sharpe_ratio": sharpe_ratio(portfolio),
        "sortino_ratio": sortino_ratio(portfolio),
        "max_drawdown": max_drawdown(portfolio),
        "tracking_error_vs_benchmark": tracking_error(portfolio, benchmark),
        "information_ratio": information_ratio(portfolio, benchmark),
        "beta_vs_benchmark": beta_vs_benchmark(portfolio, benchmark),
        "historical_var_95": historical_var(portfolio, 0.95),
        "historical_cvar_95": historical_cvar(portfolio, 0.95),
        "best_month": float(portfolio.max()),
        "worst_month": float(portfolio.min()),
        "positive_month_percentage": positive_month_percentage(portfolio),
        "correlation_to_benchmark": correlation_to_benchmark(portfolio, benchmark),
    }


def build_risk_summary(
    returns: pd.DataFrame,
    all_weights: dict[str, dict[str, pd.Series]],
    benchmark_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for profile_name, portfolios in all_weights.items():
        for portfolio_name, weights in portfolios.items():
            rows.append(
                calculate_risk_metrics(
                    profile_name,
                    portfolio_name,
                    returns,
                    weights,
                    benchmark_returns[profile_name],
                )
            )
    return pd.DataFrame(rows)


def calculate_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float],
) -> pd.DataFrame:
    """Calculate ETF-level contribution to annualized portfolio volatility."""
    weight_series = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    annual_covariance = returns.cov().values * 12.0
    weight_values = weight_series.values
    portfolio_variance = float(weight_values @ annual_covariance @ weight_values)
    portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
    if portfolio_volatility <= 1e-12:
        marginal = np.zeros_like(weight_values)
    else:
        marginal = annual_covariance @ weight_values / portfolio_volatility
    total_contribution = weight_values * marginal
    percent_contribution = np.divide(
        total_contribution,
        portfolio_volatility,
        out=np.zeros_like(total_contribution),
        where=abs(portfolio_volatility) > 1e-12,
    )
    return pd.DataFrame(
        {
            "ticker": returns.columns,
            "asset_class": [ASSET_CLASS_MAP[ticker] for ticker in returns.columns],
            "weight": weight_values,
            "marginal_risk_contribution": marginal,
            "total_risk_contribution": total_contribution,
            "percent_risk_contribution": percent_contribution,
        }
    )


def _window_max_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(np.min(drawdown))


def _rolling_metrics_for_series(
    returns: pd.Series,
    series_name: str,
    window: int = 12,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    clean = pd.Series(returns).dropna()
    rolling_return = (1.0 + clean).rolling(window).apply(np.prod, raw=True) - 1.0
    rolling_volatility = clean.rolling(window).std(ddof=1) * np.sqrt(12.0)
    rolling_sharpe = (rolling_return - risk_free_rate) / rolling_volatility.replace(0.0, np.nan)
    rolling_drawdown = clean.rolling(window).apply(_window_max_drawdown, raw=True)
    frame = pd.DataFrame(
        {
            "date": clean.index,
            "series": series_name,
            "rolling_12m_return": rolling_return.values,
            "rolling_12m_volatility": rolling_volatility.values,
            "rolling_12m_sharpe_ratio": rolling_sharpe.values,
            "rolling_max_drawdown": rolling_drawdown.values,
        }
    )
    return frame.dropna().reset_index(drop=True)


def build_rolling_risk_metrics(
    portfolio_returns_series: pd.Series,
    benchmark_returns_series: pd.Series,
    window: int = 12,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    portfolio = _rolling_metrics_for_series(
        portfolio_returns_series, "Recommended Portfolio", window, risk_free_rate
    )
    benchmark = _rolling_metrics_for_series(
        benchmark_returns_series, "Benchmark", window, risk_free_rate
    )
    return pd.concat([portfolio, benchmark], ignore_index=True)

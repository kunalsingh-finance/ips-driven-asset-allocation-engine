from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.ips_profiles import parse_benchmark_definition, profile_to_constraints
from src.utils import ASSET_CLASS_MAP, ASSET_CLASS_ORDER, ETF_UNIVERSE, RANDOM_SEED, safe_divide


ETF_WEIGHT_LIMITS: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "Balanced Growth": {
        "SPY": (0.15, None),
        "AGG": (0.15, None),
        "HYG": (None, 0.10),
        "EFA": (None, 0.20),
        "TLT": (None, 0.15),
        "GLD": (None, 0.075),
        "SGOV": (0.05, None),
    },
    "Capital Preservation": {
        "SPY": (0.10, 0.20),
        "QQQ": (None, 0.05),
        "IWM": (None, 0.05),
        "EFA": (None, 0.07),
        "EEM": (None, 0.03),
        "AGG": (0.15, None),
        "SHY": (0.10, None),
        "TLT": (None, 0.10),
        "LQD": (None, 0.15),
        "HYG": (None, 0.05),
        "SGOV": (0.10, None),
    },
    "Long-Term Endowment Growth": {
        "SPY": (0.20, None),
        "IWM": (0.03, None),
        "AGG": (0.08, None),
        "EFA": (0.05, None),
        "EEM": (0.03, None),
        "HYG": (None, 0.10),
        "VNQ": (0.03, None),
        "GLD": (0.03, None),
        "DBC": (0.02, None),
        "SGOV": (0.02, None),
    },
}

GROUP_WEIGHT_LIMITS: dict[str, list[dict[str, object]]] = {
    "Capital Preservation": [
        {"name": "core_bond_cash", "tickers": ["AGG", "SHY", "SGOV"], "minimum": 0.50, "maximum": None},
    ],
    "Long-Term Endowment Growth": [
        {"name": "international_equity", "tickers": ["EFA", "EEM"], "minimum": None, "maximum": 0.25},
        {"name": "alternatives", "tickers": ["VNQ", "GLD", "DBC"], "minimum": None, "maximum": 0.15},
    ],
}

PROFILE_ANCHOR_WEIGHTS: dict[str, dict[str, float]] = {
    "Capital Preservation": {
        "SPY": 0.15,
        "QQQ": 0.02,
        "EFA": 0.03,
        "AGG": 0.25,
        "SHY": 0.20,
        "TLT": 0.05,
        "LQD": 0.10,
        "HYG": 0.03,
        "VNQ": 0.02,
        "GLD": 0.03,
        "SGOV": 0.12,
    },
    "Balanced Growth": {
        "SPY": 0.20,
        "QQQ": 0.10,
        "IWM": 0.05,
        "EFA": 0.10,
        "EEM": 0.05,
        "AGG": 0.20,
        "TLT": 0.05,
        "SHY": 0.03,
        "LQD": 0.07,
        "HYG": 0.05,
        "VNQ": 0.025,
        "GLD": 0.025,
        "SGOV": 0.05,
    },
    "Long-Term Endowment Growth": {
        "SPY": 0.22,
        "QQQ": 0.12,
        "IWM": 0.08,
        "EFA": 0.15,
        "EEM": 0.08,
        "AGG": 0.10,
        "LQD": 0.04,
        "HYG": 0.03,
        "VNQ": 0.05,
        "GLD": 0.05,
        "DBC": 0.03,
        "SGOV": 0.05,
    },
}


def _profile_name(profile: pd.Series | dict) -> str:
    return str(dict(profile).get("profile_name", ""))


def _profile_etf_limits(profile: pd.Series | dict) -> dict[str, tuple[float | None, float | None]]:
    return ETF_WEIGHT_LIMITS.get(_profile_name(profile), {})


def _profile_group_limits(profile: pd.Series | dict) -> list[dict[str, object]]:
    return GROUP_WEIGHT_LIMITS.get(_profile_name(profile), [])


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    weight_series = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    return returns.fillna(0.0).dot(weight_series)


def class_allocation(weights: pd.Series | dict[str, float]) -> dict[str, float]:
    series = pd.Series(weights, dtype=float)
    allocation = {asset_class: 0.0 for asset_class in ASSET_CLASS_ORDER}
    for ticker, weight in series.items():
        allocation[ASSET_CLASS_MAP.get(ticker, "Other")] = allocation.get(
            ASSET_CLASS_MAP.get(ticker, "Other"), 0.0
        ) + float(weight)
    return allocation


def validate_policy_constraints(
    weights: pd.Series | dict[str, float],
    profile: pd.Series | dict,
    tolerance: float = 1e-5,
) -> tuple[bool, list[str]]:
    series = pd.Series(weights, dtype=float)
    p = dict(profile)
    messages: list[str] = []
    if abs(series.sum() - 1.0) > tolerance:
        messages.append("Weights do not sum to 100%.")
    if (series < -tolerance).any():
        messages.append("Portfolio contains short positions.")
    max_single = float(p["max_single_etf_weight"])
    if (series > max_single + tolerance).any():
        messages.append("One or more ETF weights exceed the max single ETF limit.")
    for ticker, (lower, upper) in _profile_etf_limits(profile).items():
        if ticker not in series.index:
            continue
        if lower is not None and series[ticker] < lower - tolerance:
            messages.append(f"{ticker} allocation below profile minimum.")
        if upper is not None and series[ticker] > upper + tolerance:
            messages.append(f"{ticker} allocation above profile maximum.")
    allocations = class_allocation(series)
    for asset_class, (lower, upper) in profile_to_constraints(p).items():
        value = allocations.get(asset_class, 0.0)
        if value < lower - tolerance:
            messages.append(f"{asset_class} allocation below policy minimum.")
        if value > upper + tolerance:
            messages.append(f"{asset_class} allocation above policy maximum.")
    for group in _profile_group_limits(profile):
        tickers = [ticker for ticker in group["tickers"] if ticker in series.index]
        value = float(series.loc[tickers].sum()) if tickers else 0.0
        minimum = group.get("minimum")
        maximum = group.get("maximum")
        if minimum is not None and value < float(minimum) - tolerance:
            messages.append(f"{group['name']} allocation below profile minimum.")
        if maximum is not None and value > float(maximum) + tolerance:
            messages.append(f"{group['name']} allocation above profile maximum.")
    return not messages, messages


def _tickers_by_class(tickers: list[str]) -> dict[str, list[str]]:
    return {
        asset_class: [ticker for ticker in tickers if ASSET_CLASS_MAP[ticker] == asset_class]
        for asset_class in ASSET_CLASS_ORDER
    }


def _policy_class_targets(profile: pd.Series | dict) -> dict[str, float]:
    constraints = profile_to_constraints(profile)
    lower = {asset_class: values[0] for asset_class, values in constraints.items()}
    upper = {asset_class: values[1] for asset_class, values in constraints.items()}
    allocation = lower.copy()
    remaining = 1.0 - sum(allocation.values())
    if remaining < -1e-8:
        raise ValueError("IPS lower allocation bounds sum to more than 100%.")

    room = {asset_class: upper[asset_class] - lower[asset_class] for asset_class in ASSET_CLASS_ORDER}
    while remaining > 1e-10:
        open_classes = [asset_class for asset_class in ASSET_CLASS_ORDER if room[asset_class] > 1e-10]
        if not open_classes:
            break
        total_room = sum(room[asset_class] for asset_class in open_classes)
        for asset_class in open_classes:
            add = min(room[asset_class], remaining * room[asset_class] / total_room)
            allocation[asset_class] += add
            room[asset_class] -= add
        remaining = 1.0 - sum(allocation.values())
    return allocation


def initial_policy_weights(profile: pd.Series | dict, tickers: list[str] | None = None) -> pd.Series:
    tickers = list(ETF_UNIVERSE) if tickers is None else tickers
    profile_name = _profile_name(profile)
    if profile_name in PROFILE_ANCHOR_WEIGHTS:
        anchor = pd.Series(PROFILE_ANCHOR_WEIGHTS[profile_name], dtype=float).reindex(tickers).fillna(0.0)
        anchor = anchor / anchor.sum()
        valid, _ = validate_policy_constraints(anchor, profile, tolerance=1e-6)
        if valid:
            return anchor

    max_single = float(dict(profile)["max_single_etf_weight"])
    class_targets = _policy_class_targets(profile)
    groups = _tickers_by_class(tickers)
    weights = pd.Series(0.0, index=tickers, dtype=float)
    for asset_class, class_weight in class_targets.items():
        members = groups.get(asset_class, [])
        if not members:
            continue
        equal_weight = class_weight / len(members)
        if equal_weight > max_single + 1e-8:
            raise ValueError(f"{asset_class} cannot be allocated within max single ETF limit.")
        weights.loc[members] = equal_weight
    return weights / weights.sum()


def _optimization_constraints(
    profile: pd.Series | dict,
    tickers: list[str],
) -> list[dict[str, Callable[[np.ndarray], float]]]:
    constraints: list[dict[str, Callable[[np.ndarray], float]]] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    ]
    policy_constraints = profile_to_constraints(profile)
    for asset_class, (lower, upper) in policy_constraints.items():
        indices = [i for i, ticker in enumerate(tickers) if ASSET_CLASS_MAP[ticker] == asset_class]
        constraints.append({"type": "ineq", "fun": lambda w, idx=indices, lo=lower: float(np.sum(w[idx]) - lo)})
        constraints.append({"type": "ineq", "fun": lambda w, idx=indices, hi=upper: float(hi - np.sum(w[idx]))})
    for group in _profile_group_limits(profile):
        indices = [i for i, ticker in enumerate(tickers) if ticker in group["tickers"]]
        minimum = group.get("minimum")
        maximum = group.get("maximum")
        if minimum is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w, idx=indices, lo=float(minimum): float(np.sum(w[idx]) - lo)}
            )
        if maximum is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w, idx=indices, hi=float(maximum): float(hi - np.sum(w[idx]))}
            )
    return constraints


def _annualized_return(monthly_returns: np.ndarray) -> float:
    if len(monthly_returns) == 0:
        return 0.0
    return float(np.prod(1.0 + monthly_returns) ** (12.0 / len(monthly_returns)) - 1.0)


def _annualized_volatility(monthly_returns: np.ndarray) -> float:
    if len(monthly_returns) < 2:
        return 0.0
    return float(np.std(monthly_returns, ddof=1) * np.sqrt(12.0))


def _downside_volatility(monthly_returns: np.ndarray) -> float:
    downside = monthly_returns[monthly_returns < 0]
    if len(downside) < 2:
        return 0.0
    return float(np.std(downside, ddof=1) * np.sqrt(12.0))


def _solve_portfolio(
    returns: pd.DataFrame,
    profile: pd.Series | dict,
    objective: Callable[[np.ndarray], float],
) -> pd.Series:
    tickers = list(returns.columns)
    max_single = float(dict(profile)["max_single_etf_weight"])
    etf_limits = _profile_etf_limits(profile)
    bounds = []
    for ticker in tickers:
        lower, upper = etf_limits.get(ticker, (None, None))
        lower_bound = 0.0 if lower is None else float(lower)
        upper_bound = max_single if upper is None else min(max_single, float(upper))
        bounds.append((lower_bound, upper_bound))
    constraints = _optimization_constraints(profile, tickers)
    x0 = initial_policy_weights(profile, tickers).values
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )
    if not result.success:
        return pd.Series(x0, index=tickers, dtype=float)
    weights = pd.Series(result.x, index=tickers, dtype=float).clip(lower=0.0)
    weights = weights / weights.sum()
    valid, _ = validate_policy_constraints(weights, profile, tolerance=1e-4)
    if not valid:
        return pd.Series(x0, index=tickers, dtype=float)
    return weights


def _weight_bounds(profile: pd.Series | dict, tickers: list[str]) -> tuple[pd.Series, pd.Series]:
    max_single = float(dict(profile)["max_single_etf_weight"])
    etf_limits = _profile_etf_limits(profile)
    lower = pd.Series(0.0, index=tickers, dtype=float)
    upper = pd.Series(max_single, index=tickers, dtype=float)
    for ticker, (ticker_min, ticker_max) in etf_limits.items():
        if ticker not in lower.index:
            continue
        if ticker_min is not None:
            lower[ticker] = float(ticker_min)
        if ticker_max is not None:
            upper[ticker] = min(max_single, float(ticker_max))
    return lower, upper


def _repair_weight_sum(weights: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    repaired = weights.clip(lower=lower, upper=upper)
    for _ in range(100):
        difference = 1.0 - float(repaired.sum())
        if abs(difference) < 1e-10:
            break
        if difference > 0:
            room = (upper - repaired).clip(lower=0.0)
            if room.sum() <= 1e-12:
                break
            repaired += room / room.sum() * difference
        else:
            excess = (repaired - lower).clip(lower=0.0)
            if excess.sum() <= 1e-12:
                break
            repaired -= excess / excess.sum() * abs(difference)
        repaired = repaired.clip(lower=lower, upper=upper)
    return repaired / repaired.sum()


def _committee_realism_penalty(w: np.ndarray, tickers: list[str], profile: pd.Series | dict) -> float:
    weights = pd.Series(w, index=tickers, dtype=float)
    profile_name = _profile_name(profile)
    anchor = pd.Series(PROFILE_ANCHOR_WEIGHTS.get(profile_name, {}), dtype=float).reindex(tickers).fillna(0.0)
    if anchor.sum() > 0:
        anchor = anchor / anchor.sum()

    concentration_penalty = 1.80 * float(np.sum(w**2))
    single_name_penalty = 8.00 * float(np.sum(np.maximum(w - 0.18, 0.0) ** 2))
    anchor_penalty = 1.25 * float(np.sum((weights - anchor) ** 2)) if anchor.sum() > 0 else 0.0
    high_yield_weight = float(weights.get("HYG", 0.0))
    risky_credit_penalty = 4.00 * high_yield_weight**2
    intl_concentration = float(weights.reindex(["EFA", "EEM"]).fillna(0.0).sum())
    intl_penalty = 2.00 * max(0.0, intl_concentration - 0.22) ** 2

    core_shortfall_penalty = 0.0
    for ticker, expected_floor in {"SPY": 0.12, "AGG": 0.10}.items():
        if ticker in weights.index and anchor.get(ticker, 0.0) >= expected_floor:
            core_shortfall_penalty += 18.00 * max(0.0, expected_floor - float(weights[ticker])) ** 2

    defensive_credit_penalty = 0.0
    if profile_name == "Capital Preservation":
        defensive_credit_penalty += 6.00 * high_yield_weight**2
        defensive_credit_penalty += 5.00 * float(weights.reindex(["QQQ", "IWM", "EEM"]).fillna(0.0).sum()) ** 2

    return (
        concentration_penalty
        + single_name_penalty
        + anchor_penalty
        + risky_credit_penalty
        + intl_penalty
        + core_shortfall_penalty
        + defensive_credit_penalty
    )


def minimum_volatility_portfolio(returns: pd.DataFrame, profile: pd.Series | dict) -> pd.Series:
    covariance = returns.cov().values * 12.0
    covariance += np.eye(covariance.shape[0]) * 1e-8

    def objective(w: np.ndarray) -> float:
        return float(np.sqrt(np.maximum(w @ covariance @ w, 0.0)))

    return _solve_portfolio(returns, profile, objective)


def maximum_sharpe_portfolio(
    returns: pd.DataFrame,
    profile: pd.Series | dict,
    risk_free_rate: float = 0.02,
) -> pd.Series:
    expected = returns.mean().values * 12.0
    covariance = returns.cov().values * 12.0
    covariance += np.eye(covariance.shape[0]) * 1e-8

    def objective(w: np.ndarray) -> float:
        annual_return = float(w @ expected)
        annual_vol = float(np.sqrt(np.maximum(w @ covariance @ w, 0.0)))
        sharpe = safe_divide(annual_return - risk_free_rate, annual_vol, default=-10.0)
        concentration_penalty = 0.35 * float(np.sum(w**2))
        return -sharpe + concentration_penalty + 0.15 * _committee_realism_penalty(w, list(returns.columns), profile)

    return _solve_portfolio(returns, profile, objective)


def target_volatility_portfolio(returns: pd.DataFrame, profile: pd.Series | dict) -> pd.Series:
    expected = returns.mean().values * 12.0
    covariance = returns.cov().values * 12.0
    covariance += np.eye(covariance.shape[0]) * 1e-8
    target_vol = float(dict(profile)["max_volatility_target"])

    def objective(w: np.ndarray) -> float:
        annual_return = float(w @ expected)
        annual_vol = float(np.sqrt(np.maximum(w @ covariance @ w, 0.0)))
        concentration_penalty = 0.25 * float(np.sum(w**2))
        return (
            (annual_vol - target_vol) ** 2
            - 0.04 * annual_return
            + concentration_penalty
            + 0.10 * _committee_realism_penalty(w, list(returns.columns), profile)
        )

    return _solve_portfolio(returns, profile, objective)


def ips_recommended_portfolio(
    returns: pd.DataFrame,
    profile: pd.Series | dict,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> pd.Series:
    values = returns.values
    benchmark = benchmark_returns.reindex(returns.index).fillna(0.0).values
    target_vol = float(dict(profile)["max_volatility_target"]) * 0.95
    tickers = list(returns.columns)

    def objective(w: np.ndarray) -> float:
        monthly = values @ w
        annual_return = _annualized_return(monthly)
        annual_vol = _annualized_volatility(monthly)
        downside = _downside_volatility(monthly)
        tracking_error = _annualized_volatility(monthly - benchmark)
        sharpe = safe_divide(annual_return - risk_free_rate, annual_vol, default=-5.0)
        vol_excess = max(0.0, annual_vol - target_vol)
        return (
            -0.45 * sharpe
            -0.15 * annual_return
            + 0.65 * downside
            + 0.35 * tracking_error
            + _committee_realism_penalty(w, tickers, profile)
            + 6.00 * vol_excess**2
        )

    return _solve_portfolio(returns, profile, objective)


def build_benchmark_returns(returns: pd.DataFrame, profile: pd.Series | dict) -> pd.Series:
    definition = parse_benchmark_definition(dict(profile)["benchmark_definition"])
    weights = pd.Series(definition, dtype=float).reindex(returns.columns).fillna(0.0)
    weights = weights / weights.sum()
    benchmark = portfolio_returns(returns, weights)
    benchmark.name = "Benchmark"
    return benchmark


def benchmark_weights(profile: pd.Series | dict, tickers: list[str]) -> pd.Series:
    definition = parse_benchmark_definition(dict(profile)["benchmark_definition"])
    weights = pd.Series(definition, dtype=float).reindex(tickers).fillna(0.0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights


def build_portfolio_candidates(
    returns: pd.DataFrame,
    profile: pd.Series | dict,
    benchmark_returns: pd.Series,
) -> dict[str, pd.Series]:
    candidates = {
        "Equal Weight Portfolio": initial_policy_weights(profile, list(returns.columns)),
        "Minimum Volatility Portfolio": minimum_volatility_portfolio(returns, profile),
        "Max Sharpe Portfolio": maximum_sharpe_portfolio(returns, profile),
        "Target Volatility Portfolio": target_volatility_portfolio(returns, profile),
        "IPS Recommended Portfolio": ips_recommended_portfolio(returns, profile, benchmark_returns),
    }
    return candidates


def weights_to_frame(all_weights: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    for profile_name, portfolios in all_weights.items():
        for portfolio_name, weights in portfolios.items():
            for ticker, weight in weights.items():
                rows.append(
                    {
                        "profile_name": profile_name,
                        "portfolio_name": portfolio_name,
                        "ticker": ticker,
                        "asset_class": ASSET_CLASS_MAP[ticker],
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows)


def sample_feasible_portfolios(
    returns: pd.DataFrame,
    profile: pd.Series | dict,
    n_samples: int = 600,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = list(returns.columns)
    lower, upper = _weight_bounds(profile, tickers)
    anchor = initial_policy_weights(profile, tickers)
    rows = []
    attempts = 0
    while len(rows) < n_samples and attempts < n_samples * 30:
        attempts += 1
        noise = pd.Series(rng.normal(0.0, 0.035, size=len(tickers)), index=tickers)
        weights = _repair_weight_sum(anchor + noise, lower, upper)
        valid, _ = validate_policy_constraints(weights, profile, tolerance=1e-4)
        if not valid:
            continue
        monthly = portfolio_returns(returns, weights)
        annual_return = _annualized_return(monthly.values)
        annual_vol = _annualized_volatility(monthly.values)
        rows.append(
            {
                "annualized_return": annual_return,
                "annualized_volatility": annual_vol,
                "sharpe_ratio": safe_divide(annual_return - 0.02, annual_vol, default=0.0),
                **{f"weight_{ticker}": float(weights[ticker]) for ticker in tickers},
            }
        )
    return pd.DataFrame(rows)

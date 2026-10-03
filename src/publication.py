"""Stage, publish, and verify one complete IPS data and report pack.

Publication is transactional for handled failures. Consumers use the same
nonblocking OS lock and require SUCCESS before accepting any saved artifact.
An interrupted producer leaves RUNNING, which deliberately blocks verification.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
from openpyxl import load_workbook
import pandas as pd

from src.factor_model import FACTOR_COLUMNS
from src.utils import ETF_UNIVERSE


MANIFEST_PATH = "output/run_manifest.json"
STATUS_PATH = "output/run_status.json"
MARKET_PRICES_PATH = "data/raw/daily_etf_prices.csv"
OWNED_FILES = (
    "data/raw/ips_profiles.csv", MARKET_PRICES_PATH,
    "data/processed/monthly_returns.csv", "data/processed/monthly_returns_provenance.json",
    "data/processed/fama_french_factors.csv", "data/processed/factor_provenance.json",
    "output/portfolio_weights.csv", "output/portfolio_risk_summary.csv",
    "output/recommended_allocation.csv", "output/data_provenance.json",
    "output/factor_exposure_summary.csv", "output/equity_sleeve_factor_summary.csv",
    "output/stress_test_results.csv", "output/rebalancing_trades.csv",
    "output/efficient_frontier_samples.csv", "output/risk_contribution_summary.csv",
    "output/rolling_risk_metrics.csv", "output/etf_allocation_rationale.csv",
    "output/constraints_validation.csv", "output/investment_memo.md",
    "output/investment_committee_report.xlsx",
    "charts/efficient_frontier.png", "charts/recommended_allocation.png",
    "charts/cumulative_performance_vs_benchmark.png", "charts/drawdown_comparison.png",
    "charts/factor_exposures.png", "charts/equity_sleeve_factor_exposures.png",
    "charts/risk_contribution.png", "charts/policy_drift.png", "charts/stress_test_results.png",
    "charts/rolling_return.png", "charts/rolling_volatility.png", "charts/rolling_drawdown.png",
)


class PublicationError(ValueError):
    """A pack is unavailable, inconsistent, or used by another process."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path(root: Path, relative: str) -> Path:
    """Owned paths cannot redirect publication through a symlink or escape root."""
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise PublicationError(f"IPS artifact path escapes the project root: {relative}")
    current = path
    while current != root:
        if current.is_symlink():
            raise PublicationError(f"IPS artifact paths cannot be symlinks: {relative}")
        current = current.parent
    return path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # A leftover temporary file must not hide the original write failure.


@contextmanager
def output_lock(project_root: Path):
    """Shared producer/verifier lock; never wait for a concurrent run."""
    root = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = _safe_path(root, ".ips-publication.lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PublicationError("Another IPS computation or verifier is using this project folder.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def publication_attempt(project_root: Path, mode: str, end: str):
    root = Path(project_root).resolve()
    with output_lock(root):
        run_id = uuid4().hex
        status_path = _safe_path(root, STATUS_PATH)
        started = _now()
        write_json(status_path, {"schema_version": 1, "status": "RUNNING", "run_id": run_id,
                                 "mode": mode, "requested_end": end, "started_utc": started})
        try:
            yield run_id
        except BaseException as exc:
            failed = {"schema_version": 1, "status": "FAILED", "run_id": run_id,
                      "mode": mode, "requested_end": end, "started_utc": started,
                      "finished_utc": _now(), "error": str(exc)}
            recovery = getattr(exc, "ips_recovery_path", None)
            if recovery is not None:
                failed["recovery_path"] = str(recovery)
            try:
                write_json(status_path, failed)
            except Exception as status_error:
                # RUNNING still blocks consumers. Failure recording must not mask the cause.
                exc.add_note(f"Could not record FAILED status: {status_error}")
            raise


def _required_files(mode: str) -> set[str]:
    if mode not in {"synthetic", "market"}:
        raise PublicationError("IPS pack mode must be synthetic or market.")
    return set(OWNED_FILES) if mode == "market" else set(OWNED_FILES) - {MARKET_PRICES_PATH}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicationError(f"IPS JSON control must be an object: {path.name}")
    return value


def _read_monthly(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if (set(frame.columns) != set(columns) or frame.empty or frame.index.hasnans
            or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not isinstance(frame.index, pd.DatetimeIndex)
            or not frame.index.is_month_end.all()
            or not np.isfinite(frame.to_numpy(dtype=float)).all()):
        raise PublicationError(f"IPS monthly inputs must have complete finite data and unique month-end dates: {path.name}")
    return frame.reindex(columns=columns)


def _sheet_frame(workbook, name: str) -> pd.DataFrame:
    rows = list(workbook[name].values)
    if not rows:
        raise PublicationError(f"IPS workbook sheet is empty: {name}")
    return pd.DataFrame(rows[1:], columns=rows[0]).dropna(how="all")


def _same_table(actual: pd.DataFrame, expected: pd.DataFrame, description: str) -> None:
    try:
        actual = actual.copy()
        # Pandas writes infinite ratios as Excel text; compare their numeric meaning.
        for column in expected.columns:
            if column in actual and pd.api.types.is_numeric_dtype(expected[column]):
                actual[column] = pd.to_numeric(actual[column], errors="raise")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
    except (AssertionError, ValueError, TypeError) as exc:
        raise PublicationError(f"IPS {description} disagrees with the saved CSV.") from exc


def _verify_contents(root: Path, manifest: dict) -> None:
    files = manifest["files"]
    mode = manifest["mode"]
    required = _required_files(mode)
    if not isinstance(files, dict) or set(files) != required:
        raise PublicationError("IPS manifest does not bind the complete owned file set for its mode.")
    if (manifest.get("inputs") != {key: value for key, value in files.items() if key.startswith("data/")}
            or manifest.get("outputs") != {key: value for key, value in files.items() if not key.startswith("data/")}):
        raise PublicationError("IPS manifest input/output hashes do not match its file inventory.")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise PublicationError(f"Invalid IPS artifact hash: {relative}")
        if file_hash(_safe_path(root, relative)) != expected:
            raise PublicationError(f"IPS artifact changed after publication: {relative}")
    if mode == "synthetic" and _safe_path(root, MARKET_PRICES_PATH).exists():
        raise PublicationError("An unbound market-price file remains in the synthetic pack.")

    returns_provenance = _read_json(_safe_path(root, "data/processed/monthly_returns_provenance.json"))
    factor_provenance = _read_json(_safe_path(root, "data/processed/factor_provenance.json"))
    provenance = _read_json(_safe_path(root, "output/data_provenance.json"))
    if (provenance.get("run_id") != manifest["run_id"] or provenance.get("mode") != mode
            or provenance.get("returns") != returns_provenance or provenance.get("factors") != factor_provenance
            or returns_provenance.get("mode") != mode or factor_provenance.get("mode") != mode):
        raise PublicationError("IPS saved source provenance does not bind the same run and mode.")
    if (returns_provenance.get("monthly_returns_sha256") != files["data/processed/monthly_returns.csv"]
            or factor_provenance.get("factor_data_sha256") != files["data/processed/fama_french_factors.csv"]
            or (mode == "market" and returns_provenance.get("daily_prices_sha256") != files[MARKET_PRICES_PATH])):
        raise PublicationError("IPS source provenance input hashes disagree with the saved inputs.")
    assets = returns_provenance.get("assets")
    if (not isinstance(assets, dict) or set(assets) != set(ETF_UNIVERSE)
            or any(item.get("synthetic") is not (mode == "synthetic") or item.get("symbol") != symbol
                   for symbol, item in assets.items())):
        raise PublicationError("IPS per-asset provenance does not match the declared universe and mode.")

    returns = _read_monthly(_safe_path(root, "data/processed/monthly_returns.csv"), list(ETF_UNIVERSE))
    factors = _read_monthly(_safe_path(root, "data/processed/fama_french_factors.csv"), FACTOR_COLUMNS)
    if len(returns) < 36 or not factors.index.equals(returns.index):
        raise PublicationError("IPS returns/factors must have at least 36 aligned complete months.")
    sample = {"start_date": str(returns.index.min().date()), "end_date": str(returns.index.max().date()),
              "complete_months": len(returns)}
    if (manifest.get("sample") != sample or any(provenance.get(key) != value for key, value in sample.items())
            or any(returns_provenance.get(key) != value for key, value in sample.items())
            or provenance.get("requested_end") != manifest.get("requested_end")):
        raise PublicationError("IPS sample dates/count or requested end disagree with the saved inputs.")
    requested_end = pd.Timestamp(manifest["requested_end"])
    if pd.isna(requested_end) or requested_end.strftime("%Y-%m-%d") != manifest["requested_end"]:
        raise PublicationError("IPS requested end must be a valid calendar date.")
    complete_end = requested_end if requested_end.is_month_end else requested_end.to_period("M").start_time - pd.Timedelta(days=1)
    if returns.index.max() > complete_end:
        raise PublicationError("IPS sample contains a partial or future month beyond the requested end.")

    recommended = pd.read_csv(_safe_path(root, "output/recommended_allocation.csv"))
    weights = pd.read_csv(_safe_path(root, "output/portfolio_weights.csv"))
    selected = weights.loc[weights["portfolio_name"] == "IPS Recommended Portfolio"].reset_index(drop=True)
    _same_table(recommended, selected, "recommended allocation")
    if (recommended.empty or not np.isfinite(recommended["weight"]).all()
            or (recommended["weight"] < -1e-9).any()
            or not np.allclose(recommended.groupby("profile_name")["weight"].sum(), 1.0, atol=1e-8)):
        raise PublicationError("IPS recommended allocations must be finite nonnegative full-investment weights.")

    workbook = load_workbook(_safe_path(root, "output/investment_committee_report.xlsx"), read_only=True, data_only=True)
    try:
        summary = _sheet_frame(workbook, "Source Summary")
        sections = {row["section"]: row for row in summary.to_dict("records")}
        returned, factored, publication = sections["Returns"], sections["Factors"], sections["Publication"]
        if (returned.get("monthly_returns_sha256") != files["data/processed/monthly_returns.csv"]
                or returned.get("interpretation") != mode or returned.get("complete_months") != len(returns)
                or factored.get("factor_data_sha256") != files["data/processed/fama_french_factors.csv"]
                or factored.get("mode") != mode
                or returned.get("run_id") != manifest["run_id"]
                or returned.get("start_date") != sample["start_date"]
                or returned.get("end_date") != sample["end_date"]
                or factored.get("source") != factor_provenance["source"]
                or factored.get("source_reference") != factor_provenance["source_reference"]
                or publication.get("run_id") != manifest["run_id"]
                or publication.get("requested_end") != manifest["requested_end"]
                or publication.get("start_date") != sample["start_date"]
                or publication.get("end_date") != sample["end_date"]):
            raise PublicationError("IPS workbook Source Summary is from a different run, mode, input hash, or sample.")
        _same_table(_sheet_frame(workbook, "Data Provenance"),
                    pd.DataFrame([{"asset": asset, **details} for asset, details in assets.items()]),
                    "workbook per-asset provenance")
        _same_table(_sheet_frame(workbook, "IPS Profile"),
                    pd.read_csv(_safe_path(root, "data/raw/ips_profiles.csv")), "workbook IPS profiles")
        _same_table(_sheet_frame(workbook, "Recommended Allocation"), recommended, "workbook allocation")
        _same_table(_sheet_frame(workbook, "Risk Metrics"),
                    pd.read_csv(_safe_path(root, "output/portfolio_risk_summary.csv")), "workbook risk metrics")
        methodology = _sheet_frame(workbook, "Methodology")
        references = methodology.loc[methodology["section"] == "Charts", "description"]
        chart_files = {Path(relative).name for relative in files if relative.startswith("charts/")}
        if len(references) != 1 or {str(path).replace("\\", "/") for path in str(references.iloc[0]).split("; ")} != {f"charts/{name}" for name in chart_files}:
            raise PublicationError("IPS workbook chart references are not the portable published chart set.")
    finally:
        workbook.close()
    memo = _safe_path(root, "output/investment_memo.md").read_text(encoding="utf-8")
    if (f"Run id: `{manifest['run_id']}`" not in memo
            or f"Input sample: {sample['start_date']} to {sample['end_date']}" not in memo
            or f"Source mode: **{mode}**" not in memo):
        raise PublicationError("IPS investment memo does not describe the published run, mode, and sample.")


def verify_manifest(project_root: Path) -> dict:
    """Verify without acquiring a lock; caller must hold output_lock."""
    root = Path(project_root).resolve()
    try:
        status = _read_json(_safe_path(root, STATUS_PATH))
        if status.get("status") != "SUCCESS":
            raise PublicationError("Latest IPS attempt did not complete successfully; rebuild the pack.")
        manifest = _read_json(_safe_path(root, MANIFEST_PATH))
        if (status.get("schema_version") != 1 or manifest.get("schema_version") != 1
                or not isinstance(manifest.get("run_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", manifest["run_id"])
                or manifest["run_id"] != status.get("run_id")
                or manifest.get("mode") != status.get("mode")
                or manifest.get("requested_end") != status.get("requested_end")):
            raise PublicationError("IPS run status and manifest do not identify the same complete run.")
        _verify_contents(root, manifest)
        return manifest
    except PublicationError:
        raise
    except Exception as exc:
        raise PublicationError("Missing or invalid complete IPS pack; rebuild outputs.") from exc


def verify_pack(project_root: Path) -> dict:
    with output_lock(project_root):
        return verify_manifest(project_root)


def _remove_private_directory(directory: Path, root: Path, prefix: str) -> None:
    if directory.resolve().parent != root or directory.is_symlink() or not directory.name.startswith(prefix):
        raise PublicationError("Refusing to remove a private IPS directory outside the project root.")
    shutil.rmtree(directory)


def publish_pack(stage_root: Path, project_root: Path, run_id: str, mode: str, end: str) -> dict:
    """Publish owned files only, restoring previous bytes on any handled failure.

    Caller holds output_lock. Persistent backup copies survive a failed rollback;
    the original publication exception and FAILED status describe their location.
    """
    root, stage = Path(project_root).resolve(), Path(stage_root).resolve()
    if stage == root or stage.is_relative_to(root / "output"):
        raise PublicationError("IPS publication requires a separate private stage.")
    required = _required_files(mode)
    files = {relative: file_hash(_safe_path(stage, relative)) for relative in sorted(required)}
    provenance = _read_json(_safe_path(stage, "output/data_provenance.json"))
    manifest = {"schema_version": 1, "run_id": run_id, "mode": mode, "requested_end": end,
                "published_utc": _now(), "sample": {key: provenance[key] for key in ["start_date", "end_date", "complete_months"]},
                "files": files, "inputs": {key: value for key, value in files.items() if key.startswith("data/")},
                "outputs": {key: value for key, value in files.items() if not key.startswith("data/")}}
    write_json(_safe_path(stage, MANIFEST_PATH), manifest)
    success = {"schema_version": 1, "status": "SUCCESS", "run_id": run_id,
               "mode": mode, "requested_end": end, "finished_utc": _now()}
    write_json(_safe_path(stage, STATUS_PATH), success)
    verify_manifest(stage)
    owned = (*OWNED_FILES, MANIFEST_PATH)
    # Check all target paths before mutation, including optional files to delete.
    targets = {relative: _safe_path(root, relative) for relative in owned}
    _safe_path(root, STATUS_PATH)
    recovery = root / f".ips-recovery-{run_id}"
    if not re.fullmatch(r"[0-9a-f]{32}", run_id) or recovery.resolve().parent != root or recovery.exists():
        raise PublicationError("Invalid or already occupied IPS recovery path.")
    recovery.mkdir()
    changed = []
    try:
        for relative, target in targets.items():
            if target.exists():
                backup = _safe_path(recovery, relative)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, backup)
        try:
            for relative, target in targets.items():
                source = _safe_path(stage, relative)
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    changed.append(relative)
                    source.replace(target)
                elif target.exists():
                    changed.append(relative)
                    target.unlink()
            _verify_contents(root, manifest)
            # The manifest is replaced after artifacts; SUCCESS is the final commit marker.
            write_json(_safe_path(root, STATUS_PATH), success)
        except BaseException as exc:
            rollback_errors = []
            for relative in reversed(changed):
                target, backup = targets[relative], _safe_path(recovery, relative)
                try:
                    if backup.exists():
                        shutil.copyfile(backup, target)
                    else:
                        target.unlink(missing_ok=True)
                except Exception as restore_error:
                    rollback_errors.append(f"{relative}: {restore_error}")
            if rollback_errors:
                exc.ips_recovery_path = recovery
                exc.add_note(f"Prior IPS artifact copies preserved in {recovery}; rollback errors: {'; '.join(rollback_errors)}")
            else:
                try:
                    _remove_private_directory(recovery, root, ".ips-recovery-")
                except OSError as cleanup_error:
                    exc.add_note(f"Unused IPS backups remain in {recovery}: {cleanup_error}")
            raise
    except BaseException:
        # If copying backups failed before the first mutation, prior files are intact.
        if not changed:
            try:
                _remove_private_directory(recovery, root, ".ips-recovery-")
            except OSError:
                pass
        raise
    try:
        _remove_private_directory(recovery, root, ".ips-recovery-")
    except OSError:
        # A complete verified SUCCESS pack is valid even if unused backups remain.
        pass
    return manifest

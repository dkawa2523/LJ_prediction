from __future__ import annotations

from typing import Any, Dict, Iterable

from src.common.config import ConfigError
from src.tasks import resolve_target_columns


def _require_keys(cfg: Dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ConfigError(f"{context} missing keys: {', '.join(missing)}")


def _has_target(cfg: Dict[str, Any]) -> bool:
    return bool(resolve_target_columns(cfg))


def validate_config(cfg: Dict[str, Any]) -> None:
    process = cfg.get("process", {})
    process_name = process.get("name")
    if not process_name:
        return

    if process_name == "build_dataset":
        _require_keys(cfg, ["paths", "columns", "split"], "build_dataset config")
        return

    if process_name in {"fp_train", "gnn_train", "train"}:
        _require_keys(cfg, ["data", "model", "train", "output"], f"{process_name} config")
        if not _has_target(cfg):
            raise ConfigError(f"{process_name} requires task.target_columns/target_col or data.target_columns/target_col")
        return

    if process_name in {"fp_predict", "gnn_predict", "predict"}:
        _require_keys(cfg, ["model_artifact_dir", "output", "input"], f"{process_name} config")
        return

    if process_name == "evaluate":
        _require_keys(cfg, ["model_artifact_dir", "output"], f"{process_name} config")
        return

    if process_name == "visualize":
        _require_keys(cfg, ["output"], f"{process_name} config")
        return

    if process_name == "leaderboard":
        _require_keys(cfg, ["leaderboard", "output"], f"{process_name} config")
        return

    if process_name == "audit_dataset":
        _require_keys(cfg, ["audit", "output"], f"{process_name} config")
        return

    if process_name == "collect_data":
        _require_keys(cfg, ["data_collection", "data_source", "output"], f"{process_name} config")
        return

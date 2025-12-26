from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.common.metrics import regression_metrics

MetricsFn = Callable[[Sequence[float], Sequence[float]], Dict[str, float]]


def _empty_metrics(_: Sequence[float], __: Sequence[float]) -> Dict[str, float]:
    return {}


def _normalize_columns(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            item_str = str(item).strip()
            if item_str:
                out.append(item_str)
        return out
    item_str = str(value).strip()
    return [item_str] if item_str else []


def resolve_target_columns(cfg: Dict[str, Any]) -> List[str]:
    task_cfg = cfg.get("task", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    for key in ("target_columns", "target_cols"):
        cols = _normalize_columns(task_cfg.get(key))
        if cols:
            return cols
    cols = _normalize_columns(task_cfg.get("target_col"))
    if cols:
        return cols

    for key in ("target_columns", "target_cols"):
        cols = _normalize_columns(data_cfg.get(key))
        if cols:
            return cols
    return _normalize_columns(data_cfg.get("target_col"))


def _resolve_metrics_fn(metrics_cfg: Any, task_type: str) -> MetricsFn:
    task_type = str(task_type).lower()
    base_fn = regression_metrics if task_type == "regression" else _empty_metrics

    if metrics_cfg is None:
        return base_fn

    if isinstance(metrics_cfg, str):
        key = metrics_cfg.lower()
        if key in {"regression", "default"}:
            return base_fn
        if key in {"none", "null", "off"}:
            return _empty_metrics
        raise ValueError(f"Unknown task.metrics: {metrics_cfg}")

    if isinstance(metrics_cfg, (list, tuple)):
        names = [str(item) for item in metrics_cfg]

        def _filtered(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
            metrics = base_fn(y_true, y_pred)
            return {k: v for k, v in metrics.items() if k in names}

        return _filtered

    raise ValueError(f"Unsupported task.metrics type: {type(metrics_cfg).__name__}")


def _resolve_loss_name(task_cfg: Dict[str, Any], train_cfg: Dict[str, Any], task_type: str) -> str:
    train_loss = train_cfg.get("loss")
    if train_loss is not None:
        return str(train_loss).lower()
    task_loss = task_cfg.get("loss")
    if task_loss is not None:
        return str(task_loss).lower()
    if str(task_type).lower() == "regression":
        return "mse"
    return "mse"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    target_columns: List[str]
    metrics_fn: MetricsFn
    loss_name: str

    def primary_target(self) -> Optional[str]:
        if not self.target_columns:
            return None
        return self.target_columns[0]


def resolve_task(cfg: Dict[str, Any]) -> TaskSpec:
    task_cfg = cfg.get("task", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    name = str(task_cfg.get("name") or cfg.get("task_name") or "task")
    task_type = str(task_cfg.get("type", "regression")).lower()
    target_columns = resolve_target_columns(cfg)
    metrics_fn = _resolve_metrics_fn(task_cfg.get("metrics"), task_type)
    loss_name = _resolve_loss_name(task_cfg, train_cfg, task_type)

    return TaskSpec(
        name=name,
        task_type=task_type,
        target_columns=target_columns,
        metrics_fn=metrics_fn,
        loss_name=loss_name,
    )

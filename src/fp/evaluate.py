from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.common.config import dump_yaml, load_config
from src.common.feature_pipeline import load_tabular_pipeline
from src.common.meta import build_meta, save_meta
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json
from src.fp.feature_utils import hash_cfg
from src.tasks import resolve_task
from src.utils.artifacts import compute_dataset_hash, load_meta, resolve_training_context
from src.utils.validate_config import validate_config


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)

    model_artifact_dir = Path(cfg["model_artifact_dir"])
    artifacts_dir = model_artifact_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    train_cfg_path = model_artifact_dir / "config_snapshot.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
    train_cfg = load_config(train_cfg_path)

    data_cfg = train_cfg.get("data", {})
    data_override = cfg.get("data", {})
    dataset_csv = Path(data_override.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")))
    indices_dir = Path(data_override.get("indices_dir", data_cfg.get("indices_dir", "data/processed/indices")))
    sdf_dir = Path(data_override.get("sdf_dir", data_cfg.get("sdf_dir", "data/raw/sdf_files")))
    cas_col = str(data_override.get("cas_col", data_cfg.get("cas_col", "CAS")))

    task_spec = resolve_task(cfg)
    target_col = task_spec.primary_target()
    if target_col is None:
        raise ValueError("No target column resolved from task/data config.")

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/evaluate"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "fp_evaluate")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    logger = get_logger("fp_evaluate", log_file=run_dir / "evaluate.log")

    dump_yaml(run_dir / "config.yaml", cfg)

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
    if not sdf_dir.exists():
        raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")

    train_meta = load_meta(model_artifact_dir)
    train_context = resolve_training_context(train_cfg, train_meta, model_artifact_dir)
    dataset_hash = train_context.get("dataset_hash") or compute_dataset_hash(dataset_csv, indices_dir)
    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "evaluate")),
        cfg=cfg,
        upstream_artifacts=[str(model_artifact_dir)],
        dataset_hash=dataset_hash,
        model_version=train_context.get("model_version"),
        extra={
            "task_name": train_context.get("task_name"),
            "model_name": train_context.get("model_name"),
            "featureset_name": train_context.get("featureset_name"),
        },
    )
    save_meta(run_dir, meta)

    df = pd.read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)

    pipeline = load_tabular_pipeline(artifacts_dir, train_cfg)
    feat_cfg = train_cfg.get("featurizer", {})

    cache_dir = data_cfg.get("cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else None

    cache_key = hash_cfg({"featurizer": feat_cfg, "dataset": str(dataset_csv)})
    X_all, _, _, _ = pipeline.build_features(
        df=df,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        cache_dir=cache_dir,
        cache_key=cache_key,
        logger=logger,
    )
    with open(artifacts_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    rows = []
    metrics_by_split: Dict[str, Dict[str, float]] = {}

    for split_name in ["train", "val", "test"]:
        if split_name not in indices:
            continue
        split_idx = indices[split_name]
        split_df = df.loc[split_idx]
        idx = split_df.index.to_numpy()
        X = X_all[idx]
        y = split_df[target_col].astype(float).to_numpy()
        ids = split_df[cas_col].astype(str).tolist()

        valid_mask = (~np.isnan(X).all(axis=1)) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        ids = [cid for cid, ok in zip(ids, valid_mask.tolist()) if ok]

        if len(ids) == 0:
            metrics_by_split[split_name] = {}
            continue

        X = pipeline.transform_features(X)
        preds = model.predict(X)

        metrics_by_split[split_name] = task_spec.metrics_fn(y, preds)
        for cid, yt, yp in zip(ids, y.tolist(), preds.tolist()):
            rows.append({"sample_id": cid, "y_true": yt, "y_pred": yp, "split": split_name})

    pred_df = pd.DataFrame(rows)
    pred_df["model_name"] = train_context.get("model_name")
    pred_df["model_version"] = train_context.get("model_version")
    pred_df["dataset_hash"] = dataset_hash
    pred_df["run_id"] = meta["run_id"]
    pred_path = run_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    for split_name, metrics in metrics_by_split.items():
        if not metrics:
            continue
        save_json(run_dir / f"metrics_{split_name}.json", metrics)

    split_counts = pred_df["split"].value_counts().to_dict() if "split" in pred_df.columns else {}
    save_json(
        run_dir / "metrics.json",
        {
            "by_split": metrics_by_split,
            "n_train": int(split_counts.get("train", 0)),
            "n_val": int(split_counts.get("val", 0)),
            "n_test": int(split_counts.get("test", 0)),
        },
    )

    logger.info("Done.")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate fingerprint model and write predictions/metrics.")
    ap.add_argument("--config", required=True, help="Path to configs/fp/evaluate.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)

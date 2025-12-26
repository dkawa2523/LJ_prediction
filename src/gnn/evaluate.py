from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.config import dump_yaml, load_config
from src.common.feature_pipeline import GraphFeaturePipeline, load_feature_pipeline
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.meta import build_meta, save_meta
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json
from src.gnn.models import GCNRegressor, GINRegressor, MPNNRegressor
from src.tasks import resolve_task
from src.utils.artifacts import compute_dataset_hash, load_meta, resolve_training_context
from src.utils.validate_config import validate_config

try:
    import torch
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None


def _require_pyg() -> None:
    if torch is None or DataLoader is None:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for GNN evaluation. "
            "Install torch and torch_geometric (matching your environment)."
        )


def _select_device(prefer: str) -> "torch.device":
    prefer = str(prefer).lower()
    if prefer in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    if prefer in {"cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device("cpu")


def _build_dataset(
    df: pd.DataFrame,
    indices: Dict[str, List[int]],
    split_name: str,
    sdf_dir: Path,
    cas_col: str,
    target_col: str,
    pipeline: GraphFeaturePipeline,
) -> Tuple[List[Any], List[str]]:
    split_idx = indices.get(split_name, [])
    split_df = df.loc[split_idx]
    data_list = []
    ids = []
    for cas, y in zip(split_df[cas_col].astype(str).tolist(), split_df[target_col].astype(float).tolist()):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        if mol is None or not np.isfinite(y):
            continue
        try:
            data = pipeline.featurize_mol(mol, y=y)
            data_list.append(data)
            ids.append(cas)
        except Exception:
            continue
    return data_list, ids


def run(cfg: Dict[str, Any]) -> Path:
    _require_pyg()
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
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "gnn_evaluate")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    logger = get_logger("gnn_evaluate", log_file=run_dir / "evaluate.log")

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

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)

    pipeline = load_feature_pipeline(artifacts_dir)
    if not isinstance(pipeline, GraphFeaturePipeline):
        pipeline = GraphFeaturePipeline.from_artifacts(artifacts_dir)

    train_data, train_ids = _build_dataset(df, indices, "train", sdf_dir, cas_col, target_col, pipeline)
    val_data, val_ids = _build_dataset(df, indices, "val", sdf_dir, cas_col, target_col, pipeline)
    test_data, test_ids = _build_dataset(df, indices, "test", sdf_dir, cas_col, target_col, pipeline)

    logger.info(f"Data sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    sample_data = (train_data or val_data or test_data)
    if not sample_data:
        raise RuntimeError("No valid graphs were created for evaluation.")

    model_cfg = train_cfg.get("model", {})
    model_name = str(model_cfg.get("name", "mpnn")).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    edge_mlp_hidden_dim = int(model_cfg.get("edge_mlp_hidden_dim", 128))

    in_dim = sample_data[0].x.shape[1]
    edge_dim = sample_data[0].edge_attr.shape[1] if hasattr(sample_data[0], "edge_attr") else 0
    global_dim = int(sample_data[0].u.shape[1]) if hasattr(sample_data[0], "u") else 0

    if model_name == "gcn":
        model = GCNRegressor(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, global_dim=global_dim)
    elif model_name == "gin":
        model = GINRegressor(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            global_dim=global_dim,
            edge_dim=edge_dim,
        )
    elif model_name == "mpnn":
        model = MPNNRegressor(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            global_dim=global_dim,
            edge_mlp_hidden_dim=edge_mlp_hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model.name: {model_name}")

    eval_cfg = cfg.get("eval", {})
    prefer_device = eval_cfg.get("device", train_cfg.get("train", {}).get("device", "auto"))
    device = _select_device(prefer_device)
    model = model.to(device)

    state_path = artifacts_dir / "model_best.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("train", {}).get("batch_size", 32)))
    loader_kwargs = {"batch_size": batch_size, "shuffle": False}

    def eval_loader(data_list: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        if len(data_list) == 0:
            return np.array([]), np.array([])
        loader = DataLoader(data_list, **loader_kwargs)
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch).detach().cpu().numpy().reshape(-1)
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.extend(y.tolist())
                ps.extend(pred.tolist())
        return np.asarray(ys, dtype=float), np.asarray(ps, dtype=float)

    rows = []
    metrics_by_split: Dict[str, Dict[str, float]] = {}

    for split_name, data_list, ids in [
        ("train", train_data, train_ids),
        ("val", val_data, val_ids),
        ("test", test_data, test_ids),
    ]:
        y_true, y_pred = eval_loader(data_list)
        if len(y_true) == 0:
            metrics_by_split[split_name] = {}
            continue
        metrics_by_split[split_name] = task_spec.metrics_fn(y_true, y_pred)
        for cid, yt, yp in zip(ids, y_true.tolist(), y_pred.tolist()):
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
    ap = argparse.ArgumentParser(description="Evaluate GNN model and write predictions/metrics.")
    ap.add_argument("--config", required=True, help="Path to configs/gnn/evaluate.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)

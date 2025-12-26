from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as `python scripts/predict.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.fp import predict as fp_predict
from src.gnn import predict as gnn_predict


_FP_MODELS = {"lightgbm", "lgbm", "rf", "catboost", "gpr"}
_GNN_MODELS = {"gcn", "gin", "mpnn"}


def _resolve_backend_from_model_dir(model_dir: Path) -> str:
    train_cfg_path = model_dir / "config_snapshot.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
    train_cfg = load_config(train_cfg_path)
    model_cfg = train_cfg.get("model", {})
    family = str(model_cfg.get("family", "")).lower()
    if family:
        return family
    name = str(model_cfg.get("name", "")).lower()
    if name in _FP_MODELS:
        return "fp"
    if name in _GNN_MODELS:
        return "gnn"
    raise ValueError("Unable to resolve backend from model config. Set model.family in training config.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict (dispatch to FP or GNN backend).")
    ap.add_argument("--config", required=True, help="Path to a composed predict config.")
    ap.add_argument("--query", required=True, help="CAS or formula depending on config.input.mode")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_artifact_dir = Path(cfg["model_artifact_dir"])
    backend = _resolve_backend_from_model_dir(model_artifact_dir)

    if backend == "fp":
        fp_predict.run(cfg, args.query)
    elif backend == "gnn":
        gnn_predict.run(cfg, args.query)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()

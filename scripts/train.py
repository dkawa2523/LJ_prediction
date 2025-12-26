from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as `python scripts/train.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.fp import train as fp_train
from src.gnn import train as gnn_train


_FP_MODELS = {"lightgbm", "lgbm", "rf", "catboost", "gpr"}
_GNN_MODELS = {"gcn", "gin", "mpnn"}


def _resolve_backend(cfg: Dict[str, Any]) -> str:
    process_cfg = cfg.get("process", {})
    backend = str(process_cfg.get("backend", "")).lower()
    if backend:
        return backend
    model_cfg = cfg.get("model", {})
    family = str(model_cfg.get("family", "")).lower()
    if family:
        return family
    name = str(model_cfg.get("name", "")).lower()
    if name in _FP_MODELS:
        return "fp"
    if name in _GNN_MODELS:
        return "gnn"
    raise ValueError("Unable to resolve backend. Set process.backend or model.family in config.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train model (dispatch to FP or GNN backend).")
    ap.add_argument("--config", required=True, help="Path to a composed train config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = _resolve_backend(cfg)
    if backend == "fp":
        fp_train.run(cfg)
    elif backend == "gnn":
        gnn_train.run(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()

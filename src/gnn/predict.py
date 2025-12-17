from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.common.ad import applicability_domain
from src.common.chemistry import get_elements_from_mol
from src.common.config import load_yaml
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.utils import ensure_dir, get_logger, save_json
from src.fp.featurizer_fp import morgan_bitvect
from src.gnn.featurizer_graph import featurize_mol_to_pyg
from src.gnn.models import GCNRegressor, MPNNRegressor

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _resolve_cas(mode: str, query: str, dataset_csv: Path) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"mode": mode, "query": query}
    mode = mode.lower()
    if mode == "cas":
        return query.strip(), meta
    if mode == "formula":
        df = read_csv(dataset_csv)
        matches = df[df["MolecularFormula"].astype(str) == query.strip()]
        if len(matches) == 0:
            raise ValueError(f"No CAS found for formula={query} in {dataset_csv}")
        if len(matches) > 1:
            meta["warning"] = f"Multiple entries share the same formula. Using first match (n={len(matches)}). Consider using CAS instead."
        cas = str(matches.iloc[0]["CAS"])
        meta["resolved_cas"] = cas
        return cas, meta
    raise ValueError(f"Unknown input mode: {mode}. Use 'cas' or 'formula'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict LJ parameter using GNN model with applicability-domain diagnostics.")
    ap.add_argument("--config", required=True, help="Path to configs/gnn/predict.yaml")
    ap.add_argument("--query", required=True, help="CAS or formula depending on config.input.mode")
    args = ap.parse_args()

    if torch is None:
        raise ImportError("PyTorch is required.")
    cfg = load_yaml(args.config)
    model_artifact_dir = Path(cfg["model_artifact_dir"])
    artifacts_dir = model_artifact_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    train_cfg_path = model_artifact_dir / "config_snapshot.yaml"
    train_cfg = load_yaml(train_cfg_path)

    data_cfg = train_cfg.get("data", {})
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
    dataset_csv = Path(cfg.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")))
    cas_col = str(data_cfg.get("cas_col", "CAS"))

    out_dir = ensure_dir(Path(cfg.get("output", {}).get("out_dir", "runs/predict")) / cfg.get("output", {}).get("exp_name", "gnn_predict"))
    logger = get_logger("gnn_predict", log_file=out_dir / "predict.log")

    mode = str(cfg.get("input", {}).get("mode", "formula"))
    cas, resolve_meta = _resolve_cas(mode=mode, query=args.query, dataset_csv=dataset_csv)
    logger.info(f"Resolved CAS={cas} from query={args.query} (mode={mode})")

    mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
    if mol is None:
        raise FileNotFoundError(f"SDF not found or invalid for CAS={cas} in {sdf_dir}")

    # Load featurizer config
    with open(artifacts_dir / "graph_featurizer.pkl", "rb") as f:
        gcfg = pickle.load(f)

    data = featurize_mol_to_pyg(mol, y=None, cfg=gcfg)

    model_cfg = train_cfg.get("model", {})
    model_name = str(model_cfg.get("name", "mpnn")).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    edge_mlp_hidden_dim = int(model_cfg.get("edge_mlp_hidden_dim", 128))

    in_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1] if hasattr(data, "edge_attr") else 0
    global_dim = int(data.u.shape[1]) if hasattr(data, "u") else 0

    if model_name == "gcn":
        model = GCNRegressor(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, global_dim=global_dim)
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    # Batch info required by pooling
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

    state_path = artifacts_dir / "model_best.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = float(model(data.to(device)).detach().cpu().numpy().reshape(-1)[0])

    # AD
    ad_res = None
    ad_path = artifacts_dir / "ad.pkl"
    if ad_path.exists():
        with open(ad_path, "rb") as f:
            ad_artifact = pickle.load(f)
        query_elements = sorted(get_elements_from_mol(mol).keys())
        query_fp = morgan_bitvect(mol, radius=ad_artifact["morgan_radius"], n_bits=ad_artifact["n_bits"])
        ad_res = applicability_domain(
            query_elements=query_elements,
            training_elements=ad_artifact["training_elements"],
            query_fp=query_fp,
            train_fps=ad_artifact["train_fps"],
            train_ids=ad_artifact["train_ids"],
            top_k=int(ad_artifact.get("top_k", 5)),
            tanimoto_warn_threshold=float(ad_artifact.get("tanimoto_warn_threshold", 0.5)),
        )

    print("=" * 70)
    print("LJ parameter prediction (GNN model)")
    print(f"Query: {args.query} (resolved CAS: {cas})")
    print(f"Predicted target: {pred:.6g}")
    if ad_res is not None:
        if ad_res.max_tanimoto is not None:
            print(f"Nearest-neighbor similarity (Tanimoto): {ad_res.max_tanimoto:.3f}")
        print(f"Trust score: {ad_res.trust_score}/100")
        if ad_res.warnings:
            print("\nWarnings:")
            for w in ad_res.warnings:
                print(f"  - {w}")
        if ad_res.top_k:
            print("\nTop neighbors (similarity, CAS):")
            for sim, cid in ad_res.top_k:
                print(f"  - {sim:.3f}  {cid}")
    else:
        print("AD diagnostics: not available (ad.pkl missing).")
    print("=" * 70)

    result = {
        "cas": cas,
        "query": args.query,
        "prediction": pred,
        "resolve_meta": resolve_meta,
        "ad": None if ad_res is None else ad_res.to_dict(),
    }
    save_json(out_dir / f"prediction_{cas}.json", result)
    logger.info(f"Saved prediction json to {out_dir / f'prediction_{cas}.json'}")


if __name__ == "__main__":
    main()

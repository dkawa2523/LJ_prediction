from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.ad import applicability_domain
from src.common.chemistry import get_elements_from_mol
from src.common.config import load_yaml
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.utils import ensure_dir, get_logger, save_json
from src.fp.featurizer_fp import FPConfig, featurize_mol, morgan_bitvect


def _resolve_cas(mode: str, query: str, dataset_csv: Path) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"mode": mode, "query": query}
    mode = mode.lower()
    if mode == "cas":
        return query.strip(), meta
    if mode == "formula":
        df = read_csv(dataset_csv)
        # exact match
        matches = df[df["MolecularFormula"].astype(str) == query.strip()]
        if len(matches) == 0:
            raise ValueError(f"No CAS found for formula={query} in {dataset_csv}")
        if len(matches) > 1:
            # Ambiguous (isomers). Use first but record.
            meta["warning"] = f"Multiple entries share the same formula. Using first match (n={len(matches)}). Consider using CAS instead."
        cas = str(matches.iloc[0]["CAS"])
        meta["resolved_cas"] = cas
        return cas, meta
    raise ValueError(f"Unknown input mode: {mode}. Use 'cas' or 'formula'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict LJ parameter using fingerprint model with applicability-domain diagnostics.")
    ap.add_argument("--config", required=True, help="Path to configs/fp/predict.yaml")
    ap.add_argument("--query", required=True, help="CAS (e.g. 71-43-2) or MolecularFormula (Hill) depending on config.input.mode")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_artifact_dir = Path(cfg["model_artifact_dir"])
    artifacts_dir = model_artifact_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    # Load training snapshot to guarantee feature consistency
    train_cfg_path = model_artifact_dir / "config_snapshot.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
    train_cfg = load_yaml(train_cfg_path)

    data_cfg = train_cfg.get("data", {})
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
    dataset_csv = Path(cfg.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")))

    out_dir = ensure_dir(Path(cfg.get("output", {}).get("out_dir", "runs/predict")) / cfg.get("output", {}).get("exp_name", "fp_predict"))
    logger = get_logger("fp_predict", log_file=out_dir / "predict.log")

    mode = str(cfg.get("input", {}).get("mode", "formula"))
    cas, resolve_meta = _resolve_cas(mode=mode, query=args.query, dataset_csv=dataset_csv)
    logger.info(f"Resolved CAS={cas} from query={args.query} (mode={mode})")

    mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
    if mol is None:
        raise FileNotFoundError(f"SDF not found or invalid for CAS={cas} in {sdf_dir}")

    feat_cfg = train_cfg.get("featurizer", {})
    fp_cfg = FPConfig(
        fingerprint=str(feat_cfg.get("fingerprint", "morgan")),
        morgan_radius=int(feat_cfg.get("morgan_radius", 2)),
        n_bits=int(feat_cfg.get("n_bits", 2048)),
        use_counts=bool(feat_cfg.get("use_counts", False)),
        add_descriptors=feat_cfg.get("add_descriptors", None),
    )

    x, meta = featurize_mol(mol, fp_cfg)

    # Load preprocessing + model
    with open(artifacts_dir / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    scaler = None
    scaler_path = artifacts_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    with open(artifacts_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X = x.reshape(1, -1)
    X = imputer.transform(X)
    if scaler is not None:
        X = scaler.transform(X)

    pred = float(model.predict(X).reshape(-1)[0])

    # AD
    with open(artifacts_dir / "ad.pkl", "rb") as f:
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

    # Print user-friendly summary
    print("=" * 70)
    print("LJ parameter prediction (Fingerprint model)")
    print(f"Query: {args.query} (resolved CAS: {cas})")
    print(f"Predicted target: {pred:.6g}")
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
    print("=" * 70)

    result = {
        "cas": cas,
        "query": args.query,
        "prediction": pred,
        "resolve_meta": resolve_meta,
        "ad": ad_res.to_dict(),
        "feature_meta": meta,
    }
    save_json(out_dir / f"prediction_{cas}.json", result)
    logger.info(f"Saved prediction json to {out_dir / f'prediction_{cas}.json'}")


if __name__ == "__main__":
    main()

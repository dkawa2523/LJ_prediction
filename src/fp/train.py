from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.common.ad import applicability_domain
from src.common.config import dump_yaml, load_yaml
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.metrics import regression_metrics
from src.common.plots import save_parity_plot, save_residual_plot, save_hist
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json, set_seed
from src.fp.featurizer_fp import FPConfig, featurize_mol, morgan_bitvect
from src.fp.models import get_model


def _hash_cfg(obj: Dict[str, Any]) -> str:
    s = repr(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


def _build_features(
    df: pd.DataFrame,
    sdf_dir: Path,
    cas_col: str,
    fp_cfg: FPConfig,
    cache_dir: Optional[Path],
    cache_key: str,
    logger,
) -> Tuple[np.ndarray, List[Any], List[str], Dict[str, Any]]:
    """Return X (N,D), ids (CAS), elements_list, meta.

    Notes:
      - Rows with missing/invalid SDF are returned as all-NaN feature rows.
      - Descriptor NaNs are allowed; they will be imputed later.
    """
    ensure_dir(cache_dir) if cache_dir is not None else None
    cache_path = None
    meta_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"fp_features_{cache_key}.npz"
        meta_path = cache_dir / f"fp_features_{cache_key}_meta.pkl"
        if cache_path.exists() and meta_path.exists():
            logger.info(f"Loading cached features: {cache_path}")
            npz = np.load(cache_path, allow_pickle=True)
            X = npz["X"]
            ids = npz["ids"].tolist()
            elements = npz["elements"].tolist()
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return X, ids, elements, meta

    ids = df[cas_col].astype(str).tolist()
    elements = df.get("elements", pd.Series([""] * len(df))).astype(str).tolist()
    X_list: List[np.ndarray] = []
    meta_last: Dict[str, Any] = {}

    logger.info("Featurizing molecules (fingerprint + optional descriptors) ...")
    for cas in tqdm(ids, total=len(ids)):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        if mol is None:
            X_list.append(np.array([np.nan], dtype=float))
            continue
        x, meta = featurize_mol(mol, fp_cfg)
        meta_last = meta
        X_list.append(x)

    # Determine feature dimension from first non-placeholder vector
    dims = [x.shape[0] for x in X_list if x.ndim == 1 and x.shape[0] > 1]
    if len(dims) == 0:
        raise RuntimeError("No valid molecules could be featurized. Check sdf_dir and SDF files.")
    dim = int(max(dims))
    X = np.full((len(X_list), dim), np.nan, dtype=float)
    for i, x in enumerate(X_list):
        if x.ndim != 1:
            continue
        if x.shape[0] == 1 and np.isnan(x[0]):
            continue  # placeholder
        if x.shape[0] != dim:
            x2 = np.full((dim,), np.nan, dtype=float)
            n = min(dim, x.shape[0])
            x2[:n] = x[:n]
            x = x2
        X[i] = x

    meta_last["feature_dim"] = dim

    if cache_path is not None:
        logger.info(f"Saving feature cache: {cache_path}")
        np.savez_compressed(
            cache_path,
            X=X,
            ids=np.array(ids, dtype=object),
            elements=np.array(elements, dtype=object),
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta_last, f)

    return X, ids, elements, meta_last


def main() -> None:
    ap = argparse.ArgumentParser(description="Train fingerprint-based regression model for LJ parameter.")
    ap.add_argument("--config", required=True, help="Path to configs/fp/train.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    dataset_csv = Path(data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
    indices_dir = Path(data_cfg.get("indices_dir", "data/processed/indices"))
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
    target_col = str(data_cfg.get("target_col", "lj_epsilon_over_k_K"))
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    cache_dir = data_cfg.get("cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else None

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/fp"))
    exp_name = str(out_cfg.get("exp_name", "fp_experiment"))
    run_dir = ensure_dir(run_dir_root / exp_name)
    plots_dir = ensure_dir(run_dir / "plots")

    logger = get_logger("fp_train", log_file=run_dir / "train.log")

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
    if not sdf_dir.exists():
        raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)
    for k in ["train", "val", "test"]:
        if k not in indices:
            raise FileNotFoundError(f"Split indices missing: {indices_dir}/{k}.txt")

    df_train = df.loc[indices["train"]].copy()
    df_val = df.loc[indices["val"]].copy()
    df_test = df.loc[indices["test"]].copy()
    logger.info(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    feat_cfg = cfg.get("featurizer", {})
    fp_cfg = FPConfig(
        fingerprint=str(feat_cfg.get("fingerprint", "morgan")),
        morgan_radius=int(feat_cfg.get("morgan_radius", 2)),
        n_bits=int(feat_cfg.get("n_bits", 2048)),
        use_counts=bool(feat_cfg.get("use_counts", False)),
        add_descriptors=feat_cfg.get("add_descriptors", None),
    )

    preprocess_cfg = cfg.get("preprocess", {})
    standardize = bool(preprocess_cfg.get("standardize", False))
    impute_strategy = str(preprocess_cfg.get("impute_nan", "mean"))

    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "lightgbm"))
    model_params = model_cfg.get("params", {}) or {}

    # Build features for all rows once (for caching + AD)
    cache_key = _hash_cfg({"featurizer": feat_cfg, "dataset": str(dataset_csv)})
    X_all, ids_all, elements_all, feat_meta = _build_features(
        df=df,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        fp_cfg=fp_cfg,
        cache_dir=cache_dir,
        cache_key=cache_key,
        logger=logger,
    )

    # Build masks for valid feature rows
    missing_mask = np.isnan(X_all).all(axis=1)
    if np.any(missing_mask):
        logger.warning(f"Found {int(np.sum(missing_mask))} rows with missing/invalid SDF; they will be ignored in training/eval.")

    # Utility to pick split arrays
    def pick(split_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[Any]]:
        idx = split_df.index.to_numpy()
        X = X_all[idx]
        y = split_df[target_col].astype(float).to_numpy()
        ids = split_df[cas_col].astype(str).tolist()
        elements = split_df.get("elements", pd.Series([""] * len(split_df))).astype(str).tolist()
        # filter invalid
        m = (~np.isnan(X).all(axis=1)) & np.isfinite(y)
        X = X[m]
        y = y[m]
        ids = [i for i, ok in zip(ids, m.tolist()) if ok]
        elements = [e for e, ok in zip(elements, m.tolist()) if ok]
        return X, y, elements, ids

    X_train, y_train, el_train, ids_train = pick(df_train)
    X_val, y_val, el_val, ids_val = pick(df_val)
    X_test, y_test, el_test, ids_test = pick(df_test)

    logger.info(f"After filtering invalid rows: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    if len(y_train) < 50:
        logger.warning("Training set is very small; consider loosening selectors or split strategy.")

    # Impute NaNs (descriptors may yield NaN)
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model = get_model(model_name, model_params)
    logger.info(f"Training model: {model_name} with params={model_params}")
    fit_kwargs = {}
    # Add eval_set if supported
    if model_name.lower() in {"lightgbm", "lgbm"}:
        fit_kwargs = {"eval_set": [(X_val, y_val)], "eval_metric": "rmse"}
    elif model_name.lower() == "catboost":
        fit_kwargs = {"eval_set": (X_val, y_val), "use_best_model": True}

    model.fit(X_train, y_train, **fit_kwargs)

    # Predictions
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics_val = regression_metrics(y_val, pred_val)
    metrics_test = regression_metrics(y_test, pred_test)
    logger.info(f"Val metrics: {metrics_val}")
    logger.info(f"Test metrics: {metrics_test}")

    # Plots
    if bool(out_cfg.get("plots", True)):
        save_parity_plot(y_val, pred_val, plots_dir / "parity_val.png", title="Parity (val)", xlabel="true", ylabel="pred")
        save_residual_plot(y_val, pred_val, plots_dir / "residual_val.png", title="Residual (val)")
        save_parity_plot(y_test, pred_test, plots_dir / "parity_test.png", title="Parity (test)", xlabel="true", ylabel="pred")
        save_residual_plot(y_test, pred_test, plots_dir / "residual_test.png", title="Residual (test)")
        save_hist(y_train, plots_dir / "y_train_hist.png", title="Target distribution (train)", xlabel=target_col)

        # Learning curves if available
        if model_name.lower() in {"lightgbm", "lgbm"} and hasattr(model, "evals_result_"):
            import matplotlib.pyplot as plt

            evals = model.evals_result_
            # keys can be like {'valid_0': {'rmse': [...]}}
            try:
                k0 = list(evals.keys())[0]
                metric_name = list(evals[k0].keys())[0]
                vals = evals[k0][metric_name]
                plt.figure()
                plt.plot(np.arange(1, len(vals) + 1), vals)
                plt.xlabel("iteration")
                plt.ylabel(metric_name)
                plt.title("LightGBM eval metric (val)")
                plt.tight_layout()
                plt.savefig(plots_dir / "learning_curve_val.png", dpi=200)
                plt.close()
            except Exception:
                pass
        if model_name.lower() == "catboost":
            try:
                import matplotlib.pyplot as plt

                evals = model.get_evals_result()
                # Typically {'learn': {'RMSE': [...]}, 'validation': {'RMSE': [...]}}
                if "validation" in evals:
                    metric_name = list(evals["validation"].keys())[0]
                    vals = evals["validation"][metric_name]
                    plt.figure()
                    plt.plot(np.arange(1, len(vals) + 1), vals)
                    plt.xlabel("iteration")
                    plt.ylabel(metric_name)
                    plt.title("CatBoost eval metric (val)")
                    plt.tight_layout()
                    plt.savefig(plots_dir / "learning_curve_val.png", dpi=200)
                    plt.close()
            except Exception:
                pass

    # Save artifacts
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    model_path = artifacts_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(artifacts_dir / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    if scaler is not None:
        with open(artifacts_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # AD artifacts: training fingerprints and training elements
    # Compute Morgan bitvect for all training mols in ORIGINAL dataset order to simplify indexing.
    # We'll store only training set for speed.
    train_mols = [load_sdf_mol(sdf_path_from_cas(sdf_dir, cas)) for cas in ids_train]
    train_fps = [morgan_bitvect(m, radius=fp_cfg.morgan_radius, n_bits=fp_cfg.n_bits) if m is not None else None for m in train_mols]
    # Filter None
    train_pairs = [(fp, cas) for fp, cas in zip(train_fps, ids_train) if fp is not None]
    train_fps = [p[0] for p in train_pairs]
    train_ids_for_ad = [p[1] for p in train_pairs]

    training_elements = sorted({el for e_str in el_train for el in e_str.split(",") if el})
    heavy_atoms_train = df_train["n_heavy_atoms"].dropna().astype(int).tolist()
    heavy_min = int(min(heavy_atoms_train)) if heavy_atoms_train else 0
    heavy_max = int(max(heavy_atoms_train)) if heavy_atoms_train else 0

    ad_artifact = {
        "training_elements": training_elements,
        "heavy_atom_range": [heavy_min, heavy_max],
        "morgan_radius": fp_cfg.morgan_radius,
        "n_bits": fp_cfg.n_bits,
        "train_ids": train_ids_for_ad,
        "train_fps": train_fps,  # RDKit ExplicitBitVect list (pickleable)
        "tanimoto_warn_threshold": float(cfg.get("ad", {}).get("tanimoto_warn_threshold", 0.5)),
        "top_k": int(cfg.get("ad", {}).get("top_k", 5)),
    }
    with open(artifacts_dir / "ad.pkl", "wb") as f:
        pickle.dump(ad_artifact, f)

    # Save metrics + config snapshot
    save_json(run_dir / "metrics_val.json", metrics_val)
    save_json(run_dir / "metrics_test.json", metrics_test)
    dump_yaml(run_dir / "config_snapshot.yaml", cfg)

    logger.info(f"Saved model to {model_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running as `python scripts/build_dataset.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import dump_yaml, load_config
from src.common.chemistry import elements_string, n_elements, parse_formula
from src.common.dataset_selectors import SelectorContext, apply_selectors
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas, write_csv
from src.common.lj import compute_lj
from src.common.splitters import (
    build_group_map,
    group_split,
    random_split,
    save_split_indices,
    save_split_json,
    scaffold_split,
    validate_group_leakage,
    validate_scaffold_split,
    validate_split_indices,
)
from src.common.utils import get_logger, set_seed
from src.utils.validate_config import validate_config


def _needs_mols(selectors: List[Dict[str, Any]]) -> bool:
    need = {"diversity_farthest_point", "butina_cluster"}
    for s in selectors:
        name = str(s.get("name", "")).lower()
        if name in need:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Build processed dataset: add LJ params, compute meta columns, apply selectors, make splits.")
    ap.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    ap.add_argument("--limit", type=int, default=None, help="Debug: limit number of rows loaded from raw CSV.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    paths = cfg.get("paths", {})
    raw_csv = Path(paths.get("raw_csv", "data/raw/tc_pc_tb_pubchem.csv"))
    sdf_dir = Path(paths.get("sdf_dir", "data/raw/sdf_files"))
    out_csv = Path(paths.get("out_csv", "data/processed/dataset_with_lj.csv"))
    out_indices_dir = Path(paths.get("out_indices_dir", "data/processed/indices"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger = get_logger("build_dataset", log_file=out_csv.parent / "build_dataset.log")

    if not raw_csv.exists():
        raise FileNotFoundError(f"raw_csv not found: {raw_csv}")
    if not sdf_dir.exists():
        logger.warning(f"sdf_dir does not exist: {sdf_dir} (SDF-based columns will be NaN; selectors requiring SDF will fail).")

    df = read_csv(raw_csv)
    df["raw_row_index"] = df.index.astype(int)
    logger.info(f"Loaded {df.shape[0]} rows from {raw_csv}")

    limit_rows_cfg = cfg.get("limit_rows", None)
    limit_rows = args.limit if args.limit is not None else (int(limit_rows_cfg) if limit_rows_cfg else None)
    if limit_rows is not None and int(limit_rows) > 0:
        before = df.shape[0]
        df = df.head(int(limit_rows)).copy()
        logger.warning(f"Limiting rows for debug: {before} -> {df.shape[0]}")

    # Column mapping (configurable)
    cols = cfg.get("columns", {})
    cas_col = cols.get("cas", "CAS")
    formula_col = cols.get("formula", "MolecularFormula")
    tc_col = cols.get("tc", "Tc [K]")
    pc_col = cols.get("pc", "Pc [Pa]")
    tb_col = cols.get("tb", "Tb [K]")

    # Compute formula-derived element stats
    element_counts_global: Dict[str, int] = {}
    elements_col_values = []
    n_elements_values = []

    logger.info("Parsing formulas and counting elements...")
    for formula in tqdm(df[formula_col].astype(str).tolist(), total=df.shape[0]):
        counts = parse_formula(formula)
        e_str = elements_string(counts)
        elements_col_values.append(e_str)
        n_elements_values.append(n_elements(counts))
        for el, n in counts.items():
            element_counts_global[el] = element_counts_global.get(el, 0) + 1  # count occurrence by molecule, not atom

    df["elements"] = elements_col_values
    df["n_elements"] = n_elements_values

    # Load SDF and compute size columns
    n_atoms, n_heavy = [], []
    mols: List[Any] = []
    missing_sdf = 0
    logger.info("Reading SDF files to compute n_atoms / n_heavy_atoms ...")
    for cas in tqdm(df[cas_col].astype(str).tolist(), total=df.shape[0]):
        p = sdf_path_from_cas(sdf_dir, cas)
        mol = load_sdf_mol(p) if sdf_dir.exists() else None
        if mol is None:
            missing_sdf += 1
            n_atoms.append(np.nan)
            n_heavy.append(np.nan)
        else:
            n_atoms.append(int(mol.GetNumAtoms()))
            n_heavy.append(int(mol.GetNumHeavyAtoms()))
        mols.append(mol)

    df["n_atoms"] = n_atoms
    df["n_heavy_atoms"] = n_heavy
    if missing_sdf > 0:
        logger.warning(f"Missing/invalid SDF for {missing_sdf} rows; they may be skipped later.")

    # Compute LJ parameters
    lj_cfg = cfg.get("lj", {})
    eps_method = str(lj_cfg.get("epsilon_method", "bird_critical"))
    sig_method = str(lj_cfg.get("sigma_method", "bird_critical"))
    eps_col = str(lj_cfg.get("epsilon_col", "lj_epsilon_over_k_K"))
    sig_col = str(lj_cfg.get("sigma_col", "lj_sigma_A"))

    eps_list, sig_list = [], []
    valid_eps_range = lj_cfg.get("valid_range", {}).get("epsilon_over_k_K", None)
    valid_sig_range = lj_cfg.get("valid_range", {}).get("sigma_A", None)

    logger.info(f"Computing LJ params (epsilon_method={eps_method}, sigma_method={sig_method}) ...")
    for Tc, Pc, Tb in tqdm(zip(df[tc_col].tolist(), df[pc_col].tolist(), df[tb_col].tolist()), total=df.shape[0]):
        out = compute_lj(Tc_K=_to_float_or_none(Tc), Pc_Pa=_to_float_or_none(Pc), Tb_K=_to_float_or_none(Tb),
                         epsilon_method=eps_method, sigma_method=sig_method)
        eps_list.append(out["lj_epsilon_over_k_K"])
        sig_list.append(out["lj_sigma_A"])

    df[eps_col] = eps_list
    df[sig_col] = sig_list

    # LJ validity flag
    df["lj_valid_flag"] = True
    df.loc[df[eps_col].isna() | df[sig_col].isna(), "lj_valid_flag"] = False
    if valid_eps_range:
        lo, hi = float(valid_eps_range[0]), float(valid_eps_range[1])
        df.loc[(df[eps_col] < lo) | (df[eps_col] > hi), "lj_valid_flag"] = False
    if valid_sig_range:
        lo, hi = float(valid_sig_range[0]), float(valid_sig_range[1])
        df.loc[(df[sig_col] < lo) | (df[sig_col] > hi), "lj_valid_flag"] = False

    # Apply selectors
    selectors = cfg.get("selectors", []) or []
    ctx = SelectorContext(element_counts=element_counts_global, mols=mols if _needs_mols(selectors) else None, fps=None)

    # Always filter invalid LJ first unless user opted out
    if cfg.get("filter_invalid_lj", True):
        before = df.shape[0]
        df = df[df["lj_valid_flag"]].copy()
        logger.info(f"Filtered invalid LJ rows: {before} -> {df.shape[0]}")

    if selectors:
        before = df.shape[0]
        df = apply_selectors(df, selectors=selectors, ctx=ctx)
        logger.info(f"Applied selectors: {before} -> {df.shape[0]}")

    # IMPORTANT: reset index before saving, because indices are stored separately
    # and the CSV is written with index=False.
    df = df.reset_index(drop=True)

    # Save processed dataset
    write_csv(df, out_csv)
    logger.info(f"Saved processed dataset: {out_csv} (rows={df.shape[0]})")

    # Split
    split_cfg = cfg.get("split", {})
    split_method = str(split_cfg.get("method", "random")).lower()
    ratios = split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1]))
    ratios = [float(x) for x in ratios]
    split_seed = int(split_cfg.get("seed", seed))

    if split_method == "random":
        indices = random_split(df, ratios=ratios, seed=split_seed)
    elif split_method == "scaffold":
        indices = scaffold_split(df, sdf_dir=sdf_dir, cas_col=cas_col, ratios=ratios, seed=split_seed)
        validate_scaffold_split(df, sdf_dir=sdf_dir, cas_col=cas_col, indices=indices)
    elif split_method == "group":
        group_key = split_cfg.get("group_key")
        if not group_key:
            raise ValueError("split.method=group requires split.group_key")
        group_col = cols.get(str(group_key), str(group_key))
        if group_col not in df.columns:
            raise ValueError(f"split.group_key '{group_key}' resolved to '{group_col}' not in dataset columns")
        indices = group_split(df, group_col=group_col, ratios=ratios, seed=split_seed)
        group_map = build_group_map(df, group_col=group_col)
        validate_group_leakage(indices, group_map, label=f"group:{group_col}")
    else:
        raise ValueError(f"Unknown split.method: {split_method}")

    validate_split_indices(indices)
    save_split_indices(indices, out_indices_dir)
    split_meta = {"method": split_method, "seed": split_seed, "fractions": ratios}
    if split_method == "group":
        split_meta["group_key"] = str(split_cfg.get("group_key"))
    save_split_json(indices, out_indices_dir / "split.json", metadata=split_meta)
    logger.info(f"Saved split indices to {out_indices_dir}")

    # Save snapshot of config used
    dump_yaml(out_csv.parent / "dataset_config_snapshot.yaml", cfg)
    logger.info("Done.")


def _to_float_or_none(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


if __name__ == "__main__":
    main()

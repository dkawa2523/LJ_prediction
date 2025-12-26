from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .chemistry import murcko_scaffold_smiles
from .io import load_sdf_mol, sdf_path_from_cas
from .utils import save_json


def random_split(
    df: pd.DataFrame,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    n = len(idx)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    n_test = n - n_train - n_val
    if n_train == 0:
        n_train = 1
        n_test = n - n_train - n_val

    # Keep val/test non-empty when possible to avoid downstream training failures.
    if n >= 3:
        if n_val == 0:
            n_val = 1
        if n_test == 0:
            n_test = 1
        n_train = n - n_val - n_test
        if n_train <= 0:
            # Fallback: make train at least 1 by borrowing from the largest split.
            n_train = 1
            remaining = n - n_train
            # split remaining between val/test (prefer preserving non-empty)
            n_val = max(1, min(n_val, remaining - 1))
            n_test = remaining - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}


def _normalize_group_key(value: object, row_idx: int) -> str:
    if value is None:
        return f"__missing__{row_idx}"
    if isinstance(value, float) and np.isnan(value):
        return f"__missing__{row_idx}"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return f"__missing__{row_idx}"
    return text


def build_group_map(df: pd.DataFrame, group_col: str) -> Dict[int, str]:
    group_map: Dict[int, str] = {}
    for row_idx, value in zip(df.index.tolist(), df[group_col].tolist()):
        group_map[row_idx] = _normalize_group_key(value, row_idx)
    return group_map


def _split_groups_by_count(
    groups: Iterable[List[int]],
    ratios: Sequence[float],
    seed: int,
) -> Dict[str, List[int]]:
    assert len(ratios) == 3, "ratios must be a 3-length sequence"
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = np.random.default_rng(seed)

    group_list = [list(g) for g in groups if g]
    if not group_list:
        return {"train": [], "val": [], "test": []}

    group_list.sort(key=len, reverse=True)
    rng.shuffle(group_list)

    n_total = sum(len(g) for g in group_list)
    n_train = int(ratios[0] * n_total)
    n_val = int(ratios[1] * n_total)

    train, val, test = [], [], []
    for g in group_list:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)
    return {"train": train, "val": val, "test": test}


def _validate_non_empty_splits(indices: Dict[str, List[int]], ratios: Sequence[float]) -> None:
    for split_name, ratio in zip(["train", "val", "test"], ratios):
        if ratio > 0 and len(indices.get(split_name, [])) == 0:
            raise ValueError(f"Split '{split_name}' is empty. Adjust ratios or split method.")


def group_split(
    df: pd.DataFrame,
    group_col: str,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Group-based split.

    Rows sharing the same group key are kept in the same split to prevent leakage.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    if group_col not in df.columns:
        raise ValueError(f"group_col not in dataframe: {group_col}")

    groups: Dict[str, List[int]] = {}
    group_map = build_group_map(df, group_col)
    for row_idx, key in group_map.items():
        groups.setdefault(key, []).append(row_idx)

    indices = _split_groups_by_count(groups.values(), ratios=ratios, seed=seed)
    _validate_non_empty_splits(indices, ratios)
    return indices


def scaffold_split(
    df: pd.DataFrame,
    sdf_dir: str | Path,
    cas_col: str = "CAS",
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Murcko-scaffold split.

    Groups molecules by scaffold and assigns groups to train/val/test.
    This helps reduce train-test leakage by placing similar scaffolds together.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    scaffolds: Dict[str, List[int]] = {}
    for row_idx, cas in zip(df.index.tolist(), df[cas_col].astype(str).tolist()):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        scaf = murcko_scaffold_smiles(mol) if mol is not None else ""
        scaffolds.setdefault(scaf, []).append(row_idx)

    indices = _split_groups_by_count(scaffolds.values(), ratios=ratios, seed=seed)
    _validate_non_empty_splits(indices, ratios)
    return indices


def validate_split_indices(indices: Dict[str, List[int]]) -> None:
    seen = set()
    duplicates = set()
    for split_name, idxs in indices.items():
        for idx in idxs:
            if idx in seen:
                duplicates.add(idx)
            else:
                seen.add(idx)
    if duplicates:
        dup_list = sorted(duplicates)
        raise ValueError(f"Duplicate indices across splits: {dup_list[:5]}")


def validate_group_leakage(indices: Dict[str, List[int]], group_map: Dict[int, str], label: str) -> None:
    group_to_split: Dict[str, str] = {}
    leakage: Dict[str, List[str]] = {}
    for split_name, idxs in indices.items():
        for idx in idxs:
            group = group_map.get(idx, "")
            prev = group_to_split.get(group)
            if prev is None:
                group_to_split[group] = split_name
            elif prev != split_name:
                leakage.setdefault(group, [prev]).append(split_name)
    if leakage:
        sample = list(leakage.items())[:3]
        raise ValueError(f"{label} leakage across splits detected: {sample}")


def validate_scaffold_split(
    df: pd.DataFrame,
    sdf_dir: str | Path,
    cas_col: str,
    indices: Dict[str, List[int]],
) -> None:
    scaffold_map: Dict[int, str] = {}
    for row_idx, cas in zip(df.index.tolist(), df[cas_col].astype(str).tolist()):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        scaf = murcko_scaffold_smiles(mol) if mol is not None else ""
        scaffold_map[row_idx] = scaf
    validate_group_leakage(indices, scaffold_map, label="scaffold")


def save_split_json(
    indices: Dict[str, List[int]],
    out_path: str | Path,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "indices": {
            "train": indices.get("train", []),
            "val": indices.get("val", []),
            "test": indices.get("test", []),
        }
    }
    if metadata:
        payload.update(metadata)
    save_json(out_path, payload)


def save_split_indices(indices: Dict[str, List[int]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, idxs in indices.items():
        p = out_dir / f"{split_name}.txt"
        with p.open("w", encoding="utf-8") as f:
            for i in idxs:
                f.write(f"{i}\n")


def load_split_indices(indices_dir: str | Path) -> Dict[str, List[int]]:
    indices_dir = Path(indices_dir)
    out = {}
    for split_name in ["train", "val", "test"]:
        p = indices_dir / f"{split_name}.txt"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            out[split_name] = [int(line.strip()) for line in f if line.strip()]
    return out

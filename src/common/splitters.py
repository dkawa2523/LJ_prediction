from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .chemistry import murcko_scaffold_smiles
from .io import load_sdf_mol, sdf_path_from_cas


def random_split(
    df: pd.DataFrame,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
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
    rng = np.random.default_rng(seed)

    scaffolds: Dict[str, List[int]] = {}
    for row_idx, cas in zip(df.index.tolist(), df[cas_col].astype(str).tolist()):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        scaf = murcko_scaffold_smiles(mol) if mol is not None else ""
        scaffolds.setdefault(scaf, []).append(row_idx)

    # Sort scaffold groups by size descending to pack large scaffolds first
    groups = list(scaffolds.values())
    groups.sort(key=len, reverse=True)
    # Shuffle groups with same size for randomness
    # (deterministic but not critical; we do a simple shuffle of entire list after stable sort)
    rng.shuffle(groups)

    n_total = df.shape[0]
    n_train = int(ratios[0] * n_total)
    n_val = int(ratios[1] * n_total)

    train, val, test = [], [], []
    for g in groups:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)

    # Safety: if val is empty (can happen for extreme grouping), fallback to random split
    if len(val) == 0 or len(test) == 0:
        return random_split(df, ratios=ratios, seed=seed)

    return {"train": train, "val": val, "test": test}


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

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def _configure_matplotlib_cache() -> None:
    """
    Ensure Matplotlib uses a writable config/cache directory.

    Some environments (e.g. sandboxed runs, containers) cannot write to $HOME,
    which can make Matplotlib imports very slow or noisy due to repeated cache builds.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_root = Path.cwd() / ".cache"
    cache_dir = cache_root / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


_configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np


def save_learning_curve(loss_train: Sequence[float], loss_val: Sequence[float], out_path: str | Path, ylabel: str = "loss") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(loss_train) + 1)
    plt.figure()
    plt.plot(x, loss_train, label="train")
    if loss_val is not None and len(loss_val) == len(loss_train):
        plt.plot(x, loss_val, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_parity_plot(y_true, y_pred, out_path: str | Path, title: Optional[str] = None, xlabel: str = "true", ylabel: str = "pred") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residual_plot(y_true, y_pred, out_path: str | Path, title: Optional[str] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, resid, s=12, alpha=0.7)
    plt.axhline(0.0)
    plt.xlabel("true")
    plt.ylabel("residual (pred-true)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_hist(values, out_path: str | Path, title: str, xlabel: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    plt.figure()
    plt.hist(v, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

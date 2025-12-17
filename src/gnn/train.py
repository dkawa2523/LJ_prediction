from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config import dump_yaml, load_yaml
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.metrics import regression_metrics
from src.common.plots import save_learning_curve, save_parity_plot, save_residual_plot, save_hist
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json, set_seed
from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg
from src.gnn.models import GCNRegressor, MPNNRegressor

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Adam = None
    DataLoader = None

try:
    from torch_geometric.typing import WITH_TORCH_SCATTER, WITH_TORCH_SPARSE
except Exception:  # pragma: no cover
    WITH_TORCH_SCATTER = False
    WITH_TORCH_SPARSE = False


def _require_pyg():
    if torch is None or DataLoader is None:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for GNN training. "
            "Install torch and torch_geometric (matching your environment)."
        )


def _select_device(cfg_train: Dict[str, Any]) -> "torch.device":
    prefer = str(cfg_train.get("device", "auto")).lower()
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
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def _warn_if_pyg_extensions_missing(logger) -> None:
    if WITH_TORCH_SCATTER and WITH_TORCH_SPARSE:
        return
    missing = []
    if not WITH_TORCH_SCATTER:
        missing.append("torch_scatter")
    if not WITH_TORCH_SPARSE:
        missing.append("torch_sparse")
    logger.warning(
        "PyG optional extensions are missing (%s). Training can be extremely slow and may look like a hang. "
        "Install matching wheels from https://data.pyg.org/whl/ or follow the official PyG install guide.",
        ", ".join(missing),
    )


def _set_mps_memory_env(cfg_train: Dict[str, Any], logger) -> None:
    """
    Optionally set MPS memory watermark env var.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO must be set before the first MPS allocation to be effective.
    Leaving it unset is safest; lowering it can avoid OOMs but may reduce usable memory.
    Setting it to 0.0 removes the limit and may destabilize the system.
    """
    if torch is None:
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_built()):
        return
    ratio = cfg_train.get("mps_high_watermark_ratio", None)
    if ratio is None:
        return
    try:
        ratio_f = float(ratio)
    except Exception:
        logger.warning("Invalid train.mps_high_watermark_ratio=%r (expected float). Ignoring.", ratio)
        return
    import os

    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        logger.info("PYTORCH_MPS_HIGH_WATERMARK_RATIO is already set; leaving as-is.")
        return
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(ratio_f)
    logger.warning("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=%s (config).", ratio_f)


def _is_oom_error(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "mps backend out of memory" in s


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GNN model (PyTorch Geometric) for LJ parameter regression.")
    ap.add_argument("--config", required=True, help="Path to configs/gnn/train.yaml")
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

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/gnn"))
    exp_name = str(out_cfg.get("exp_name", "gnn_experiment"))
    run_dir = ensure_dir(run_dir_root / exp_name)
    plots_dir = ensure_dir(run_dir / "plots")
    artifacts_dir = ensure_dir(run_dir / "artifacts")

    logger = get_logger("gnn_train", log_file=run_dir / "train.log")

    try:
        _require_pyg()
    except Exception as e:
        logger.error(str(e))
        raise
    _warn_if_pyg_extensions_missing(logger)
    _set_mps_memory_env(cfg.get("train", {}) or {}, logger)

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)

    # Featurizer config
    feat_cfg = cfg.get("featurizer", {})
    gcfg = GraphFeaturizerConfig(
        node_features=list(feat_cfg.get("node_features", ["atomic_num", "degree", "formal_charge", "aromatic", "num_h", "in_ring"])),
        edge_features=list(feat_cfg.get("edge_features", ["bond_type", "conjugated", "aromatic"])),
        use_3d_pos=bool(feat_cfg.get("use_3d_pos", True)),
        add_global_descriptors=feat_cfg.get("add_global_descriptors", None),
    )

    def build_dataset(split_name: str) -> List[Any]:
        split_df = df.loc[indices[split_name]]
        data_list = []
        for cas, y in tqdm(zip(split_df[cas_col].astype(str).tolist(), split_df[target_col].astype(float).tolist()), total=len(split_df), desc=f"featurize {split_name}"):
            mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
            if mol is None or not np.isfinite(y):
                continue
            try:
                data = featurize_mol_to_pyg(mol, y=y, cfg=gcfg)
                data_list.append(data)
            except Exception:
                continue
        return data_list

    train_data = build_dataset("train")
    val_data = build_dataset("val")
    test_data = build_dataset("test")

    logger.info(f"Data sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    if len(train_data) == 0:
        raise RuntimeError("No valid training graphs were created. Check sdf_dir, target values, and featurizer settings.")

    # Model config
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "mpnn")).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    edge_mlp_hidden_dim = int(model_cfg.get("edge_mlp_hidden_dim", 128))

    in_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if hasattr(train_data[0], "edge_attr") else 0
    global_dim = int(train_data[0].u.shape[1]) if hasattr(train_data[0], "u") else 0

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

    train_cfg = cfg.get("train", {})
    device = _select_device(train_cfg)
    num_threads = train_cfg.get("num_threads", None)
    if num_threads is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(
        f"Model params: {n_params/1e6:.2f}M (~{(n_params*4)/1e6:.1f} MB fp32, excluding optimizer state)"
    )
    logger.info(f"torch={torch.__version__} device={device} num_threads={torch.get_num_threads()}")
    model = model.to(device)

    epochs = int(train_cfg.get("epochs", 200))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    loss_name = str(train_cfg.get("loss", "mse")).lower()

    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    loader_kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent_workers}
    if num_workers > 0 and "prefetch_factor" in train_cfg:
        loader_kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
    logger.info(f"DataLoader: batches(train)={len(train_loader)} batch_size={batch_size} num_workers={num_workers} pin_memory={pin_memory}")

    best_val_rmse = float("inf")
    best_path = artifacts_dir / "model_best.pt"
    history_train, history_val = [], []

    patience = int(train_cfg.get("early_stopping", {}).get("patience", 20))
    bad_epochs = 0

    def eval_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch).detach().cpu().numpy()
                y = batch.y.view(-1).detach().cpu().numpy()
                ys.append(y)
                ps.append(pred)
        if not ys:
            return {}, np.array([]), np.array([])
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        return regression_metrics(y_true, y_pred), y_true, y_pred

    log_interval_sec = float(train_cfg.get("log_interval_sec", 30.0))
    show_pbar = bool(train_cfg.get("progress_bar", True))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    max_batches_per_epoch = int(max_batches_per_epoch) if max_batches_per_epoch is not None else None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        logger.info(f"Epoch {epoch:04d} start")
        epoch_iter = train_loader
        if show_pbar:
            epoch_iter = tqdm(train_loader, total=len(train_loader), desc=f"train epoch {epoch}", leave=False)
        import time

        start_t = time.monotonic()
        last_log_t = start_t
        n_batches = len(train_loader)

        for step, batch in enumerate(epoch_iter, start=1):
            try:
                batch = batch.to(device)
                opt.zero_grad()
                pred = model(batch)
                y = batch.y.view(-1)
                loss = criterion(pred, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            except RuntimeError as e:
                if _is_oom_error(e):
                    try:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps" and hasattr(torch, "mps"):
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                    logger.error(
                        "Out of memory on device=%s during training. "
                        "Try reducing train.batch_size / model.hidden_dim / model.num_layers, "
                        "or set train.device: cpu in the config.",
                        device,
                    )
                raise
            if max_batches_per_epoch is not None and step >= max_batches_per_epoch:
                break

            now = time.monotonic()
            if log_interval_sec > 0 and (now - last_log_t) >= log_interval_sec:
                elapsed = now - start_t
                avg_loss = float(np.mean(losses[-10:])) if losses else float("nan")
                eta = (elapsed / step) * (n_batches - step) if step > 0 else float("nan")
                logger.info(
                    f"Epoch {epoch:04d} step {step}/{n_batches} "
                    f"avg_loss(last10)={avg_loss:.6g} elapsed={elapsed:.1f}s eta~{eta:.1f}s"
                )
                last_log_t = now
        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_metrics, _, _ = eval_loader(val_loader)
        val_rmse = val_metrics.get("rmse", float("inf"))

        history_train.append(train_loss)
        history_val.append(val_rmse)
        logger.info(f"Epoch {epoch:04d}: train_loss={train_loss:.6g} val_rmse={val_rmse:.6g}")

        if (val_rmse < best_val_rmse) or (not best_path.exists()):
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best
    model.load_state_dict(torch.load(best_path, map_location=device))

    val_metrics, yv, pv = eval_loader(val_loader)
    test_metrics, yt, pt = eval_loader(test_loader)
    logger.info(f"Best val metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    # Plots
    if bool(out_cfg.get("plots", True)):
        save_learning_curve(history_train, history_val, plots_dir / "learning_curve.png", ylabel="loss / rmse")
        if len(yv) > 0:
            save_parity_plot(yv, pv, plots_dir / "parity_val.png", title="Parity (val)", xlabel="true", ylabel="pred")
            save_residual_plot(yv, pv, plots_dir / "residual_val.png", title="Residual (val)")
        if len(yt) > 0:
            save_parity_plot(yt, pt, plots_dir / "parity_test.png", title="Parity (test)", xlabel="true", ylabel="pred")
            save_residual_plot(yt, pt, plots_dir / "residual_test.png", title="Residual (test)")
        save_hist([d.y.item() for d in train_data], plots_dir / "y_train_hist.png", title="Target distribution (train)", xlabel=target_col)

    # Save artifacts
    with open(artifacts_dir / "graph_featurizer.pkl", "wb") as f:
        pickle.dump(gcfg, f)
    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
    save_json(run_dir / "metrics_val.json", val_metrics)
    save_json(run_dir / "metrics_test.json", test_metrics)

    # AD artifacts (for inference-time applicability domain)
    try:
        from src.fp.featurizer_fp import morgan_bitvect

        train_df_for_ad = df.loc[indices["train"]].copy()
        train_ids_for_ad = train_df_for_ad[cas_col].astype(str).tolist()
        train_elements_str = train_df_for_ad.get("elements", pd.Series([""] * len(train_df_for_ad))).astype(str).tolist()
        training_elements = sorted({el for e_str in train_elements_str for el in e_str.split(",") if el})

        train_mols = [load_sdf_mol(sdf_path_from_cas(sdf_dir, cas)) for cas in train_ids_for_ad]
        train_fps = [morgan_bitvect(m, radius=2, n_bits=2048) if m is not None else None for m in train_mols]
        train_pairs = [(fp, cas) for fp, cas in zip(train_fps, train_ids_for_ad) if fp is not None]
        train_fps = [p[0] for p in train_pairs]
        train_ids_for_ad = [p[1] for p in train_pairs]

        heavy_atoms_train = train_df_for_ad["n_heavy_atoms"].dropna().astype(int).tolist()
        heavy_min = int(min(heavy_atoms_train)) if heavy_atoms_train else 0
        heavy_max = int(max(heavy_atoms_train)) if heavy_atoms_train else 0

        ad_artifact = {
            "training_elements": training_elements,
            "heavy_atom_range": [heavy_min, heavy_max],
            "morgan_radius": 2,
            "n_bits": 2048,
            "train_ids": train_ids_for_ad,
            "train_fps": train_fps,
            "tanimoto_warn_threshold": float(cfg.get("ad", {}).get("tanimoto_warn_threshold", 0.5)),
            "top_k": int(cfg.get("ad", {}).get("top_k", 5)),
        }
        with open(artifacts_dir / "ad.pkl", "wb") as f:
            pickle.dump(ad_artifact, f)
        logger.info(f"Saved AD artifact to {artifacts_dir / 'ad.pkl'}")
    except Exception as e:
        logger.warning(f"Failed to create AD artifact (will disable AD in predict): {e}")

    logger.info(f"Saved best model to {best_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

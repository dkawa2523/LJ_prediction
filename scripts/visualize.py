from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python scripts/visualize.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import dump_yaml, load_config
from src.common.meta import build_meta, save_meta
from src.common.plots import save_hist, save_parity_plot, save_residual_plot
from src.common.utils import ensure_dir, get_logger
from src.utils.validate_config import validate_config


def _resolve_predictions_path(cfg) -> Path:
    input_cfg = cfg.get("input", {})
    pred_path = input_cfg.get("predictions_path")
    if pred_path:
        return Path(pred_path)
    eval_dir = input_cfg.get("evaluate_run_dir")
    if eval_dir:
        return Path(eval_dir) / "predictions.csv"
    raise ValueError("visualize requires input.predictions_path or input.evaluate_run_dir")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize predictions (parity/residual/hist).")
    ap.add_argument("--config", required=True, help="Path to a composed visualize config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    output_cfg = cfg.get("output", {})
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(output_cfg.get("exp_name", experiment_cfg.get("name", "visualize")))
    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/visualize")) / exp_name)
    plots_dir = ensure_dir(out_dir / "plots")
    logger = get_logger("visualize", log_file=out_dir / "visualize.log")

    dump_yaml(out_dir / "config.yaml", cfg)
    save_meta(
        out_dir,
        build_meta(process_name=str(cfg.get("process", {}).get("name", "visualize")), cfg=cfg),
    )

    pred_path = _resolve_predictions_path(cfg)
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {pred_path}")

    df = pd.read_csv(pred_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("predictions.csv must contain y_true and y_pred columns for visualization.")

    plots_cfg = cfg.get("plots", {})
    splits = plots_cfg.get("splits", ["val", "test"])
    include_train_hist = bool(plots_cfg.get("include_train_hist", True))

    has_split = "split" in df.columns
    for split_name in splits:
        if has_split:
            split_df = df[df["split"] == split_name]
        else:
            if split_name != "all":
                continue
            split_df = df
        if split_df.empty:
            continue
        save_parity_plot(
            split_df["y_true"].to_numpy(),
            split_df["y_pred"].to_numpy(),
            plots_dir / f"parity_{split_name}.png",
            title=f"Parity ({split_name})",
            xlabel="true",
            ylabel="pred",
        )
        save_residual_plot(
            split_df["y_true"].to_numpy(),
            split_df["y_pred"].to_numpy(),
            plots_dir / f"residual_{split_name}.png",
            title=f"Residual ({split_name})",
        )

    if include_train_hist and has_split:
        train_df = df[df["split"] == "train"]
        if not train_df.empty:
            save_hist(train_df["y_true"].to_numpy(), plots_dir / "y_train_hist.png", title="Target distribution (train)", xlabel="y_true")

    logger.info("Done.")


if __name__ == "__main__":
    main()

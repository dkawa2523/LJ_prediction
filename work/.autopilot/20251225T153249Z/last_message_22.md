Using skills: S90_artifacts_versioning + S95_tests_ci to align artifacts and add contract tests.

Prepared a proposed artifact-contract implementation (new artifacts helper, train/predict/evaluate updates, contract tests). I couldn’t apply changes because the sandbox is read‑only.

**Change Plan**
- `src/utils/artifacts.py`: add artifact hashing/meta helpers + contract validation utilities.
- `src/fp/train.py`: use artifact meta, write `metrics.json`, emit `model/model.ckpt` + `preprocess.pkl`.
- `src/gnn/train.py`: use artifact meta, write `metrics.json`, copy `model/model.ckpt`.
- `src/fp/predict.py`: save `predictions.csv`, add `model_version`/`run_id` in meta and outputs.
- `src/gnn/predict.py`: same as FP predict with model versioning.
- `src/fp/evaluate.py`: add `metrics.json`, enrich predictions columns, use artifact meta.
- `src/gnn/evaluate.py`: same as FP evaluate.
- `scripts/visualize.py`: use artifact meta + upstream info.
- `scripts/audit_dataset.py`: use artifact meta.
- `tests/contract/test_artifact_contract.py`: add required file/key/column contract tests.

**Proposed Diff**
```diff
diff --git a/src/utils/artifacts.py b/src/utils/artifacts.py
new file mode 100644
--- /dev/null
+++ b/src/utils/artifacts.py
@@
+from __future__ import annotations
+
+import csv
+import hashlib
+import json
+import os
+import subprocess
+from pathlib import Path
+from typing import Any, Dict, Iterable, Optional
+from uuid import uuid4
+
+from src.common.meta import utc_now_iso
+from src.common.utils import load_json
+
+REQUIRED_META_KEYS = {
+    "run_id",
+    "process_name",
+    "created_at",
+    "git_sha",
+    "dataset_hash",
+    "config_hash",
+    "task_name",
+    "model_name",
+    "featureset_name",
+    "upstream_artifacts",
+    "tags",
+}
+
+REQUIRED_PREDICTION_COLUMNS = {"sample_id", "y_pred"}
+
+PROCESS_REQUIRED_FILES = {
+    "train": [Path("metrics.json"), Path("model") / "model.ckpt"],
+    "evaluate": [Path("metrics.json"), Path("predictions.csv")],
+    "predict": [Path("predictions.csv")],
+    "visualize": [Path("plots")],
+}
+
+
+def _hash_file(path: Path) -> str:
+    h = hashlib.sha256()
+    with path.open("rb") as f:
+        for chunk in iter(lambda: f.read(1024 * 1024), b""):
+            h.update(chunk)
+    return h.hexdigest()
+
+
+def hash_dict(obj: Dict[str, Any]) -> str:
+    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
+    return hashlib.sha256(payload).hexdigest()
+
+
+def compute_config_hash(cfg: Dict[str, Any]) -> str:
+    return hash_dict(cfg)
+
+
+def resolve_dataset_paths(cfg: Dict[str, Any]) -> tuple[Optional[Path], Optional[Path]]:
+    data_cfg = cfg.get("data", {})
+    dataset_csv = data_cfg.get("dataset_csv") or cfg.get("dataset_csv")
+    indices_dir = data_cfg.get("indices_dir") or cfg.get("indices_dir")
+    return (Path(dataset_csv) if dataset_csv else None, Path(indices_dir) if indices_dir else None)
+
+
+def compute_dataset_hash(dataset_csv: Optional[Path], indices_dir: Optional[Path] = None) -> Optional[str]:
+    if dataset_csv is None:
+        return None
+    dataset_csv = Path(dataset_csv)
+    if not dataset_csv.exists():
+        return None
+    h = hashlib.sha256()
+    h.update(_hash_file(dataset_csv).encode("utf-8"))
+    if indices_dir is not None:
+        indices_dir = Path(indices_dir)
+        for name in ("train.txt", "val.txt", "test.txt"):
+            p = indices_dir / name
+            if p.exists():
+                h.update(name.encode("utf-8"))
+                h.update(_hash_file(p).encode("utf-8"))
+    return h.hexdigest()
+
+
+def resolve_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
+    env_sha = os.getenv("GIT_SHA") or os.getenv("GIT_COMMIT")
+    if env_sha:
+        return env_sha
+    try:
+        out = subprocess.check_output(
+            ["git", "rev-parse", "HEAD"],
+            cwd=repo_root or Path.cwd(),
+            stderr=subprocess.DEVNULL,
+        )
+        sha = out.decode("utf-8").strip()
+        return sha or None
+    except Exception:
+        return None
+
+
+def resolve_task_name(cfg: Dict[str, Any]) -> Optional[str]:
+    task_cfg = cfg.get("task", {})
+    name = task_cfg.get("name") or task_cfg.get("task_name")
+    return str(name) if name else None
+
+
+def resolve_model_name(cfg: Dict[str, Any]) -> Optional[str]:
+    model_cfg = cfg.get("model", {})
+    name = model_cfg.get("name") or model_cfg.get("family")
+    return str(name) if name else None
+
+
+def resolve_featureset_name(cfg: Dict[str, Any]) -> Optional[str]:
+    explicit = cfg.get("featureset_name")
+    if explicit:
+        return str(explicit)
+    feat_cfg = cfg.get("featurizer", {})
+    if not feat_cfg:
+        return None
+    base = feat_cfg.get("fingerprint") or feat_cfg.get("name") or "featurizer"
+    suffix = hash_dict(feat_cfg)[:8]
+    return f"{base}_{suffix}"
+
+
+def resolve_tags(cfg: Dict[str, Any]) -> list[str]:
+    tags = cfg.get("tags")
+    if tags is None:
+        tags = cfg.get("experiment", {}).get("tags")
+    if tags is None:
+        return []
+    if isinstance(tags, str):
+        return [tags]
+    return [str(t) for t in tags]
+
+
+def build_artifact_meta(
+    cfg: Dict[str, Any],
+    process_name: Optional[str] = None,
+    upstream_artifacts: Optional[Iterable[str]] = None,
+    meta_source_cfg: Optional[Dict[str, Any]] = None,
+    extra: Optional[Dict[str, Any]] = None,
+) -> Dict[str, Any]:
+    meta_source = meta_source_cfg or cfg
+    dataset_csv, indices_dir = resolve_dataset_paths(cfg)
+    if dataset_csv is None and meta_source is not cfg:
+        dataset_csv, indices_dir = resolve_dataset_paths(meta_source)
+    meta = {
+        "run_id": uuid4().hex,
+        "process_name": process_name or str(cfg.get("process", {}).get("name", "unknown")),
+        "created_at": utc_now_iso(),
+        "git_sha": resolve_git_sha(),
+        "dataset_hash": compute_dataset_hash(dataset_csv, indices_dir),
+        "config_hash": compute_config_hash(cfg),
+        "task_name": resolve_task_name(meta_source),
+        "model_name": resolve_model_name(meta_source),
+        "featureset_name": resolve_featureset_name(meta_source),
+        "upstream_artifacts": [str(x) for x in (upstream_artifacts or [])],
+        "tags": resolve_tags(cfg),
+    }
+    if extra:
+        meta.update(extra)
+    return meta
+
+
+def load_upstream_meta(model_artifact_dir: str | Path) -> Dict[str, Any]:
+    path = Path(model_artifact_dir) / "meta.json"
+    if path.exists():
+        return load_json(path)
+    return {}
+
+
+def resolve_model_version(model_meta: Dict[str, Any]) -> Optional[str]:
+    if not model_meta:
+        return None
+    return model_meta.get("run_id") or model_meta.get("config_hash")
+
+
+def validate_meta_dict(meta: Dict[str, Any]) -> None:
+    missing = REQUIRED_META_KEYS - set(meta)
+    if missing:
+        raise ValueError(f"meta.json missing keys: {sorted(missing)}")
+
+
+def validate_meta_file(path: str | Path) -> None:
+    validate_meta_dict(load_json(path))
+
+
+def validate_predictions_file(path: str | Path) -> None:
+    with Path(path).open("r", encoding="utf-8") as f:
+        reader = csv.reader(f)
+        header = next(reader, [])
+    missing = REQUIRED_PREDICTION_COLUMNS - set(header)
+    if missing:
+        raise ValueError(f"predictions.csv missing columns: {sorted(missing)}")
+
+
+def validate_run_dir(run_dir: str | Path, process_name: str) -> None:
+    run_dir = Path(run_dir)
+    missing = []
+    for rel in (Path("config.yaml"), Path("meta.json")):
+        if not (run_dir / rel).exists():
+            missing.append(str(rel))
+    for rel in PROCESS_REQUIRED_FILES.get(process_name, []):
+        path = run_dir / rel
+        if rel.name == "plots":
+            if not path.exists() or not path.is_dir():
+                missing.append(str(rel))
+        else:
+            if not path.exists():
+                missing.append(str(rel))
+    if missing:
+        raise FileNotFoundError(f"artifact contract missing: {sorted(missing)}")
+    validate_meta_file(run_dir / "meta.json")
+    if process_name in {"evaluate", "predict"}:
+        validate_predictions_file(run_dir / "predictions.csv")
```

```diff
diff --git a/src/fp/train.py b/src/fp/train.py
--- a/src/fp/train.py
+++ b/src/fp/train.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_artifact_meta
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
+    dump_yaml(run_dir / "config.yaml", cfg)
+    artifact_meta = build_artifact_meta(cfg, process_name=str(cfg.get("process", {}).get("name", "train")))
+    save_meta(run_dir, artifact_meta)
@@
     if scaler is not None:
         with open(artifacts_dir / "scaler.pkl", "wb") as f:
             pickle.dump(scaler, f)
+
+    model_dir = ensure_dir(run_dir / "model")
+    with open(model_dir / "model.ckpt", "wb") as f:
+        pickle.dump(model, f)
+    with open(model_dir / "preprocess.pkl", "wb") as f:
+        pickle.dump({"imputer": imputer, "scaler": scaler}, f)
@@
-    save_json(run_dir / "metrics_val.json", metrics_val)
-    save_json(run_dir / "metrics_test.json", metrics_test)
+    save_json(run_dir / "metrics_val.json", metrics_val)
+    save_json(run_dir / "metrics_test.json", metrics_test)
+    save_json(
+        run_dir / "metrics.json",
+        {
+            "val": metrics_val,
+            "test": metrics_test,
+            "n_train": len(y_train),
+            "n_val": len(y_val),
+            "n_test": len(y_test),
+            "seed": seed,
+        },
+    )
```

```diff
diff --git a/src/gnn/train.py b/src/gnn/train.py
--- a/src/gnn/train.py
+++ b/src/gnn/train.py
@@
-import argparse
-import pickle
+import argparse
+import pickle
+import shutil
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_artifact_meta
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
+    dump_yaml(run_dir / "config.yaml", cfg)
+    artifact_meta = build_artifact_meta(cfg, process_name=str(cfg.get("process", {}).get("name", "train")))
+    save_meta(run_dir, artifact_meta)
@@
-    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
-    save_json(run_dir / "metrics_val.json", val_metrics)
-    save_json(run_dir / "metrics_test.json", test_metrics)
+    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
+    save_json(run_dir / "metrics_val.json", val_metrics)
+    save_json(run_dir / "metrics_test.json", test_metrics)
+    save_json(
+        run_dir / "metrics.json",
+        {
+            "val": val_metrics,
+            "test": test_metrics,
+            "n_train": len(train_data),
+            "n_val": len(val_data),
+            "n_test": len(test_data),
+            "seed": seed,
+        },
+    )
@@
-    logger.info(f"Saved best model to {best_path}")
+    model_dir = ensure_dir(run_dir / "model")
+    model_ckpt_path = model_dir / "model.ckpt"
+    shutil.copy2(best_path, model_ckpt_path)
+
+    logger.info(f"Saved best model to {best_path}")
```

```diff
diff --git a/src/fp/predict.py b/src/fp/predict.py
--- a/src/fp/predict.py
+++ b/src/fp/predict.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, save_json
+from src.utils.artifacts import build_artifact_meta, load_upstream_meta, resolve_model_version
@@
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    dump_yaml(out_dir / "config.yaml", cfg)
+    model_meta = load_upstream_meta(model_artifact_dir)
+    model_version = resolve_model_version(model_meta)
+    artifact_meta = build_artifact_meta(
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        upstream_artifacts=[str(model_artifact_dir)],
+        meta_source_cfg=train_cfg,
+        extra={"model_version": model_version},
+    )
+    save_meta(out_dir, artifact_meta)
@@
-    x, meta = featurize_mol(mol, fp_cfg)
+    x, feature_meta = featurize_mol(mol, fp_cfg)
@@
-    result = {
-        "cas": cas,
-        "query": query,
-        "prediction": pred,
-        "resolve_meta": resolve_meta,
-        "ad": ad_res.to_dict(),
-        "feature_meta": meta,
-    }
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": artifact_meta.get("model_name"),
+        "model_version": model_version,
+        "dataset_hash": artifact_meta.get("dataset_hash"),
+        "run_id": artifact_meta.get("run_id"),
+    }
+    pred_path = out_dir / "predictions.csv"
+    pd.DataFrame([pred_row]).to_csv(pred_path, index=False)
+    logger.info(f"Saved predictions to {pred_path}")
+
+    result = {
+        "cas": cas,
+        "query": query,
+        "prediction": pred,
+        "resolve_meta": resolve_meta,
+        "ad": ad_res.to_dict(),
+        "feature_meta": feature_meta,
+        "model_name": artifact_meta.get("model_name"),
+        "model_version": model_version,
+        "run_id": artifact_meta.get("run_id"),
+        "dataset_hash": artifact_meta.get("dataset_hash"),
+    }
```

```diff
diff --git a/src/gnn/predict.py b/src/gnn/predict.py
--- a/src/gnn/predict.py
+++ b/src/gnn/predict.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, save_json
+from src.utils.artifacts import build_artifact_meta, load_upstream_meta, resolve_model_version
@@
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    dump_yaml(out_dir / "config.yaml", cfg)
+    model_meta = load_upstream_meta(model_artifact_dir)
+    model_version = resolve_model_version(model_meta)
+    artifact_meta = build_artifact_meta(
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        upstream_artifacts=[str(model_artifact_dir)],
+        meta_source_cfg=train_cfg,
+        extra={"model_version": model_version},
+    )
+    save_meta(out_dir, artifact_meta)
@@
-    result = {
-        "cas": cas,
-        "query": query,
-        "prediction": pred,
-        "resolve_meta": resolve_meta,
-        "ad": None if ad_res is None else ad_res.to_dict(),
-    }
+    meta_model_name = artifact_meta.get("model_name") or model_name
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": meta_model_name,
+        "model_version": model_version,
+        "dataset_hash": artifact_meta.get("dataset_hash"),
+        "run_id": artifact_meta.get("run_id"),
+    }
+    pred_path = out_dir / "predictions.csv"
+    pd.DataFrame([pred_row]).to_csv(pred_path, index=False)
+    logger.info(f"Saved predictions to {pred_path}")
+
+    result = {
+        "cas": cas,
+        "query": query,
+        "prediction": pred,
+        "resolve_meta": resolve_meta,
+        "ad": None if ad_res is None else ad_res.to_dict(),
+        "model_name": meta_model_name,
+        "model_version": model_version,
+        "run_id": artifact_meta.get("run_id"),
+        "dataset_hash": artifact_meta.get("dataset_hash"),
+    }
```

```diff
diff --git a/src/fp/evaluate.py b/src/fp/evaluate.py
--- a/src/fp/evaluate.py
+++ b/src/fp/evaluate.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, save_json
+from src.utils.artifacts import build_artifact_meta, load_upstream_meta, resolve_model_version
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(
-        run_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "evaluate")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    dump_yaml(run_dir / "config.yaml", cfg)
+    model_meta = load_upstream_meta(model_artifact_dir)
+    model_version = resolve_model_version(model_meta)
+    artifact_meta = build_artifact_meta(
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "evaluate")),
+        upstream_artifacts=[str(model_artifact_dir)],
+        meta_source_cfg=train_cfg,
+        extra={"model_version": model_version},
+    )
+    save_meta(run_dir, artifact_meta)
@@
-    rows = []
-    metrics_by_split: Dict[str, Dict[str, float]] = {}
+    rows = []
+    metrics_by_split: Dict[str, Dict[str, float]] = {}
+    split_counts: Dict[str, int] = {}
+    model_name = artifact_meta.get("model_name")
+    dataset_hash = artifact_meta.get("dataset_hash")
+    run_id = artifact_meta.get("run_id")
@@
-        if len(ids) == 0:
-            metrics_by_split[split_name] = {}
-            continue
+        if len(ids) == 0:
+            metrics_by_split[split_name] = {}
+            split_counts[split_name] = 0
+            continue
@@
-        metrics_by_split[split_name] = regression_metrics(y, preds)
-        for cid, yt, yp in zip(ids, y.tolist(), preds.tolist()):
-            rows.append({"sample_id": cid, "y_true": yt, "y_pred": yp, "split": split_name})
+        metrics_by_split[split_name] = regression_metrics(y, preds)
+        split_counts[split_name] = len(ids)
+        for cid, yt, yp in zip(ids, y.tolist(), preds.tolist()):
+            rows.append(
+                {
+                    "sample_id": cid,
+                    "y_true": yt,
+                    "y_pred": yp,
+                    "split": split_name,
+                    "model_name": model_name,
+                    "model_version": model_version,
+                    "dataset_hash": dataset_hash,
+                    "run_id": run_id,
+                }
+            )
@@
     for split_name, metrics in metrics_by_split.items():
         if not metrics:
             continue
         save_json(run_dir / f"metrics_{split_name}.json", metrics)
+
+    metrics_payload: Dict[str, Any] = {"by_split": metrics_by_split}
+    for split_name in ("train", "val", "test"):
+        if split_name in split_counts:
+            metrics_payload[f"n_{split_name}"] = split_counts[split_name]
+    save_json(run_dir / "metrics.json", metrics_payload)
```

```diff
diff --git a/src/gnn/evaluate.py b/src/gnn/evaluate.py
--- a/src/gnn/evaluate.py
+++ b/src/gnn/evaluate.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, save_json
+from src.utils.artifacts import build_artifact_meta, load_upstream_meta, resolve_model_version
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(
-        run_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "evaluate")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    dump_yaml(run_dir / "config.yaml", cfg)
+    model_meta = load_upstream_meta(model_artifact_dir)
+    model_version = resolve_model_version(model_meta)
+    artifact_meta = build_artifact_meta(
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "evaluate")),
+        upstream_artifacts=[str(model_artifact_dir)],
+        meta_source_cfg=train_cfg,
+        extra={"model_version": model_version},
+    )
+    save_meta(run_dir, artifact_meta)
@@
-    rows = []
-    metrics_by_split: Dict[str, Dict[str, float]] = {}
+    rows = []
+    metrics_by_split: Dict[str, Dict[str, float]] = {}
+    split_counts: Dict[str, int] = {}
+    meta_model_name = artifact_meta.get("model_name") or model_name
+    dataset_hash = artifact_meta.get("dataset_hash")
+    run_id = artifact_meta.get("run_id")
@@
-        if len(y_true) == 0:
-            metrics_by_split[split_name] = {}
-            continue
+        if len(y_true) == 0:
+            metrics_by_split[split_name] = {}
+            split_counts[split_name] = 0
+            continue
         metrics_by_split[split_name] = regression_metrics(y_true, y_pred)
+        split_counts[split_name] = len(ids)
         for cid, yt, yp in zip(ids, y_true.tolist(), y_pred.tolist()):
-            rows.append({"sample_id": cid, "y_true": yt, "y_pred": yp, "split": split_name})
+            rows.append(
+                {
+                    "sample_id": cid,
+                    "y_true": yt,
+                    "y_pred": yp,
+                    "split": split_name,
+                    "model_name": meta_model_name,
+                    "model_version": model_version,
+                    "dataset_hash": dataset_hash,
+                    "run_id": run_id,
+                }
+            )
@@
     for split_name, metrics in metrics_by_split.items():
         if not metrics:
             continue
         save_json(run_dir / f"metrics_{split_name}.json", metrics)
+
+    metrics_payload: Dict[str, Any] = {"by_split": metrics_by_split}
+    for split_name in ("train", "val", "test"):
+        if split_name in split_counts:
+            metrics_payload[f"n_{split_name}"] = split_counts[split_name]
+    save_json(run_dir / "metrics.json", metrics_payload)
```

```diff
diff --git a/scripts/visualize.py b/scripts/visualize.py
--- a/scripts/visualize.py
+++ b/scripts/visualize.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_artifact_meta, load_upstream_meta
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(process_name=str(cfg.get("process", {}).get("name", "visualize"))),
-    )
+    dump_yaml(out_dir / "config.yaml", cfg)
+    input_cfg = cfg.get("input", {})
+    upstream_artifacts = []
+    extra_meta = {}
+    eval_dir = input_cfg.get("evaluate_run_dir")
+    if eval_dir:
+        upstream_artifacts.append(str(eval_dir))
+        eval_meta = load_upstream_meta(eval_dir)
+        if eval_meta.get("dataset_hash"):
+            extra_meta["dataset_hash"] = eval_meta["dataset_hash"]
+        if eval_meta.get("model_name"):
+            extra_meta["model_name"] = eval_meta["model_name"]
+        if eval_meta.get("featureset_name"):
+            extra_meta["featureset_name"] = eval_meta["featureset_name"]
+        if eval_meta.get("model_version"):
+            extra_meta["model_version"] = eval_meta["model_version"]
+    artifact_meta = build_artifact_meta(
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "visualize")),
+        upstream_artifacts=upstream_artifacts,
+        extra=extra_meta or None,
+    )
+    save_meta(out_dir, artifact_meta)
```

```diff
diff --git a/scripts/audit_dataset.py b/scripts/audit_dataset.py
--- a/scripts/audit_dataset.py
+++ b/scripts/audit_dataset.py
@@
-from src.common.meta import build_meta, save_meta
+from src.common.meta import save_meta
+from src.utils.artifacts import build_artifact_meta
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "audit_dataset"))))
+    dump_yaml(run_dir / "config.yaml", cfg)
+    artifact_meta = build_artifact_meta(cfg, process_name=str(cfg.get("process", {}).get("name", "audit_dataset")))
+    save_meta(run_dir, artifact_meta)
```

```diff
diff --git a/tests/contract/test_artifact_contract.py b/tests/contract/test_artifact_contract.py
new file mode 100644
--- /dev/null
+++ b/tests/contract/test_artifact_contract.py
@@
+from __future__ import annotations
+
+from pathlib import Path
+
+from src.common.config import load_config
+from src.common.utils import save_json
+from src.utils.artifacts import (
+    REQUIRED_META_KEYS,
+    build_artifact_meta,
+    validate_predictions_file,
+    validate_run_dir,
+)
+
+
+def test_meta_contract_keys() -> None:
+    cfg = load_config(Path("configs/fp/train_fixture.yaml"))
+    meta = build_artifact_meta(cfg, process_name="train")
+    missing = REQUIRED_META_KEYS - set(meta)
+    assert not missing
+    assert meta["dataset_hash"]
+    assert meta["config_hash"]
+    assert meta["task_name"]
+
+
+def test_predictions_contract_columns(tmp_path: Path) -> None:
+    pred_path = tmp_path / "predictions.csv"
+    pred_path.write_text("sample_id,y_pred\nA,1.0\n", encoding="utf-8")
+    validate_predictions_file(pred_path)
+
+
+def test_validate_train_run_dir_contract(tmp_path: Path) -> None:
+    cfg = load_config(Path("configs/fp/train_fixture.yaml"))
+    run_dir = tmp_path / "run"
+    run_dir.mkdir()
+    (run_dir / "config.yaml").write_text("{}", encoding="utf-8")
+    meta = build_artifact_meta(cfg, process_name="train")
+    save_json(run_dir / "meta.json", meta)
+    save_json(run_dir / "metrics.json", {"val": {"rmse": 1.0}})
+    model_dir = run_dir / "model"
+    model_dir.mkdir()
+    (model_dir / "model.ckpt").write_bytes(b"test")
+    validate_run_dir(run_dir, "train")
```

**Tests Added/Updated**
- `tests/contract/test_artifact_contract.py` (meta keys, predictions columns, train run-dir contract check). Not run here (read‑only).

**Verification Commands**
- `pytest -q tests/contract/test_artifact_contract.py`
- `pytest -q`

**Compatibility Impact**
- config: no changes.
- CLI: no changes.
- artifact: new files `metrics.json`, `model/model.ckpt`, `predictions.csv` (predict) and extra columns in `predictions.csv`; existing files preserved.

**Task Status**
- `work/queue.json` should be updated to `blocked` for task 020 due to read‑only sandbox preventing file edits. I couldn’t update it directly.

If you want me to apply this patch, please rerun with write access.  
Next steps I can take once writable:
1) Apply the patch and update `work/queue.json`.
2) Extend the contract to `build_dataset` artifacts (if you want to include dataset_hash/index outputs now).
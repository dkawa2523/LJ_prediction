Using skills S90_artifacts_versioning and S95_tests_ci because this task is about artifact contract implementation plus contract tests.  
Read‑only sandbox prevents applying edits or updating `work/queue.json`; below are proposed changes and diffs you can apply when write access is available.

**Plan**
- Add `src/utils/artifacts.py` for hashing, meta building, model/prediction helpers, and contract validation.
- Update training/predict/evaluate to write `metrics.json`, `predictions.csv`, and the new `model/` artifacts while preserving legacy `artifacts/` fallback.
- Update visualize/audit to emit contract‑compliant `meta.json`.
- Add contract tests under `tests/contract/`.

**Proposed Diffs**

```diff
diff --git a/src/utils/artifacts.py b/src/utils/artifacts.py
new file mode 100644
index 0000000..1a2b3c4
--- /dev/null
+++ b/src/utils/artifacts.py
@@
+from __future__ import annotations
+
+import csv
+import hashlib
+import json
+import pickle
+import shutil
+from dataclasses import asdict, is_dataclass
+from pathlib import Path
+from typing import Any, Dict, Iterable, Optional
+from uuid import uuid4
+
+from src.common.meta import utc_now_iso
+from src.common.utils import ensure_dir, load_json, save_json
+
+REQUIRED_META_KEYS = [
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
+]
+
+REQUIRED_PREDICTION_COLUMNS = ["sample_id", "y_pred"]
+
+
+def _json_default(obj: Any) -> Any:
+    if isinstance(obj, Path):
+        return str(obj)
+    if is_dataclass(obj):
+        return asdict(obj)
+    return repr(obj)
+
+
+def _stable_dumps(obj: Any) -> str:
+    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=True)
+
+
+def _short_hash(obj: Any, length: int = 12) -> str:
+    return hashlib.sha256(_stable_dumps(obj).encode("utf-8")).hexdigest()[:length]
+
+
+def compute_config_hash(cfg: Dict[str, Any]) -> str:
+    return hashlib.sha256(_stable_dumps(cfg).encode("utf-8")).hexdigest()
+
+
+def _update_hash_with_file(hasher: "hashlib._Hash", path: Path) -> None:
+    with path.open("rb") as f:
+        for chunk in iter(lambda: f.read(1024 * 1024), b""):
+            hasher.update(chunk)
+
+
+def compute_dataset_hash(dataset_csv: Path, indices_dir: Optional[Path] = None) -> str:
+    dataset_csv = Path(dataset_csv)
+    if not dataset_csv.exists():
+        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
+    hasher = hashlib.sha256()
+    hasher.update(b"dataset_csv\0")
+    _update_hash_with_file(hasher, dataset_csv)
+    if indices_dir:
+        indices_dir = Path(indices_dir)
+        if indices_dir.exists():
+            for path in sorted(indices_dir.glob("*.txt")):
+                hasher.update(path.name.encode("utf-8"))
+                hasher.update(b"\0")
+                _update_hash_with_file(hasher, path)
+    return hasher.hexdigest()
+
+
+def _find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
+    start = start or Path(__file__).resolve()
+    for p in [start, *start.parents]:
+        if (p / ".git").exists():
+            return p
+    return None
+
+
+def resolve_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
+    repo_root = repo_root or _find_repo_root()
+    if repo_root is None:
+        return None
+    head_path = repo_root / ".git" / "HEAD"
+    if not head_path.exists():
+        return None
+    head = head_path.read_text(encoding="utf-8").strip()
+    if head.startswith("ref: "):
+        ref = head.split(" ", 1)[1].strip()
+        ref_path = repo_root / ".git" / ref
+        if ref_path.exists():
+            return ref_path.read_text(encoding="utf-8").strip()
+        packed = repo_root / ".git" / "packed-refs"
+        if packed.exists():
+            for line in packed.read_text(encoding="utf-8").splitlines():
+                if not line or line.startswith("#") or line.startswith("^"):
+                    continue
+                sha, name = line.split(" ", 1)
+                if name.strip() == ref:
+                    return sha.strip()
+        return None
+    return head if head else None
+
+
+def resolve_model_name(cfg: Dict[str, Any]) -> str:
+    model_cfg = cfg.get("model", {}) or {}
+    name = model_cfg.get("name") or model_cfg.get("family")
+    return str(name) if name else "unknown"
+
+
+def resolve_task_name(cfg: Dict[str, Any]) -> str:
+    task_cfg = cfg.get("task", {}) or {}
+    name = task_cfg.get("name") or task_cfg.get("target_col")
+    if name:
+        return str(name)
+    data_cfg = cfg.get("data", {}) or {}
+    target_col = data_cfg.get("target_col")
+    return str(target_col) if target_col else "unknown"
+
+
+def resolve_featureset_name(cfg: Dict[str, Any]) -> str:
+    featureset_cfg = cfg.get("featureset", {}) or {}
+    name = featureset_cfg.get("name")
+    if name:
+        return str(name)
+    featurizer_cfg = cfg.get("featurizer", {}) or {}
+    name = featurizer_cfg.get("name")
+    if name:
+        return str(name)
+    if featurizer_cfg:
+        return f"featurizer_{_short_hash(featurizer_cfg)}"
+    return "unknown"
+
+
+def _resolve_tags(cfg: Optional[Dict[str, Any]]) -> list[str]:
+    if cfg is None:
+        return []
+    meta_cfg = cfg.get("meta", {}) or {}
+    tags = meta_cfg.get("tags")
+    if tags is None:
+        tags = cfg.get("tags")
+    if tags is None:
+        return []
+    if isinstance(tags, (list, tuple)):
+        return [str(t) for t in tags]
+    return [str(tags)]
+
+
+def _coerce_str(value: Optional[Any]) -> str:
+    if value is None:
+        return "unknown"
+    if isinstance(value, str) and not value.strip():
+        return "unknown"
+    return str(value)
+
+
+def _resolve_dataset_hash(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
+    if cfg is None:
+        return None
+    data_cfg = cfg.get("data", {}) or {}
+    dataset_csv = data_cfg.get("dataset_csv") or cfg.get("dataset_csv")
+    indices_dir = data_cfg.get("indices_dir") or cfg.get("indices_dir")
+    if not dataset_csv:
+        return None
+    try:
+        return compute_dataset_hash(Path(dataset_csv), Path(indices_dir) if indices_dir else None)
+    except FileNotFoundError:
+        return None
+
+
+def build_meta(
+    process_name: str,
+    cfg: Optional[Dict[str, Any]] = None,
+    upstream_artifacts: Optional[Iterable[str]] = None,
+    extra: Optional[Dict[str, Any]] = None,
+    *,
+    run_id: Optional[str] = None,
+    created_at: Optional[str] = None,
+    dataset_hash: Optional[str] = None,
+    config_hash: Optional[str] = None,
+    git_sha: Optional[str] = None,
+    task_name: Optional[str] = None,
+    model_name: Optional[str] = None,
+    featureset_name: Optional[str] = None,
+    tags: Optional[Iterable[str]] = None,
+) -> Dict[str, Any]:
+    resolved_dataset_hash = dataset_hash or _resolve_dataset_hash(cfg)
+    resolved_config_hash = config_hash or (compute_config_hash(cfg) if cfg else None)
+    resolved_git_sha = git_sha or resolve_git_sha()
+    resolved_task_name = task_name or (resolve_task_name(cfg) if cfg else None)
+    resolved_model_name = model_name or (resolve_model_name(cfg) if cfg else None)
+    resolved_featureset_name = featureset_name or (resolve_featureset_name(cfg) if cfg else None)
+    meta = {
+        "run_id": _coerce_str(run_id) if run_id else uuid4().hex,
+        "process_name": str(process_name),
+        "created_at": created_at or utc_now_iso(),
+        "git_sha": _coerce_str(resolved_git_sha),
+        "dataset_hash": _coerce_str(resolved_dataset_hash),
+        "config_hash": _coerce_str(resolved_config_hash),
+        "task_name": _coerce_str(resolved_task_name),
+        "model_name": _coerce_str(resolved_model_name),
+        "featureset_name": _coerce_str(resolved_featureset_name),
+        "upstream_artifacts": list(upstream_artifacts or []),
+        "tags": list(tags) if tags is not None else _resolve_tags(cfg),
+    }
+    if extra:
+        meta.update(extra)
+    return meta
+
+
+def save_meta(run_dir: str | Path, meta: Dict[str, Any]) -> Path:
+    run_dir = Path(run_dir)
+    path = run_dir / "meta.json"
+    save_json(path, meta)
+    return path
+
+
+def ensure_model_dir(run_dir: str | Path) -> Path:
+    return ensure_dir(Path(run_dir) / "model")
+
+
+def copy_model_checkpoint(source: str | Path, model_dir: str | Path) -> Path:
+    source = Path(source)
+    model_dir = Path(model_dir)
+    model_dir.mkdir(parents=True, exist_ok=True)
+    dest = model_dir / "model.ckpt"
+    shutil.copy2(source, dest)
+    return dest
+
+
+def save_preprocess(
+    path: str | Path,
+    *,
+    imputer=None,
+    scaler=None,
+    extra: Optional[Dict[str, Any]] = None,
+) -> Optional[Path]:
+    payload = {}
+    if imputer is not None:
+        payload["imputer"] = imputer
+    if scaler is not None:
+        payload["scaler"] = scaler
+    if extra:
+        payload.update(extra)
+    if not payload:
+        return None
+    path = Path(path)
+    with path.open("wb") as f:
+        pickle.dump(payload, f)
+    return path
+
+
+def save_featurizer_state(
+    path: str | Path,
+    featurizer_cfg: Dict[str, Any],
+    feature_meta: Optional[Dict[str, Any]] = None,
+    extra: Optional[Dict[str, Any]] = None,
+) -> Path:
+    payload = {"featurizer": featurizer_cfg}
+    if feature_meta:
+        payload["feature_meta"] = feature_meta
+    if extra:
+        payload.update(extra)
+    path = Path(path)
+    save_json(path, payload)
+    return path
+
+
+def resolve_model_artifacts_dir(model_artifact_dir: str | Path) -> Path:
+    model_artifact_dir = Path(model_artifact_dir)
+    for candidate in [model_artifact_dir / "model", model_artifact_dir / "artifacts"]:
+        if candidate.exists():
+            return candidate
+    raise FileNotFoundError(f"No model artifacts directory found under: {model_artifact_dir}")
+
+
+def validate_artifact_contract(run_dir: str | Path, process_name: str) -> None:
+    run_dir = Path(run_dir)
+    required = [run_dir / "config.yaml", run_dir / "meta.json"]
+    missing = [str(p) for p in required if not p.exists()]
+    if missing:
+        raise ValueError(f"Missing required artifacts: {missing}")
+
+    meta = load_json(run_dir / "meta.json")
+    missing_keys = [k for k in REQUIRED_META_KEYS if k not in meta]
+    if missing_keys:
+        raise ValueError(f"meta.json missing keys: {missing_keys}")
+    if meta.get("process_name") and str(meta.get("process_name")) != str(process_name):
+        raise ValueError(f"process_name mismatch: meta={meta.get('process_name')} expected={process_name}")
+
+    if process_name == "train":
+        if not (run_dir / "metrics.json").exists():
+            raise ValueError("train requires metrics.json")
+        if not (run_dir / "model" / "model.ckpt").exists():
+            raise ValueError("train requires model/model.ckpt")
+    if process_name == "evaluate":
+        if not (run_dir / "metrics.json").exists():
+            raise ValueError("evaluate requires metrics.json")
+        _require_prediction_columns(run_dir / "predictions.csv")
+    if process_name == "predict":
+        _require_prediction_columns(run_dir / "predictions.csv")
+    if process_name == "visualize":
+        if not (run_dir / "plots").exists():
+            raise ValueError("visualize requires plots/")
+
+
+def _require_prediction_columns(path: Path) -> None:
+    if not path.exists():
+        raise ValueError(f"predictions.csv missing: {path}")
+    with path.open("r", encoding="utf-8", newline="") as f:
+        reader = csv.DictReader(f)
+        headers = reader.fieldnames or []
+    missing_cols = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in headers]
+    if missing_cols:
+        raise ValueError(f"predictions.csv missing columns: {missing_cols}")
```

```diff
diff --git a/src/fp/train.py b/src/fp/train.py
index 5a7c1c1..d7b9f0a 100644
--- a/src/fp/train.py
+++ b/src/fp/train.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
-
-    if not dataset_csv.exists():
+    dump_yaml(run_dir / "config.yaml", cfg)
+
+    if not dataset_csv.exists():
         raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
@@
-    df = read_csv(dataset_csv)
+    dataset_hash = artifacts_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "train")),
+        cfg=cfg,
+        dataset_hash=dataset_hash,
+    )
+    artifacts_utils.save_meta(run_dir, run_meta)
+
+    df = read_csv(dataset_csv)
@@
-    save_json(run_dir / "metrics_val.json", metrics_val)
-    save_json(run_dir / "metrics_test.json", metrics_test)
+    save_json(run_dir / "metrics.json", {"val": metrics_val, "test": metrics_test})
+    save_json(run_dir / "metrics_val.json", metrics_val)
+    save_json(run_dir / "metrics_test.json", metrics_test)
     dump_yaml(run_dir / "config_snapshot.yaml", cfg)
+
+    model_dir = artifacts_utils.ensure_model_dir(run_dir)
+    artifacts_utils.copy_model_checkpoint(model_path, model_dir)
+    artifacts_utils.save_preprocess(model_dir / "preprocess.pkl", imputer=imputer, scaler=scaler)
+    artifacts_utils.save_featurizer_state(
+        model_dir / "featurizer_state.json",
+        featurizer_cfg=cfg.get("featurizer", {}),
+        feature_meta=feat_meta,
+    )
```

```diff
diff --git a/src/gnn/train.py b/src/gnn/train.py
index 76f5c2f..c3ae8f1 100644
--- a/src/gnn/train.py
+++ b/src/gnn/train.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
+    dump_yaml(run_dir / "config.yaml", cfg)
+
+    if not dataset_csv.exists():
+        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
+    if not indices_dir.exists():
+        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
+
+    dataset_hash = artifacts_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "train")),
+        cfg=cfg,
+        dataset_hash=dataset_hash,
+    )
+    artifacts_utils.save_meta(run_dir, run_meta)
@@
-    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
-    save_json(run_dir / "metrics_val.json", val_metrics)
-    save_json(run_dir / "metrics_test.json", test_metrics)
+    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
+    save_json(run_dir / "metrics.json", {"val": val_metrics, "test": test_metrics})
+    save_json(run_dir / "metrics_val.json", val_metrics)
+    save_json(run_dir / "metrics_test.json", test_metrics)
@@
     except Exception as e:
         logger.warning(f"Failed to create AD artifact (will disable AD in predict): {e}")
+
+    model_dir = artifacts_utils.ensure_model_dir(run_dir)
+    artifacts_utils.copy_model_checkpoint(best_path, model_dir)
+    artifacts_utils.save_featurizer_state(
+        model_dir / "featurizer_state.json",
+        featurizer_cfg={
+            "node_features": gcfg.node_features,
+            "edge_features": gcfg.edge_features,
+            "use_3d_pos": gcfg.use_3d_pos,
+            "add_global_descriptors": gcfg.add_global_descriptors,
+        },
+    )
```

```diff
diff --git a/src/fp/predict.py b/src/fp/predict.py
index 6c0a1ee..d2dbe8b 100644
--- a/src/fp/predict.py
+++ b/src/fp/predict.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, load_json, save_json
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    artifacts_dir = model_artifact_dir / "artifacts"
-    if not artifacts_dir.exists():
-        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")
+    model_dir = artifacts_utils.resolve_model_artifacts_dir(model_artifact_dir)
@@
-    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/predict")) / exp_name)
-    logger = get_logger("fp_predict", log_file=out_dir / "predict.log")
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/predict")) / exp_name)
+    logger = get_logger("fp_predict", log_file=out_dir / "predict.log")
+    dump_yaml(out_dir / "config.yaml", cfg)
+
+    dataset_hash = artifacts_utils.compute_dataset_hash(dataset_csv)
+    model_meta = {}
+    model_meta_path = model_artifact_dir / "meta.json"
+    if model_meta_path.exists():
+        model_meta = load_json(model_meta_path)
+    model_version = str(model_meta.get("run_id") or model_meta.get("config_hash") or "unknown")
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        cfg=cfg,
+        upstream_artifacts=[str(model_artifact_dir)],
+        dataset_hash=dataset_hash,
+        model_name=artifacts_utils.resolve_model_name(train_cfg),
+        task_name=artifacts_utils.resolve_task_name(train_cfg),
+        featureset_name=artifacts_utils.resolve_featureset_name(train_cfg),
+        extra={"model_version": model_version, "model_run_id": model_meta.get("run_id")},
+    )
+    artifacts_utils.save_meta(out_dir, run_meta)
@@
-    with open(artifacts_dir / "imputer.pkl", "rb") as f:
-        imputer = pickle.load(f)
-    scaler = None
-    scaler_path = artifacts_dir / "scaler.pkl"
-    if scaler_path.exists():
-        with open(scaler_path, "rb") as f:
-            scaler = pickle.load(f)
-    with open(artifacts_dir / "model.pkl", "rb") as f:
-        model = pickle.load(f)
+    legacy_dir = model_artifact_dir / "artifacts"
+    preprocess_path = model_dir / "preprocess.pkl"
+    if preprocess_path.exists():
+        with open(preprocess_path, "rb") as f:
+            preprocess = pickle.load(f)
+        imputer = preprocess["imputer"]
+        scaler = preprocess.get("scaler")
+    else:
+        with open(legacy_dir / "imputer.pkl", "rb") as f:
+            imputer = pickle.load(f)
+        scaler = None
+        scaler_path = legacy_dir / "scaler.pkl"
+        if scaler_path.exists():
+            with open(scaler_path, "rb") as f:
+                scaler = pickle.load(f)
+
+    model_path = model_dir / "model.ckpt"
+    if not model_path.exists():
+        model_path = model_dir / "model.pkl"
+    if not model_path.exists():
+        model_path = legacy_dir / "model.pkl"
+    with open(model_path, "rb") as f:
+        model = pickle.load(f)
@@
-    with open(artifacts_dir / "ad.pkl", "rb") as f:
+    with open(legacy_dir / "ad.pkl", "rb") as f:
         ad_artifact = pickle.load(f)
@@
-    result = {
+    result = {
         "cas": cas,
         "query": query,
         "prediction": pred,
+        "model_version": model_version,
         "resolve_meta": resolve_meta,
         "ad": ad_res.to_dict(),
         "feature_meta": meta,
     }
     save_json(out_dir / f"prediction_{cas}.json", result)
     logger.info(f"Saved prediction json to {out_dir / f'prediction_{cas}.json'}")
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": run_meta["model_name"],
+        "model_version": model_version,
+        "dataset_hash": dataset_hash,
+        "run_id": run_meta["run_id"],
+        "query": query,
+    }
+    pred_path = out_dir / "predictions.csv"
+    pd.DataFrame([pred_row]).to_csv(pred_path, index=False)
+    logger.info("Saved predictions csv to %s", pred_path)
     return out_dir
```

```diff
diff --git a/src/gnn/predict.py b/src/gnn/predict.py
index 55b8b1b..ce2b4c5 100644
--- a/src/gnn/predict.py
+++ b/src/gnn/predict.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, load_json, save_json
@@
-from src.gnn.featurizer_graph import featurize_mol_to_pyg
+from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    artifacts_dir = model_artifact_dir / "artifacts"
-    if not artifacts_dir.exists():
-        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")
+    model_dir = artifacts_utils.resolve_model_artifacts_dir(model_artifact_dir)
@@
-    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/predict")) / exp_name)
-    logger = get_logger("gnn_predict", log_file=out_dir / "predict.log")
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/predict")) / exp_name)
+    logger = get_logger("gnn_predict", log_file=out_dir / "predict.log")
+    dump_yaml(out_dir / "config.yaml", cfg)
+
+    dataset_hash = artifacts_utils.compute_dataset_hash(dataset_csv)
+    model_meta = {}
+    model_meta_path = model_artifact_dir / "meta.json"
+    if model_meta_path.exists():
+        model_meta = load_json(model_meta_path)
+    model_version = str(model_meta.get("run_id") or model_meta.get("config_hash") or "unknown")
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        cfg=cfg,
+        upstream_artifacts=[str(model_artifact_dir)],
+        dataset_hash=dataset_hash,
+        model_name=artifacts_utils.resolve_model_name(train_cfg),
+        task_name=artifacts_utils.resolve_task_name(train_cfg),
+        featureset_name=artifacts_utils.resolve_featureset_name(train_cfg),
+        extra={"model_version": model_version, "model_run_id": model_meta.get("run_id")},
+    )
+    artifacts_utils.save_meta(out_dir, run_meta)
@@
-    with open(artifacts_dir / "graph_featurizer.pkl", "rb") as f:
-        gcfg = pickle.load(f)
+    legacy_dir = model_artifact_dir / "artifacts"
+    featurizer_state_path = model_dir / "featurizer_state.json"
+    if featurizer_state_path.exists():
+        state = load_json(featurizer_state_path)
+        featurizer_cfg = state.get("featurizer", state)
+        gcfg = GraphFeaturizerConfig(**featurizer_cfg)
+    else:
+        with open(legacy_dir / "graph_featurizer.pkl", "rb") as f:
+            gcfg = pickle.load(f)
@@
-    state_path = artifacts_dir / "model_best.pt"
+    state_path = model_dir / "model.ckpt"
+    if not state_path.exists():
+        state_path = model_dir / "model_best.pt"
+    if not state_path.exists():
+        state_path = legacy_dir / "model_best.pt"
     model.load_state_dict(torch.load(state_path, map_location=device))
@@
-    ad_path = artifacts_dir / "ad.pkl"
+    ad_path = legacy_dir / "ad.pkl"
@@
-    result = {
+    result = {
         "cas": cas,
         "query": query,
         "prediction": pred,
+        "model_version": model_version,
         "resolve_meta": resolve_meta,
         "ad": None if ad_res is None else ad_res.to_dict(),
     }
     save_json(out_dir / f"prediction_{cas}.json", result)
     logger.info(f"Saved prediction json to {out_dir / f'prediction_{cas}.json'}")
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": run_meta["model_name"],
+        "model_version": model_version,
+        "dataset_hash": dataset_hash,
+        "run_id": run_meta["run_id"],
+        "query": query,
+    }
+    pred_path = out_dir / "predictions.csv"
+    pd.DataFrame([pred_row]).to_csv(pred_path, index=False)
+    logger.info("Saved predictions csv to %s", pred_path)
     return out_dir
```

```diff
diff --git a/src/fp/evaluate.py b/src/fp/evaluate.py
index 9b3552a..55a8e7b 100644
--- a/src/fp/evaluate.py
+++ b/src/fp/evaluate.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, load_json, save_json
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    artifacts_dir = model_artifact_dir / "artifacts"
-    if not artifacts_dir.exists():
-        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")
+    model_dir = artifacts_utils.resolve_model_artifacts_dir(model_artifact_dir)
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(
-        run_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "evaluate")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
-
     if not dataset_csv.exists():
         raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
@@
+    dataset_hash = artifacts_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    model_meta = {}
+    model_meta_path = model_artifact_dir / "meta.json"
+    if model_meta_path.exists():
+        model_meta = load_json(model_meta_path)
+    model_version = str(model_meta.get("run_id") or model_meta.get("config_hash") or "unknown")
+    model_name = artifacts_utils.resolve_model_name(train_cfg)
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "evaluate")),
+        cfg=cfg,
+        upstream_artifacts=[str(model_artifact_dir)],
+        dataset_hash=dataset_hash,
+        model_name=model_name,
+        task_name=artifacts_utils.resolve_task_name(train_cfg),
+        featureset_name=artifacts_utils.resolve_featureset_name(train_cfg),
+        extra={"model_version": model_version, "model_run_id": model_meta.get("run_id")},
+    )
+    dump_yaml(run_dir / "config.yaml", cfg)
+    artifacts_utils.save_meta(run_dir, run_meta)
@@
-    with open(artifacts_dir / "imputer.pkl", "rb") as f:
-        imputer = pickle.load(f)
-    scaler = None
-    scaler_path = artifacts_dir / "scaler.pkl"
-    if scaler_path.exists():
-        with open(scaler_path, "rb") as f:
-            scaler = pickle.load(f)
-    with open(artifacts_dir / "model.pkl", "rb") as f:
-        model = pickle.load(f)
+    legacy_dir = model_artifact_dir / "artifacts"
+    preprocess_path = model_dir / "preprocess.pkl"
+    if preprocess_path.exists():
+        with open(preprocess_path, "rb") as f:
+            preprocess = pickle.load(f)
+        imputer = preprocess["imputer"]
+        scaler = preprocess.get("scaler")
+    else:
+        with open(legacy_dir / "imputer.pkl", "rb") as f:
+            imputer = pickle.load(f)
+        scaler = None
+        scaler_path = legacy_dir / "scaler.pkl"
+        if scaler_path.exists():
+            with open(scaler_path, "rb") as f:
+                scaler = pickle.load(f)
+
+    model_path = model_dir / "model.ckpt"
+    if not model_path.exists():
+        model_path = model_dir / "model.pkl"
+    if not model_path.exists():
+        model_path = legacy_dir / "model.pkl"
+    with open(model_path, "rb") as f:
+        model = pickle.load(f)
@@
-    pred_df = pd.DataFrame(rows)
+    pred_df = pd.DataFrame(rows)
+    if not pred_df.empty:
+        pred_df["model_name"] = model_name
+        pred_df["model_version"] = model_version
+        pred_df["dataset_hash"] = dataset_hash
+        pred_df["run_id"] = run_meta["run_id"]
     pred_path = run_dir / "predictions.csv"
     pred_df.to_csv(pred_path, index=False)
@@
     for split_name, metrics in metrics_by_split.items():
         if not metrics:
             continue
         save_json(run_dir / f"metrics_{split_name}.json", metrics)
+    save_json(run_dir / "metrics.json", metrics_by_split)
```

```diff
diff --git a/src/gnn/evaluate.py b/src/gnn/evaluate.py
index d840faa..7b5bda2 100644
--- a/src/gnn/evaluate.py
+++ b/src/gnn/evaluate.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.common.utils import ensure_dir, get_logger, save_json
-from src.gnn.featurizer_graph import featurize_mol_to_pyg
+from src.common.utils import ensure_dir, get_logger, load_json, save_json
+from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    artifacts_dir = model_artifact_dir / "artifacts"
-    if not artifacts_dir.exists():
-        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")
+    model_dir = artifacts_utils.resolve_model_artifacts_dir(model_artifact_dir)
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(
-        run_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "evaluate")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
-
     if not dataset_csv.exists():
         raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
@@
+    dataset_hash = artifacts_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    model_meta = {}
+    model_meta_path = model_artifact_dir / "meta.json"
+    if model_meta_path.exists():
+        model_meta = load_json(model_meta_path)
+    model_version = str(model_meta.get("run_id") or model_meta.get("config_hash") or "unknown")
+    model_name = artifacts_utils.resolve_model_name(train_cfg)
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "evaluate")),
+        cfg=cfg,
+        upstream_artifacts=[str(model_artifact_dir)],
+        dataset_hash=dataset_hash,
+        model_name=model_name,
+        task_name=artifacts_utils.resolve_task_name(train_cfg),
+        featureset_name=artifacts_utils.resolve_featureset_name(train_cfg),
+        extra={"model_version": model_version, "model_run_id": model_meta.get("run_id")},
+    )
+    dump_yaml(run_dir / "config.yaml", cfg)
+    artifacts_utils.save_meta(run_dir, run_meta)
@@
-    with open(artifacts_dir / "graph_featurizer.pkl", "rb") as f:
-        gcfg = pickle.load(f)
+    legacy_dir = model_artifact_dir / "artifacts"
+    featurizer_state_path = model_dir / "featurizer_state.json"
+    if featurizer_state_path.exists():
+        state = load_json(featurizer_state_path)
+        featurizer_cfg = state.get("featurizer", state)
+        gcfg = GraphFeaturizerConfig(**featurizer_cfg)
+    else:
+        with open(legacy_dir / "graph_featurizer.pkl", "rb") as f:
+            gcfg = pickle.load(f)
@@
-    state_path = artifacts_dir / "model_best.pt"
+    state_path = model_dir / "model.ckpt"
+    if not state_path.exists():
+        state_path = model_dir / "model_best.pt"
+    if not state_path.exists():
+        state_path = legacy_dir / "model_best.pt"
     model.load_state_dict(torch.load(state_path, map_location=device))
@@
-    pred_df = pd.DataFrame(rows)
+    pred_df = pd.DataFrame(rows)
+    if not pred_df.empty:
+        pred_df["model_name"] = model_name
+        pred_df["model_version"] = model_version
+        pred_df["dataset_hash"] = dataset_hash
+        pred_df["run_id"] = run_meta["run_id"]
     pred_path = run_dir / "predictions.csv"
     pred_df.to_csv(pred_path, index=False)
@@
     for split_name, metrics in metrics_by_split.items():
         if not metrics:
             continue
         save_json(run_dir / f"metrics_{split_name}.json", metrics)
+    save_json(run_dir / "metrics.json", metrics_by_split)
```

```diff
diff --git a/scripts/visualize.py b/scripts/visualize.py
index 3cc0c3b..3d9048c 100644
--- a/scripts/visualize.py
+++ b/scripts/visualize.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(process_name=str(cfg.get("process", {}).get("name", "visualize"))),
-    )
-
     pred_path = _resolve_predictions_path(cfg)
     if not pred_path.exists():
         raise FileNotFoundError(f"predictions.csv not found: {pred_path}")
 
     df = pd.read_csv(pred_path)
@@
+    dataset_hash = None
+    model_name = None
+    model_version = None
+    if "dataset_hash" in df.columns and not df["dataset_hash"].empty:
+        dataset_hash = str(df["dataset_hash"].iloc[0])
+    if "model_name" in df.columns and not df["model_name"].empty:
+        model_name = str(df["model_name"].iloc[0])
+    if "model_version" in df.columns and not df["model_version"].empty:
+        model_version = str(df["model_version"].iloc[0])
+
+    dump_yaml(out_dir / "config.yaml", cfg)
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "visualize")),
+        cfg=cfg,
+        dataset_hash=dataset_hash,
+        model_name=model_name,
+        extra={"model_version": model_version} if model_version else None,
+    )
+    artifacts_utils.save_meta(out_dir, run_meta)
```

```diff
diff --git a/scripts/audit_dataset.py b/scripts/audit_dataset.py
index 2ac7c9d..f7b14a2 100644
--- a/scripts/audit_dataset.py
+++ b/scripts/audit_dataset.py
@@
-from src.common.meta import build_meta, save_meta
@@
-from src.utils.validate_config import validate_config
+from src.utils import artifacts as artifacts_utils
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "audit_dataset"))))
+    dump_yaml(run_dir / "config.yaml", cfg)
+    data_cfg = cfg.get("data", {}) or {}
+    dataset_csv = data_cfg.get("dataset_csv")
+    indices_dir = data_cfg.get("indices_dir")
+    dataset_hash = None
+    if dataset_csv:
+        dataset_hash = artifacts_utils.compute_dataset_hash(
+            Path(dataset_csv),
+            Path(indices_dir) if indices_dir else None,
+        )
+    run_meta = artifacts_utils.build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "audit_dataset")),
+        cfg=cfg,
+        dataset_hash=dataset_hash,
+    )
+    artifacts_utils.save_meta(run_dir, run_meta)
```

```diff
diff --git a/tests/contract/test_artifact_contract.py b/tests/contract/test_artifact_contract.py
new file mode 100644
index 0000000..cc8cabc
--- /dev/null
+++ b/tests/contract/test_artifact_contract.py
@@
+from __future__ import annotations
+
+import csv
+from pathlib import Path
+
+import pytest
+
+from src.common.utils import save_json
+from src.utils import artifacts as artifacts_utils
+
+
+def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
+    with path.open("w", encoding="utf-8", newline="") as f:
+        writer = csv.DictWriter(f, fieldnames=fieldnames)
+        writer.writeheader()
+        for row in rows:
+            writer.writerow(row)
+
+
+def _build_meta(process_name: str) -> dict[str, object]:
+    cfg = {
+        "task": {"name": "dummy"},
+        "model": {"name": "dummy"},
+        "featurizer": {"fingerprint": "morgan"},
+    }
+    return artifacts_utils.build_meta(
+        process_name=process_name,
+        cfg=cfg,
+        dataset_hash="deadbeef",
+        git_sha="test",
+    )
+
+
+def test_contract_train_passes(tmp_path: Path) -> None:
+    run_dir = tmp_path / "run"
+    run_dir.mkdir()
+    (run_dir / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
+    save_json(run_dir / "meta.json", _build_meta("train"))
+    model_dir = run_dir / "model"
+    model_dir.mkdir()
+    (model_dir / "model.ckpt").write_bytes(b"abc")
+    save_json(run_dir / "metrics.json", {"val": {"rmse": 1.0}})
+    artifacts_utils.validate_artifact_contract(run_dir, "train")
+
+
+def test_contract_predict_passes(tmp_path: Path) -> None:
+    run_dir = tmp_path / "run"
+    run_dir.mkdir()
+    (run_dir / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
+    save_json(run_dir / "meta.json", _build_meta("predict"))
+    _write_csv(run_dir / "predictions.csv", [{"sample_id": "A", "y_pred": 1.0}], ["sample_id", "y_pred"])
+    artifacts_utils.validate_artifact_contract(run_dir, "predict")
+
+
+def test_contract_missing_meta_keys_fails(tmp_path: Path) -> None:
+    run_dir = tmp_path / "run"
+    run_dir.mkdir()
+    (run_dir / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
+    save_json(run_dir / "meta.json", {"run_id": "only"})
+    _write_csv(run_dir / "predictions.csv", [{"sample_id": "A", "y_pred": 1.0}], ["sample_id", "y_pred"])
+    with pytest.raises(ValueError):
+        artifacts_utils.validate_artifact_contract(run_dir, "predict")
```

**Tests**
- Added `tests/contract/test_artifact_contract.py`.

**Verification Commands**
- `pytest -q tests/contract/test_artifact_contract.py`
- `pytest -q`

**Compatibility**
- Adds new `model/` artifacts and `metrics.json` while keeping legacy `artifacts/` fallback for predict/evaluate.
- Predict now emits `predictions.csv` (additive; existing JSON output preserved).
- `meta.json` gains required keys across processes.

**Queue Update**
- I did not update `work/queue.json` due to read‑only sandbox. Recommendation: set task `020` to `blocked` with reason “needs write access to apply artifact contract changes,” or rerun with write access and I can apply and mark `done`.

If you want me to apply these changes, rerun with write access and I’ll implement and update `work/queue.json`.
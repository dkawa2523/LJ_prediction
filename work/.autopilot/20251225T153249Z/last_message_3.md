**Status**
- Using S90_artifacts_versioning and S95_tests_ci to align artifacts with `docs/04` and add contract tests.
- Read-only sandbox + approval_policy=never: I can’t apply edits or update `work/queue.json`.
- Task file `work/tasks/020_artifact_contract_impl.md` still says blocked; queue shows 010/015 done, so please confirm unblocked and enable write access if you want me to apply the patch.

**Change Plan**
- `src/utils/artifacts.py`: add hash/meta helpers and shared writers for config/metrics/predictions.
- `src/fp/train.py`: write contract-compliant meta + metrics.json; add `model/` outputs.
- `src/gnn/train.py`: same as FP with GNN-specific featurizer state.
- `src/fp/predict.py` and `src/gnn/predict.py`: write `predictions.csv` with required columns and meta containing model_version.
- `tests/contract/test_artifact_contract.py`: add contract tests for required files/keys.

**Proposed Diff**

```diff
*** Add File: src/utils/artifacts.py
+from __future__ import annotations
+
+import hashlib
+import json
+from pathlib import Path
+from typing import Any, Dict, Iterable, Optional
+from uuid import uuid4
+
+from src.common.config import dump_yaml
+from src.common.meta import utc_now_iso
+from src.common.utils import load_json, save_json
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
+
+def _json_dumps(obj: Any) -> str:
+    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
+
+
+def hash_dict(obj: Dict[str, Any]) -> str:
+    return hashlib.sha256(_json_dumps(obj).encode("utf-8")).hexdigest()
+
+
+def hash_config(cfg: Dict[str, Any]) -> str:
+    return hash_dict(cfg)
+
+
+def _hash_file(path: Path, hasher: "hashlib._Hash") -> None:
+    with path.open("rb") as f:
+        for chunk in iter(lambda: f.read(1024 * 1024), b""):
+            hasher.update(chunk)
+
+
+def hash_paths(paths: Iterable[Path]) -> str:
+    hasher = hashlib.sha256()
+    for path in paths:
+        p = Path(path)
+        hasher.update(p.name.encode("utf-8"))
+        _hash_file(p, hasher)
+    return hasher.hexdigest()
+
+
+def dataset_hash(dataset_csv: Path, indices_dir: Optional[Path] = None, allow_missing: bool = False) -> str:
+    dataset_csv = Path(dataset_csv)
+    paths = []
+    if dataset_csv.exists():
+        paths.append(dataset_csv)
+    else:
+        if allow_missing:
+            return "unknown"
+        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
+
+    if indices_dir is not None:
+        indices_dir = Path(indices_dir)
+        if indices_dir.exists():
+            for p in sorted(indices_dir.glob("*.txt")):
+                paths.append(p)
+        elif not allow_missing:
+            raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
+
+    return hash_paths(paths)
+
+
+def _resolve_git_dir(repo_root: Path) -> Optional[Path]:
+    git_path = repo_root / ".git"
+    if git_path.is_dir():
+        return git_path
+    if git_path.is_file():
+        text = git_path.read_text(encoding="utf-8").strip()
+        if text.startswith("gitdir:"):
+            rel = text.split(":", 1)[1].strip()
+            return (repo_root / rel).resolve()
+    return None
+
+
+def get_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
+    repo_root = repo_root or Path(__file__).resolve().parents[2]
+    git_dir = _resolve_git_dir(repo_root)
+    if git_dir is None:
+        return None
+    head = git_dir / "HEAD"
+    if not head.exists():
+        return None
+    head_text = head.read_text(encoding="utf-8").strip()
+    if head_text.startswith("ref:"):
+        ref = head_text.split(" ", 1)[1].strip()
+        ref_path = git_dir / ref
+        if ref_path.exists():
+            return ref_path.read_text(encoding="utf-8").strip()
+        return None
+    return head_text
+
+
+def resolve_task_name(cfg: Dict[str, Any]) -> str:
+    return str(cfg.get("task", {}).get("name", "unknown"))
+
+
+def resolve_model_name(cfg: Dict[str, Any]) -> str:
+    return str(cfg.get("model", {}).get("name", "unknown"))
+
+
+def resolve_featureset_name(cfg: Dict[str, Any]) -> str:
+    explicit = cfg.get("featureset_name") or cfg.get("featureset", {}).get("name")
+    if explicit:
+        return str(explicit)
+
+    feat_cfg = cfg.get("featurizer", {}) or {}
+    if not feat_cfg:
+        return "unknown"
+    if "fingerprint" in feat_cfg:
+        base = f"fp_{feat_cfg.get('fingerprint', 'fp')}"
+        if feat_cfg.get("add_descriptors"):
+            base += "_desc"
+    elif "node_features" in feat_cfg or "edge_features" in feat_cfg:
+        base = "gnn_graph"
+        if feat_cfg.get("add_global_descriptors"):
+            base += "_desc"
+    else:
+        base = "featurizer"
+    return f"{base}_{hash_dict(feat_cfg)[:8]}"
+
+
+def build_meta(
+    process_name: str,
+    cfg: Dict[str, Any],
+    dataset_hash_value: str,
+    upstream_artifacts: Optional[Iterable[str]] = None,
+    context_cfg: Optional[Dict[str, Any]] = None,
+    tags: Optional[Iterable[str]] = None,
+    extra: Optional[Dict[str, Any]] = None,
+) -> Dict[str, Any]:
+    context_cfg = context_cfg or cfg
+    tag_list = list(tags or context_cfg.get("tags", []) or [])
+    meta = {
+        "run_id": uuid4().hex,
+        "process_name": process_name,
+        "created_at": utc_now_iso(),
+        "git_sha": get_git_sha(),
+        "dataset_hash": dataset_hash_value,
+        "config_hash": hash_config(cfg),
+        "task_name": resolve_task_name(context_cfg),
+        "model_name": resolve_model_name(context_cfg),
+        "featureset_name": resolve_featureset_name(context_cfg),
+        "upstream_artifacts": list(upstream_artifacts or []),
+        "tags": tag_list,
+    }
+    if extra:
+        meta.update(extra)
+    return meta
+
+
+def write_config(run_dir: str | Path, cfg: Dict[str, Any]) -> Path:
+    run_dir = Path(run_dir)
+    path = run_dir / "config.yaml"
+    dump_yaml(path, cfg)
+    return path
+
+
+def write_meta(run_dir: str | Path, meta: Dict[str, Any]) -> Path:
+    run_dir = Path(run_dir)
+    path = run_dir / "meta.json"
+    save_json(path, meta)
+    return path
+
+
+def write_metrics(run_dir: str | Path, metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
+    run_dir = Path(run_dir)
+    path = run_dir / filename
+    save_json(path, metrics)
+    return path
+
+
+def write_predictions(run_dir: str | Path, rows: Iterable[Dict[str, Any]], filename: str = "predictions.csv") -> Path:
+    import pandas as pd
+
+    run_dir = Path(run_dir)
+    df = pd.DataFrame(list(rows))
+    missing = REQUIRED_PREDICTION_COLUMNS - set(df.columns)
+    if missing:
+        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
+    path = run_dir / filename
+    df.to_csv(path, index=False)
+    return path
+
+
+def load_meta(run_dir: str | Path) -> Dict[str, Any]:
+    path = Path(run_dir) / "meta.json"
+    if not path.exists():
+        return {}
+    return load_json(path)
```

```diff
--- a/src/fp/train.py
+++ b/src/fp/train.py
@@
-import argparse
-import pickle
+import argparse
+import pickle
+import shutil
 from pathlib import Path
 from typing import Any, Dict, List, Optional, Tuple
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import dump_yaml, load_config
@@
-from src.common.utils import ensure_dir, get_logger, save_json, set_seed
+from src.common.utils import ensure_dir, get_logger, save_json, set_seed
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_meta, dataset_hash, write_config, write_meta, write_metrics
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
+    write_config(run_dir, cfg)
@@
     if not sdf_dir.exists():
         raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")
 
+    dataset_hash_value = dataset_hash(dataset_csv, indices_dir)
+    meta = build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "train")),
+        cfg=cfg,
+        dataset_hash_value=dataset_hash_value,
+    )
+    write_meta(run_dir, meta)
+
@@
     with open(artifacts_dir / "ad.pkl", "wb") as f:
         pickle.dump(ad_artifact, f)
 
+    model_dir = ensure_dir(run_dir / "model")
+    shutil.copyfile(model_path, model_dir / "model.ckpt")
+    preprocess = {"imputer": imputer, "scaler": scaler}
+    with open(model_dir / "preprocess.pkl", "wb") as f:
+        pickle.dump(preprocess, f)
+    save_json(model_dir / "featurizer_state.json", {"type": "fp", "config": fp_cfg.__dict__})
+
@@
     save_json(run_dir / "metrics_val.json", metrics_val)
     save_json(run_dir / "metrics_test.json", metrics_test)
+    metrics_summary = {
+        "val": metrics_val,
+        "test": metrics_test,
+        "n_train": len(y_train),
+        "n_val": len(y_val),
+        "n_test": len(y_test),
+        "seed": seed,
+    }
+    write_metrics(run_dir, metrics_summary)
```

```diff
--- a/src/gnn/train.py
+++ b/src/gnn/train.py
@@
-import argparse
-import pickle
+import argparse
+import pickle
+import shutil
+from dataclasses import asdict
 from pathlib import Path
 from typing import Any, Dict, List, Optional
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import dump_yaml, load_config
@@
-from src.common.utils import ensure_dir, get_logger, save_json, set_seed
+from src.common.utils import ensure_dir, get_logger, save_json, set_seed
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_meta, dataset_hash, write_config, write_meta, write_metrics
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
+    write_config(run_dir, cfg)
+    dataset_hash_value = dataset_hash(dataset_csv, indices_dir)
+    meta = build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "train")),
+        cfg=cfg,
+        dataset_hash_value=dataset_hash_value,
+    )
+    write_meta(run_dir, meta)
@@
     dump_yaml(run_dir / "config_snapshot.yaml", cfg)
     save_json(run_dir / "metrics_val.json", val_metrics)
     save_json(run_dir / "metrics_test.json", test_metrics)
+    metrics_summary = {
+        "val": val_metrics,
+        "test": test_metrics,
+        "n_train": len(train_data),
+        "n_val": len(val_data),
+        "n_test": len(test_data),
+        "seed": seed,
+    }
+    write_metrics(run_dir, metrics_summary)
+
+    model_dir = ensure_dir(run_dir / "model")
+    shutil.copyfile(best_path, model_dir / "model.ckpt")
+    save_json(model_dir / "featurizer_state.json", {"type": "gnn", "config": asdict(gcfg)})
```

```diff
--- a/src/fp/predict.py
+++ b/src/fp/predict.py
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import load_config
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_meta, dataset_hash, load_meta, write_config, write_meta, write_predictions
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    write_config(out_dir, cfg)
+    train_meta = load_meta(model_artifact_dir)
+    train_data_cfg = train_cfg.get("data", {})
+    train_dataset_csv = Path(train_data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
+    indices_dir = train_data_cfg.get("indices_dir")
+    indices_dir = Path(indices_dir) if indices_dir else None
+    dataset_hash_value = train_meta.get("dataset_hash") or dataset_hash(train_dataset_csv, indices_dir, allow_missing=True)
+    model_version = str(train_meta.get("model_version") or train_meta.get("run_id") or model_artifact_dir.name)
+    meta = build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        cfg=cfg,
+        dataset_hash_value=dataset_hash_value,
+        upstream_artifacts=[str(model_artifact_dir)],
+        context_cfg=train_cfg,
+        extra={"model_version": model_version},
+    )
+    write_meta(out_dir, meta)
@@
-    result = {
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": meta["model_name"],
+        "model_version": model_version,
+        "dataset_hash": dataset_hash_value,
+        "run_id": meta["run_id"],
+    }
+    write_predictions(out_dir, [pred_row])
+
+    result = {
         "cas": cas,
         "query": query,
         "prediction": pred,
         "resolve_meta": resolve_meta,
         "ad": ad_res.to_dict(),
         "feature_meta": meta,
     }
```

```diff
--- a/src/gnn/predict.py
+++ b/src/gnn/predict.py
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import load_config
@@
-from src.utils.validate_config import validate_config
+from src.utils.artifacts import build_meta, dataset_hash, load_meta, write_config, write_meta, write_predictions
+from src.utils.validate_config import validate_config
@@
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    write_config(out_dir, cfg)
+    train_meta = load_meta(model_artifact_dir)
+    train_data_cfg = train_cfg.get("data", {})
+    train_dataset_csv = Path(train_data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
+    indices_dir = train_data_cfg.get("indices_dir")
+    indices_dir = Path(indices_dir) if indices_dir else None
+    dataset_hash_value = train_meta.get("dataset_hash") or dataset_hash(train_dataset_csv, indices_dir, allow_missing=True)
+    model_version = str(train_meta.get("model_version") or train_meta.get("run_id") or model_artifact_dir.name)
+    meta = build_meta(
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        cfg=cfg,
+        dataset_hash_value=dataset_hash_value,
+        upstream_artifacts=[str(model_artifact_dir)],
+        context_cfg=train_cfg,
+        extra={"model_version": model_version},
+    )
+    write_meta(out_dir, meta)
@@
-    result = {
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": meta["model_name"],
+        "model_version": model_version,
+        "dataset_hash": dataset_hash_value,
+        "run_id": meta["run_id"],
+    }
+    write_predictions(out_dir, [pred_row])
+
+    result = {
         "cas": cas,
         "query": query,
         "prediction": pred,
         "resolve_meta": resolve_meta,
         "ad": None if ad_res is None else ad_res.to_dict(),
     }
```

```diff
*** Add File: tests/contract/test_artifact_contract.py
+from __future__ import annotations
+
+import json
+
+import pandas as pd
+
+from src.utils.artifacts import REQUIRED_META_KEYS, build_meta, write_config, write_meta, write_metrics, write_predictions
+
+
+def _dummy_cfg():
+    return {
+        "task": {"name": "lj_epsilon"},
+        "model": {"name": "lightgbm"},
+        "featurizer": {"fingerprint": "morgan"},
+    }
+
+
+def test_train_artifact_contract_minimum(tmp_path):
+    cfg = _dummy_cfg()
+    meta = build_meta(process_name="train", cfg=cfg, dataset_hash_value="dummyhash")
+    write_config(tmp_path, cfg)
+    write_meta(tmp_path, meta)
+    write_metrics(tmp_path, {"val": {"rmse": 1.0}})
+    model_dir = tmp_path / "model"
+    model_dir.mkdir()
+    (model_dir / "model.ckpt").write_bytes(b"dummy")
+
+    meta_loaded = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
+    for key in REQUIRED_META_KEYS:
+        assert key in meta_loaded
+
+    assert (tmp_path / "config.yaml").exists()
+    assert (tmp_path / "metrics.json").exists()
+    assert (model_dir / "model.ckpt").exists()
+
+
+def test_predict_artifact_contract_minimum(tmp_path):
+    cfg = _dummy_cfg()
+    meta = build_meta(process_name="predict", cfg=cfg, dataset_hash_value="dummyhash", extra={"model_version": "v1"})
+    write_config(tmp_path, cfg)
+    write_meta(tmp_path, meta)
+    write_predictions(tmp_path, [{"sample_id": "71-43-2", "y_pred": 0.123}])
+    pred_path = tmp_path / "predictions.csv"
+    df = pd.read_csv(pred_path)
+    assert {"sample_id", "y_pred"}.issubset(set(df.columns))
```

**Tests Added/Updated**
- `tests/contract/test_artifact_contract.py`

**Validation Commands**
- `pytest tests/contract/test_artifact_contract.py`
- (Optional, writes artifacts) `python scripts/train.py --config configs/fp/train_fixture.yaml`

**Compatibility Impact**
- Config: no changes required.
- CLI: no behavior changes.
- Artifacts: adds `model/` with `model.ckpt`, `preprocess.pkl`, `featurizer_state.json`; adds `metrics.json`; predict now writes `predictions.csv` with required columns; existing `artifacts/` + `prediction_*.json` + `config_snapshot.yaml` remain.

**Queue Status**
- I can’t update `work/queue.json` in read-only; this should remain `blocked` until you enable write access and confirm unblocked.

If you want me to apply this, pick one:
1) Enable write access and confirm task 020 is unblocked.  
2) Ask me to apply the patch and run the validation commands.
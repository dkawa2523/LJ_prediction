Using S90_artifacts_versioning and S95_tests_ci to align artifact outputs with the contract and add contract tests.

Read-only sandbox blocked file edits and `work/queue.json` updates; below is a proposed plan and diff you can apply.

**Change Plan**
- `src/utils/artifacts.py`: add shared artifact helpers (hashing, meta writer, model/preprocess/featurizer packaging, predictions CSV).
- `src/fp/train.py`: replace meta/config writes with artifact helpers and emit `metrics.json` + `model/` artifacts.
- `src/gnn/train.py`: same as FP train, plus `metrics.json` + `model/` artifacts.
- `src/fp/predict.py`: write contract-compliant `meta.json` and `predictions.csv`.
- `src/gnn/predict.py`: same as FP predict.
- `tests/contract/test_artifacts_contract.py`: add contract tests for required keys/files and predictions CSV.

**Proposed Diff**
`src/utils/artifacts.py`
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
+import pickle
+import shutil
+from dataclasses import asdict, is_dataclass
+from pathlib import Path
+from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
+from uuid import uuid4
+
+from src.common.config import dump_yaml
+from src.common.meta import utc_now_iso
+from src.common.utils import ensure_dir, load_json, save_json
+
+
+def _json_default(obj: Any) -> Any:
+    if is_dataclass(obj):
+        return asdict(obj)
+    if isinstance(obj, Path):
+        return str(obj)
+    return str(obj)
+
+
+def _get_nested(cfg: Mapping[str, Any], *keys: str) -> Optional[Any]:
+    cur: Any = cfg
+    for key in keys:
+        if not isinstance(cur, Mapping):
+            return None
+        cur = cur.get(key)
+        if cur is None:
+            return None
+    return cur
+
+
+def compute_config_hash(cfg: Mapping[str, Any]) -> str:
+    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
+    return hashlib.sha256(payload).hexdigest()
+
+
+def _update_hash_from_file(hasher: "hashlib._Hash", path: Path) -> None:
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
+    _update_hash_from_file(hasher, dataset_csv)
+    if indices_dir is not None:
+        indices_dir = Path(indices_dir)
+        if indices_dir.exists():
+            for path in sorted(indices_dir.glob("*.txt")):
+                hasher.update(path.name.encode("utf-8"))
+                _update_hash_from_file(hasher, path)
+    return hasher.hexdigest()
+
+
+def get_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
+    for key in ("GIT_SHA", "GITHUB_SHA", "CI_COMMIT_SHA"):
+        value = os.environ.get(key)
+        if value:
+            return value.strip()
+    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
+    git_dir = repo_root / ".git"
+    head = git_dir / "HEAD"
+    if not head.exists():
+        return None
+    head_value = head.read_text(encoding="utf-8").strip()
+    if head_value.startswith("ref:"):
+        ref = head_value.split(" ", 1)[1].strip()
+        ref_path = git_dir / ref
+        if ref_path.exists():
+            return ref_path.read_text(encoding="utf-8").strip()
+        packed_refs = git_dir / "packed-refs"
+        if packed_refs.exists():
+            for line in packed_refs.read_text(encoding="utf-8").splitlines():
+                if not line or line.startswith("#") or line.startswith("^"):
+                    continue
+                sha, ref_name = line.split(" ", 1)
+                if ref_name.strip() == ref:
+                    return sha
+        return None
+    if len(head_value) >= 7:
+        return head_value
+    return None
+
+
+def resolve_featureset_name(cfg: Mapping[str, Any]) -> str:
+    name = _get_nested(cfg, "featureset", "name") or cfg.get("featureset_name")
+    if name:
+        return str(name)
+    feat_cfg = cfg.get("featurizer", {}) if isinstance(cfg, Mapping) else {}
+    if isinstance(feat_cfg, Mapping):
+        if "fingerprint" in feat_cfg:
+            parts = [str(feat_cfg.get("fingerprint", "fp"))]
+            if feat_cfg.get("morgan_radius") is not None:
+                parts.append(f"r{feat_cfg.get('morgan_radius')}")
+            if feat_cfg.get("n_bits") is not None:
+                parts.append(f"b{feat_cfg.get('n_bits')}")
+            if feat_cfg.get("add_descriptors"):
+                parts.append("desc")
+            return "_".join(parts)
+        if feat_cfg.get("node_features") or feat_cfg.get("edge_features"):
+            return "graph"
+    return "unknown"
+
+
+def resolve_model_version(cfg: Mapping[str, Any], fallback: Optional[str] = None) -> str:
+    experiment_name = _get_nested(cfg, "experiment", "name") or _get_nested(cfg, "output", "exp_name")
+    if experiment_name:
+        return str(experiment_name)
+    if fallback:
+        return str(fallback)
+    return "unknown"
+
+
+def build_meta_from_cfg(
+    process_cfg: Mapping[str, Any],
+    process_name: str,
+    dataset_hash: Optional[str] = None,
+    upstream_artifacts: Optional[Iterable[str]] = None,
+    tags: Optional[Iterable[str]] = None,
+    extra: Optional[Mapping[str, Any]] = None,
+    info_cfg: Optional[Mapping[str, Any]] = None,
+    repo_root: Optional[Path] = None,
+) -> Dict[str, Any]:
+    info_cfg = info_cfg or process_cfg
+    meta = {
+        "run_id": uuid4().hex,
+        "process_name": process_name,
+        "created_at": utc_now_iso(),
+        "git_sha": get_git_sha(repo_root) or "unknown",
+        "dataset_hash": dataset_hash or "unknown",
+        "config_hash": compute_config_hash(process_cfg),
+        "task_name": str(_get_nested(info_cfg, "task", "name") or "unknown"),
+        "model_name": str(_get_nested(info_cfg, "model", "name") or "unknown"),
+        "featureset_name": resolve_featureset_name(info_cfg),
+        "upstream_artifacts": list(upstream_artifacts or []),
+        "tags": list(tags or []),
+    }
+    if extra:
+        meta.update(extra)
+    return meta
+
+
+def write_run_metadata(
+    run_dir: str | Path,
+    cfg: Mapping[str, Any],
+    process_name: str,
+    dataset_hash: Optional[str] = None,
+    upstream_artifacts: Optional[Iterable[str]] = None,
+    tags: Optional[Iterable[str]] = None,
+    extra: Optional[Mapping[str, Any]] = None,
+    info_cfg: Optional[Mapping[str, Any]] = None,
+) -> Dict[str, Any]:
+    run_dir = ensure_dir(run_dir)
+    dump_yaml(run_dir / "config.yaml", dict(cfg))
+    meta = build_meta_from_cfg(
+        process_cfg=cfg,
+        process_name=process_name,
+        dataset_hash=dataset_hash,
+        upstream_artifacts=upstream_artifacts,
+        tags=tags,
+        extra=extra,
+        info_cfg=info_cfg,
+    )
+    save_json(run_dir / "meta.json", meta)
+    return meta
+
+
+def write_metrics(run_dir: str | Path, metrics: Mapping[str, Any], filename: str = "metrics.json") -> Path:
+    run_dir = Path(run_dir)
+    path = run_dir / filename
+    save_json(path, dict(metrics))
+    return path
+
+
+def write_model_artifact(run_dir: str | Path, model_path: str | Path) -> Path:
+    model_path = Path(model_path)
+    if not model_path.exists():
+        raise FileNotFoundError(f"model file not found: {model_path}")
+    model_dir = ensure_dir(Path(run_dir) / "model")
+    dest = model_dir / "model.ckpt"
+    shutil.copy2(model_path, dest)
+    return dest
+
+
+def write_preprocess_bundle(
+    run_dir: str | Path,
+    imputer: Any = None,
+    scaler: Any = None,
+) -> Optional[Path]:
+    bundle: Dict[str, Any] = {}
+    if imputer is not None:
+        bundle["imputer"] = imputer
+    if scaler is not None:
+        bundle["scaler"] = scaler
+    if not bundle:
+        return None
+    model_dir = ensure_dir(Path(run_dir) / "model")
+    path = model_dir / "preprocess.pkl"
+    with path.open("wb") as f:
+        pickle.dump(bundle, f)
+    return path
+
+
+def write_featurizer_state(
+    run_dir: str | Path,
+    featurizer_state: Optional[Any],
+) -> Optional[Path]:
+    if featurizer_state is None:
+        return None
+    if is_dataclass(featurizer_state):
+        featurizer_state = asdict(featurizer_state)
+    if not isinstance(featurizer_state, Mapping):
+        return None
+    model_dir = ensure_dir(Path(run_dir) / "model")
+    path = model_dir / "featurizer_state.json"
+    save_json(path, dict(featurizer_state))
+    return path
+
+
+def write_predictions_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
+    if not rows:
+        raise ValueError("rows is empty")
+    required = {"sample_id", "y_pred"}
+    for row in rows:
+        missing = required - set(row.keys())
+        if missing:
+            raise ValueError(f"predictions row missing required keys: {sorted(missing)}")
+    all_keys = set().union(*(row.keys() for row in rows))
+    extra_keys = sorted(all_keys - required)
+    fieldnames = ["sample_id", "y_pred"] + extra_keys
+    path = Path(path)
+    path.parent.mkdir(parents=True, exist_ok=True)
+    with path.open("w", newline="", encoding="utf-8") as f:
+        writer = csv.DictWriter(f, fieldnames=fieldnames)
+        writer.writeheader()
+        for row in rows:
+            writer.writerow(dict(row))
+    return path
+
+
+def load_meta(path: str | Path) -> Dict[str, Any]:
+    path = Path(path)
+    if not path.exists():
+        return {}
+    return load_json(path)
```

`src/fp/train.py`
```diff
diff --git a/src/fp/train.py b/src/fp/train.py
--- a/src/fp/train.py
+++ b/src/fp/train.py
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import dump_yaml, load_config
@@
-from src.common.utils import ensure_dir, get_logger, save_json, set_seed
+from src.common.utils import ensure_dir, get_logger, save_json, set_seed
 from src.fp.feature_utils import build_features, hash_cfg
 from src.fp.featurizer_fp import FPConfig, morgan_bitvect
 from src.fp.models import get_model
+from src.utils import artifacts as artifact_utils
 from src.utils.validate_config import validate_config
@@
-    logger = get_logger("fp_train", log_file=run_dir / "train.log")
-
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
-
-    if not dataset_csv.exists():
+    logger = get_logger("fp_train", log_file=run_dir / "train.log")
+
+    if not dataset_csv.exists():
         raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
     if not indices_dir.exists():
         raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
     if not sdf_dir.exists():
         raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")
+
+    dataset_hash = artifact_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    artifact_utils.write_run_metadata(
+        run_dir,
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "train")),
+        dataset_hash=dataset_hash,
+        extra={"model_version": artifact_utils.resolve_model_version(cfg)},
+    )
@@
     save_json(run_dir / "metrics_val.json", metrics_val)
     save_json(run_dir / "metrics_test.json", metrics_test)
     dump_yaml(run_dir / "config_snapshot.yaml", cfg)
+
+    metrics_summary = {
+        "val": metrics_val,
+        "test": metrics_test,
+        "n_train": int(len(y_train)),
+        "n_val": int(len(y_val)),
+        "n_test": int(len(y_test)),
+        "seed": seed,
+    }
+    artifact_utils.write_metrics(run_dir, metrics_summary)
+    artifact_utils.write_model_artifact(run_dir, model_path)
+    artifact_utils.write_preprocess_bundle(run_dir, imputer=imputer, scaler=scaler)
+    artifact_utils.write_featurizer_state(run_dir, feat_cfg)
```

`src/gnn/train.py`
```diff
diff --git a/src/gnn/train.py b/src/gnn/train.py
--- a/src/gnn/train.py
+++ b/src/gnn/train.py
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import dump_yaml, load_config
@@
-from src.common.utils import ensure_dir, get_logger, save_json, set_seed
+from src.common.utils import ensure_dir, get_logger, save_json, set_seed
 from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg
 from src.gnn.models import GCNRegressor, MPNNRegressor
+from src.utils import artifacts as artifact_utils
 from src.utils.validate_config import validate_config
@@
-    logger = get_logger("gnn_train", log_file=run_dir / "train.log")
-
-    dump_yaml(run_dir / "config.yaml", cfg)
-    save_meta(run_dir, build_meta(process_name=str(cfg.get("process", {}).get("name", "train"))))
+    logger = get_logger("gnn_train", log_file=run_dir / "train.log")
+
+    if not dataset_csv.exists():
+        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
+    if not indices_dir.exists():
+        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
+    if not sdf_dir.exists():
+        raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")
+
+    dataset_hash = artifact_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    artifact_utils.write_run_metadata(
+        run_dir,
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "train")),
+        dataset_hash=dataset_hash,
+        extra={"model_version": artifact_utils.resolve_model_version(cfg)},
+    )
@@
     save_json(run_dir / "metrics_val.json", val_metrics)
     save_json(run_dir / "metrics_test.json", test_metrics)
+
+    metrics_summary = {
+        "val": val_metrics,
+        "test": test_metrics,
+        "n_train": int(len(train_data)),
+        "n_val": int(len(val_data)),
+        "n_test": int(len(test_data)),
+        "seed": seed,
+    }
+    artifact_utils.write_metrics(run_dir, metrics_summary)
+    artifact_utils.write_model_artifact(run_dir, best_path)
+    artifact_utils.write_featurizer_state(run_dir, feat_cfg)
```

`src/fp/predict.py`
```diff
diff --git a/src/fp/predict.py b/src/fp/predict.py
--- a/src/fp/predict.py
+++ b/src/fp/predict.py
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import load_config
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, save_json
 from src.fp.featurizer_fp import FPConfig, featurize_mol, morgan_bitvect
+from src.utils import artifacts as artifact_utils
 from src.utils.validate_config import validate_config
@@
-    logger = get_logger("fp_predict", log_file=out_dir / "predict.log")
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    logger = get_logger("fp_predict", log_file=out_dir / "predict.log")
+    model_meta = artifact_utils.load_meta(model_artifact_dir / "meta.json")
+    dataset_hash = model_meta.get("dataset_hash")
+    train_data_cfg = train_cfg.get("data", {})
+    train_dataset_csv = Path(train_data_cfg.get("dataset_csv", dataset_csv))
+    indices_value = train_data_cfg.get("indices_dir")
+    train_indices_dir = Path(indices_value) if indices_value else None
+    if not dataset_hash and train_dataset_csv.exists():
+        dataset_hash = artifact_utils.compute_dataset_hash(
+            train_dataset_csv,
+            train_indices_dir if train_indices_dir and train_indices_dir.exists() else None,
+        )
+    model_version = model_meta.get("model_version") or artifact_utils.resolve_model_version(
+        train_cfg, fallback=model_artifact_dir.name
+    )
+    run_meta = artifact_utils.write_run_metadata(
+        out_dir,
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        dataset_hash=dataset_hash,
+        upstream_artifacts=[str(model_artifact_dir)],
+        extra={"model_version": model_version},
+        info_cfg=train_cfg,
+    )
@@
-    result = {
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": run_meta.get("model_name"),
+        "model_version": model_version,
+        "dataset_hash": run_meta.get("dataset_hash"),
+        "run_id": run_meta.get("run_id"),
+    }
+    artifact_utils.write_predictions_csv(out_dir / "predictions.csv", [pred_row])
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

`src/gnn/predict.py`
```diff
diff --git a/src/gnn/predict.py b/src/gnn/predict.py
--- a/src/gnn/predict.py
+++ b/src/gnn/predict.py
@@
-from src.common.config import dump_yaml, load_config
-from src.common.meta import build_meta, save_meta
+from src.common.config import load_config
@@
-from src.common.utils import ensure_dir, get_logger, save_json
+from src.common.utils import ensure_dir, get_logger, save_json
 from src.fp.featurizer_fp import morgan_bitvect
 from src.gnn.featurizer_graph import featurize_mol_to_pyg
 from src.gnn.models import GCNRegressor, MPNNRegressor
+from src.utils import artifacts as artifact_utils
 from src.utils.validate_config import validate_config
@@
-    logger = get_logger("gnn_predict", log_file=out_dir / "predict.log")
-    dump_yaml(out_dir / "config.yaml", cfg)
-    save_meta(
-        out_dir,
-        build_meta(
-            process_name=str(cfg.get("process", {}).get("name", "predict")),
-            upstream_artifacts=[str(model_artifact_dir)],
-        ),
-    )
+    logger = get_logger("gnn_predict", log_file=out_dir / "predict.log")
+    model_meta = artifact_utils.load_meta(model_artifact_dir / "meta.json")
+    dataset_hash = model_meta.get("dataset_hash")
+    train_data_cfg = train_cfg.get("data", {})
+    train_dataset_csv = Path(train_data_cfg.get("dataset_csv", dataset_csv))
+    indices_value = train_data_cfg.get("indices_dir")
+    train_indices_dir = Path(indices_value) if indices_value else None
+    if not dataset_hash and train_dataset_csv.exists():
+        dataset_hash = artifact_utils.compute_dataset_hash(
+            train_dataset_csv,
+            train_indices_dir if train_indices_dir and train_indices_dir.exists() else None,
+        )
+    model_version = model_meta.get("model_version") or artifact_utils.resolve_model_version(
+        train_cfg, fallback=model_artifact_dir.name
+    )
+    run_meta = artifact_utils.write_run_metadata(
+        out_dir,
+        cfg,
+        process_name=str(cfg.get("process", {}).get("name", "predict")),
+        dataset_hash=dataset_hash,
+        upstream_artifacts=[str(model_artifact_dir)],
+        extra={"model_version": model_version},
+        info_cfg=train_cfg,
+    )
@@
-    result = {
+    pred_row = {
+        "sample_id": cas,
+        "y_pred": pred,
+        "model_name": run_meta.get("model_name"),
+        "model_version": model_version,
+        "dataset_hash": run_meta.get("dataset_hash"),
+        "run_id": run_meta.get("run_id"),
+    }
+    artifact_utils.write_predictions_csv(out_dir / "predictions.csv", [pred_row])
+
+    result = {
         "cas": cas,
         "query": query,
         "prediction": pred,
         "resolve_meta": resolve_meta,
         "ad": None if ad_res is None else ad_res.to_dict(),
     }
```

`tests/contract/test_artifacts_contract.py`
```diff
diff --git a/tests/contract/test_artifacts_contract.py b/tests/contract/test_artifacts_contract.py
new file mode 100644
--- /dev/null
+++ b/tests/contract/test_artifacts_contract.py
@@
+from __future__ import annotations
+
+from pathlib import Path
+
+from src.utils import artifacts as artifact_utils
+
+
+def _write_indices(indices_dir: Path) -> None:
+    splits = {"train": [0], "val": [1], "test": [2]}
+    for name, rows in splits.items():
+        content = "\n".join(str(r) for r in rows) + "\n"
+        (indices_dir / f"{name}.txt").write_text(content, encoding="utf-8")
+
+
+def test_train_contract_artifacts(tmp_path: Path) -> None:
+    dataset_csv = tmp_path / "dataset.csv"
+    dataset_csv.write_text(
+        "CAS,lj_epsilon_over_k_K\nA,1.0\nB,2.0\nC,3.0\n",
+        encoding="utf-8",
+    )
+    indices_dir = tmp_path / "indices"
+    indices_dir.mkdir()
+    _write_indices(indices_dir)
+
+    run_dir = tmp_path / "run"
+    artifacts_dir = run_dir / "artifacts"
+    artifacts_dir.mkdir(parents=True)
+    model_path = artifacts_dir / "model.pkl"
+    model_path.write_bytes(b"model")
+
+    cfg = {
+        "process": {"name": "train"},
+        "data": {"dataset_csv": str(dataset_csv), "indices_dir": str(indices_dir)},
+        "task": {"name": "lj_epsilon"},
+        "model": {"name": "lightgbm"},
+        "featurizer": {"fingerprint": "morgan", "morgan_radius": 2, "n_bits": 128},
+    }
+
+    dataset_hash = artifact_utils.compute_dataset_hash(dataset_csv, indices_dir)
+    meta = artifact_utils.write_run_metadata(run_dir, cfg, process_name="train", dataset_hash=dataset_hash)
+    artifact_utils.write_metrics(
+        run_dir,
+        {"val": {"rmse": 1.0}, "test": {"rmse": 1.2}, "n_train": 1, "n_val": 1, "n_test": 1, "seed": 42},
+    )
+    artifact_utils.write_model_artifact(run_dir, model_path)
+    artifact_utils.write_preprocess_bundle(run_dir, imputer={"dummy": True}, scaler={"dummy": False})
+    artifact_utils.write_featurizer_state(run_dir, cfg["featurizer"])
+
+    assert (run_dir / "config.yaml").exists()
+    assert (run_dir / "meta.json").exists()
+    assert (run_dir / "metrics.json").exists()
+    assert (run_dir / "model" / "model.ckpt").exists()
+    assert (run_dir / "model" / "preprocess.pkl").exists()
+    assert (run_dir / "model" / "featurizer_state.json").exists()
+
+    required = {
+        "run_id",
+        "process_name",
+        "created_at",
+        "git_sha",
+        "dataset_hash",
+        "config_hash",
+        "task_name",
+        "model_name",
+        "featureset_name",
+        "upstream_artifacts",
+        "tags",
+    }
+    assert required.issubset(set(meta.keys()))
+
+
+def test_predictions_csv_contract(tmp_path: Path) -> None:
+    out_path = tmp_path / "predictions.csv"
+    rows = [{"sample_id": "A", "y_pred": 1.23, "model_name": "lightgbm"}]
+    artifact_utils.write_predictions_csv(out_path, rows)
+
+    header = out_path.read_text(encoding="utf-8").splitlines()[0]
+    assert header.startswith("sample_id,y_pred")
```

**Tests Added/Updated**
- `tests/contract/test_artifacts_contract.py`

**Verification Commands**
- `pytest -q tests/contract/test_artifacts_contract.py`

**Compatibility Impact**
- Config: no schema changes; `meta.json` now includes required keys plus `config_hash`.
- CLI: no CLI changes; additional artifact files are written.
- Artifacts: train adds `model/model.ckpt`, `model/preprocess.pkl`, `model/featurizer_state.json`, `metrics.json`; predict adds `predictions.csv`; `meta.json` is enriched.

**Queue Status**
- Unable to update `work/queue.json` due to read-only; set task `020` to `blocked` with reason “sandbox read-only prevented applying changes”.

If you want me to apply this directly, the next steps I can take are:
1) Rerun with write access so I can apply the patch and update `work/queue.json`.
2) Run `pytest -q tests/contract/test_artifacts_contract.py` to validate the contract test.
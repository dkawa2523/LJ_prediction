Using S90_artifacts_versioning and S95_tests_ci because this task is the artifact contract + contract tests.

**Status**
- Blocked per `work/tasks/020_artifact_contract_impl.md` (run dir standardization not finalized).
- `work/queue.json` shows 010/015 done; please confirm unblocking and the final run-dir naming so I can proceed.

**1) Change Plan (File-Level)**
- `src/utils/artifacts.py`: add helpers to compute `config_hash`/`dataset_hash`, assemble meta, and write contract artifacts (`config.yaml`, `meta.json`, `metrics.json`, `predictions.csv`, `model/*`).
- `src/common/meta.py`: expand `build_meta` to include required keys from docs/04 (git_sha, dataset_hash, config_hash, task/model/featureset, tags).
- `src/fp/train.py`: write `model/` + `metrics.json` via artifact helpers; align filenames/locations to docs/04.
- `src/gnn/train.py`: same as above for GNN artifacts.
- `src/fp/predict.py`: emit `predictions.csv` with required columns and enrich meta with `model_version`.
- `src/gnn/predict.py`: same as above for GNN predictions.
- `tests/contract/test_artifacts_contract.py`: contract tests for required files and `meta.json` keys.

**2) Implementation (Diff)**
- No changes applied (task blocked).

**3) Tests Added/Updated**
- None (task blocked).

**4) Verification Commands**
- After unblocking and implementation:
  - `pytest tests/contract/test_artifacts_contract.py`
  - `python scripts/train.py --config configs/fp/train.yaml`
  - `python scripts/predict.py --config configs/fp/predict.yaml --query 64-17-5`

**5) Compatibility Impact**
- None now. Planned changes will adjust artifact layout and meta contents to match docs/04.

**6) work/queue.json Status Update**
- Not updated (read-only sandbox, approval_policy=never). It should be set to `blocked` with the task-file reason unless you unblock this task.

If 010/015 are indeed complete, tell me the finalized run-dir convention and I’ll proceed with the implementation.
Using S90_artifacts_versioning and S95_tests_ci because this task is the artifact contract implementation plus contract tests.

No code changes yet because `work/tasks/020_artifact_contract_impl.md` marks the task as blocked (run dir standardization unresolved), and this session is read-only so I can’t update `work/queue.json`. I reviewed current artifact outputs in `src/fp/train.py`, `src/gnn/train.py`, `src/fp/predict.py`, `src/gnn/predict.py`, `src/fp/evaluate.py`, `src/gnn/evaluate.py`, and `src/common/meta.py` to prep a file-level plan.

**Change Plan (pending unblock)**
- `src/utils/artifacts.py`: add helpers to write `config.yaml`, `meta.json`, `metrics.json`, `predictions.csv`, and place model files under `model/` using the finalized run dir convention.
- `src/common/meta.py`: extend `build_meta` to include required keys (`dataset_hash`, `config_hash`, `git_sha`, etc.) and accept `model_version`.
- `src/common/utils.py` (or new `src/common/hashing.py`): implement dataset/config hashing used by meta and contract tests.
- `src/fp/train.py` and `src/gnn/train.py`: route artifact writes through helpers; emit `model/model.ckpt` plus a consolidated `metrics.json` (retain `config_snapshot.yaml` for inference).
- `src/fp/predict.py` and `src/gnn/predict.py`: write `predictions.csv` with required columns and meta containing model version/run info.
- `src/fp/evaluate.py`, `src/gnn/evaluate.py`, `tests/contract/test_artifacts.py`: align evaluate outputs to `metrics.json` contract and add contract tests for required files/keys.

**Implementation (Diff)**
- No diff applied (blocked + read-only sandbox).

**Tests Added/Updated**
- None (blocked).

**Verification Commands (after unblock)**
```bash
pytest tests/contract/test_artifacts.py
pytest tests
```

**Compatibility Impact**
- None yet.
- Planned changes will add/rename artifact files (`metrics.json`, `model/model.ckpt`, `predictions.csv`) and may affect downstream consumers expecting current names (`metrics_val.json`, `metrics_test.json`, `model.pkl`, `model_best.pt`).

**Queue Status**
- Task remains blocked per `work/tasks/020_artifact_contract_impl.md`.
- `work/queue.json` still shows `in_progress`; I can’t update it in read-only mode.

Next steps:
1) Confirm whether run dir standardization from 010/015 is finalized and update `work/tasks/020_artifact_contract_impl.md` (and `work/queue.json`) to unblock.  
2) Re-run with write access so I can implement the artifact contract and tests.
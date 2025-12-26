Using S90_artifacts_versioning (artifact contract) then S95_tests_ci (contract tests).

**Status**
- `work/tasks/020_artifact_contract_impl.md` marks this task blocked until run-dir naming is finalized; per `docs/00_INVARIANTS.md` I can’t implement yet.
- `work/queue.json` shows 010/015 are done, so there’s a mismatch; please confirm the run-dir standard and whether this task is now unblocked.

**Change Plan (when unblocked)**
- `src/utils/artifacts.py`: add helpers to emit `config.yaml`, `meta.json`, `metrics.json`, `model.ckpt`, `predictions.csv` per `docs/04_ARTIFACTS_AND_VERSIONING.md`.
- `src/fp/train.py` and `src/gnn/train.py`: use helpers to write required train artifacts under the finalized run dir.
- `src/fp/predict.py` and `src/gnn/predict.py`: write `predictions.csv` + meta fields (including model_version, dataset_hash, config_hash).
- `tests/contract/test_artifact_contract.py`: add contract checks for required files/keys.

**Implementation**
- No changes made (blocked).

**Tests**
- No tests added/updated (blocked).

**Verification Commands**
- Planned: `pytest -q tests/contract/test_artifact_contract.py`.

**Compatibility Impact**
- None yet. If/when implemented, train/predict artifact filenames/locations will change to match `docs/04_ARTIFACTS_AND_VERSIONING.md` (e.g., new required files; potential renames from `model.pkl`/`metrics_val.json`).

**Queue Status**
- I couldn’t update `work/queue.json` to `blocked` due to read-only sandbox. If you want, set task 020 status to `blocked` with the reason from `work/tasks/020_artifact_contract_impl.md`.

If you want me to proceed, please pick one:
1) Confirm the run-dir standard is fixed and explicitly unblock this task.
2) Update `work/tasks/020_artifact_contract_impl.md` (and optionally `work/queue.json`) to reflect the unblocked status, then I’ll implement.
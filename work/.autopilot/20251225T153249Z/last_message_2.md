Using S90_artifacts_versioning then S95_tests_ci because this task is about artifact outputs and contract tests.

**Status**
- Blocked per `work/tasks/020_artifact_contract_impl.md` (output dir/naming not fixed). Note: `work/queue.json` shows 010/015 done, so please confirm unblocking or update the task file.
- Sandbox is read-only; no file edits or queue updates performed.

**Change Plan (File-level)**
- `src/utils/artifacts.py`: add shared artifact save helpers (config/meta/metrics/model/prediction).
- `scripts/train.py`: save `config.yaml`, `meta.json`, `metrics.json`, `model/` via helpers.
- `scripts/predict.py`: save `predictions.csv` and meta fields (model_version, etc.).
- `tests/contract/`: add artifact contract tests for required files/keys.

**Implementation (Diff)**
- None; blocked.

**Tests Added/Updated**
- None; blocked.

**Verification Commands**
- Not run. Once unblocked: `pytest -q tests/contract`.

**Compatibility Impact**
- None yet. Planned changes will standardize train/predict artifacts.

**Queue Update**
- Could not update `work/queue.json` (read-only). Task should remain `blocked` per task file until unblocked.

Next steps:
1) Confirm task 020 is unblocked and update `work/tasks/020_artifact_contract_impl.md` (or tell me to proceed despite it).
2) Re-run with write access so I can implement and update `work/queue.json`.
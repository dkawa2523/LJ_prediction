Using S90_artifacts_versioning and S95_tests_ci because this task is about artifact contracts and contract tests.

**Status**
- Task 020 is blocked per `work/tasks/020_artifact_contract_impl.md` (waiting on 010/015 output dir standardization). Even though `work/queue.json` shows 010/015 as done, the task file is the source of truth—please update it to unblock or confirm you want me to proceed.

**Change Plan (Provisional, pending unblock)**
- `src/utils/artifacts.py` add common save helpers for config/meta/metrics/artifacts.
- `scripts/train.py` call artifact save for `config.yaml`, `meta.json`, `metrics.json`, `model.ckpt`.
- `scripts/predict.py` save `predictions.csv` and update meta with `model_version`.
- `tests/contract/` add contract tests for required files/keys.

**Implementation (Diff)**
- None — blocked; no changes made.

**Tests Added/Updated**
- None.

**Verify**
- N/A (no changes).

**Compatibility Impact**
- None (no changes).

**Queue Update**
- Not updated; sandbox is read-only and approval policy is never. If you want the task marked, please set task 020 in `work/queue.json` to `blocked` with the reason, or to `in_progress` after unblocking.

Next steps:
1) Update `work/tasks/020_artifact_contract_impl.md` to remove the Blocked section (or confirm you want to proceed anyway), and point me to the finalized run/output directory convention from 010/015.
2) If you want me to update `work/queue.json`, allow write access or update it yourself.
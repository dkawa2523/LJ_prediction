Using skills: S90_artifacts_versioning and S95_tests_ci because this task is about artifact contract implementation and contract tests.

**Change Plan**
- `src/utils/artifacts.py`: add shared save helpers for `config.yaml`, `meta.json`, `metrics.json`, `model/model.ckpt`, `predictions.csv` per `docs/04_ARTIFACTS_AND_VERSIONING.md`
- `scripts/train.py`: call helpers to emit required train artifacts
- `scripts/predict.py`: call helpers to emit `predictions.csv` and meta with model_version
- `tests/contract/test_artifact_contract.py`: validate required files/keys/columns

**Implementation**
- N/A — blocked by `work/tasks/020_artifact_contract_impl.md` (run dir standardization not confirmed) and the sandbox is read-only.

**Tests**
- None added/updated (blocked).

**Verify**
- Not run (blocked).

**Compatibility**
- No changes.

**Queue**
- Unable to update `work/queue.json` in read-only mode; suggested status: `blocked` with reason “Process/run dir standardization pending per `work/tasks/020_artifact_contract_impl.md`.” Note: `work/queue.json` shows 010/015 as done, which conflicts with the block reason.

Next steps:
1) Confirm whether task 020 should be unblocked by updating `work/tasks/020_artifact_contract_impl.md` now that 010/015 are marked done in `work/queue.json`.
2) Rerun with write access so I can implement the changes and update `work/queue.json`.
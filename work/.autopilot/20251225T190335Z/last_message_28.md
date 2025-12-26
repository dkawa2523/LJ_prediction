Using skills: S90_artifacts_versioning and S95_tests_ci (artifact contract + contract test guidance).

**Status**
- `work/tasks/020_artifact_contract_impl.md` marks this task as blocked until output dir standardization (010/015) is confirmed.
- `work/queue.json` shows 010/015 done, but the task file still says blocked; the task file is the source of truth.
- Read-only sandbox prevents me from updating `work/queue.json`.

**Change Plan**
- After unblock: add `src/utils/artifacts.py`, update `scripts/train.py` and `scripts/predict.py`, add contract tests under `tests/contract`.

**Implementation**
- No code changes; blocked.

**Tests**
- None added.

**Verification**
- Not run.

**Compatibility**
- None (no changes).

**Action Needed**
1) If 010/015 is truly complete, update `work/tasks/020_artifact_contract_impl.md` to remove the Blocked section so I can proceed.
2) If still blocked, update `work/queue.json` to set task 020 status to `blocked` with the reason.
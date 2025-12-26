Using S90_artifacts_versioning + S95_tests_ci to follow the artifact contract and testing workflow.

**Status**
- Blocked per `work/tasks/020_artifact_contract_impl.md` (output dir/Process standardization not confirmed); no code changes made.
- Note: `work/queue.json` shows 010/015 as done, but the task file still declares this blocked, so I’m following the task file as the source of truth.

**1) Change Plan (file-level)**
- Blocked; once unblocked I will touch `src/utils/artifacts.py`, `scripts/train.py`, `scripts/predict.py`, and `tests/contract/`.

**2) Implementation (diff)**
- N/A (blocked).

**3) Tests Added/Updated**
- N/A (blocked).

**4) Verification Commands**
- N/A (blocked).

**5) Compatibility Impact (config/CLI/artifact)**
- None (no changes).

**6) work/queue.json status**
- Not updated; sandbox is read-only. Should set task 020 to `blocked` with the reason from `work/tasks/020_artifact_contract_impl.md` until the output dir standardization is confirmed.

Next steps you may want:
1) Confirm that output dir naming is fixed and update `work/tasks/020_artifact_contract_impl.md` to remove the block so I can implement.
2) Re-run with write access if you want me to update `work/queue.json` and make code changes.
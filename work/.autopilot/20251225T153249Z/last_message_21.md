Using S90_artifacts_versioning and S95_tests_ci because this task is about artifact contracts plus contract tests.

Task 020 is blocked per `work/tasks/020_artifact_contract_impl.md` (output dir/naming not fixed). Even though `work/queue.json` shows 010/015 as done, the task file is the source of truth, so I did not make code changes.

**Change Plan (files)**  
- Blocked now; once unblocked I would touch `src/utils/artifacts.py`, `scripts/train.py`, `scripts/predict.py`, and `tests/contract/` to implement the artifact contract.

**Implementation (diff)**  
- No changes; blocked.

**Tests Added/Updated**  
- None; blocked.

**Validation Command**  
- N/A (would run `pytest tests/contract` after unblocking).

**Compatibility Impact (config/CLI/artifact)**  
- None; no changes applied.

**Queue Status Update**  
- I could not update `work/queue.json` to `blocked` because the environment is read‑only. Please update it manually or grant write access.

Next steps if you want me to proceed:
1) Confirm 010/015 are complete and update the Blocked section in `work/tasks/020_artifact_contract_impl.md`, or tell me to ignore the block.
2) If you want me to update `work/queue.json`, provide a writable environment.
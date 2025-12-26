Using S90_artifacts_versioning and S95_tests_ci because this task is about artifact contracts and contract tests.

**Status**
- `work/tasks/020_artifact_contract_impl.md` marks this task as blocked until 010/015 are complete, so I’m pausing implementation.
- `work/queue.json` shows 010/015 as done, which conflicts with the task file; please confirm whether Task 020 is unblocked and what the finalized run/artifact output directory convention is.

**Change Plan**
- No file changes while blocked; once unblocked, expected touch points are `src/utils/artifacts.py`, `scripts/train.py`, `scripts/predict.py`, and `tests/contract/**`.

**Implementation (Diff)**
- No changes (blocked).

**Tests**
- None added/updated (blocked).

**Verification Command**
- Not run (blocked).

**Compatibility Impact**
- None (no changes).

**Queue Update**
- Required: set task 020 status to `blocked` in `work/queue.json`. I can’t update it in read‑only sandbox; please update it or allow write access.

Next steps if you want me to proceed:
1) Confirm Task 020 is unblocked and share the finalized run/artifact output directory convention.
2) Enable write access so I can implement and update `work/queue.json`.
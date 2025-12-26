Using S90_artifacts_versioning and S95_tests_ci because this task is the artifact contract + contract tests.

Task 020 is still blocked per `work/tasks/020_artifact_contract_impl.md` (output dir standardization not fixed). Even though `work/queue.json` shows 010/015 as done, the task file is the single source of truth, so I’m not implementing until that block is cleared.

**1) Change Plan (File-Level)**
- `src/utils/artifacts.py`: add shared artifact save helpers (config/meta/metrics/model/predictions).
- `scripts/train.py`: call artifact save on training end.
- `scripts/predict.py`: call predictions + meta save.
- `tests/contract/test_artifacts_contract.py`: add artifact contract checks per docs/04.

**2) Implementation (Diff)**
- No changes; blocked and sandbox is read-only.

**3) Tests Added/Updated**
- None; blocked.

**4) Verification Commands**
- Not run; blocked.

**5) Compatibility Impact (config/CLI/artifact)**
- None; no changes.

**6) Queue Update**
- Should set task 020 to `blocked` with the reason in `work/tasks/020_artifact_contract_impl.md`. I can’t edit `work/queue.json` in read‑only mode; if you want me to update it, please grant write access or make the change and I’ll proceed.

If the block is actually resolved, please update `work/tasks/020_artifact_contract_impl.md` to remove the Blocked section and tell me to continue.
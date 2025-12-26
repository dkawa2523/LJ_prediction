Using S90_artifacts_versioning → S95_tests_ci because this task is about artifact contract outputs and the required contract tests.

Task 020 is blocked per `work/tasks/020_artifact_contract_impl.md` (output dir/ naming not finalized), so I made no code changes.

**Change Plan**
- `src/utils/artifacts.py`: add shared writers for `config.yaml`, `meta.json`, `metrics.json`, `model.ckpt`, `predictions.csv`
- `scripts/train.py`: call artifact writers for train outputs
- `scripts/predict.py`: call artifact writers for predictions + meta (model_version, etc.)
- `tests/contract/test_artifact_contract.py`: validate required files/keys per `docs/04_ARTIFACTS_AND_VERSIONING.md`

**Implementation**
- Blocked; no diff produced.

**Tests**
- None added or updated (blocked).

**Verify**
- `pytest tests/contract -q` (after unblocked)

**Compatibility**
- None yet; planned changes will standardize train/predict artifacts per `docs/04_ARTIFACTS_AND_VERSIONING.md`.

**Queue**
- `work/queue.json` should mark task `020` as `blocked` with the reason from `work/tasks/020_artifact_contract_impl.md`. I can’t update it because the filesystem is read‑only.

Open question: `work/queue.json` shows 010/015 as done, but the task file still blocks on them. Do you want me to proceed and update `work/tasks/020_artifact_contract_impl.md` to unblock, or should we keep it blocked until output dir conventions are explicitly fixed?
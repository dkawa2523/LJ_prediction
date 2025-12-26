Using S90_artifacts_versioning and S95_tests_ci to align artifact contract implementation and tests.

**Blocker**
- `work/tasks/020_artifact_contract_impl.md` explicitly marks this task as blocked (010/015 not fixed). That file is the source of truth, so I can’t implement until it’s updated.
- I’m in a read‑only sandbox, so I can’t update `work/queue.json` myself.

**Change Plan (pending unblock)**
- `src/utils/artifacts.py`: add shared helpers to write `config.yaml`, `meta.json`, `metrics.json`, `model.ckpt`, and `predictions.csv`.
- `scripts/train.py`: call artifact helpers at train completion; ensure `model/` and `metrics.json` are emitted.
- `scripts/predict.py`: write `predictions.csv` and include `model_version` in `meta.json`.
- `tests/contract/test_artifact_contract.py`: verify required files and required meta/prediction keys per `docs/04_ARTIFACTS_AND_VERSIONING.md`.

**Implementation**
- No changes made (blocked, plus read‑only).

**Tests**
- Not run (no changes).

**Verification Command**
- `pytest -q` (once the contract tests are added).

**Compatibility Impact**
- None (no changes).

**Queue Status Update**
- Please update `work/queue.json` to set task 020 to `blocked` with the task file’s reason, or update `work/tasks/020_artifact_contract_impl.md` to clear the block and tell me to proceed.
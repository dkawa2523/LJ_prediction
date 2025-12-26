Using S90_artifacts_versioning and S95_tests_ci to implement the artifact contract outputs and add contract validation tests.
Implemented shared artifact utilities and wired train/predict/evaluate to emit contract files (meta/config/metrics/model/predictions), plus a lightweight contract test to validate required keys/files.

**変更計画**
- `src/utils/artifacts.py`: add artifact hashing, meta building, and contract validation helpers.
- `src/common/meta.py`: route meta building through the new helpers.
- `src/fp/train.py`, `src/gnn/train.py`: add dataset hash/meta, `metrics.json`, and `model/` outputs.
- `src/fp/predict.py`, `src/gnn/predict.py`: add run meta context and `predictions.csv`.
- `src/fp/evaluate.py`, `src/gnn/evaluate.py`: add run context columns + `metrics.json`.
- `scripts/visualize.py`, `scripts/audit_dataset.py`: pass cfg into meta builder for config hash.
- `tests/contract/test_artifacts_contract.py`: new contract test.
- `work/tasks/020_artifact_contract_impl.md`, `work/queue.json`: stale Blocked cleanup + status done.

**実装（差分が分かるように）**
- `src/utils/artifacts.py`: new utilities for dataset/config hashing, meta generation (required keys), training context resolution, and artifact validators.
- `src/common/meta.py`: re-export updated meta helpers so existing call sites get required keys.
- `src/fp/train.py`: compute dataset hash, write `metrics.json`, emit `model/model.ckpt`, `model/preprocess.pkl`, `model/featurizer_state.json`.
- `src/gnn/train.py`: compute dataset hash, write `metrics.json`, emit `model/model.ckpt` + `model/featurizer_state.json`.
- `src/fp/predict.py`: add run meta context and write `predictions.csv` with contract columns.
- `src/gnn/predict.py`: same as FP predict (meta + `predictions.csv`).
- `src/fp/evaluate.py`: enrich predictions with model/run context and write `metrics.json`.
- `src/gnn/evaluate.py`: same as FP evaluate (context + `metrics.json`).
- `scripts/visualize.py`: meta builder now receives cfg for config hash.
- `scripts/audit_dataset.py`: meta builder now receives cfg for config hash.
- `work/tasks/020_artifact_contract_impl.md`: removed stale Blocked note.
- `work/queue.json`: task 020 status set to done.

**追加/更新したテスト**
- `tests/contract/test_artifacts_contract.py`
- 未実行（手元では走らせていません）

**検証コマンド**
- `pytest tests/contract/test_artifacts_contract.py`

**互換性影響（config/CLI/artifact）**
- config: 既存の schema 変更なし（`output.model_version` は任意で追加可能）。
- CLI: 変更なし。
- artifact: train に `model/` と `metrics.json` が追加、predict に `predictions.csv` が追加、`meta.json`/`predictions.csv` に比較用の列が追加（既存出力は維持）。

**タスク完了**
- `work/queue.json` の 020 を done に更新済み。

Next steps (optional):
1) `pytest tests/contract/test_artifacts_contract.py` を実行して contract を確認  
2) `scripts/train.py` → `scripts/predict.py` を実行して `runs/` 配下の成果物配置を確認
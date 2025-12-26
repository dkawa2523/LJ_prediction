S90_artifacts_versioning と S95_tests_ci を適用: artifact 契約実装と contract test の作業だからです。  
`work/tasks/020_artifact_contract_impl.md` が Blocked 指定なので進められません（`work/queue.json` で 010/015 が done でもタスクファイルが単一の正）。加えてこの環境は read-only のため書き込みできません。

**変更計画**
- `src/utils/artifacts.py` に artifact 保存ヘルパー（`config.yaml`/`meta.json`/`metrics.json`/`model.ckpt`/`predictions.csv`、`config_hash`/`dataset_hash`/`git_sha` 生成）を追加
- `src/fp/train.py` と `src/gnn/train.py` で保存先を contract 仕様に合わせ、`model/` 配下へ `model.ckpt` を保存
- `src/fp/predict.py` と `src/gnn/predict.py` で `predictions.csv` を契約列（`sample_id`,`y_pred` 等）で保存し、`meta.json` に `model_version` 等を追記
- `tests/contract/test_artifacts_contract.py` を新設し、必須ファイル・必須キーの検証を追加

**実装**
- なし（Blocked + read-only）

**追加/更新したテスト**
- なし（Blocked）

**検証コマンド**
- なし（Blocked）

**互換性影響（config/CLI/artifact）**
- なし（未変更）

**work/queue.json**
- 未更新。`work/tasks/020_artifact_contract_impl.md` の Blocked 理由を反映して `work/queue.json` の 020 を `blocked` にする必要がありますが、read-only のため実施不可です。

次の一手（必要なら番号で返してください）
1) `work/tasks/020_artifact_contract_impl.md` の Blocked を解除（または根拠を更新）して作業再開を指示  
2) 書き込み可能な環境（workspace-write）で再実行して変更を適用
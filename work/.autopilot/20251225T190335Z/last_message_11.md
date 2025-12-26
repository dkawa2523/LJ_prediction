S90_artifacts_versioning と S95_tests_ci を使用します（artifact契約実装とcontract testの整合のため）。

**Status**
- `work/tasks/020_artifact_contract_impl.md` で blocked 指定（010/015の完了待ち）が明記されています
- `work/queue.json` では 010/015 が done になっており矛盾があるため、どちらを正とするか確認が必要です
- `approval_policy=never` + read-only のため、編集や `work/queue.json` 更新ができません
- この状態では実装に進めないため変更は行っていません

**Change Plan**
- `src/utils/artifacts.py` に共通ヘルパー（config_hash/dataset_hash算出、`config.yaml`/`meta.json`/`metrics.json`/`model/`/`predictions.csv` 書き出し）を追加
- `scripts/train.py` で backend 実行後に成果物を契約レイアウトへ正規化し、`docs/04` 必須キーを meta に付与
- `scripts/predict.py` で `predictions.csv` を契約列で出力し、meta に model_version 等を付与
- `tests/contract/test_artifacts_contract.py` で必須ファイル/必須キーの検査を追加

**Implementation**
未実施（blocked + read-only のため差分なし）。

**Tests**
未追加/未更新。

**Verification**
未実行。解除後は `pytest tests/contract -q` を想定。

**Compatibility Impact**
現時点の影響なし。実装後は成果物のファイル名/配置が `docs/04_ARTIFACTS_AND_VERSIONING.md` に合わせて変わります。

**Queue Update**
`work/queue.json` の task 020 を `blocked` に更新（理由は `work/tasks/020_artifact_contract_impl.md` に既存）する必要がありますが、read-only のため未反映です。

Next steps:
1) 010/015 が本当に完了しているなら `work/tasks/020_artifact_contract_impl.md` の blocked を解消（もしくは理由更新）してください  
2) 書き込み可能な環境で再実行するか、先に `work/queue.json` を blocked に更新してから実装に入らせてください
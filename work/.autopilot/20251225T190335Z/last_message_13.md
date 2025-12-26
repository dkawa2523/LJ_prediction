S90_artifacts_versioning と S95_tests_ci を使用（artifact 契約と contract test の指針に合わせるため）。

**状況**
- `work/tasks/020_artifact_contract_impl.md` が Blocked 指定のため、単一の正として実装は進められません。
- `work/queue.json` では 010/015 が done ですが、タスク定義が優先のため不整合の解消が必要です。
- 環境が read-only のため、ファイル更新（`work/queue.json` 含む）を実行できません。

**変更計画（ファイル単位）**
- `src/utils/artifacts.py` を新規追加し、config/meta/metrics/predictions 保存と hash 生成を共通化。
- `src/fp/train.py` と `src/gnn/train.py` で `config.yaml`/`meta.json`/`metrics.json`/`model.ckpt` を契約どおりに出力。
- `src/fp/predict.py` と `src/gnn/predict.py` で `predictions.csv` を契約列つきで出力し、`meta.json` に `model_version` など必須キーを付与。
- `tests/contract/test_artifact_contract.py` を追加し、必須ファイル/キーの検査を実装。

**実装（差分）**
- なし（Blocked のため未実施）

**追加/更新したテスト**
- なし（Blocked のため未実施）

**検証コマンド**
- 未実施。実装後は `pytest tests/contract` を想定。

**互換性影響（config/CLI/artifact）**
- 予定: artifact 構成（`metrics.json`/`predictions.csv`/`meta.json` のキーと列）に影響。CLI/Config への影響は最小想定。

**work/queue.json**
- read-only のため更新不可。`id=020` を `blocked` にし、理由に task の Blocked を反映する必要があります。

次に進めてもよいか確認したいです。010/015 が完了済みなら、`work/tasks/020_artifact_contract_impl.md` の Blocked を解除して進めてOKですか？  
進める場合の選択肢:
1) Blocked 解除＆書き込み可能なモードで再実行  
2) Blocked のまま保留
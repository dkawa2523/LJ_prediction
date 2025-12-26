S90_artifacts_versioning と S95_tests_ci を使用しています（artifact契約とcontract test対応のため）。  
`work/tasks/020_artifact_contract_impl.md` に blocked が明記されているため、実装は保留しています。`work/queue.json` では 010/015 が done ですが、タスクファイルが唯一の正とあるので現状はブロック扱いのままにしています。続行して良いか、タスクファイルの更新/解除をご指示ください。

**変更計画（ファイル単位）**
- `src/utils/artifacts.py`: artifact保存の共通関数（`config.yaml`, `meta.json`, `metrics.json`, `model.ckpt`, `predictions.csv`）とハッシュ算出を追加
- `src/common/meta.py`: `meta.json` 必須キーの拡張（または artifacts helper 経由で補完）
- `src/fp/train.py` / `src/gnn/train.py`: train成果物を契約に沿って保存（`model/` 配下と `metrics.json` など）
- `src/fp/predict.py` / `src/gnn/predict.py`: `predictions.csv` と `meta.json`（`model_version` 等）保存
- `tests/contract/test_artifact_contract.py`: 必須ファイル/キー/列の検査
- `scripts/train.py` / `scripts/predict.py`: 必要なら保存先解決の補助
- 注記: S90のAllowed Changesに `src/fp/*`/`src/gnn/*` が含まれていないため、変更許可の範囲をご確認ください

**実装（差分が分かるように）**
- 未実施（blocked）

**追加/更新したテスト**
- 未追加（blocked）

**検証コマンド**
- 未実行（blocked）
- 予定: `pytest tests/contract -q`

**互換性影響（config/CLI/artifact）**
- 変更なし（未実装）
- 実装時は artifact の保存場所/命名が変わるため下流参照に影響が出る可能性あり

**work/queue.json 更新**
- 変更不可（read-only）
- タスク完了ではないため、更新するなら `blocked` と理由（run dir標準化未確定）を記載してください

次の一手はどれにしますか？
1) `work/tasks/020_artifact_contract_impl.md` の Blocked を解除して進める  
2) 010/015 の出力ディレクトリ仕様を共有してから進める  
3) 今回はここで止める
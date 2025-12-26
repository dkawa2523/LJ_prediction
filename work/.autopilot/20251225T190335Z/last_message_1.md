S90_artifacts_versioning と S95_tests_ci を使用（artifact契約とcontract testのワークフローに従うため）。

**Status**
- `work/tasks/020_artifact_contract_impl.md` が blocked 指定で、出力先/命名規約が未確定のため着手不可です。
- `work/queue.json` では 010/015 が done で不整合があるので、出力ディレクトリ規約が確定済みか確認が必要です。
- 環境が read-only + approval_policy=never のため、ファイル編集や `work/queue.json` 更新ができません。

**Outputs**
- 変更計画（ファイル単位）: `src/utils/artifacts.py` に保存共通関数、`src/fp/train.py` と `src/gnn/train.py` に train 成果物保存、`src/fp/predict.py` と `src/gnn/predict.py` に `predictions.csv` 保存、`tests/contract/test_artifacts_contract.py` に必須ファイル/キー検査を追加。
- 実装（差分が分かるように）: なし（blocked + read-only）。
- 追加/更新したテスト: なし。
- 検証コマンド: 未実行。解除後は `pytest tests/contract -q` を想定。
- 互換性影響（config/CLI/artifact）: なし（未実装）。実装時は `docs/04_ARTIFACTS_AND_VERSIONING.md` に沿ったartifact配置変更が発生します。
- `work/queue.json` 更新: 編集不可。タスク020は unblock が確認できるまで `blocked` へ更新が必要です。

次の進め方を教えてください:
1) 出力ディレクトリ/命名規約が確定済みか（確定なら規約内容）
2) 書き込み可能モードで再実行して実装を進めてよいか
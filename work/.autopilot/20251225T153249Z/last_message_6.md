S90_artifacts_versioning → S95_tests_ci を使用（artifact契約実装とcontract testのため）。  
`work/tasks/020_artifact_contract_impl.md` の Blocked 指定に従い停止したため、コード変更はありません。

**変更計画**
- `src/utils/artifacts.py`: artifact保存の共通関数（blocked解除後に実装）
- `scripts/train.py`: `config.yaml`/`meta.json`/`metrics.json`/`model.ckpt` 保存追加
- `scripts/predict.py`: `predictions.csv` 保存と `meta.json` 追記
- `tests/contract/` 配下: contract test 追加

**実装（差分）**
- 変更なし（blocked）

**追加/更新したテスト**
- なし

**検証コマンド**
- なし

**互換性影響（config/CLI/artifact）**
- なし（変更未実施）

**work/queue.json**
- 読み取り専用のため更新不可。`work/queue.json` の task `id: "020"` を `status: "blocked"` に更新し、理由は `work/tasks/020_artifact_contract_impl.md` の Blocked を参照する形で記録してください。

`work/queue.json` では 010/015 が done なので、ブロック解除済みとして進めて良いか確認させてください。進める場合は `work/tasks/020_artifact_contract_impl.md` の Blocked を更新/削除のうえ再指示ください。
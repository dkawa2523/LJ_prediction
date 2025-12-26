S90_artifacts_versioning → S95_tests_ci を使用（artifact契約とcontract testが対象のため）。タスクの単一正は `work/tasks/020_artifact_contract_impl.md` で、現在は blocked 記載のため着手しませんでした。加えて、この環境は read-only のため `work/queue.json` を更新できません。

**変更計画**
- blocked: `work/tasks/020_artifact_contract_impl.md` の unblock 後に `src/utils/artifacts.py`, `scripts/train.py`, `scripts/predict.py`, `tests/contract/**` を実装予定

**実装（差分）**
- なし（blocked）

**追加/更新したテスト**
- なし（blocked）

**検証コマンド**
- 未実行（blocked）。想定: `pytest tests/contract -q`

**互換性影響（config/CLI/artifact）**
- なし（未変更）

**work/queue.json 更新**
- `work/queue.json` を task 020: `blocked` + reason に更新すべきだが、read-only のため実施不可

次のステップ案:
1) `work/tasks/020_artifact_contract_impl.md` の blocked を解除（010/015完了済みなら反映）  
2) 書き込み可能な環境で `work/queue.json` を更新してから着手指示をください
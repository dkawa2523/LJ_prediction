Using skills: S90_artifacts_versioning and S95_tests_ci — this task is exactly artifacts + contract tests.

Blocked: Task 020 is explicitly blocked until 010/015 finalize the Process/run output dir + config merge. Without that, artifact paths/naming can’t be implemented safely.

**変更計画（ファイル単位）**
- Pending after unblock: `src/utils/artifacts.py` (new), `scripts/train.py`, `scripts/predict.py`, `tests/contract/` (contract checks)

**実装（差分）**
- 変更なし（blocked、read-onlyのため未編集）

**追加/更新したテスト**
- なし

**検証コマンド**
- なし

**互換性影響（config/CLI/artifact）**
- なし（未実装）

**work/queue.json**
- 未更新。`work/tasks/020_artifact_contract_impl.md` の Blocked 理由（010/015未完了）で status を `blocked` に更新してください。

次の一手が必要なら、010/015完了後に出力ディレクトリ規約を教えてください。そこで実装に進みます。
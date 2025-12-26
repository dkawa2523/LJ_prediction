Using skills S90 (Artifacts & Versioning) and S95 (Tests & CI) to align with artifact contract and contract-test expectations.  
Task 020 is blocked per `work/tasks/020_artifact_contract_impl.md` because Process/run dir standardization (010/015) is unfinished; in read-only mode I cannot update `work/queue.json`.

**変更計画（ファイル単位）**
- Blocked until output directory/naming is fixed; no file changes planned yet.

**実装（差分が分かるように）**
- 変更なし（blocked）。

**追加/更新したテスト**
- 変更なし（blocked）。

**検証コマンド**
- なし（unblocked後に `pytest tests/contract` を想定）。

**互換性影響（config/CLI/artifact）**
- 影響なし（変更なし）。将来的に artifact 出力先と必須ファイル/キーが統一される予定。

**work/queue.json 更新**
- 変更不可（read-only）。`work/queue.json` の status を `blocked` にし、理由を「010/015 未完了で出力先規約未確定」と追記してください。

次のアクション案:
1) 010/015 を完了して Process/run dir 規約を確定
2) 本タスクに戻って artifact 保存共通関数と contract test を実装
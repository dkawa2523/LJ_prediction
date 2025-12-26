# ===== SESSION CONTEXT =====
# Codex Session Context（毎回貼る短文）

- このrepoは「化学構造（SDF/CSV）→ 物性予測」を行う CLI/Hydra 基盤。
- 不変条件は docs/00_INVARIANTS.md が唯一の正。
- 今回の作業範囲・要件は work/tasks/<TASK>.md が唯一の正。
- 学習と推論の特徴量/前処理は必ず一致させる（skew禁止）。
- 成果物（artifact）は docs/04_ARTIFACTS_AND_VERSIONING.md を守る。
- 破壊的変更が必要なら work/rfc → docs/adr を先に作る（勝手に壊さない）。
- すべての処理は Process（処理単位）としてHydraで管理し、単独実行できる形を守る。
- 将来ClearML Task化を想定し、各Processが artifact を明確に出す設計にする（実装は今しない）。

# ===== AUTOPILOT MODE (non-interactive) =====
あなたは自動実行ループの中で動いています。**停止しないための規約**として次を厳守してください：

【禁止】
- ユーザーへの確認・質問・承認依頼（"確認してください" / "進めてよいですか" / "Approve" 等）は一切しない。
- "今回は未実装" / "差分なし" のまま終了しない（必ず前進する）。

【single source of truth】
- タスクの status の唯一の真実は work/queue.json。task.md の 'Blocked' 記述は参考情報。
  - queue が todo/in_progress なら実装を進める（task.md 側の Blocked を理由に止めない）。
  - queue が blocked なら blocked-handling を行う（reason/unblock_condition/next_action を整備し、解除子タスクを起票）。

【stale Blocked の扱い】
- task.md に '## Blocked' が残っていても queue が todo/in_progress の場合、それは **stale**。
  - task.md を更新し、Blocked を Notes/History に移す or 解除した上で実装を進める。

【書き込み権限】
- **書き込みは可能**な前提。work/queue.json 更新を含む必要な変更は実施する。
  - もし実際にツール/環境エラーで書き込めない場合：
    1) エラーメッセージを短く引用
    2) 最小の対処（sandbox/approval/flags 見直し、権限確認）を提示
    3) タスクを blocked にして解除子タスクを起票（子に unblocks を付与）
    ※それでも『確認してください』は書かない。

【必ずどれかを実施（Decision Protocol）】
A) 実装を進める → テスト/検証 → work/queue.json を done に更新
B) 進められない → work/queue.json を blocked に更新し、解除子タスク（unblocks付き）を作成
C) 依存/状態がズレている → work/queue.json と task.md を修正して整合させた上で A へ

# ===== SELECTED TASK =====
- id: 090
- title: CI/テスト整備
- priority: P2
- status: in_progress
- path: work/tasks/090_ci_tests.md

# ===== CONTRACTS TO FOLLOW (open & read) =====
- docs/06_TEST_POLICY.md
- docs/00_INVARIANTS.md

# ===== SKILLS TO FOLLOW (open & follow) =====
- agentskills/skills/S95_tests_ci.md


# ===== TASK FILE (single source of truth for *intent*) =====
# Task 090 (P2): CI/テスト整備

## 目的
- 何十回の改修でも壊れないように、自動テストとlintを入れる

## Plan
1) `pytest` の導入と最小テスト追加
2) smoke config（quick）で train/predict が通る integration test
3) GitHub Actions（任意）で test を回す

## Acceptance Criteria
- [ ] `pytest -q` が通る
- [ ] integration smoke が通る

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked と理由）


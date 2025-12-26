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
- id: 075
- title: 結果集計Process（leaderboard/比較レポート生成）
- priority: P0
- status: in_progress
- path: work/tasks/075_leaderboard_process.md

# ===== CONTRACTS TO FOLLOW (open & read) =====
- docs/00_INVARIANTS.md
- docs/04_ARTIFACTS_AND_VERSIONING.md
- docs/10_PROCESS_CATALOG.md

# ===== SKILLS TO FOLLOW (open & follow) =====
- agentskills/skills/S70_evaluator_metrics.md
- agentskills/skills/S10_hydra_config.md
- agentskills/skills/S95_tests_ci.md


# ===== TASK FILE (single source of truth for *intent*) =====
# Task 075 (P0): 結果集計Process（leaderboard/比較レポート生成）

## 目的（Why）
- モデル/特徴量/タスクの比較を「人力でrunフォルダを見に行く」状態から脱却する。
- multirunで大量実験したとき、**どれが良いか即分かる**ようにする。
- 将来ClearMLに移行しても、同じ “比較レポート” の発想で繋げられる。

## 背景（Context）
- 比較の正しさには、(a)同一split、(b)同一指標定義、(c)metaが揃っていることが必要。
- `metrics.json` / `meta.json` / `config.yaml` を契約化しているので、集計は自動化できる。

## スコープ（Scope）
### In scope
- **新Process `leaderboard`（または `aggregate_results`）** を追加（1 script = 1 process）
  - `scripts/leaderboard.py`（例）
- 入力：`runs/` 配下（または指定root）をスキャンし、以下が揃うrunを集計
  - `meta.json`
  - `metrics.json`
  - （任意）`config.yaml`
- 出力（artifact）
  - `leaderboard.csv`
  - `leaderboard.md`（上位N件、条件付き）
  - `plots/`（任意：metric vs timeなど）

### Out of scope（今回はやらない）
- ClearML SDKでのアップロード（設計準拠だけ）
- 完全なWeb UI

## 影響（Contract Impact）
- Process追加なので `docs/10_PROCESS_CATALOG.md` を更新
- 既存runを壊さない（read-onlyで集計する）

## 実装計画（Plan）
1) `docs/10_PROCESS_CATALOG.md` に leaderboard を追記（入力/出力）
2) `scripts/leaderboard.py` を追加
   - `leaderboard.root_dir`（デフォルト runs/）
   - `leaderboard.metric_key`（例 r2）
   - `leaderboard.sort_order`（desc）
   - `leaderboard.filters`（task/model/features、期間など）
3) `configs/process/leaderboard.yaml`（入口）と `configs/leaderboard/default.yaml` を追加
4) テスト追加
   - `tests/` でテンポラリrunディレクトリを作り、ダミーの meta/metrics を置いて集計できること

## 受け入れ条件（Acceptance Criteria）
- [ ] `python scripts/leaderboard.py ...` が単独で実行できる（Hydra管理）
- [ ] `leaderboard.csv` が生成される
- [ ] 필터（task/model/featuresの少なくとも1つ）が動く
- [ ] `docs/10_PROCESS_CATALOG.md` が更新されている
- [ ] pytestに最低1つ追加テスト

## 検証手順（How to Verify）
- 例：
  - `python scripts/leaderboard.py leaderboard.root_dir=runs leaderboard.metric_key=r2`
  - `pytest -q`

## メモ
- これができると、HPOやアンサンブルの比較が一気に回しやすくなる

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked と理由）


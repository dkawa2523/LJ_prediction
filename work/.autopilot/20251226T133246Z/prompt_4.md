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
- id: 035
- title: split戦略の追加（random/scaffold/group、seed固定、漏洩防止）
- priority: P0
- status: in_progress
- path: work/tasks/035_split_strategies.md

# ===== CONTRACTS TO FOLLOW (open & read) =====
- docs/00_INVARIANTS.md
- docs/02_DATA_CONTRACTS.md
- docs/03_CONFIG_CONVENTIONS.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

# ===== SKILLS TO FOLLOW (open & follow) =====
- agentskills/skills/S20_dataset_loader.md
- agentskills/skills/S10_hydra_config.md
- agentskills/skills/S95_tests_ci.md


# ===== TASK FILE (single source of truth for *intent*) =====
# Task 035 (P0): split戦略の追加（random/scaffold/group、seed固定、漏洩防止）

## 目的（Why）
- R²の比較が意味を持つように、splitを戦略的に選べるようにする。
- splitが変わるとスコアが大きく変動するため、**同一splitで比較**できる仕組みが必須。

## 背景（Context）
- random splitだけだと簡単に“高スコア”が出る（漏洩/類似分子が跨る）。
- 分子タスクでは scaffold split が標準的な比較になりやすい。
- 将来の比較（multirun, ClearML）でも split の追跡が必須（dataset_hash + split.json）。

## スコープ（Scope）
### In scope
- `build_dataset`（またはsplit生成部分）に split戦略を追加
  - `random`（seed固定）
  - `scaffold`（Murcko scaffoldでグルーピングし、グループ単位で割当）
  - `group`（指定列：例 cas / inchikey / formula 等、同一グループは同一split）
- split成果物の保存（契約）
  - `split.json` などとして artifact に保存（docs/04準拠）
- splitのバリデーション
  - split間重複がない
  - scaffold split では同一scaffoldが跨らない（少なくとも検査できる）

### Out of scope（今回はやらない）
- Nested CV、完全なk-fold運用（これはP1でもOK）
  - ただし “将来追加できる拡張ポイント” は設計しておく

## 影響（Contract Impact）
- dataset artifactの内容が増える（split保存）
- 既存のsplit生成がある場合、互換性を壊さずに拡張する（旧キーを残す/変換する）

## 実装計画（Plan）
1) `src/data/splitting.py`（案）を追加し、splitメソッドを関数として実装
   - `make_split_random(...)`
   - `make_split_scaffold(...)`（RDKit: Murcko scaffold）
   - `make_split_group(...)`
2) `configs/dataset/*.yaml` に以下のキーを追加（例）
   - `dataset.split.method: random|scaffold|group`
   - `dataset.split.seed: 0`
   - `dataset.split.fractions: [0.8, 0.1, 0.1]`
   - `dataset.split.group_key: "cas"`（groupの場合）
3) `build_dataset` Process に組込み、splitを生成して artifact に保存
4) テスト追加（重要）
   - 小さな分子集合で scaffold split を作り、同一scaffoldが跨らないことを検査
   - seed固定で split が再現されることを検査

## 受け入れ条件（Acceptance Criteria）
- [ ] random/scaffold/group を config で切替できる
- [ ] seed固定で split が再現される
- [ ] split成果物が artifact として保存される（docs/04準拠）
- [ ] scaffold split の漏洩検査ができる（テストで担保）
- [ ] pytestに最低1つ追加テスト

## 検証手順（How to Verify）
- 例：
  - `python scripts/build_dataset.py dataset.split.method=scaffold dataset.split.seed=0`
  - `python scripts/build_dataset.py dataset.split.method=group dataset.split.group_key=cas`
  - `pytest -q`

## メモ
- このタスク完了後、train/evaluate は “dataset artifactのsplitを必ず使う” 方針に寄せる（比較可能性）

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked と理由）


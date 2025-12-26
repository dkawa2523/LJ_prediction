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
- id: 025
- title: データ監査＆漏洩/重複検知（比較の信頼性確保）
- priority: P0
- status: in_progress
- path: work/tasks/025_dataset_audit.md

# ===== CONTRACTS TO FOLLOW (open & read) =====
- docs/00_INVARIANTS.md
- docs/02_DATA_CONTRACTS.md
- docs/04_ARTIFACTS_AND_VERSIONING.md
- docs/10_PROCESS_CATALOG.md

# ===== SKILLS TO FOLLOW (open & follow) =====
- agentskills/skills/S20_dataset_loader.md
- agentskills/skills/S70_evaluator_metrics.md
- agentskills/skills/S95_tests_ci.md


# ===== TASK FILE (single source of truth for *intent*) =====
# Task 025 (P0): データ監査＆漏洩/重複検知（比較の信頼性確保）

## 目的（Why）
- R²などの比較が「本当に意味のある比較」になるように、データの地雷（重複/漏洩/単位混在/ラベル異常/無効構造）を可視化して潰す。
- “精度改善の前に、評価が正しいことを保証する” がP0。

## 背景（Context）
- 1万件規模でR²が低い原因は、モデル以前に「データ問題」であることが多い。
- split漏洩（同一分子がtrainとtestに混入等）があると、比較が壊れる。
- 将来ClearMLで Process を Task 化するため、監査も独立Process化する。

## スコープ（Scope）
### In scope
- **新Process `audit_dataset`** を追加（1 script = 1 process）
  - 例: `scripts/audit_dataset.py`
- auditの結果を **artifactとして保存**
  - `audit_report.json`（機械可読）
  - `audit_report.md`（人間可読）
  - `plots/`（分布/外れ値など）
- 監査項目（最低限）
  - 構造の妥当性（RDKitでMol生成できない行）
  - **重複検知**（canonical SMILES / InChIKey）
  - ターゲット欠損/異常（NaN、極端値、分布）
  - splitがある場合：**split間重複/漏洩検知**
  - 主要統計（件数、元素種分布、分子量分布、ターゲット統計）

### Out of scope（今回はやらない）
- 自動修復（塩除去や中性化で直す等）を全面的にやる
  - ただし「修復候補の検出」はOK
- ClearML SDK の導入（設計はdocs/12に従うが実装はしない）

## 影響（Contract Impact）
- Invariantsに抵触しない（Process追加は推奨）
- artifact契約に追加（auditの成果物）→ **docs/10_PROCESS_CATALOG.md を更新する**

## 実装計画（Plan）
1) `docs/10_PROCESS_CATALOG.md` に `audit_dataset` を追記（入力/出力を明記）
2) `scripts/audit_dataset.py` を追加
   - 入力：
     - raw CSV/SDF 直読み OR `build_dataset` artifact（どちらか、configで選択）
   - 出力：
     - `audit/audit_report.json`
     - `audit/audit_report.md`
     - `plots/*.png`
3) `src/data/audit.py`（案）に監査ロジックを分離
   - canonical SMILES、InChIKeyの計算
   - 重複クラスタの抽出
   - split漏洩チェック（同一キーが複数splitに存在）
4) `configs/process/audit_dataset.yaml`（入口）と `configs/audit/default.yaml`（詳細）を追加
5) テスト追加
   - 小さなCSV（数件）で audit が走り、reportに必須キーが入ること

## 受け入れ条件（Acceptance Criteria）
- [ ] `python scripts/audit_dataset.py ...` が単独で実行できる（Hydra管理）
- [ ] `audit_report.json` に最低限のキーが入る（例：invalid_mol_count, duplicate_groups, target_stats）
- [ ] splitがある場合、split漏洩が検知できる（0件でもOK）
- [ ] `docs/10_PROCESS_CATALOG.md` が更新されている
- [ ] pytestで最低1つテストが追加されている

## 検証手順（How to Verify）
- 例：
  - `python scripts/audit_dataset.py dataset=quick audit=default`
  - `pytest -q`

## メモ
- 監査結果を見て次タスクを起票する（例：重複排除、単位統一、外れ値処理など）

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked と理由）


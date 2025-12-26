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

# ===== SELECTED TASK =====
- id: 020
- title: Artifact 契約の実装
- priority: P0
- status: in_progress
- path: work/tasks/020_artifact_contract_impl.md

# ===== CONTRACTS TO FOLLOW (open & read) =====
- docs/00_INVARIANTS.md
- docs/04_ARTIFACTS_AND_VERSIONING.md
- docs/10_PROCESS_CATALOG.md

# ===== SKILLS TO FOLLOW (open & follow) =====
- agentskills/skills/S90_artifacts_versioning.md
- agentskills/skills/S95_tests_ci.md


# ===== TASK FILE (single source of truth) =====
# Task 020 (P0): Artifact 契約をコードに実装

## 目的
- 「どのデータ/設定/コードで学習したモデルか」を必ず追跡できるようにする

## Plan
1) `src/utils/artifacts.py`（案）を作り、artifact 保存の共通関数を実装
2) 学習終了時に `config.yaml`, `meta.json`, `metrics.json`, `model.ckpt` を保存
3) 推論時に `predictions.csv` を保存し、meta に model_version 等を書き込む
4) contract test を追加（必須ファイル/キーの検査）

## Acceptance Criteria
- [ ] docs/04 の必須成果物が生成される
- [ ] contract test が通る

## Blocked
- reason: Process/run dir の標準化（010/015）が未完了で、artifact 出力先と命名規約が確定していない
- unblock_condition: 010/015 が完了し、Processごとの出力ディレクトリとconfig合成が固定される
- next_action: 010/015 を先に完了し、artifact 保存先の確定後に本タスクへ戻る

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked と理由）


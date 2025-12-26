# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: 050

状況:
- 直前の出力が「未実装/差分なし/確認お願いします/blockedだから保留」等で停止している可能性があります。
- ここでは “前に進める” ことが最優先です。

指示（必ずどれか）:
A) 実装を進める（最小でもよい）→ テスト/検証 → work/queue.json を done に更新
B) 本当に進められないなら、work/queue.json を blocked に更新し、解除子タスクを起票（子に unblocks を付与）
C) task.md が stale (Blocked が残っている等) なら task.md/queue.json を修正して整合させてから A へ

禁止:
- 「確認をお願いします」「進めてよいですか」等の質問
- “差分なし” で終了

参考: 直前の最終メッセージ（抜粋）
```
Using S20_dataset_loader to keep collection output column mapping and missing-field logging aligned with the dataset contracts.

Implemented the collect_data process with a plugin registry, dummy data source, mapping formatter, and cache, plus configs and a unit test; secrets are env-only and raw outputs land in the run dir with optional export.

**Change Plan**
- `src/data_collection/*`: new package for client/formatter/cache/registry/runner layers.
- `scripts/collect_data.py`: new CLI entrypoint for the collect_data process.
- `configs/collect_data.yaml`, `configs/process/collect_data.yaml`, `configs/data_collection/default.yaml`, `configs/data_source/dummy.yaml`: config groups to switch `data_source` via config.
- `src/utils/validate_config.py`: validate collect_data config shape.
- `tests/test_data_collection.py`: unit test for dummy collection output.
- `work/queue.json`: task 050 marked done.

**Implementation Details**
- `src/data_collection/runner.py`: orchestration, cache usage, raw CSV/SDF output, optional export, meta.json with data hash.
- `src/data_collection/formatters/mapping.py`: column_map formatting with missing-field logging and sample_id generation.
- `src/data_collection/utils.py`: env-only API key enforcement (no secrets in code).
- `src/data_collection/registry.py`: lazy-load dummy source to avoid circular imports.

**Tests**
- `pytest -k data_collection` (pass)

**Validation Command**
- `pytest -k data_collection`

**Compatibility Impact**
- Config: new `collect_data` config groups; existing configs unchanged.
- CLI: new `scripts/collect_data.py`.
- Artifacts: new `runs/collect_data/<exp>/` with `config.yaml`, `meta.json`, and `raw/` outputs; export to `data/raw` only when enabled.

Next steps (optional):
1) Run `python scripts/collect_data.py --config configs/collect_data.yaml` to generate stub outputs.
2) Add a real data source (e.g., PubChem) by implementing a new client/formatter and a `configs/data_source/<name>.yaml`.
```

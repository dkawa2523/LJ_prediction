# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: 075

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
Added a new leaderboard aggregation process that scans run artifacts, extracts a target metric across split formats, filters by meta fields, and writes `leaderboard.csv`/`leaderboard.md` with proper `config.yaml`/`meta.json` in a dedicated run dir. This is wired as a new Process with configs, validation, docs update, and a focused pytest.

Using skills: S70_evaluator_metrics (metrics aggregation patterns), S10_hydra_config (process config wiring), S95_tests_ci (pytest coverage).

**Change Plan**
- `src/common/leaderboard.py` add aggregation logic (scan runs, metric extraction, filtering, CSV/MD output, meta).
- `scripts/leaderboard.py` add CLI entrypoint.
- `configs/leaderboard.yaml`, `configs/leaderboard/default.yaml`, `configs/process/leaderboard.yaml` add process/config defaults.
- `src/utils/validate_config.py` validate leaderboard configs.
- `docs/10_PROCESS_CATALOG.md` add leaderboard row.
- `tests/test_leaderboard.py` add coverage for filter + by_split extraction.
- `work/queue.json` mark task 075 done.

**Implementation (diff)**
- `src/common/leaderboard.py`
```diff
+def run(cfg: Dict[str, Any]) -> Path:
+    validate_config(cfg)
+    ...
+    rows, upstream = _collect_rows(...)
+    sorted_rows = _sort_rows(...)
+    _write_csv(.../leaderboard.csv)
+    _build_markdown(.../leaderboard.md)
+    save_meta(...)
+
+def _extract_metric(...):
+    # supports val/test and by_split formats
+    ...
+
+def _matches_filters(...):
+    # task/model/features/tags/period filters
+    ...
```

- `scripts/leaderboard.py`
```diff
+def main() -> None:
+    ap.add_argument("--config", required=True, ...)
+    cfg = load_config(args.config)
+    run(cfg)
```

- `configs/leaderboard.yaml`
```yaml
defaults:
  - process: leaderboard
  - leaderboard: default
  - hydra: default
  - _self_
experiment:
  name: "leaderboard"
output:
  exp_name: "leaderboard"
```

- `configs/leaderboard/default.yaml`
```yaml
leaderboard:
  root_dir: "runs"
  metric_key: "r2"
  metric_split: "val"
  sort_order: "desc"
  top_n: 20
  filters:
    task_name: null
    model_name: null
    featureset_name: null
    process_name: null
    dataset_hash: null
    run_id: null
    git_sha: null
    tags: []
    created_after: null
    created_before: null
```

- `src/utils/validate_config.py`
```diff
+    if process_name == "leaderboard":
+        _require_keys(cfg, ["leaderboard", "output"], f"{process_name} config")
+        return
```

- `docs/10_PROCESS_CATALOG.md`
```diff
+| leaderboard | scripts/leaderboard.py | runs/ (meta.json + metrics.json) | leaderboard.csv, leaderboard.md | Reporting Task |
```

**Tests Added/Updated**
- `tests/test_leaderboard.py` covers filtering by `model_name` and `by_split` metric extraction.

**Verification Commands**
- `pytest -q tests/test_leaderboard.py` (ran)
- `python scripts/leaderboard.py --config configs/leaderboard.yaml` (not run)

**Compatibility Impact**
- Config: new group `leaderboard` + entry `configs/leaderboard.yaml`.
- CLI: new `scripts/leaderboard.py --config ...`.
- Artifacts: outputs `leaderboard.csv`, `leaderboard.md`, `config.yaml`, `meta.json`, `leaderboard.log` under `runs/leaderboard/<exp_name>`.

**Queue**
- Marked task 075 as done in `work/queue.json`.

Next steps:
1) Run `python scripts/leaderboard.py --config configs/leaderboard.yaml` on real `runs/` and adjust `leaderboard.filters` as needed.
2) (Optional) add plots in `runs/leaderboard/<exp_name>/plots/` if you want visual comparisons.
```

# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: 025

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
Using S20_dataset_loader, S70_evaluator_metrics, and S95_tests_ci to align audit data handling, artifact tracking, and tests.  
Added dataset hash tracking to audit outputs/meta, documented audit artifacts, and tightened the audit test so audit artifacts are tied to the audited input and the contract is explicit.

**Change Plan**
- `src/data/audit.py`: compute dataset hash and include it in JSON/MD reports.
- `scripts/audit_dataset.py`: save meta after audit using the report’s dataset hash.
- `tests/test_audit_dataset.py`: assert dataset hash presence/shape.
- `docs/04_ARTIFACTS_AND_VERSIONING.md`: document audit artifacts.
- `work/queue.json`: set task 025 to done.

**Implementation**
- `src/data/audit.py`
```diff
-from src.common.splitters import load_split_indices
+from src.common.splitters import load_split_indices
+from src.utils.artifacts import compute_dataset_hash
...
     if sdf_dir is not None and not sdf_dir.exists():
         raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")
+    dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)
...
-        "dataset_csv": str(dataset_csv),
+        "dataset_csv": str(dataset_csv),
+        "dataset_hash": dataset_hash,
...
-    lines = ["# Dataset Audit Report", "", f"- Total rows: {report.get('total_rows')}"]
+    lines = ["# Dataset Audit Report", "", f"- Total rows: {report.get('total_rows')}"]
+    if report.get("dataset_hash"):
+        lines.append(f"- Dataset hash: {report.get('dataset_hash')}")
```
- `scripts/audit_dataset.py`
```diff
-    save_meta(run_dir, build_meta(process_name=..., cfg=cfg))
-
-    report, report_md, plot_data = audit_dataset(cfg)
+    report, report_md, plot_data = audit_dataset(cfg)
+    meta = build_meta(process_name=..., cfg=cfg, dataset_hash=report.get("dataset_hash"))
+    save_meta(run_dir, meta)
```
- `tests/test_audit_dataset.py`
```diff
     assert report["invalid_mol_count"] == 1
+    assert isinstance(report.get("dataset_hash"), str)
+    assert len(report["dataset_hash"]) == 64
```
- `docs/04_ARTIFACTS_AND_VERSIONING.md`
```diff
+### 3.7 audit_dataset
+- `audit/`
+  - `audit_report.json`
+  - `audit_report.md`
+- `plots/`（分布/外れ値など）
```

**Tests**
- Updated: `tests/test_audit_dataset.py`
- Not run (not requested).

**Verification Commands**
- `python scripts/audit_dataset.py --config configs/audit_dataset_fixture.yaml`
- `pytest -q tests/test_audit_dataset.py`

**Compatibility Impact**
- Artifacts: `audit_report.json` and `audit_report.md` now include `dataset_hash` (additive).
- Meta: audit `meta.json` now uses the audited dataset hash.
- No config/CLI changes.

**Queue Update**
- `work/queue.json`: task `025` set to `done`.

**Next Steps**
1) Run the audit command above to generate a fresh report.
2) Run the targeted pytest command to validate the new assertion.
```

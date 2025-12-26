# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: 090

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
Using skill S95_tests_ci to add pytest smoke coverage for task 090.

Added a CLI-level FP train/predict smoke test using fixture configs and artifact validation, and marked the task done in `work/queue.json`.

**Change Plan**
- `tests/integration/test_smoke_fp_train_predict.py`: add FP train/predict CLI smoke test with fixture config + artifact validation.
- `work/queue.json`: set task 090 status to done after verification.

**Implementation**
```diff
+++ tests/integration/test_smoke_fp_train_predict.py
+def test_smoke_fp_train_predict(tmp_path: Path) -> None:
+    pytest.importorskip("rdkit")
+    pytest.importorskip("sklearn")
+    ...
+    subprocess.run([sys.executable, "scripts/train.py", "--config", ...], check=True)
+    validate_train_artifacts(...)
+    subprocess.run([sys.executable, "scripts/predict.py", "--config", ..., "--query", "64-17-5"], check=True)
+    validate_predict_artifacts(...)
```

**Tests Added/Updated**
- `tests/integration/test_smoke_fp_train_predict.py`

**Verification**
- `pytest -q` (16 passed)

**Compatibility Impact**
- config/CLI/artifact: none; test-only changes.

**Queue**
- `work/queue.json`: task 090 marked done.

Next step (optional):
1) Add a GitHub Actions workflow to run `pytest -q` on push/PR.
```

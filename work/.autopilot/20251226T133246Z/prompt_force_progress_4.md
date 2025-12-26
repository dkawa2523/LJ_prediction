# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: 035

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
Using S20_dataset_loader, S10_hydra_config, S95_tests_ci because this adds split logic, updates dataset configs, and introduces new tests.
Implemented configurable random/scaffold/group splits with validation and `split.json` artifact output, plus config updates and reproducibility/leakage tests.
- **変更計画** `src/common/splitters.py` `scripts/build_dataset.py` `configs/dataset/default.yaml` `configs/dataset/quick.yaml` `configs/dataset/fixture.yaml` `tests/test_splitters.py` `work/queue.json`
- **実装** `src/common/splitters.py` `scripts/build_dataset.py` の主要差分
```diff
-    idx = df.index.to_numpy()
+    idx = df.index.to_numpy().copy()
+def group_split(
+    df: pd.DataFrame,
+    group_col: str,
+    ratios: Sequence[float] = (0.8, 0.1, 0.1),
+    seed: int = 42,
+) -> Dict[str, List[int]]:
+def validate_split_indices(indices: Dict[str, List[int]]) -> None:
+def save_split_json(indices: Dict[str, List[int]], out_path: str | Path, metadata: Optional[Dict[str, object]] = None) -> None:
```
```diff
-    ratios = split_cfg.get("ratios", [0.8, 0.1, 0.1])
+    ratios = split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1]))
+    elif split_method == "group":
+        group_key = split_cfg.get("group_key")
+        if not group_key:
+            raise ValueError("split.method=group requires split.group_key")
+        group_col = cols.get(str(group_key), str(group_key))
+        if group_col not in df.columns:
+            raise ValueError(f"split.group_key '{group_key}' resolved to '{group_col}' not in dataset columns")
+        indices = group_split(df, group_col=group_col, ratios=ratios, seed=split_seed)
+        group_map = build_group_map(df, group_col=group_col)
+        validate_group_leakage(indices, group_map, label=f"group:{group_col}")
+    save_split_json(indices, out_indices_dir / "split.json", metadata=split_meta)
```
- **追加/更新したテスト** `tests/test_splitters.py`（`test_random_split_reproducible` `test_group_split_keeps_groups` `test_scaffold_split_no_leakage`）
- **検証コマンド** `pytest -q tests/test_splitters.py`（pass）
- **互換性影響** `split.fractions` を追加し `split.ratios` は後方互換で保持、`split.group_key` は group 方法で必須、`split.json` が `indices` に追加され dataset_hash が変わる、scaffold/group は空split時にエラー
- **Queue更新** `work/queue.json` の task 035 を done に更新

Next steps:
1) `pytest -q` で全体テストを確認
2) `configs/dataset/default.yaml` の `split.method` を `group` に切り替えて `python scripts/build_dataset.py --config configs/dataset.yaml` を実行し split.json を確認
```

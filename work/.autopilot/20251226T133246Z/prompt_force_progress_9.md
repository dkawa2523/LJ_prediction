# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: 070

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
Using skill S40_featurizer, I added a stub pretrained-embedding featurizer plus a tabular pipeline selector so FP train/eval/predict can switch featuresets without train/infer skew. The stub uses deterministic SMILES hashing and is saved/loaded via existing feature pipeline artifacts, with AD falling back to default Morgan settings when embeddings are used.

**Plan**
- `src/fp/featurizer_embedding.py`: add deterministic stub embedding featurizer.
- `src/common/feature_pipeline.py`: add embedding pipeline plus tabular pipeline resolver/loader.
- `src/fp/train.py`: select tabular pipeline and keep AD defaults when embeddings are used.
- `src/fp/evaluate.py`, `src/fp/predict.py`: load the tabular pipeline from artifacts/config.
- `src/fp/feature_utils.py`: add embedding feature builder and cache path.
- `configs/features/pretrained_embedding_stub.yaml`: expose featureset selection.
- `tests/test_feature_pipeline.py`: add embedding roundtrip test.
- `src/utils/artifacts.py`: resolve featureset name for embeddings.

**Implementation**
- `src/fp/featurizer_embedding.py`
```diff
+@dataclass
+class PretrainedEmbeddingConfig:
+    name: str = "pretrained_embedding"
+    backend: str = "stub"
+    embedding_dim: int = 256
+    seed: int = 0
+    normalize: bool = True
+
+def featurize_mol(mol, cfg: PretrainedEmbeddingConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
+    smiles = Chem.MolToSmiles(mol, canonical=True)
+    vec = _stub_embed_smiles(smiles, int(cfg.embedding_dim), int(cfg.seed))
+    ...
```
- `src/common/feature_pipeline.py`, `src/fp/train.py`
```diff
+class EmbeddingFeaturePipeline:
+    ...
+def resolve_tabular_pipeline(cfg: Dict[str, Any]) -> FingerprintFeaturePipeline | EmbeddingFeaturePipeline:
+    if is_pretrained_embedding(feat_cfg):
+        return EmbeddingFeaturePipeline.from_config(cfg)
+
-    pipeline = FingerprintFeaturePipeline.from_config(cfg)
+    pipeline = resolve_tabular_pipeline(cfg)
+    if isinstance(pipeline, FingerprintFeaturePipeline):
+        ad_radius = pipeline.fp_cfg.morgan_radius
+    else:
+        ad_radius = int(ad_cfg.get("morgan_radius", 2))
```
- `configs/features/pretrained_embedding_stub.yaml`
```diff
+featurizer:
+  name: "pretrained_embedding"
+  backend: "stub"
+  embedding_dim: 256
+  seed: 0
+  normalize: true
```

**Tests**
- `tests/test_feature_pipeline.py`: added `test_embedding_pipeline_roundtrip`.
- `python -m pytest tests/test_feature_pipeline.py -k embedding` failed with `No module named pytest`.

**Verification**
- `python -m pytest tests/test_feature_pipeline.py -k embedding`

**Compatibility**
- Config: new features group `configs/features/pretrained_embedding_stub.yaml` (select via `features=pretrained_embedding_stub`); optional `ad.morgan_radius`/`ad.n_bits` apply for embedding AD.
- CLI: no new flags; FP train/eval/predict auto-select tabular pipeline from `featurizer`.
- Artifacts: `feature_pipeline.pkl` may store `EmbeddingFeaturePipeline`; `model/featurizer_state.json` now includes a `type` key.

**Queue**
- `work/queue.json`: task 070 set to done.

**Next Steps**
1. Run pytest in the project venv once `pytest` is available.
2. Try a baseline run with `features=pretrained_embedding_stub` to validate end-to-end behavior.
```

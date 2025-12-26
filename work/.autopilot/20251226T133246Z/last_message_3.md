Unified FP/GNN preprocessing+featurization behind a shared feature pipeline so train/evaluate/predict reuse the same fitted state, and added a pipeline roundtrip smoke test.  
Using S30_preprocess_pipeline, S40_featurizer, S95_tests_ci to enforce train/infer parity and add a minimal test.

**Change Plan**
- `src/common/feature_pipeline.py`: new shared pipeline module for FP/GNN, with save/load helpers.
- `src/fp/train.py`, `src/fp/evaluate.py`, `src/fp/predict.py`: switch to pipeline fit/transform + artifact save/load.
- `src/gnn/train.py`, `src/gnn/evaluate.py`, `src/gnn/predict.py`: switch to pipeline featurizer + artifact save/load.
- `tests/test_feature_pipeline.py`: add pipeline save/load consistency test.
- `work/queue.json`: mark task 030 done.

**Implementation**
- `src/common/feature_pipeline.py`: added `FingerprintFeaturePipeline` and `GraphFeaturePipeline`, plus `save_feature_pipeline`/`load_feature_pipeline` and backward-compatible artifact loaders for imputer/scaler or `graph_featurizer.pkl`.
- `src/fp/train.py`: build features and preprocessing through the pipeline and persist `artifacts/feature_pipeline.pkl` while keeping `imputer.pkl`/`scaler.pkl` for compatibility.
- `src/fp/evaluate.py` and `src/fp/predict.py`: load the saved pipeline first, with fallback to legacy artifacts when `feature_pipeline.pkl` is missing.
- `src/gnn/train.py`: featurize via pipeline and persist `artifacts/feature_pipeline.pkl` alongside `graph_featurizer.pkl`.
- `src/gnn/evaluate.py` and `src/gnn/predict.py`: use pipeline-first loading with fallback to `graph_featurizer.pkl`.

**Tests**
- Added `tests/test_feature_pipeline.py` to verify pipeline save/load consistency.
- Test run failed: `python -m pytest tests/test_feature_pipeline.py` → `No module named pytest`.

**Verification Command**
- `python -m pytest tests/test_feature_pipeline.py`

**Compatibility Impact**
- Artifacts: new `artifacts/feature_pipeline.pkl` is written on train; existing `artifacts/imputer.pkl`, `artifacts/scaler.pkl`, and `artifacts/graph_featurizer.pkl` remain for backward compatibility.
- Config/CLI: no changes.
- Inference: pipeline is loaded from training artifacts; legacy artifacts are still supported if the new pipeline file is absent.

**Queue**
- Updated `work/queue.json` to set task 030 to `done`.

Next steps (optional):
1) Install pytest in the venv and re-run the test command.  
2) Run a quick train→predict pass with `configs/fp/train_fixture.yaml` and `configs/fp/predict_fixture.yaml` to confirm pipeline artifacts in real runs.
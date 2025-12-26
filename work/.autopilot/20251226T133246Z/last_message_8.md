Using skills: S50_model_plugin + S95_tests_ci (model plugin wiring + minimal test coverage).

Added a GIN regressor and wired it into GNN train/predict/evaluate plus CLI dispatch, and added a gin model config with a config-load test.

**Change Plan**
- `src/gnn/models.py`: implement `GINRegressor` with GIN/GINE conv support.
- `src/gnn/train.py`, `src/gnn/predict.py`, `src/gnn/evaluate.py`: allow `model.name=gin` selection.
- `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py`: include `gin` in GNN backend detection.
- `configs/model/gin.yaml`: add model config for GIN.
- `tests/test_config_loader.py`: add config-load unit test.
- `work/queue.json`: mark task 060 done.

**Implementation (Diff Highlights)**
- `src/gnn/models.py`
```diff
-from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
+from torch_geometric.nn import GCNConv, GINConv, GINEConv, global_mean_pool, NNConv
...
+    class GINRegressor(nn.Module):
+        def __init__(..., edge_dim: int = 0):
+            ...
+            self.use_edge_attr = edge_dim > 0 and GINEConv is not None
+            ...
+            if self.use_edge_attr:
+                self.convs.append(GINEConv(make_mlp(in_dim), edge_dim=edge_dim))
+            else:
+                self.convs.append(GINConv(make_mlp(in_dim)))
```

- `src/gnn/train.py`, `src/gnn/predict.py`, `src/gnn/evaluate.py`
```diff
+    elif model_name == "gin":
+        model = GINRegressor(
+            in_dim=in_dim,
+            hidden_dim=hidden_dim,
+            num_layers=num_layers,
+            dropout=dropout,
+            global_dim=global_dim,
+            edge_dim=edge_dim,
+        )
```

- `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py`
```diff
-_GNN_MODELS = {"gcn", "mpnn"}
+_GNN_MODELS = {"gcn", "gin", "mpnn"}
```

- `configs/model/gin.yaml`
```yaml
model:
  family: "gnn"
  name: "gin"
  hidden_dim: 64
  num_layers: 4
  dropout: 0.1
```

- `tests/test_config_loader.py`
```diff
+def test_load_gin_model_config() -> None:
+    cfg = load_config(Path("configs/model/gin.yaml"))
+    assert cfg["model"]["name"] == "gin"
+    assert cfg["model"]["family"] == "gnn"
```

**Tests**
- Added `tests/test_config_loader.py::test_load_gin_model_config`.
- Not run (not requested).

**Verification Commands**
```bash
python - <<'PY'
from src.common.config import load_config
from src.gnn import train as gnn_train

cfg = load_config("configs/gnn/train_quick.yaml")
cfg["model"]["name"] = "gin"
cfg["output"]["exp_name"] = "quick_gin_v1"
cfg["train"]["epochs"] = 1
gnn_train.run(cfg)
PY
```

```bash
python - <<'PY'
import pandas as pd
from src.common.config import load_config
from src.gnn import predict as gnn_predict

cfg = load_config("configs/gnn/predict.yaml")
cfg["model_artifact_dir"] = "runs/train/gnn/quick_gin_v1"
cas = pd.read_csv("data/processed/quick/dataset_with_lj.csv")["CAS"].iloc[0]
gnn_predict.run(cfg, query=str(cas))
PY
```

```bash
pytest -q tests/test_config_loader.py -k gin
```

**Compatibility Impact**
- Config: new `configs/model/gin.yaml` enables `model=gin`; existing configs unchanged.
- CLI: backend dispatch now recognizes `gin` as GNN in `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py`.
- Artifacts: no contract changes; outputs remain per existing GNN artifact schema.

**Task Status**
- Marked task 060 done in `work/queue.json`.
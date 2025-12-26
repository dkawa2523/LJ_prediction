from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import GCNConv, GINConv, GINEConv, global_mean_pool, NNConv
except Exception:  # pragma: no cover
    GCNConv = None
    GINConv = None
    GINEConv = None
    global_mean_pool = None
    NNConv = None


def _require_pyg():
    if torch is None or nn is None or F is None:
        raise ImportError("PyTorch is required for GNN models. Please install torch.")
    if global_mean_pool is None:
        raise ImportError(
            "PyTorch Geometric is required for GNN models. "
            "Please install torch_geometric (matching your torch/CUDA)."
        )

if nn is None:  # pragma: no cover

    class GCNRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class GINRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

    class MPNNRegressor:
        def __init__(self, *args, **kwargs):
            _require_pyg()

else:

    class GCNRegressor(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0, global_dim: int = 0):
            super().__init__()
            _require_pyg()
            if GCNConv is None:
                raise ImportError("GCNConv is unavailable. Please install torch_geometric.")
            self.dropout = float(dropout)
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            g = global_mean_pool(x, batch)
            if hasattr(data, "u"):
                # data.u shape (batch, global_dim)
                g = torch.cat([g, data.u], dim=-1)
            out = self.head(g)
            return out.view(-1)

    class MPNNRegressor(nn.Module):
        """
        A simple message passing model using NNConv with edge-conditioned filters.
        """

        def __init__(
            self,
            in_dim: int,
            edge_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            global_dim: int = 0,
            edge_mlp_hidden_dim: int = 128,
        ):
            super().__init__()
            _require_pyg()
            if NNConv is None:
                raise ImportError("NNConv is unavailable. Please install torch_geometric.")
            self.dropout = float(dropout)

            if edge_mlp_hidden_dim <= 0:
                raise ValueError("edge_mlp_hidden_dim must be > 0")

            def make_edge_nn(out_dim: int) -> nn.Module:
                # Map edge_attr -> (in_channels * out_channels) weight matrix for NNConv.
                return nn.Sequential(
                    nn.Linear(edge_dim, edge_mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(edge_mlp_hidden_dim, out_dim),
                )

            self.conv1 = NNConv(in_dim, hidden_dim, make_edge_nn(hidden_dim * in_dim), aggr="mean")

            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(NNConv(hidden_dim, hidden_dim, make_edge_nn(hidden_dim * hidden_dim), aggr="mean"))

            self.head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            g = global_mean_pool(x, batch)
            if hasattr(data, "u"):
                g = torch.cat([g, data.u], dim=-1)
            out = self.head(g)
            return out.view(-1)

    class GINRegressor(nn.Module):
        """
        Graph Isomorphism Network (GIN/GINE) regressor.
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            global_dim: int = 0,
            edge_dim: int = 0,
        ):
            super().__init__()
            _require_pyg()
            if GINConv is None:
                raise ImportError("GINConv is unavailable. Please install torch_geometric.")
            self.dropout = float(dropout)
            self.use_edge_attr = edge_dim > 0 and GINEConv is not None

            def make_mlp(input_dim: int) -> nn.Module:
                return nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

            self.convs = nn.ModuleList()
            if self.use_edge_attr:
                self.convs.append(GINEConv(make_mlp(in_dim), edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GINEConv(make_mlp(hidden_dim), edge_dim=edge_dim))
            else:
                self.convs.append(GINConv(make_mlp(in_dim)))
                for _ in range(num_layers - 1):
                    self.convs.append(GINConv(make_mlp(hidden_dim)))

            self.head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_attr = getattr(data, "edge_attr", None)
            for conv in self.convs:
                if self.use_edge_attr:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            g = global_mean_pool(x, batch)
            if hasattr(data, "u"):
                g = torch.cat([g, data.u], dim=-1)
            out = self.head(g)
            return out.view(-1)

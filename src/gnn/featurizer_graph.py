from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.common.descriptors import calc_descriptors, descriptors_to_array

try:
    import torch
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    torch = None
    Data = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


class GraphFeaturizerError(ValueError):
    pass


@dataclass
class GraphFeaturizerConfig:
    node_features: List[str]
    edge_features: List[str]
    use_3d_pos: bool = True
    add_global_descriptors: Optional[List[str]] = None


def _require_pyg():
    if Data is None or torch is None:
        raise ImportError(
            "torch_geometric is required for GNN pipeline. "
            "Please install PyTorch Geometric (matching your torch/CUDA)."
        )


def _edge_feature_dim(feature_names: Sequence[str]) -> int:
    dim = 0
    for name in feature_names:
        if name == "bond_type":
            dim += 4  # single/double/triple/aromatic
        elif name in {"conjugated", "aromatic"}:
            dim += 1
        else:
            raise GraphFeaturizerError(f"Unknown edge feature: {name}")
    return dim


def featurize_mol_to_pyg(mol, y: Optional[float], cfg: GraphFeaturizerConfig) -> "Data":
    """Convert an RDKit Mol to PyG Data."""
    _require_pyg()
    if Chem is None:
        raise ImportError("RDKit is required for reading SDF and building graphs.")
    if mol is None:
        raise GraphFeaturizerError("mol is None")

    # Node features
    x_list: List[List[float]] = []
    for atom in mol.GetAtoms():
        feats: List[float] = []
        for name in cfg.node_features:
            if name == "atomic_num":
                feats.append(float(atom.GetAtomicNum()))
            elif name == "degree":
                feats.append(float(atom.GetDegree()))
            elif name == "formal_charge":
                feats.append(float(atom.GetFormalCharge()))
            elif name == "aromatic":
                feats.append(float(atom.GetIsAromatic()))
            elif name == "num_h":
                feats.append(float(atom.GetTotalNumHs(includeNeighbors=True)))
            elif name == "in_ring":
                feats.append(float(atom.IsInRing()))
            else:
                raise GraphFeaturizerError(f"Unknown node feature: {name}")
        x_list.append(feats)

    x = torch.tensor(np.array(x_list, dtype=np.float32), dtype=torch.float32)

    # Edges
    edge_index = []
    edge_attr_list: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # undirected => add both directions
        for (u, v) in [(i, j), (j, i)]:
            edge_index.append([u, v])
            edge_attr_list.append(_bond_features(bond, cfg.edge_features))

    edge_attr_dim = _edge_feature_dim(cfg.edge_features)
    if len(edge_index) == 0:
        edge_index_t = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
    else:
        edge_index_t = torch.tensor(np.array(edge_index, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr_list, dtype=np.float32), dtype=torch.float32)

    # 3D positions
    pos = None
    if cfg.use_3d_pos:
        conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        if conf is not None and conf.Is3D():
            coords = []
            for k in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(k)
                coords.append([p.x, p.y, p.z])
            pos = torch.tensor(np.array(coords, dtype=np.float32), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr)
    if pos is not None:
        data.pos = pos
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)

    # Global descriptors
    if cfg.add_global_descriptors:
        desc = calc_descriptors(mol, cfg.add_global_descriptors)
        u = descriptors_to_array(desc, cfg.add_global_descriptors).astype(np.float32)
        data.u = torch.tensor(u.reshape(1, -1), dtype=torch.float32)
        data.u_names = cfg.add_global_descriptors

    return data


def _bond_features(bond, feature_names: Sequence[str]) -> List[float]:
    feats: List[float] = []
    btype = bond.GetBondType()
    bond_onehot = [
        float(btype == Chem.rdchem.BondType.SINGLE),
        float(btype == Chem.rdchem.BondType.DOUBLE),
        float(btype == Chem.rdchem.BondType.TRIPLE),
        float(btype == Chem.rdchem.BondType.AROMATIC),
    ]
    for name in feature_names:
        if name == "bond_type":
            feats.extend(bond_onehot)
        elif name == "conjugated":
            feats.append(float(bond.GetIsConjugated()))
        elif name == "aromatic":
            feats.append(float(bond.GetIsAromatic()))
        else:
            raise GraphFeaturizerError(f"Unknown edge feature: {name}")
    return feats

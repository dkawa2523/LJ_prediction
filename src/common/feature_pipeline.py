from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.fp.feature_utils import build_embedding_features, build_features
from src.fp.featurizer_embedding import PretrainedEmbeddingConfig, featurize_mol as featurize_embedding_mol
from src.fp.featurizer_fp import FPConfig, featurize_mol
from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg

PIPELINE_VERSION = 1
FEATURE_PIPELINE_FILENAME = "feature_pipeline.pkl"


class FeaturePipelineError(RuntimeError):
    pass


def _normalize_list(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def is_pretrained_embedding(feat_cfg: Dict[str, Any]) -> bool:
    name = str(feat_cfg.get("name", "")).lower()
    if name in {"pretrained_embedding", "pretrained-embedding", "embedding"}:
        return True
    kind = str(feat_cfg.get("type", "")).lower()
    if kind in {"pretrained_embedding", "embedding"}:
        return True
    if feat_cfg.get("embedding_dim") is not None:
        return True
    return False


def build_fp_config(feat_cfg: Dict[str, Any]) -> FPConfig:
    add_desc = _normalize_list(feat_cfg.get("add_descriptors", None))
    return FPConfig(
        fingerprint=str(feat_cfg.get("fingerprint", "morgan")),
        morgan_radius=int(feat_cfg.get("morgan_radius", 2)),
        n_bits=int(feat_cfg.get("n_bits", 2048)),
        use_counts=bool(feat_cfg.get("use_counts", False)),
        add_descriptors=add_desc,
    )


def build_embedding_config(feat_cfg: Dict[str, Any]) -> PretrainedEmbeddingConfig:
    return PretrainedEmbeddingConfig(
        name=str(feat_cfg.get("name", "pretrained_embedding")),
        backend=str(feat_cfg.get("backend", "stub")),
        embedding_dim=int(feat_cfg.get("embedding_dim", 256)),
        seed=int(feat_cfg.get("seed", 0)),
        normalize=bool(feat_cfg.get("normalize", True)),
    )


def build_graph_config(feat_cfg: Dict[str, Any]) -> GraphFeaturizerConfig:
    node_features = feat_cfg.get(
        "node_features",
        ["atomic_num", "degree", "formal_charge", "aromatic", "num_h", "in_ring"],
    )
    edge_features = feat_cfg.get("edge_features", ["bond_type", "conjugated", "aromatic"])
    return GraphFeaturizerConfig(
        node_features=[str(v) for v in node_features],
        edge_features=[str(v) for v in edge_features],
        use_3d_pos=bool(feat_cfg.get("use_3d_pos", True)),
        add_global_descriptors=_normalize_list(feat_cfg.get("add_global_descriptors", None)),
    )


@dataclass
class FingerprintFeaturePipeline:
    fp_cfg: FPConfig = field(default_factory=FPConfig)
    impute_strategy: str = "mean"
    standardize: bool = False
    imputer: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None
    feature_meta: Optional[Dict[str, Any]] = None
    pipeline_type: str = field(default="fp", init=False)
    version: int = field(default=PIPELINE_VERSION, init=False)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FingerprintFeaturePipeline":
        feat_cfg = cfg.get("featurizer", {}) or {}
        preprocess_cfg = cfg.get("preprocess", {}) or {}
        return cls(
            fp_cfg=build_fp_config(feat_cfg),
            impute_strategy=str(preprocess_cfg.get("impute_nan", "mean")),
            standardize=bool(preprocess_cfg.get("standardize", False)),
        )

    def build_features(
        self,
        df,
        sdf_dir: Path,
        cas_col: str,
        cache_dir: Optional[Path],
        cache_key: str,
        logger,
    ) -> Tuple[np.ndarray, list[Any], list[str], Dict[str, Any]]:
        X, ids, elements, meta = build_features(
            df=df,
            sdf_dir=sdf_dir,
            cas_col=cas_col,
            fp_cfg=self.fp_cfg,
            cache_dir=cache_dir,
            cache_key=cache_key,
            logger=logger,
        )
        self.feature_meta = meta
        return X, ids, elements, meta

    def fit(self, X: np.ndarray) -> None:
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imp = self.imputer.fit_transform(X)
        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X_imp)
        else:
            self.scaler = None

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        if self.imputer is None:
            raise FeaturePipelineError("FingerprintFeaturePipeline is not fitted (imputer is missing).")
        X_imp = self.imputer.transform(X)
        if self.scaler is not None:
            X_imp = self.scaler.transform(X_imp)
        return X_imp

    def fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform_features(X)

    def featurize_mol(self, mol) -> Tuple[np.ndarray, Dict[str, Any]]:
        return featurize_mol(mol, self.fp_cfg)

    def transform_mol(self, mol) -> Tuple[np.ndarray, Dict[str, Any]]:
        x, meta = self.featurize_mol(mol)
        X = self.transform_features(x.reshape(1, -1))
        return X.reshape(-1), meta

    def save_preprocess_artifacts(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.imputer is None:
            raise FeaturePipelineError("Cannot save preprocess artifacts: imputer is missing.")
        with open(artifacts_dir / "imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)
        if self.scaler is not None:
            with open(artifacts_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

    def load_preprocess_artifacts(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        imputer_path = artifacts_dir / "imputer.pkl"
        if not imputer_path.exists():
            raise FileNotFoundError(f"imputer.pkl not found: {imputer_path}")
        with open(imputer_path, "rb") as f:
            self.imputer = pickle.load(f)
        scaler_path = artifacts_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

    def featurizer_state(self) -> Dict[str, Any]:
        return {"type": "fingerprint", "config": asdict(self.fp_cfg), "feature_meta": self.feature_meta}


@dataclass
class EmbeddingFeaturePipeline:
    emb_cfg: PretrainedEmbeddingConfig = field(default_factory=PretrainedEmbeddingConfig)
    impute_strategy: str = "mean"
    standardize: bool = False
    imputer: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None
    feature_meta: Optional[Dict[str, Any]] = None
    pipeline_type: str = field(default="embedding", init=False)
    version: int = field(default=PIPELINE_VERSION, init=False)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "EmbeddingFeaturePipeline":
        feat_cfg = cfg.get("featurizer", {}) or {}
        preprocess_cfg = cfg.get("preprocess", {}) or {}
        return cls(
            emb_cfg=build_embedding_config(feat_cfg),
            impute_strategy=str(preprocess_cfg.get("impute_nan", "mean")),
            standardize=bool(preprocess_cfg.get("standardize", False)),
        )

    def build_features(
        self,
        df,
        sdf_dir: Path,
        cas_col: str,
        cache_dir: Optional[Path],
        cache_key: str,
        logger,
    ) -> Tuple[np.ndarray, list[Any], list[str], Dict[str, Any]]:
        X, ids, elements, meta = build_embedding_features(
            df=df,
            sdf_dir=sdf_dir,
            cas_col=cas_col,
            emb_cfg=self.emb_cfg,
            cache_dir=cache_dir,
            cache_key=cache_key,
            logger=logger,
        )
        self.feature_meta = meta
        return X, ids, elements, meta

    def fit(self, X: np.ndarray) -> None:
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imp = self.imputer.fit_transform(X)
        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X_imp)
        else:
            self.scaler = None

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        if self.imputer is None:
            raise FeaturePipelineError("EmbeddingFeaturePipeline is not fitted (imputer is missing).")
        X_imp = self.imputer.transform(X)
        if self.scaler is not None:
            X_imp = self.scaler.transform(X_imp)
        return X_imp

    def fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform_features(X)

    def featurize_mol(self, mol) -> Tuple[np.ndarray, Dict[str, Any]]:
        return featurize_embedding_mol(mol, self.emb_cfg)

    def transform_mol(self, mol) -> Tuple[np.ndarray, Dict[str, Any]]:
        x, meta = self.featurize_mol(mol)
        X = self.transform_features(x.reshape(1, -1))
        return X.reshape(-1), meta

    def save_preprocess_artifacts(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.imputer is None:
            raise FeaturePipelineError("Cannot save preprocess artifacts: imputer is missing.")
        with open(artifacts_dir / "imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)
        if self.scaler is not None:
            with open(artifacts_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

    def load_preprocess_artifacts(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        imputer_path = artifacts_dir / "imputer.pkl"
        if not imputer_path.exists():
            raise FileNotFoundError(f"imputer.pkl not found: {imputer_path}")
        with open(imputer_path, "rb") as f:
            self.imputer = pickle.load(f)
        scaler_path = artifacts_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

    def featurizer_state(self) -> Dict[str, Any]:
        return {"type": "pretrained_embedding", "config": asdict(self.emb_cfg), "feature_meta": self.feature_meta}


@dataclass
class GraphFeaturePipeline:
    graph_cfg: GraphFeaturizerConfig
    pipeline_type: str = field(default="gnn", init=False)
    version: int = field(default=PIPELINE_VERSION, init=False)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GraphFeaturePipeline":
        feat_cfg = cfg.get("featurizer", {}) or {}
        return cls(build_graph_config(feat_cfg))

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> "GraphFeaturePipeline":
        artifacts_dir = Path(artifacts_dir)
        cfg_path = artifacts_dir / "graph_featurizer.pkl"
        if not cfg_path.exists():
            raise FileNotFoundError(f"graph_featurizer.pkl not found: {cfg_path}")
        with open(cfg_path, "rb") as f:
            gcfg = pickle.load(f)
        if not isinstance(gcfg, GraphFeaturizerConfig):
            raise FeaturePipelineError("graph_featurizer.pkl does not contain a GraphFeaturizerConfig.")
        return cls(graph_cfg=gcfg)

    def featurize_mol(self, mol, y: Optional[float] = None):
        return featurize_mol_to_pyg(mol, y=y, cfg=self.graph_cfg)


def resolve_tabular_pipeline(cfg: Dict[str, Any]) -> FingerprintFeaturePipeline | EmbeddingFeaturePipeline:
    feat_cfg = cfg.get("featurizer", {}) or {}
    if is_pretrained_embedding(feat_cfg):
        return EmbeddingFeaturePipeline.from_config(cfg)
    return FingerprintFeaturePipeline.from_config(cfg)


def save_feature_pipeline(pipeline: Any, artifacts_dir: Path) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / FEATURE_PIPELINE_FILENAME
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    return path


def load_feature_pipeline(artifacts_dir: Path) -> Optional[Any]:
    artifacts_dir = Path(artifacts_dir)
    path = artifacts_dir / FEATURE_PIPELINE_FILENAME
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tabular_pipeline(
    artifacts_dir: Path, train_cfg: Dict[str, Any]
) -> FingerprintFeaturePipeline | EmbeddingFeaturePipeline:
    pipeline = load_feature_pipeline(artifacts_dir)
    if isinstance(pipeline, (FingerprintFeaturePipeline, EmbeddingFeaturePipeline)) and pipeline.imputer is not None:
        return pipeline
    pipeline = resolve_tabular_pipeline(train_cfg)
    pipeline.load_preprocess_artifacts(artifacts_dir)
    return pipeline

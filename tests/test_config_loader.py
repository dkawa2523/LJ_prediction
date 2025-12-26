from pathlib import Path

from src.common.config import load_config


def test_load_config_resolves_groups_from_root() -> None:
    cfg = load_config(Path("configs/fp/train_fixture.yaml"))

    assert cfg["process"]["name"] == "train"
    assert cfg["process"]["backend"] == "fp"
    assert cfg["output"]["run_dir"] == "runs/train/fp"
    assert cfg["paths"]["raw_csv"] == "tests/fixtures/data/raw/tc_pc_tb_fixture.csv"


def test_load_gin_model_config() -> None:
    cfg = load_config(Path("configs/model/gin.yaml"))

    assert cfg["model"]["name"] == "gin"
    assert cfg["model"]["family"] == "gnn"

# Raw data directory (not tracked in git)

この `data/raw/` 配下は **生データ置き場**です。容量・ライセンス・機密性の観点から、デフォルトでは `.gitignore` で **git管理対象外**にしています。

必要に応じて以下を配置してください：

- `tc_pc_tb_pubchem.csv`（例: `data/raw/tc_pc_tb_pubchem.csv`）
- `sdf_files/`（SDFはCAS一致のファイル名：例 `data/raw/sdf_files/71-43-2.sdf`）

設定でパスは変更できます：`configs/dataset.yaml` / `configs/*/*.yaml` の `paths:` / `data:` を参照。


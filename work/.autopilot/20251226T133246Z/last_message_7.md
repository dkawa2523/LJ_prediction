Using S20_dataset_loader to keep collection output column mapping and missing-field logging aligned with the dataset contracts.

Implemented the collect_data process with a plugin registry, dummy data source, mapping formatter, and cache, plus configs and a unit test; secrets are env-only and raw outputs land in the run dir with optional export.

**Change Plan**
- `src/data_collection/*`: new package for client/formatter/cache/registry/runner layers.
- `scripts/collect_data.py`: new CLI entrypoint for the collect_data process.
- `configs/collect_data.yaml`, `configs/process/collect_data.yaml`, `configs/data_collection/default.yaml`, `configs/data_source/dummy.yaml`: config groups to switch `data_source` via config.
- `src/utils/validate_config.py`: validate collect_data config shape.
- `tests/test_data_collection.py`: unit test for dummy collection output.
- `work/queue.json`: task 050 marked done.

**Implementation Details**
- `src/data_collection/runner.py`: orchestration, cache usage, raw CSV/SDF output, optional export, meta.json with data hash.
- `src/data_collection/formatters/mapping.py`: column_map formatting with missing-field logging and sample_id generation.
- `src/data_collection/utils.py`: env-only API key enforcement (no secrets in code).
- `src/data_collection/registry.py`: lazy-load dummy source to avoid circular imports.

**Tests**
- `pytest -k data_collection` (pass)

**Validation Command**
- `pytest -k data_collection`

**Compatibility Impact**
- Config: new `collect_data` config groups; existing configs unchanged.
- CLI: new `scripts/collect_data.py`.
- Artifacts: new `runs/collect_data/<exp>/` with `config.yaml`, `meta.json`, and `raw/` outputs; export to `data/raw` only when enabled.

Next steps (optional):
1) Run `python scripts/collect_data.py --config configs/collect_data.yaml` to generate stub outputs.
2) Add a real data source (e.g., PubChem) by implementing a new client/formatter and a `configs/data_source/<name>.yaml`.
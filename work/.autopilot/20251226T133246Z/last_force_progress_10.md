Using skill S95_tests_ci to finalize CI/test setup for task 090.

Added a minimal GitHub Actions pytest workflow so CI runs `pytest -q`, and synced task metadata to reflect completion.
- `.github/workflows/pytest.yml`: new CI job with minimal deps and `pytest -q`.
- `work/tasks/090_ci_tests.md`: checked off acceptance criteria and noted the workflow.
- `work/queue.json`: refreshed `updated_at` to reflect the completion.

Tests: `pytest -q`

Optional next step: add an rdkit-enabled CI job so chemistry-dependent tests run on CI.
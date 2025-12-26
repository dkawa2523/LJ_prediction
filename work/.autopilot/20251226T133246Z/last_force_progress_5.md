Marked task 075 as complete and recorded the verification in `work/tasks/075_leaderboard_process.md`, and refreshed the queue timestamp in `work/queue.json` to reflect the progress.

- Updated acceptance checkboxes and added the executed pytest command in `work/tasks/075_leaderboard_process.md`
- Bumped `updated_at` in `work/queue.json` after completing verification

Tests run:
- `pytest -q tests/test_leaderboard.py`

Next step (optional):
1) `python scripts/leaderboard.py --config configs/leaderboard.yaml` to generate a real leaderboard from `runs/`.
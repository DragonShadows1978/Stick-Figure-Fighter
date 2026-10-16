# Archive

Historical material from the AtlasForge development cycles, kept for reference.
None of this is imported by the game, the engine, or the tests.

- `scripts/` — one-off training runs (`train_cycle*`), matchup/report generators,
  screenshot capture scripts and visual-effect smoke tests. They import modules
  from the repository root, so run them from the root (`python3 archive/scripts/foo.py`)
  if you ever need one. The capture/visual scripts also depend on a `vision_tool`
  module that was never committed.
- `vision/` — X11 shared-memory screen capture bindings used by those scripts.
- `research/`, `artifacts/` — per-cycle research notes and implementation plans.

The task.toml and environment/Dockerfile of `gabeorlanski/code_search` from the Harbor package store dataset
`gabeorlanski/slopcodebench` (36 tasks, every one with per-step `artifacts` on each `[[steps]]` entry), as pulled on
2026-10-22 with `gym dataset pull harbor:gabeorlanski/slopcodebench`. The step folders hold one-line instructions
and stub `test.sh` scripts in place of the dataset's own; `tests/` (pytest helpers and data) and `README.md` are
left out. Used by the loader and environment server tests for the per-step artifacts shape.

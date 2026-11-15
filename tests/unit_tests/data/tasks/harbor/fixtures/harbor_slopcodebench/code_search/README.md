A synthetic task in the shape of the Harbor hub dataset `gabeorlanski/slopcodebench` (36 tasks, every one with
per-step `artifacts` on each `[[steps]]` entry, gating two reward keys with `min_reward`, on a build-mode Dockerfile
with a `SHELL` line), written for NeMo Gym's loader and environment server tests. `task.toml`, `environment/Dockerfile`,
the step instructions and the stub `test.sh` scripts are all NeMo Gym's own text: the step names, verifier budgets,
resources, exclude list and reward gates are the tests' own and nothing is copied from the dataset's files (the two
`task.toml` share only TOML table headers, schema keys and generic values such as `workdir = "/app"` and the two-hour
agent budget), so no source revision or third-party license applies. Only the shape is shared; the dataset's `tests/`
helpers and data, `README.md` and real step scripts have no counterpart here.

# Description

NVARC is an ARC-AGI style resource server with two modes:
- `transductive`: the model outputs the grid directly
- `inductive`: the model outputs Python code implementing `transform()`

Data links: local example dataset in `data/example.jsonl`

In inductive mode, `python_max_concurrency` limits the number of submitted programs running in each server worker. Requests waiting for a slot do not consume their configured `python_timeout_seconds`; that timeout starts after admission. The `gym.verify.nvarc.python.queue` and `gym.verify.nvarc.python.execute` spans expose these phases separately when verify tracing is enabled.

# Licensing information
Code: Apache 2.0
Data: example data included in-repo; train/validation paths are configured but not committed

Dependencies
- `nemo_gym`: Apache 2.0

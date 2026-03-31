# Description

NVARC is an ARC-AGI style resource server with two modes:
- `transductive`: the model outputs the grid directly
- `inductive`: the model outputs Python code implementing `transform()`

Data links: local example dataset in `data/example.jsonl`

# Licensing information
Code: Apache 2.0
Data: example data included in-repo; train/validation paths are configured but not committed

Dependencies
- `nemo_gym`: Apache 2.0

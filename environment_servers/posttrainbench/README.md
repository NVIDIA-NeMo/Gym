# PostTrainBench environment server

The `harbor` environment server with PostTrainBench's run score. Episodes run exactly as on `harbor`
(see `environment_servers/harbor/README.md`). The server reads the taskset that
`benchmarks/posttrainbench/prepare.py` writes (`taskset_file`: task folder, digests, per-task baselines and
upstream's benchmark weights) and adds `ptb/weighted_avg` and per-benchmark means to the metrics. A task with
no scored rollout in a repeat counts as its base model's zero-shot baseline, as upstream's `collect.py` counts a
missing run. See `benchmarks/posttrainbench/README.md` for the benchmark.

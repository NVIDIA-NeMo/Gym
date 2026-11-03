# PostTrainBench environment server

The `harbor` environment server with PostTrainBench's run score. Episodes run exactly as on `harbor`
(see `environment_servers/harbor/README.md`). The server reads the taskset that
`benchmarks/posttrainbench/prepare.py` writes (`taskset_file`: task folder, digests, per-task baselines and
upstream's benchmark weights) and adds `ptb/weighted_avg` and per-benchmark means to the metrics. The weighted
score is withheld while any expected task/repeat has no scored rollout. An explicit no-model submission instead
receives its baseline. See `benchmarks/posttrainbench/README.md` for details.

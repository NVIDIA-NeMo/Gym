# Collaborative Intelligence resources server

Scores Collaborative Intelligence rows from the score metadata `collab_intel_agent` returns. Rows whose scenario the
policy failed alone, or whose run failed on the evaluation side (a harness crash, a judge failure), are masked. A
policy timeout (`agent_timeout`) or context-window error (`model_context_limit`) is a measured fail. Each failed row
names its `failure_kind` and `failure_reason`. Setup, data and the run command are in
[benchmarks/collab_intel/README.md](../../benchmarks/collab_intel/README.md).

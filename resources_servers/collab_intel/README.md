# Collaborative Intelligence resources server

Scores Collaborative Intelligence rows from the score metadata `collab_intel_agent` returns. Only rows whose run failed
on the evaluation side (a harness crash, a judge failure) are masked. A row whose scenario the policy failed alone is a
measured zero with `alone_pass=false`, and the key metric `collab_intel/score` leaves it out, so `mean/reward` is not
the benchmark score. A policy timeout (`agent_timeout`) or context-window error (`model_context_limit`) is a measured
fail. Each failed row names its `failure_kind` and `failure_reason`. Setup, data and the run command are in
[benchmarks/collab_intel/README.md](../../benchmarks/collab_intel/README.md).

# VS-Bench reference resources server

This verifier delegates strategic reasoning and perception scoring to the
pinned VS-Bench reference source. Perception runs in a Bubblewrap sandbox:
the upstream scorer evaluates model output as Python. See the
[benchmark README](../../benchmarks/vs_bench/README.md) for source identity,
the separately identified KAZ correction, deployment prerequisites, and
scoring limitations.

The verifier fixture uses the real upstream reasoning parser. Before running
`gym env test vs_bench_reasoning`, set `VS_BENCH_AUTHOR_ROOT` to a validated
source export and `VS_BENCH_AUTHOR_PYTHON` to its Python 3.10 interpreter.

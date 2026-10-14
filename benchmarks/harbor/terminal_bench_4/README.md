# Terminal Bench 4 on the Harbor task shape

The 66 Terminal Bench 4 tasks also run as plain Harbor task folders through the generic `harbor` environment
server, without the `terminal_bench_4` benchmark's own server. The dataset comes from Harbor's package store,
pinned by content hash. The legacy benchmark folder, `benchmarks/terminal_bench_4/`, is untouched.

```sh
gym dataset validate harbor:terminal-bench/terminal-bench@4.0.0 --sandbox opensandbox \
  --exclude-tasks fp8-rmsnorm-gemm,jax-speedrun-gpu,math-eval-grader
gym eval run datasets/terminal-bench-4.0.0 --agent miniswe_sandboxed_agent --sandbox opensandbox \
  --exclude-tasks fp8-rmsnorm-gemm,jax-speedrun-gpu,math-eval-grader \
  --model-type openai_model --model <model> --model-url <url>
```

The first command fetches the packages into `datasets/terminal-bench-4.0.0/` and runs every reference
solution through the separate verifier. `--agent terminus_2_sandboxed_agent` runs Terminus 2 instead.
The 11 Compose tasks are skipped automatically until Compose groups land, and the 3 H100 tasks are
excluded above until per-task GPU routing lands; `--only-tasks` selects a subset.

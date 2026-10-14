# Terminal Bench 2.1 on the Harbor task shape

The 89 Terminal Bench 2.1 tasks run as Harbor task folders through the generic `harbor` environment server,
fetched from the Harbor hub exactly as published, without the `tests/test.sh` and `solution/solve.sh` fixes
the `terminal_bench_2_1` server applies at run time. Results on this path are therefore compared with
Harbor's own numbers for the dataset, not one to one with that server; tasks whose published tests or
solutions need those fixes score as they do in Harbor. The legacy benchmark folder,
`benchmarks/terminal_bench_2_1/`, is untouched.

Check the reference solutions, then run a model:

```bash
gym dataset validate harbor:terminal-bench/terminal-bench-2-1 --sandbox opensandbox
gym eval run harbor:terminal-bench/terminal-bench-2-1 --agent terminus_2_sandboxed_agent --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url>
```

The first fetch writes the task folders into the shared datasets folder (`datasets/`); later fetches keep every
folder whose content still matches. `gym dataset validate --output <file>` writes the oracle rollouts where a
later `gym dev compare` can find them. To match the legacy server's sandbox sizing for a like-for-like
comparison, add an overlay that sets `sandbox_resources_override` on the generated
`harbor_<taskset>_environment` block (see `environment_servers/harbor/configs/harbor.yaml` for the knobs).

# Terminal Bench 4 on the Harbor task shape

The 66 Terminal Bench 4 tasks also run as plain Harbor task folders through the generic `harbor` environment
server, without the `terminal_bench_4` benchmark's own server. The dataset comes from Harbor's package store,
pinned by content hash. The legacy benchmark folder, `benchmarks/terminal_bench_4/`, is untouched.

```sh
gym dataset validate harbor:terminal-bench/terminal-bench@4.0.0 --sandbox opensandbox \
  --config environments/harbor/terminal_bench_4/harbor_sandbox.yaml
gym eval run harbor:terminal-bench/terminal-bench@4.0.0 --agent miniswe_sandboxed_agent --sandbox opensandbox \
  --config environments/harbor/terminal_bench_4/harbor_sandbox.yaml \
  --model-type openai_model --model <model> --model-url <url>
```

`harbor_sandbox.yaml` overlays the shipped OpenSandbox block with the status-poll cadence the legacy
benchmark's server runs at (0.25 s backing off to 2 s). The shipped cap of 30 s is sized for about 1,500
concurrent SWE-bench samples and adds about 11 s per TB4 task between the agent's last command and the
trajectory write.

Both commands resolve the same reference through the package store. The first fetch writes the task
folders into `environments/harbor/terminal-bench-4.0.0/` and every later fetch keeps each folder whose content hash
still matches, so the model run reuses the validated tasks. `gym dataset validate` runs every reference
solution through the separate verifier. `--agent terminus_2_sandboxed_agent` runs Terminus 2 instead.
The 11 Compose tasks start sidecar images whose entrypoint, command, user and ports live in the image, not the
task. Preparing the dataset records them from the registry into `compose-images.json` next to the task folders,
pinning each image to the linux/amd64 digest it resolved to; entries already recorded are kept. The recording
the legacy benchmark's server uses, `benchmarks/terminal_bench_4/compose-images.json`, matches the freshly
recorded one for 37 of the 38 images; `redis:7-alpine` is a floating tag whose startup configuration is
identical but whose current build is newer. Place the benchmark's file in the dataset folder before the first
prepare to pin the same redis build. The 3 H100 tasks run on a GPU deployment when the harbor environment
server's `gpu_sandbox_provider` names its config block (for example `sandbox_gpu`); without one they run on the
default provider and fail to schedule. `--exclude-tasks` and `--only-tasks` select subsets.

Settings live in four places (see `environment_servers/harbor/README.md`). For a like-for-like comparison
with the legacy benchmark's server, which sets `CIRCLE_NODE_TOTAL=3` for `nextjs-performance`, write that
override beside the fetched tasks; the loader applies it and nothing else changes:

```toml
# environments/harbor/terminal-bench-4.0.0/dataset.toml
[gym.tasks."nextjs-performance".environment]
env = { CIRCLE_NODE_TOTAL = "3" }
```

Deployment settings go in the `sandbox` and `sandbox_gpu` provider blocks of the merged configuration (`env.yaml` or a `--config` overlay): Compose networking and
the shared-memory label as documented in the harbor README, and, for `gym dataset validate` on the three images
that run as a non-root user, `images.rewrites` pointing them at derived images with writable `/solution` and
`/logs` folders. Model runs use the upstream images.

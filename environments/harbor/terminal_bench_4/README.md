# Terminal Bench 4 on the Harbor task shape

The 66 Terminal Bench 4 tasks also run as plain Harbor task folders through the generic `harbor` environment
server, without the `terminal_bench_4` benchmark's own server. The dataset comes from Harbor's package store,
pinned by tag (`@4.0.0`); the pull records each task's content hash in `manifest.toml` and refuses a task
folder that no longer matches it. The `terminal_bench_4/miniswe` benchmark runs this entry with mini-SWE at the Artificial
Analysis settings (`gym eval run --benchmark terminal_bench_4/miniswe`; see `benchmarks/terminal_bench_4/README.md`).

```sh
gym dataset pull terminal_bench_4
gym dataset validate terminal_bench_4 --sandbox opensandbox \
  --config environments/harbor/terminal_bench_4/harbor_sandbox.yaml
gym eval run terminal_bench_4 --agent mini_swe_agent/mini_swe_agent_sandboxed --sandbox opensandbox \
  --config environments/harbor/terminal_bench_4/harbor_sandbox.yaml \
  --model-type openai_model --model <model> --model-url <url>
```

`dataset.toml` here declares the dataset's source, `harbor:terminal-bench/terminal-bench@4.0.0`. The pull fetches
it into this folder (the 66 task folders and `manifest.toml` are gitignored beside this README), a pull of that
reference itself lands here too, and the two commands above resolve the entry name to this folder; the
`harbor:...` reference still works as a target and fetches into the same place. A checkout that fetched the reference
before this entry declared it holds `environments/harbor/terminal-bench-4.0.0/` as well: nothing reads it any more
(the pull says so while it is there), so move it aside or delete it; a `dataset.toml` written into that old folder
(the `nextjs-performance` override an earlier version of this README placed there) is no longer gitignored and shows
as untracked, so move its keys below the `[gym]` table here and delete it. Runs through the entry name use
`terminal_bench_4` as the taskset, as the rows' `task_source` and in the `results/harbor/terminal_bench_4/` path, and
their dataset digest covers this folder's `dataset.toml`, so they differ from runs of the bare reference made before.
To pull another version, change `source` here, delete only the pulled task folders and `manifest.toml` in this
folder, and pull again.

`harbor_sandbox.yaml` overlays the shipped OpenSandbox block with the status-poll cadence the legacy
benchmark's server runs at (0.25 s backing off to 2 s). The shipped cap of 30 s is sized for about 1,500
concurrent SWE-bench samples and adds about 11 s per TB4 task between the agent's last command and the
trajectory write.

Every fetch keeps each task folder whose content hash still matches and refuses one that was edited, so the
model run reuses the validated tasks. `gym dataset validate` runs every reference
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

Settings live in four places (see `fern/versions/latest/pages/agent-interactions/harbor.mdx`, "Configure the server"). `dataset.toml` sets
`CIRCLE_NODE_TOTAL=3` for `nextjs-performance`, as the legacy benchmark's server does, so its test sharding matches
the task's two-CPU allocation. It sets it for the agent's sandbox and for the separate verifier's, which gets only its
own `[verifier.environment].env`; the loader applies both and nothing else changes.

Deployment settings go in the `sandbox` and `sandbox_gpu` provider blocks of the merged configuration (`env.yaml` or a `--config` overlay): Compose networking and
the shared-memory label as documented in `fern/versions/latest/pages/data/task-folders/compose-tasks.mdx`, and, for `gym dataset validate` on the three images
that run as a non-root user, `images.rewrites` pointing them at derived images with writable `/solution` and
`/logs` folders. Model runs use the upstream images.

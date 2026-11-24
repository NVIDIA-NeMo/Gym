# Terminal-Bench-Science on the Harbor task shape

The 70 Terminal-Bench-Science v0.1 tasks run as plain Harbor task folders through the generic `harbor` environment
server. The dataset comes from Harbor's package store, pinned by the same digest
`benchmarks/terminal_bench_science/manifest.json` records for the resources server's presets, so both paths run the
same packages; the pull records each task's content hash in `manifest.toml` and refuses a task folder that no longer
matches it. Run it through the `terminal_bench_science` benchmark, which carries the server settings a
full run needs (see `benchmarks/terminal_bench_science/README.md`, "Run"):

```sh
gym dataset pull terminal_bench_science
gym dataset validate terminal_bench_science --sandbox opensandbox \
  --exclude-tasks inverse-lithography,noisy-blackbox-optimization,protein-active-learning
gym eval run --benchmark terminal_bench_science --sandbox opensandbox \
  --exclude-tasks inverse-lithography,noisy-blackbox-optimization,protein-active-learning \
  --model-type vllm_model --model <model> --model-url <url>
```

Both commands refuse to start while preparation leaves a task out, so they exclude the three Compose tasks this
route does not support (see Known limits). A direct `gym eval run terminal_bench_science --agent ...` runs without
the benchmark's settings: the server's 21600-second episode cap would cut every 8-hour run short and retry it, and the
run gets the default harness setup allowance, no sandbox labels or disk limit, and the harness's own step limit. Use
it only for short smoke runs. `--agent-type` on the benchmark runs another harness with the same server settings;
only `mini_swe_agent` has been run on this entry.

`dataset.toml` here declares the source. The pull fetches it into this folder (the 70 task folders and `manifest.toml`
are gitignored beside this README), and the commands above resolve the entry name to this folder.

## Images

No package declares a prebuilt image: each task's agent image is built from `environment/`, its separate verifier's
from `tests/`, and the five Compose tasks' sidecars from their own Dockerfiles. Preparing the dataset builds them
with buildah, podman or docker and pushes them under `builds.repository` (default `$NEMO_GYM_BUILD_REPOSITORY`), then
records them in `built-images.json` next to the task folders; see `environment_servers/harbor/README.md` and
`fern/versions/latest/pages/data/task-folders/harbor-tasks.mdx`. Without a build tool, a task whose images are not
recorded is skipped with the loader's message. The sandbox deployment must be able to pull from that repository.

## Overrides

`dataset.toml` changes only verifier settings, the ones the resources server's presets apply
(`benchmarks/terminal_bench_science/README.md`, "Deviations from the packages"):

- **Verifier timeouts.** `finite-free-stam` and `gen-turan-paths` get 1800 s and `certified-sparse-regression` 7200 s.
- **Verifier CPUs and memory.** The 58 packages whose `[verifier.environment]` declares neither get the task's agent
  allocation. Harbor leaves them unlimited on Docker; a sandbox provider applies its own default (1 CPU and 2 GiB on
  OpenSandbox).
- **Verifier network.** Every package denies its verifier all egress with `[verifier.environment].network_mode =
  "no-network"`. Gym's loader applies a verifier restriction from `[verifier].network_mode`, as an egress policy on the
  separate verifier's sandbox, and refuses one on `[verifier.environment]`, so each task's restriction is set on
  `[verifier]` and its verifier environment is public. The verifier sandbox still has no egress. The policy needs the
  OpenSandbox provider.

## Known limits

- **Unsupported Compose tasks.** Preparation skips `inverse-lithography` (`fabricator.init`),
  `noisy-blackbox-optimization` (`init`, `pids_limit`, `read_only`, `cap_drop`) and `protein-active-learning`
  (`networks.task.internal: true`). The Harbor route rejects these settings and does not apply the resources
  server's per-task Compose overrides or ignored settings. `gym eval run` and `gym dataset validate` refuse to start
  while these tasks are left out, so pass
  `--exclude-tasks inverse-lithography,noisy-blackbox-optimization,protein-active-learning` (or `--skip-unloadable`
  to `gym eval run`). The excluded tasks have no rollout or reward and are absent from the default score
  denominator; the result is a partial benchmark, not a full 70-task result.
- **Untested Compose tasks.** `longitudinal-clinical-agent` and `tamp-skill-planning` have no known Compose field
  rejection after image preparation, but neither has been run on this route. Their sidecars run as separate sandboxes.
- **Disk.** Without the benchmark's settings, a sandbox's disk limit is the package's `storage_mb` (at most 20 GiB);
  OpenSandbox evicts a sandbox that writes past its limit. The `terminal_bench_science` benchmark raises it to
  100 GiB, as the resources server does.

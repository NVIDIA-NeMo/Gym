# Terminal-Bench-Science on the Harbor task shape

The 70 Terminal-Bench-Science v0.1 tasks run as plain Harbor task folders through the generic `harbor` environment
server. The dataset comes from Harbor's package store, pinned by the same digest
`benchmarks/terminal_bench_science/manifest.json` records for the resources server's presets, so both paths run the
same packages; the pull records each task's content hash in `manifest.toml` and refuses a task folder that no longer
matches it. The `terminal_bench_science/miniswe` benchmark runs this entry with mini-SWE
(`gym eval run --benchmark terminal_bench_science/miniswe`; see `benchmarks/terminal_bench_science/README.md`).

```sh
gym dataset pull terminal_bench_science
gym dataset validate terminal_bench_science --sandbox opensandbox
gym eval run terminal_bench_science --agent mini_swe_agent/mini_swe_agent_sandboxed --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url>
```

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

- **Compose tasks.** `inverse-lithography`, `longitudinal-clinical-agent`, `noisy-blackbox-optimization`,
  `protein-active-learning` and `tamp-skill-planning` run sidecars. The Harbor server starts them as separate sandboxes and does not
  apply the resources server's per-task Compose settings (`compose_service_overrides`, `compose_ignored_settings`,
  capability probes). `protein-active-learning`'s non-root `submission` service reaches `assay` by name, which a
  non-root sandbox cannot add to `/etc/hosts`. These tasks have not been run on this path.
- **Disk.** A sandbox's disk limit is the package's `storage_mb`. The resources server raises the limit to 100 GiB
  (`sandbox_min_disk_gib`); OpenSandbox evicts a sandbox that writes past its limit.

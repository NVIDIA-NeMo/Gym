# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. Hub tasks run unchanged.

## What the server does

- `/seed_session`: checks the row's folder digest, starts the task image as a sandbox, creates the working
  directory, and returns a `SandboxAccess` the agent harness borrows.
- `/verify`: uploads `tests/` into the same sandbox, runs `bash /tests/test.sh`, and reads
  `/logs/verifier/reward.json` or `reward.txt`.
- `/close_session`: stops the sandbox.

Reward rule: when `test.sh` ran, the sample counts. A missing or invalid reward file scores 0 with a
`failure_kind`. Only a Gym-side failure (sandbox lost, transfer failed) sets `mask_sample`.

Supported today: single-step tasks with a prebuilt `docker_image` or a base-image-only Dockerfile
(`FROM` plus `WORKDIR`/`ENV`/`USER`/`LABEL`); shared verifier mode, and separate verifier mode when
`[verifier.environment]` names a prebuilt image (the agent's `/logs/artifacts` and `artifacts` entries are
copied into the verifier sandbox first, sidecar hooks and artifacts included). Compose environments start as
a sandbox group when a `compose-images.json` with the sidecar images' recorded OCI configuration sits next to
the task folders, written by `gym dataset fetch`; on OpenSandbox the provider block also needs
`networking.enabled: true` (with `loopback_forwarding`) and `runtime_requirements.shm_size_metadata_key`, as
`benchmarks/terminal_bench_4/resources.yaml` shows. `[environment.healthcheck]` is polled at seed.
Tasks that declare GPUs run on the `gpu_sandbox_provider` block when one is set. The task's
`[[environment.mcp_servers]]` and `skills_dir` are written to `/tmp/.nemo-gym/task.json` in the agent's
sandbox for harnesses that drive task MCP servers from inside it (mini-SWE does). Dockerfile builds and
multi-step tasks come in later milestones.

## Run

```bash
gym eval run harbor:hello-world --agent hermes_agent --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches the task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>-<tag>/`, checked against Gym's task digest and recorded in
`manifest.toml`. A local task folder, or a folder of task folders, works the same way.

## Where settings live

Four homes, one owner each. A setting sits on this server's config only if every dataset on the same
deployment wants the same value; a unit test pins that list.

| Home | File | Owner | Holds |
|---|---|---|---|
| Task | `task.toml` | task author | image, resources, timeouts, env, verifier, agent user (read-only once fetched) |
| Dataset | `dataset.toml`, `[gym]` table | dataset author | `[gym.defaults]` `timeout_multiplier`, `resource_multiplier`; `[gym.tasks."<id>".environment]` overrides in Harbor's own fields (`env` merges) |
| Deployment | the `sandbox:` provider block | operator | image rewrites and registry credentials (`images`), root setup commands (`setup`), networking, shared-memory labels |
| Run | run config | the run | provider selection, GPU routing, repeats, output paths, `sandbox_resources_override` for parity runs |

`gym dataset init <name>` writes a dataset in this shape, and `gym dataset validate <folder>` with no extra
config is the test that a dataset is self-contained.

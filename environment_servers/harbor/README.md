# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. The supported subset is listed below.

## What the server does

This is an environment server: one `/run` call owns one whole episode, and no resources server takes part.

- Seed: checks the row's digest against the taskset mapping, snapshots and checks the task files once per
  digest, starts the task image as a sandbox through the configured provider, and creates the working
  directory. Concurrent episodes share the snapshot. Later source edits cannot change an already cached task;
  re-materialize and restart the server to use edited tasks.
- Agent: opens a session on the configured agent server with a `SandboxAccess` for that sandbox (the same
  seed request every sandboxed harness already accepts), posts the task's `responses_create_params` to it,
  then closes the session and keeps the harness's observations for the result.
- Verify: clears `/tests` and `/logs/verifier`, uploads the snapshot's `tests/`, runs `bash /tests/test.sh`
  in the same sandbox, and downloads the verifier logs.
- Cleanup: stops the sandbox after the episode, whatever happened before. Cancellation and the episode
  deadline also reach this cleanup; a failed stop keeps the sandbox owned by the server for a later retry
  or shutdown, and a redelivered episode cannot provision while an owned sandbox of the same episode remains.

A task that cannot be set up is a handled episode failure with `stage: seed`: terminal when the task or row
is at fault (unknown task, changed folder, unsupported settings), retryable when the sandbox provider failed.
A harness that cannot be seeded, fails its turn or fails to close is `stage: agent` or `stage: cleanup`, with
its observations and any usable response attached. Verification never runs before the harness has let go.

When `test.sh` ran, the sample counts. A missing or invalid reward file scores 0 with a `failure_kind`.
Only a required infrastructure failure (sandbox lost, reward transfer failed) sets `mask_sample`.
Diagnostic logs are best effort: unsafe archive entries are skipped securely, and a diagnostic
transfer failure cannot discard an already measured reward. Verifier timeouts score zero and retain
available logs. An existing `reward.json`
takes precedence even if malformed; `reward.txt` is read only when JSON is absent. Boolean JSON rewards
are numeric, as in Harbor. Gym selects the `reward` key or the only key; multiple keys without `reward`
are an authoring error (a terminal `stage: verification` failure) because Gym requires one scalar score.

## Supported task behavior

Supported today: single-step tasks with a prebuilt `docker_image`, a pull-mode Dockerfile (`FROM` plus
`WORKDIR`/`ENV`/`USER`/`LABEL`) or an overlay-mode Dockerfile (the same plus `RUN` lines, which the server
runs in the pulled base image at seed, before the healthcheck and the agent, each with the `ENV`, `WORKDIR`
and `USER` in effect at that line; `ENV`, `WORKDIR` and `USER` resolve at prepare from the base image's
recorded configuration the way `docker build` resolves them, so preparing fails when the registry cannot
resolve the base image. A `RUN` line that exits non-zero or outruns `[environment].build_timeout_sec` is a
422 with the command and the tail of its output; a provider that cannot run one is a 503, and in a separate
verifier's sandbox either masks the sample with `harbor:overlay_failed`); shared verifier mode, and
separate verifier mode in `[verifier.environment].docker_image` or, without one, the task image (the agent's
`/logs/artifacts` and `artifacts` entries are copied into the verifier sandbox first, sidecar hooks and
artifacts included). Compose environments start as
a sandbox group when a `compose-images.json` with the sidecar images' recorded OCI configuration sits next to
the task folders, written by `gym dataset fetch`; on OpenSandbox the provider block also needs
`networking.enabled: true` (with `loopback_forwarding`) and `runtime_requirements.shm_size_metadata_key` (see
[Compose tasks on OpenSandbox](#compose-tasks-on-opensandbox)). `[environment.healthcheck]` is polled at seed.
Tasks that declare GPUs run on the `gpu_sandbox_provider` block when one is set. The task's
`[[environment.mcp_servers]]` and `skills_dir` are written to `/tmp/.nemo-gym/task.json` in the agent's
sandbox for harnesses that drive task MCP servers from inside it (mini-SWE does). Dockerfile builds and
multi-step tasks come in later milestones.

| Setting or feature | Behavior in this layer |
| --- | --- |
| Image | Prebuilt `environment.docker_image`, or a base-image-only Dockerfile (`FROM`, literal absolute `WORKDIR`, `ENV`, `LABEL`). Dockerfile builds are deferred. |
| Verification | Single-step verification, shared or separate. A separate verifier runs in `[verifier.environment].docker_image`, else in the task image, for the episode-sized sandbox lifetime; a verifier that needs a `tests/Dockerfile` build is rejected at seed. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values fail the episode before provisioning. Resolved values are not added to dataset rows, task snapshots, or the episode result. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | Public networking only. Restricted network modes, host allowlists and deny rules are rejected by the loader rather than silently ignored. |
| Artifacts and hooks | Verifier logs are downloaded. `[[verifier.collect]]` hooks run in the agent's sandbox in both modes (a non-zero exit warns, a hook the sandbox cannot run masks). In separate mode `/logs/artifacts` and the `artifacts` entries are copied into the verifier, each under its own staging root, with their `exclude` rules applied, replacing what the verifier image ships at those paths; a probe the sandbox cannot answer masks rather than counting as an absent artifact. Sidecar hooks and artifacts are not collected yet. |
| Image ENTRYPOINT | No Harbor-compatible entrypoint lifecycle is guaranteed; startup follows the selected Gym sandbox provider. Tasks must not rely on an entrypoint running. |
| Other features | Multi-step tasks, in-sandbox MCP tools, Windows and TPU environments are unsupported. |

Clearing `/tests` and `/logs/verifier` before verification is a deliberate difference from Harbor's
single-step shared mode, which merges tests. It prevents agent-planted reward files or test helpers
from being used. Linked verification roots (including `/logs`) are unlinked before preparation,
without deleting their former targets. Helpers baked into `/tests` in an image are also removed; ship required helpers in
the task's `tests/` directory instead. Preparation and transfers attempt root, with a fallback to the
default image user when switching identity is unavailable. Target directory ownership and modes are
preserved during extraction.

## Compose tasks on OpenSandbox

The shipped `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml` (what `--sandbox opensandbox`
loads) has no networking block, and the Compose adapter refuses to start a group without one. Put an overlay
like this one after it in `--config` (or in a `+config_paths` entry). The three blocks are the deployment's:
service-to-service networking, the label the cluster reads a service's `shm_size` from, and the capability
labels and probes a task that needs `SYS_PTRACE` is scheduled with. The optional second provider block shows
how `gpu_sandbox_provider` routes the tasks that declare GPUs to a GPU pool while the rest stay on `sandbox`.

```yaml
sandbox:
  opensandbox:
    networking:
      enabled: true
      loopback_forwarding: true
      # Runs in every container the provider creates, sidecars included: it must no-op where it does not apply.
      setup_command: "command -v python3 >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1 || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3)"
    runtime_requirements:
      shm_size_metadata_key: nemo.nvidia.com/shm
      capability_metadata:
        SYS_PTRACE:
          nemo.nvidia.com/ptrace: "true"
      capability_probes:
        SYS_PTRACE: "awk '/^CapEff:/ { print $2 }' /proc/self/status | { read caps; test $((0x$caps & (1 << 19))) -ne 0; }"

# Optional: a second provider block for GPU tasks, named by the server's `gpu_sandbox_provider`.
sandbox_gpu:
  opensandbox: ${sandbox.opensandbox}

harbor_environment_server:
  environment_servers:
    harbor:
      gpu_sandbox_provider: sandbox_gpu
```

## Configuration and artifacts

See [`configs/harbor.yaml`](configs/harbor.yaml) for this layer's supported settings. Verifier logs default
to `${results_dir}/harbor/verifier/<generated-directory>`, using Gym's absolute results root. An explicit
relative `artifacts_dir` is relative to that root; an absolute path is used directly. The server reserves
a fresh directory for each verification; episode identifiers are never used as host paths.
New directories honor the server process's umask and inherited directory permissions, including group
access for shared results. Local log-directory creation or containment failures exclude the sample
with `failure_kind: persistence_failed` and identify the affected path.
Measured outcomes include an absolute `verifier_logs_dir` naming the collection location; that directory
can be empty if optional log collection fails.

Hermes honors the task's `[agent].timeout_sec` via the materialized `agent_timeout_sec` metadata,
using the smaller of that deadline and its configured runner limit. The verifier uses
`[verifier].timeout_sec`. Sandbox lifetime covers the episode budget (`default_episode_timeout_seconds`),
the final cleanup budget (`cleanup_timeout_seconds`) and `sandbox_ttl_slack_s`, so setup and queued agent
work also fit; both budgets must be finite. Sandboxes are normally stopped as soon as the episode ends.

Resource settings come from the task. `sandbox_resources_override`, `gpu_sandbox_provider`,
`request_gpu_type` and `verifier_grace_s` belong to later stack layers and are not configuration options
in this PR.

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>/` (`latest`), `<name>-<tag>/` or `<name>-<12 digest chars>/` (a digest pin), checked against Gym's task digest and recorded in
`manifest.toml`. A local task folder, or a folder of task folders, works the same way. The generated run config declares the materialized rows as the agent's
validation dataset and binds the per-taskset environment server to that agent. Running model episodes
requires a harness that accepts `sandbox_access` in its session seed and leaves the sandbox to the server
when its session is closed.

## Example data

`data/tasks/` contains five small Apache-2.0 terminal tasks: copying a file, numeric sorting,
deduplicating lines, uppercasing text, and selecting the first line. Each uses `ubuntu:24.04`, a
60-second agent limit, and a 30-second verifier limit. The verifier compares the output byte for
byte with a fixture held outside the agent workdir.

`data/example.jsonl` contains their materialized rows. `data/example_rollouts.jsonl` records actual
Qwen2.5-0.5B-Instruct commands, execution results, and verifier outputs; `data/example_metrics.json`
records their measured scores and runtime. These are smoke examples from a single-command harness,
not a Hermes benchmark. They used llama.cpp on CPU and Enroot sandboxes with an in-process handoff
because Enroot cannot serialize sandbox access. Provider expiration and the distributed handoff
were not exercised by these examples.

To run the bundled tasks with Hermes and a configured sandbox provider and model:

```bash
gym eval run environment_servers/harbor/data/tasks --agent hermes_agent --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The loader generates the task mapping and dataset rows for that run.

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

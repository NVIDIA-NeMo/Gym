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

| Setting or feature | Behavior in this layer |
| --- | --- |
| Image | Prebuilt `environment.docker_image`, or a base-image-only Dockerfile (`FROM`, literal absolute `WORKDIR`, `ENV`, `LABEL`). Dockerfile builds are deferred. |
| Verification | Single-step, shared-sandbox verification. Explicit `verifier.environment_mode = "separate"` or a separate verifier environment is rejected at seed. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values fail the episode before provisioning. Resolved values are not added to dataset rows, task snapshots, or the episode result. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | Public networking only. Restricted network modes, host allowlists and deny rules are rejected by the loader rather than silently ignored. |
| Artifacts and hooks | Verifier logs are downloaded. Artifact `exclude` rules and shared-mode collect hooks are not implemented in this layer. |
| Image ENTRYPOINT | No Harbor-compatible entrypoint lifecycle is guaranteed; startup follows the selected Gym sandbox provider. Tasks must not rely on an entrypoint running. |
| Other features | Multi-step tasks, in-sandbox MCP tools, Windows and TPU environments are unsupported. |

Clearing `/tests` and `/logs/verifier` before verification is a deliberate difference from Harbor's
single-step shared mode, which merges tests. It prevents agent-planted reward files or test helpers
from being used. Linked verification roots (including `/logs`) are unlinked before preparation,
without deleting their former targets. Helpers baked into `/tests` in an image are also removed; ship required helpers in
the task's `tests/` directory instead. Preparation and transfers attempt root, with a fallback to the
default image user when switching identity is unavailable. Target directory ownership and modes are
preserved during extraction.

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
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). A local task folder, or a folder of task
folders, works the same way. The generated run config declares the materialized rows as the agent's
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

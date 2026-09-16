# Harbor Training

This agent runs a Harbor 0.23.0 task as a Gym Responses API agent. Harbor's
OpenCode model traffic is sent to the configured Gym model server, including
Gym's per-rollout correlation prefix when token capture is enabled.

Set `HARBOR_DATASET_PATH` to a materialized Harbor dataset. The base
configuration uses a Docker-capable worker; the OpenSandbox overlay replaces
that environment. Dataset manifests are external inputs, not packaged data.

## Configuration

Training and evaluation share one `harbor_agent_general` entry point and the
typed ATIF converter from [Gym PR #2596](https://github.com/NVIDIA-NeMo/Gym/pull/2596).
Use `harbor_jobs_dir`, `harbor_dataset`, `harbor_agent`, `harbor_environment`,
`harbor_verifier`, and `harbor_ray_task_num_cpus` for the Harbor configuration.

Set `model_server` and `model_api_key` to route policy calls through Gym, and
enable `token_id_capture`. These are optional for direct-provider evaluation.
Policy alerts and filesystem auditing are independently optional.
Verifier behavior is configured through Harbor's verifier model.

The OpenCode wrappers declare Harbor's typed `options_model`, including their
title, preinstalled-image, alert, process-limit and audit options. Invalid kwargs
fail Harbor's preflight before a trial starts. The agent package pins Harbor
because its separate Gym server environment resolves dependencies independently
of the training framework's lockfile.

`harbor_debug` and `harbor_max_retries` control Harbor job execution.
A nonempty reward dictionary must contain `harbor_reward_key` (default `reward`).
Multi-step tasks retain all ATIF trajectories and use Harbor's configured
trial-level reward aggregation.

ATIF is the human-readable transcript, not a substitute for captured inference
inputs. Training consumes independent exact-call token records, preserving
retokenization boundaries and subagent calls. `atif_conversion` reports any
lossy transcript conversions separately from token-capture coverage.

### Migrating Existing Training Configurations

Use `responses_api_agents/harbor_agent_general` for config paths and Python
import paths. The previous training integration's unprefixed `jobs_dir`,
`dataset`, `agent`, `environment`, `verifier`, and `ray_task_num_cpus` fields
become the corresponding `harbor_*` fields above. Do not rename fields nested
inside the Harbor models themselves.

External Gym JSONL manifests must also select the registered agent:

```json
{"agent_ref":{"type":"responses_api_agents","name":"harbor_agent_general"},"task_name":"example-task","responses_create_params":{"input":[]}}
```

Changing the config alone does not rewrite existing `agent_ref` values. Update
those references when preparing your manifest, preserving task payloads and
any dataset exclusions. Dataset contents and local credentials are not part of
this package.

## OpenSandbox

Add these config paths to run any prebuilt-image Harbor dataset through Gym's
OpenSandbox provider:

```yaml
config_paths:
  - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
  - responses_api_agents/harbor_agent_general/configs/harbor_agent_opencode_compaction.yaml
  - nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
  - responses_api_agents/harbor_agent_general/configs/harbor_agent.yaml
  - responses_api_agents/harbor_agent_general/configs/harbor_agent_opensandbox.yaml
```

The OpenCode compaction overlay preserves vLLM's HTTP context-overflow error
instead of translating it into an empty `finish_reason=length` completion.
OpenCode uses that typed error to compact the session and retry; ordinary
nonempty output-length truncation is unchanged.

Set `HARBOR_DATASET_PATH` to the Harbor task root and configure
`OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the target service. The
OpenSandbox layer requests 0.25 CPU and 512 MiB per sandbox, with burstable
limits of 4 CPU and 64 GiB. The OpenSandbox runtime must be authorized to pull
the prebuilt images referenced by the task configs. `HARBOR_BENCHMARK_NAME`
optionally sets the per-sandbox benchmark metadata label.

`OPENSANDBOX_PROTOCOL` defaults to `http`, and
`OPENSANDBOX_USE_SERVER_PROXY` defaults to `true`. Supply credentials through
your job's secret management; never put resolved keys in committed YAML files.

OpenSandbox volume and sandbox-copy string values support
`{context_id}`, `{environment_name}`, `{task_name}`, `{task_id}`, and
`{session_id}`. `{context_id}` is Harbor's trial UUID and is shared by the
policy and separate-verifier environments. `{task_name}` aliases Harbor's
`environment_name`. Unknown placeholders are rejected during Gym agent
configuration and again before the provider create call; they are never passed
through literally.

Before increasing rollout concurrency, create one sandbox using the same
image, entrypoint, provider metadata, and rendered volumes as the intended
training run. Confirm the task-specific mount exists and is readable inside
the sandbox, then scale through a small batch before using the full queue.

The direct EFS workspace path is
`<EFS host>/<owner>/nemo-gym-harbor-artifacts/<run>/{context_id}`. The policy
mounts that physical source read-write at `/app`; the separate verifier mounts
the exact same source read-only at `/app`. Task data is an independent
task-specific S3 volume mounted read-only at `/data` in both roles. The policy
creates `/app/data -> /data`; the verifier validates the inherited link against
its independent clean-data mount without modifying read-only `/app`. Harbor
passes only an integrity marker between roles, so policy files are never
downloaded, uploaded, archived, or reconstructed. After verifier termination,
a short-lived cleanup sandbox mounts only the run parent and removes the exact
context-ID directory.

## Test

```bash
uv sync --project responses_api_agents/harbor_agent_general
uv run --project responses_api_agents/harbor_agent_general \
  pytest responses_api_agents/harbor_agent_general/tests
```

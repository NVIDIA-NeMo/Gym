# anyswe_agent

AnySWE runs a Gym agent inside a SWE task image, extracts its repository patch, and
grades the patch in a fresh sandbox. The agent and grader use the shared Gym
sandbox API, so Docker, OpenSandbox, and other registered providers use the same
execution path.

Supported datasets are SWE-bench, SWE-bench Multilingual, and R2E-Gym. This can
be extended to other datasets, for example by refering `swe_agents`.

## Quickstart

Create `env.yaml` for the model:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Set the task-image format to images containing the AnySWE runtime:

```bash
export ANYSWE_CONTAINER_FORMATTER='registry.example.com/anyswe/swebench:{instance_id}'
```

Prepare five rows. The sandbox provider pulls the task images.

```bash
python3 responses_api_agents/anyswe_agent/prepare.py --limit 5

gym env start \
  --config responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent anyswe_hermes \
  --input responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
  --output results/anyswe_rollouts.jsonl \
  --limit 5
```

The quickstart uses OpenSandbox. Set `OPENSANDBOX_API_KEY`,
`OPENSANDBOX_DOMAIN`, and `NEMO_GYM_SANDBOX_MODEL_BASE_URL` before starting the
environment. The model URL must be reachable from the task sandbox.

## Agent configuration

Any compatible Gym agent can be selected without changing the SWE environment:

```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs:
  max_turns: 100
  terminal_backend: local
```

By default, each task image provides `/agent_deps_mount/bin/python` with NeMo
Gym and the configured agent module importable. This baked path is recommended
for large evaluation waves.

For development, set `upload_agent_runtime: true`. AnySWE builds the portable
runtime with `setup_scripts/<agent>_deps.sh` and uploads it to each sandbox.
This is convenient for smoke tests but expensive at scale. A future shared
runtime mechanism can be added independently once its interface is settled.

The included Hermes, Claude Code, OpenClaw, and OpenCode configurations share
the same sandbox and grading path.

## Sandbox providers

The agent references a provider-independent `sandbox` block:

```yaml
sandbox_provider: sandbox
sandbox_spec:
  resources:
    cpu: 2
    memory_mib: 4096
```

The quickstart defines that block with the shipped OpenSandbox config. To use
local Docker instead, replace the OpenSandbox config path with
`nemo_gym/sandbox/providers/docker/configs/docker.yaml`. Both providers use the
same `container_formatter` and `sandbox_spec` fields.

An equivalent inline OpenSandbox configuration is:

```yaml
sandbox_model_base_url: ${oc.env:NEMO_GYM_SANDBOX_MODEL_BASE_URL}
sandbox_provider:
  opensandbox:
    connection:
      domain: ${oc.env:OPENSANDBOX_DOMAIN}
      protocol: http
      api_key: ${oc.env:OPENSANDBOX_API_KEY}
      use_server_proxy: true
    create:
      timeout_s: 1200
    operations:
      command_retries: 5
sandbox_spec:
  ttl_s: 7200
  ready_timeout_s: 1200
  resources:
    cpu: 2
    memory_mib: 4096
```

Private images can pass `image_auth` through
`sandbox_spec.provider_options` when the OpenSandbox provider includes registry
authentication support.

## Data

`prepare.py` writes `data/swebench_verified.jsonl`. Task images are resolved by
`container_formatter` when each sandbox starts.

## Reward and masking

SWE-bench and SWE-bench Multilingual use the official `make_test_spec` and
`get_eval_report` path. R2E-Gym requires every fail-to-pass and pass-to-pass test
to report `PASSED`.

Resolved patches receive reward `1`; other completed attempts receive `0`.
Agent timeouts, evaluation timeouts, sandbox failures, and accidental successes
after max-iteration or context-window termination set `mask_sample=true`.

Each rollout includes the trajectory, patch, reward, grading fields, and
`mask_sample`.

# anyswe_agent

AnySWE runs a Gym agent inside a SWE task image, extracts its repository patch, and
grades the patch in a fresh sandbox. The agent and grader use the shared Gym
sandbox API, so Docker, OpenSandbox, and other registered providers use the same
execution path.

Supported datasets are SWE-bench, SWE-bench Multilingual, and R2E-Gym.

## Quickstart

Create `env.yaml` for the policy model:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Prepare five rows. The sandbox provider pulls the task images.

```bash
python3 responses_api_agents/anyswe_agent/prepare.py --limit 5

gym env start \
  --config responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent anyswe_hermes \
  --input responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
  --output results/anyswe_rollouts.jsonl \
  --limit 5
```

The shipped agent configs use Docker with host networking by default. For a
remote provider, set `NEMO_GYM_SANDBOX_MODEL_BASE_URL` to a policy endpoint
reachable from the task sandbox.

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

`runtime_delivery: upload` installs Gym and the agent dependencies once, then
uploads that portable environment to each sandbox. `runtime_delivery: preloaded`
expects the image to provide `/agent_deps_mount`.

Add `setup_scripts/<agent_dir>_deps.sh` when using upload delivery with a new
agent.

The included Hermes, Claude Code, OpenClaw, and OpenCode configurations share
the same sandbox and grading path.

## Sandbox providers

Docker is the local default:

```yaml
sandbox_provider:
  docker:
    network: host
sandbox_spec:
  resources:
    cpu: 2
    memory_mib: 4096
```

OpenSandbox uses the same `container_formatter` and `sandbox_spec` fields:

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

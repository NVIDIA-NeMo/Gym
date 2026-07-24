# anyswe_agent

AnySWE runs a Gym agent inside a SWE task image, extracts its repository patch,
and grades the patch in a fresh OpenSandbox environment. It supports SWE-bench,
SWE-bench Multilingual, and R2E-Gym.

## Run

Create `env.yaml` for the policy model:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Set the OpenSandbox credentials, the model URL reachable from its sandboxes, and
the task-image format:

```bash
export OPENSANDBOX_API_KEY=...
export OPENSANDBOX_DOMAIN=...
export NEMO_GYM_SANDBOX_MODEL_BASE_URL=...
export ANYSWE_CONTAINER_FORMATTER='registry.example.com/anyswe/swebench:{instance_id}'
```

Prepare the dataset, start the environment, and collect rollouts:

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

`prepare.py` writes `data/swebench_verified.jsonl`. Each row must resolve to a
task image through its `image` field or `container_formatter`.

## Agents

The included configurations run Hermes Agent, Claude Code, Pi, OpenClaw, or OpenCode with
the same sandbox and grading path. Configure another Gym agent with:

```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs:
  max_turns: 100
  terminal_backend: local
```

For large runs, bake `/agent_deps_mount/bin/python`, NeMo Gym, and the selected
agent into each task image. `agent_runtime_source` controls other delivery modes:

- `baked` uses the runtime in the task image and is the default.
- `auto` builds the portable runtime with `setup_scripts/<agent>_deps.sh` once
  and uploads the resulting archive to each sandbox.
- A local tarball path uploads a prebuilt runtime.
- An HTTP(S) URL downloads a prebuilt runtime inside the sandbox.

Each archive must unpack the runtime directly under `/agent_deps_mount`.

Private task images can pass registry credentials through
`sandbox_spec.provider_options.image_auth`.

## Reward

SWE-bench and SWE-bench Multilingual use the official `make_test_spec` and
`get_eval_report` path. R2E-Gym requires every fail-to-pass and pass-to-pass test
to report `PASSED`.

Resolved patches receive reward `1`; other completed attempts receive `0`.
Agent timeouts, evaluation timeouts, sandbox failures, and accidental successes
after max-iteration or context-window termination set `mask_sample=true`.

Each rollout includes the trajectory, patch, reward, grading fields, and
`mask_sample`.

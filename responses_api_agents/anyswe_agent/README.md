# anyswe_agent

Runs any Gym agent inside a SWE-style task container and evaluates the resulting
`git diff HEAD` patch with the dataset harness. Works with `hermes_agent`,
`claude_code_agent`, or another compatible Gym agent.

# Quickstart

From the repo root, create `env.yaml` for the policy model server:

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Prepare a 5 examples (tasks+images):

```bash
python responses_api_agents/anyswe_agent/prepare.py --limit 5
```

Start the environment:

```bash
ng_run "+config_paths=[responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  ++anyswe_hermes.responses_api_agents.anyswe_agent.container_formatter='responses_api_agents/anyswe_agent/data/sifs/{instance_id}.sif'
```

Collect rollouts:

```bash
ng_collect_rollouts \
  +agent_name=anyswe_hermes \
  +input_jsonl_fpath=responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
  +output_jsonl_fpath=results/anyswe_rollouts.jsonl \
  +limit=5
```

Each rollout row contains `reward`, the full trajectory, and `mask_sample` for
timeouts or unreliable rewards.

# Agent wiring

Point the config at the Gym agent server:

```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs: {max_turns: 100, terminal_backend: local}
```

Agent dependencies install once at startup into a portable prefix mounted inside
the task container. Add `setup_scripts/<agent_dir>_deps.sh` for new agents.


# Dataset and images

`prepare.py` writes `data/swebench_verified.jsonl` and builds
`data/sifs/{instance_id}.sif`.

```bash
python responses_api_agents/anyswe_agent/prepare.py
```

Image builds require `apptainer`, network access to the SWE-bench registry, and
substantial disk space. Use `--limit` and `--jobs N` while iterating. Dataset
prep requires `pip install datasets`.

Supported datasets: SWE-bench, SWE-bench Multilingual, R2E-Gym.

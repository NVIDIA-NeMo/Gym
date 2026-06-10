# anyswe_agent

Agent-agnostic SWE-bench runner. Runs SWE-bench (plus SWE-bench Multilingual and
R2E-Gym) with any NeMo-Gym agent server: `hermes_agent`, `claude_code_agent`,
or your own.

The chosen agent runs natively inside each task's SWEBench Apptainer container
(using its own tools on `/testbed`); the patch is always `git diff HEAD`, and the
eval harness runs concurrently in a second container.

# How agents are wired

Point the config at any Gym agent server: no `anyswe_agent` code changes:

```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs: {max_turns: 100, terminal_backend: local}
```

Each agent's dependencies install once at startup into a relocatable prefix
(portable CPython + the agent package/CLI) mounted read-only into the container:

- `setup_scripts/hermes_agent_deps.sh`
- `setup_scripts/claude_code_agent_deps.sh`

To add a new agent: add a `setup_scripts/<agent_dir>_deps.sh` and a config.

# Dataset & image prep

One command builds both the dataset JSONL and the task images:

```bash
# 5 instances + their 5 SIFs: fast smoke test
python responses_api_agents/anyswe_agent/prepare.py --limit 5

# full SWE-bench Verified (500 tasks) + all SIFs: hundreds of GB, slow
python responses_api_agents/anyswe_agent/prepare.py

# dataset only (no images), or a single instance
python responses_api_agents/anyswe_agent/prepare.py --no-images
python responses_api_agents/anyswe_agent/prepare.py --instance-id django__django-13741
```

This writes `data/swebench_verified.jsonl` and builds `data/sifs/{instance_id}.sif`,
so the agent's `container_formatter` is just `data/sifs/{instance_id}.sif`.

Prerequisites for images: `apptainer` on PATH + network access to the SWE-bench
registry. Each SIF is multiple GB. Use `--limit` while iterating, `--jobs N` to
parallelize builds. Dataset prep needs `pip install datasets`.

# Rollout collection

## Create env.yaml

```
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Launch nemo gym servers

```bash
ng_run "+config_paths=[responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  ++anyswe_hermes.responses_api_agents.anyswe_agent.container_formatter='responses_api_agents/anyswe_agent/data/sifs/{instance_id}.sif'
```

Swap to claude-code with `configs/anyswe_claude_code.yaml` (needs a model server
exposing the Anthropic Messages API).

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=anyswe_hermes \
    +input_jsonl_fpath=responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
    +output_jsonl_fpath=results/anyswe_rollouts.jsonl \
    +limit=5
```

Each rollout row carries `reward` (1.0 if resolved), the full trajectory, and
`mask_sample` (set when the reward is unreliable: agent/eval timeout).

# Config notes

| Field | Meaning |
|-------|---------|
| `agent_server_module` / `agent_server_class` / `agent_config_class` | Which Gym agent to run in-container |
| `agent_kwargs` | Extra kwargs forwarded to the agent's config |
| `container_formatter` | Apptainer SIF path template, `{instance_id}` substituted |
| `skip_eval` | Run the agent but skip grading (collect trajectories only) |

Reward is `1.0` iff the eval harness reports `resolved`. Supported datasets:
SWE-bench, SWE-bench Multilingual, R2E-Gym.

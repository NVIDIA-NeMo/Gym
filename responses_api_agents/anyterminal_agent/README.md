# anyterminal_agent

Runs any Gym agent inside a Terminal Bench task container and evaluates the result
by running the task's `tests/test.sh` in the same container. Shipped profiles cover
Hermes, OpenClaw, OpenCode, Pi, Claude Code, NeMo Fabric, and Terminus 2.

Unlike `anyswe_agent` (which runs agent and eval in two concurrent containers),
anyterminal runs everything sequentially in one container: agent finishes, then
`test.sh` runs and writes a reward to `/logs/verifier/reward.txt`. The test
directory is mounted read-only so the agent cannot tamper with the tests before
they run.

## Multi-harness training

`configs/anyterminal_multi_harness.yaml` owns one Terminal-Bench dataset under the
neutral source route `anyterminal_multi_harness`. Use `fan_out` to run every source
task through all P0 harnesses in this order:

1. OpenCode
2. OpenClaw
3. Pi
4. Hermes

Each task/harness pair is independent. In GRPO, all sibling generations for one
pair stay on the same harness, so group-relative advantages never mix harnesses.
Use `agent_pool` instead only when the desired behavior is selecting one harness
per source task.

Prepare four real Terminal-Bench 2.1 tasks from an existing checkout and collate them:

```bash
python responses_api_agents/anyterminal_agent/prepare.py \
  --tasks-cache /path/to/terminal-bench-2-1 \
  --dataset-name tasks \
  --task-name fix-git log-summary-date-ranges configure-git-webserver modernize-scientific-stack

gym dataset collate \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml \
  --output-dir data/anyterminal_multi_harness \
  --mode train_preparation
```

Start all four harnesses and collect one rollout per task and harness. No `--agent`
is needed; `fan_out` creates the cross-product:

```bash
gym env start \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_multi_harness.yaml \
  '+fan_out={anyterminal_multi_harness:[anyterminal_opencode,anyterminal_openclaw,anyterminal_pi,anyterminal_hermes]}' \
  --input data/anyterminal_multi_harness/train.jsonl \
  --output results/anyterminal_multi_harness.jsonl
```

For an Enroot cluster, use `configs/anyterminal_multi_harness_enroot.yaml` for both
commands. The host needs Enroot, and compute nodes need registry access for the task
images. The same profile can be placed in NeMo RL's `env.nemo_gym.config_paths`; use
`env.nemo_gym.fan_out.anyterminal_multi_harness` with the four target names and
`token_capture.enabled: true` for policy training with external harnesses.

## Prerequisites

Every task runs inside the configured Gym sandbox provider. The base profiles default
to Docker. The multi-harness cluster profile selects the built-in Enroot provider.
Apptainer is also supported through an inline provider config.

```bash
apt-get update && apt-get install -y wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.2/apptainer_1.4.2_amd64.deb
apt-get install -y ./apptainer_1.4.2_amd64.deb
apptainer --version
```

## Quickstart

**1. Prepare the dataset** — downloads tasks via Harbor and writes the input JSONL:

```bash
# Download tasks + build dataset + build SIFs (default)
python responses_api_agents/anyterminal_agent/prepare.py

# Skip SIF builds — Apptainer will pull docker:// images at runtime
python responses_api_agents/anyterminal_agent/prepare.py --no-build-sif

# Build SIFs into a custom directory
python responses_api_agents/anyterminal_agent/prepare.py --sif-dir /shared/sifs

# Smoke test — first 5 tasks only
python responses_api_agents/anyterminal_agent/prepare.py --limit 5 --no-build-sif
```

Requires the `harbor` CLI on PATH. Tasks are downloaded automatically and cached at
`~/.cache/harbor/tasks/terminal-bench/`; subsequent runs skip the download.

**2. Start the environment** with Hermes and a model server:

```bash
gym env start \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_hermes.yaml \
  --model-type vllm_model
```

If you pre-built SIFs into a custom directory, override `container_formatter`:

```bash
gym env start --config ... \
  ++anyterminal_hermes.responses_api_agents.anyterminal_agent.container_formatter=/shared/sifs/{task_name}.sif
```

**3. Collect rollouts:**

```bash
gym eval run --no-serve \
  --agent anyterminal_hermes \
  --input responses_api_agents/anyterminal_agent/data/terminal_bench.jsonl \
  --output results/anyterminal_rollouts.jsonl
```

Each rollout row contains `reward` (0.0 or 1.0), the full agent trajectory, and
`mask_sample` (set when a timeout made the reward unreliable).

## Agent wiring

Swap the agent by changing three fields in the YAML (or overriding on the CLI):


```yaml
agent_server_module: responses_api_agents.hermes_agent.app
agent_server_class: HermesAgent
agent_config_class: HermesAgentConfig
agent_kwargs:
  max_turns: 30
  terminal_backend: local
```

Agent dependencies are installed once at startup into a portable Python prefix
mounted read-only inside the task container at `/agent_deps_mount`. To support a
new agent, add `responses_api_agents/<agent_dir>/scripts/<agent_dir>_deps.sh` (see
`responses_api_agents/hermes_agent/scripts/hermes_agent_deps.sh` for the pattern).

## Container images

Each Terminal Bench task specifies a Docker image in its `task.toml`. You can
either:

- **Pull at runtime** (default): Docker, Enroot, or Apptainer imports the task's
  `docker://<image>`. This requires registry access on compute nodes.
- **Pre-build SIFs** (`prepare.py --build-image --image-dir PATH`): Apptainer
  converts each image to a `.sif` file. Point `container_formatter` at that
  directory for faster or air-gapped runs.

## Key config options

| Field | Default | Description |
|---|---|---|
| `container_formatter` | `docker://{docker_image}` | Runtime image or pre-built SIF path template |
| `sandbox_provider` | `{docker: {}}` | Inline provider config or named provider block |
| `agent_runtime_source` | `auto` | Build runtime, use a baked runtime, URL, or archive |
| `tb_agent_timeout` | `1800` | Seconds before the agent is killed |
| `tb_eval_timeout` | `300` | Seconds for `test.sh` to complete |
| `concurrency` | `256` | Max concurrent tasks dispatched to Ray |

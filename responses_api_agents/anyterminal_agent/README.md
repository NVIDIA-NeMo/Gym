# anyterminal_agent

Runs any Gym agent inside a Terminal Bench task container and evaluates the result
by running the task's `tests/test.sh` in the same container. Works with
`hermes_agent`, `claude_code_agent`, or any other compatible Gym agent.

Unlike `anyswe_agent` (which runs agent and eval in two concurrent containers),
anyterminal runs everything sequentially in one container: agent finishes, then
`test.sh` runs and writes a reward to `/logs/verifier/reward.txt`. The test
directory is mounted read-only so the agent cannot tamper with the tests before
they run.

## Prerequisites

Every task runs inside an [Apptainer](https://apptainer.org/) (formerly Singularity) container,
so Apptainer must be installed on each machine that runs rollouts. It is not bundled with Gym.

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

If you pre-built SIFs into a custom directory, override `tb_sif_dir`:

```bash
gym env start --config ... \
  ++anyterminal_hermes.responses_api_agents.anyterminal_agent.tb_sif_dir=/shared/sifs
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
new agent, add `setup_scripts/<agent_dir>_deps.sh` (see `hermes_agent_deps.sh`
for the pattern).

## Container images

Each Terminal Bench task specifies a Docker image in its `task.toml`. You can
either:

- **Pull at runtime** (default, `tb_sif_dir: null`): Apptainer pulls
  `docker://<image>` on first use. Requires internet access on compute nodes.
- **Pre-build SIFs** (`prepare.py --sif-dir PATH`): Converts each image to a
  `.sif` file. Faster and works on air-gapped clusters.

## Key config options

| Field | Default | Description |
|---|---|---|
| `tb_tasks_cache_dir` | `~/.cache/harbor/tasks` | Where Harbor stores downloaded task definitions |
| `tb_sif_dir` | `null` | Pre-built SIF directory; `null` = pull docker:// at runtime |
| `tb_agent_timeout` | `1800` | Seconds before the agent is killed |
| `tb_eval_timeout` | `300` | Seconds for `test.sh` to complete |
| `apptainer_memory_limit_mb` | `32768` | Per-container memory cap via `ulimit -v` |
| `concurrency` | `256` | Max concurrent tasks dispatched to Ray |

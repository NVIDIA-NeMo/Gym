# SWE-bench ProRL Usage Guide

This document describes how to run SWE-bench tasks with ProRL-Agent-Server in NeMo-Gym.

Use the NeMo-Gym environment in every terminal:

```bash
source .venv/bin/activate
```

---

## Prerequisites

- ProRL-Agent-Server repository is cloned and available
- NeMo-Gym environment is set up (`uv venv && uv sync --extra dev`) [already configured]
- `env.yaml` is configured with model endpoint information

---

## Step 1: Start ProRL-Agent-Server + vLLM

In the first terminal, run:

```bash
LAUNCH_SCRIPT=/path/to/prorl_agent/scripts/tests/host_prorl_agent_on_single_node.sh
bash $LAUNCH_SCRIPT
```

This script starts ProRL-Agent-Server (listening on `:8006` by default) and the vLLM inference service.

---

## Step 2: Get Node IP and Update Config

```bash
NODE_IP=$(hostname --ip-address | awk '{print $1}')
echo $NODE_IP
```

Replace the output IP address in the following two files:

**`responses_api_agents/prorl_agent/configs/swebench_prorl.yaml`**

```yaml
prorl_url: http://<NODE_IP>:8006
```

**`env.yaml`**

```yaml
policy_base_url: http://<NODE_IP>:8100/v1
```

---

## Step 3: Start Gym prorl agent Service

Open a new terminal and run from the NeMo-Gym root directory:

```bash
export RAY_TMPDIR=/tmp

ng_run \
    "+config_paths=[responses_api_agents/prorl_agent/configs/swebench_prorl.yaml,responses_api_models/vllm_model/configs/vllm_model_chat_completion_api.yaml]"
```

> **Note:** `RAY_TMPDIR=/tmp` avoids Ray AF_UNIX socket path exceeding the 107-byte limit on Lustre long paths.
>
> When Gym prorl agent starts, it automatically calls ProRL-Agent-Server's `/start` and `/add_llm_server` endpoints to complete registration.

---

## Step 4: Run a Single Test

Open a new terminal.

### 4.1 Verify Service Health

```bash
ng_status
```

Ensure both `prorl agent` and `policy_model` show healthy.

### 4.2 Run a Single Sample

```bash
ng_collect_rollouts \
    +agent_name=prorl_agent \
    +limit=1 \
    +input_jsonl_fpath=data/example.jsonl \
    +output_jsonl_fpath=results/prorl_single.jsonl \
    +num_repeats=1 \
    "+responses_create_params={temperature: 0.0}"
```

Results are saved to `results/prorl_single.jsonl`; each record includes fields such as `reward` and `response.output_text`.

---

## Configuration Reference

| Parameter | Location | Description |
|-----------|----------|-------------|
| `prorl_url` | `swebench_prorl.yaml` | ProRL-Agent-Server URL; use the actual IP for multi-node setups |
| `concurrency` | `swebench_prorl.yaml` | Concurrency level; must not exceed ProRL Server's `--max-run-workers` |
| `swebench_agent_timeout` | `swebench_prorl.yaml` | Per-task timeout in seconds; default 2700 (45 minutes) |
| `policy_base_url` | `env.yaml` | vLLM inference service URL |
| `policy_model_name` | `env.yaml` | Model path or name |

---

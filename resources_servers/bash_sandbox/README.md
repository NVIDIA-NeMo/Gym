# Bash Sandbox Resources Server

`bash_sandbox` is a tool-execution resources server used by `gdpval_agent`. It provides a
Linux sandbox session per task and exposes tools like command execution, web search/fetch,
file upload, and `finish` file persistence.

For GDPVal, this server can run in two modes:
- **Passthrough mode (default):** used to collect committee-model outputs (precompute phase);
  verification returns a passthrough reward.
- **Judge mode (optional):** compares evaluated model outputs against precomputed committee
  model outputs and computes reward from majority verdicts.

## Phase 1 (required): collect committee outputs in passthrough mode

Before evaluating any checkpoint with judge mode, you should first run committee models and save
their outputs with `judge.enabled: false`.

1. Keep `judge.enabled: false` in `resources_servers/bash_sandbox/configs/bash_sandbox.yaml`.
2. Start servers with the GDPVal agent and a model server:

```bash
ng_run "+config_paths=[resources_servers/bash_sandbox/configs/bash_sandbox.yaml,responses_api_models/local_vllm_model/configs/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml]" \
  ++bash_sandbox_agent.responses_api_agents.gdpval_agent.model_server.name=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

3. Collect rollouts as usual:

```bash
ng_collect_rollouts \
  +agent_name=bash_sandbox_agent \
  +input_jsonl_fpath=responses_api_agents/gdpval_agent/data/train.jsonl \
  +output_jsonl_fpath=responses_api_agents/gdpval_agent/data/train_rollouts.jsonl
```

## Phase 2: evaluate checkpoints with judge enabled

Judge mode requires a 2-phase workflow:

1. **Precompute committee outputs** over the target dataset.
2. **Enable judge** and point it at those output directories.

### Phase 1: precompute committee outputs

#### Step 1: prepare JSONL tasks

Use the GDPVal client helper:

```bash
python responses_api_agents/gdpval_agent/client.py prepare \
  --output-jsonl /path/to/committee_tasks.jsonl \
  --split train \
  --output-dir /path/to/committee_outputs/MyCommitteeModel \
  --validate
```

Optional filters:
- `--limit N`
- `--task-ids id1,id2,...`

`--output-dir` must match the `judge.committee_models[].output_dir` value used in Phase 2.

#### Step 2: run committee model with judge disabled

Run with `judge.enabled: false`, but point `gdpval_agent.model_server.name` at the committee
model server instance:

```bash
ng_run "+config_paths=[resources_servers/bash_sandbox/configs/bash_sandbox.yaml,responses_api_models/local_vllm_model/configs/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml]" \
  ++bash_sandbox_agent.responses_api_agents.gdpval_agent.model_server.name=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  ++bash_sandbox_resources_server.resources_servers.bash_sandbox.judge.enabled=false
```

Then generate committee rollouts:

```bash
ng_collect_rollouts \
  +agent_name=bash_sandbox_agent \
  +input_jsonl_fpath=/path/to/committee_tasks.jsonl \
  +output_jsonl_fpath=/path/to/committee_rollouts.jsonl
```

Each task creates:

```text
/path/to/committee_outputs/MyCommitteeModel/
  task_<task_id>/
    finish_params.json
    <output files...>
    <office-output>.pdf
    reference_files/
```

Important finish behavior:
- `finish_params.json` is used as the "committee attempted this task" sentinel.
- Office files (`.docx`, `.pptx`, `.xlsx`) are converted to PDF siblings for judging.

### Enable judge and evaluate checkpoint outputs

At this point, committee outputs should already exist on disk (from Phase 1). Judge mode is for
checkpoint evaluation against those precomputed outputs, not for generating committee outputs.

Configure `resources_servers/bash_sandbox/configs/bash_sandbox.yaml`:

```yaml
bash_sandbox_resources_server:
  resources_servers:
    bash_sandbox:
      judge:
        enabled: true
        judge_model_name: gemini-3-pro-preview
        gcp_project_id: your-gcp-project
        gcp_location: global
        thinking_budget: 5000
        max_output_tokens: 65535
        num_trials: 4
        max_concurrent_judgements: 10
        committee_models:
          - name: MyCommitteeModel
            output_dir: /path/to/committee_outputs/MyCommitteeModel

        # Alternative to VertexAI path:
        # nvidia_openai_api_key: "sk-..."
        # nvidia_openai_model: "gcp/google/gemini-3-pro-preview"
```

NVIDIA OpenAI integration rule:
- set both `nvidia_openai_api_key` and `nvidia_openai_model`, or set neither.
- if both are set, `judge_model_name` / `gcp_project_id` / `gcp_location` are ignored.

### Reward semantics

For each committee model:
- evaluated model wins majority of trials -> `1.0`
- tie (equal wins or all `TIE`) -> `0.5`
- committee model wins majority -> `0.0`

Final task reward is the mean across committee models with `success=True`.

A committee model is excluded from the mean if:
- `task_<task_id>/finish_params.json` is missing, or
- all judge retries fail / verdict is unparsable (`success=False`).

If all committee models are excluded, reward falls back to `1.0`.

### Fallback behavior

| Condition | Reward |
|-----------|--------|
| `judge.enabled: false` | `1.0` |
| No committee models configured | `1.0` |
| No committee output for task | `1.0` |
| All committee verdicts excluded | `1.0` |

## Session affinity for multi-worker deployments

Session state is process-local. With multiple resources workers, route all requests for a
session to the same worker.

1. Run workers on separate URLs (example: `http://host:8001`, `http://host:8002`).
2. Set `worker_urls` in resources server config:

```yaml
worker_urls:
  - "http://host:8001"
  - "http://host:8002"
```

3. Ensure callers pass `affinity_key=session_id` consistently (the GDPVal agent already does).

If `num_workers: 1`, `worker_urls` is optional.

## Requirements

- LibreOffice is required for office-to-PDF conversion used by judge prompts.
  - Debian/Ubuntu example: `apt-get install -y libreoffice`

## Licensing information

Code: Apache 2.0  
Data: Depends on the dataset used (for GDPVal, see the `openai/gdpval` dataset terms).

Dependencies:
- `nemo_gym`: Apache 2.0

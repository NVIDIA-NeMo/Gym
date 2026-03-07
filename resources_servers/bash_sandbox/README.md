# Description

Data links: ?

# GDPVal judge: pre-computing committee outputs and running evaluation

The judge compares the evaluated model's outputs against **pre-computed** outputs from one or
more committee models.  The two-phase workflow is:

1. **Pre-compute** — run the committee model(s) against the full dataset offline and store
   outputs in a directory tree the judge can read at verify time.
2. **Configure & run** — point the judge config at those directories and enable judging.

## Phase 1: Pre-compute committee model outputs

### Step 1 — Prepare the task JSONL

Use `client.py prepare` (from `responses_api_agents/gdpval_agent/`) to convert the HuggingFace
`openai/gdpval` dataset into a JSONL file compatible with `ng_collect_rollouts`:

```bash
python responses_api_agents/gdpval_agent/client.py prepare \
    --output-jsonl /path/to/committee_tasks.jsonl \
    --split train \                        # or "validation"
    --output-dir /path/to/committee_outputs/MyCommitteeModel
```

The `--output-dir` here is the root directory under which the committee model's per-task
outputs will be written.  It must match `judge.committee_models[].output_dir` in the YAML
(see Phase 2).

Optional flags:
- `--limit N` — use only the first N tasks (useful for smoke tests)
- `--task-ids id1,id2,...` — restrict to specific task IDs
- `--validate` — validate the produced JSONL before writing

### Step 2 — Run the committee model (judge disabled)

Start the bash-sandbox resources server with **`judge.enabled: false`** (or omit the judge
section entirely) plus the GDPVal agent pointed at the **committee** model server (not the
policy model), then drive the JSONL through `ng_collect_rollouts`.

The agent config (`gdpval_agent.yaml`) references the policy model server by default.  For
precomputation you must point it at the committee model's inference endpoint instead.  The
simplest approach is to create a separate config file (e.g. `gdpval_agent_committee.yaml`)
that overrides the `model_server` section to target the committee model:

```yaml
# gdpval_agent_committee.yaml  — overrides only the model server
gdpval_agent:
  responses_api_agents:
    gdpval_agent:
      model_server:
        type: responses_api_models
        name: committee_model_server   # must be defined in your global server config
```

Then run:

```bash
ng_collect_rollouts \
    --config responses_api_agents/gdpval_agent/configs/gdpval_agent_committee.yaml \
    --dataset-jsonl /path/to/committee_tasks.jsonl \
    --rollouts-jsonl /path/to/committee_rollouts.jsonl
```

Each task creates a directory:

```
/path/to/committee_outputs/MyCommitteeModel/
  task_<task_id>/
      finish_params.json        ← written automatically by the finish tool
      <output files …>
      <output files …>.pdf      ← PDF siblings written automatically for office files
      reference_files/          ← reference files, if any
```

Two things happen automatically inside the `finish` tool call for each task:

- **`finish_params.json`** is written — the judge uses this as a completion sentinel to decide
  whether the committee model attempted a given task.
- **Office files** (`.docx`, `.pptx`, `.xlsx`) are converted to PDF siblings in-place using
  LibreOffice. The judge can only read office files via their `.pdf` sibling; without
  conversion the file is silently excluded from the judge prompt.  Conversion failures are
  logged as warnings and do not abort the task.

> **LibreOffice must be installed** on the machine running the resources server.  Install it
> with `apt-get install -y libreoffice` (or equivalent).

## Phase 2: Configure and run with the judge

Edit `resources_servers/bash_sandbox/configs/bash_sandbox.yaml`:

```yaml
bash_sandbox_resources_server:
  resources_servers:
    bash_sandbox:
      judge:
        enabled: true
        # --- VertexAI / Gemini path (default) ---
        judge_model_name: gemini-3-pro-preview
        gcp_project_id: your-gcp-project
        gcp_location: global
        thinking_budget: 5000
        max_output_tokens: 65535
        num_trials: 4                  # trials per committee model per task (position-swapped)
        max_concurrent_judgements: 10
        committee_models:
          - name: MyCommitteeModel          # human-readable name used in logs/verdicts
            output_dir: /path/to/committee_outputs/MyCommitteeModel
          # add more committee models here if desired

        # --- NVIDIA / OpenAI-compatible path (alternative to VertexAI) ---
        # Omit both fields to use the VertexAI path above.
        # Set both to switch to the NVIDIA OpenAI endpoint instead;
        # when set, judge_model_name / gcp_project_id / gcp_location are ignored.
        # nvidia_openai_api_key: "sk-..."
        # nvidia_openai_model: "gcp/google/gemini-3-pro-preview"
```

Both fields `nvidia_openai_api_key` and `nvidia_openai_model` must be set together or omitted
together — setting only one raises a `ValueError` at startup.  When both are set,
`judge_model_name`, `gcp_project_id`, and `gcp_location` are unused.

### Reward semantics

For each task the judge runs `num_trials` comparisons (alternating which model is "A" and
which is "B" to reduce position bias). For each committee model the per-task reward is:

| Outcome | Reward |
|---------|--------|
| Evaluated model wins majority of trials | 1.0 |
| Tie (equal wins, or all TIE verdicts) | 0.5 |
| Committee model wins majority | 0.0 |

When multiple committee models are configured, the final reward is the **mean** across all
committee models whose verdict `success=True`. A committee model is excluded from the mean
(silently) in two cases:

- it has no output for the task (`task_<task_id>/finish_params.json` is absent)
- it ran judging but all API retries were exhausted or produced no parseable verdict
  (`success=False`)

If **all** committee models are excluded the judge falls back to `reward = 1.0`.

### Fallback behaviour

| Condition | Reward |
|-----------|--------|
| `judge.enabled: false` | 1.0 (passthrough) |
| No committee models configured | 1.0 (passthrough) |
| No committee model has output for the task | 1.0 (passthrough) |
| All committee model verdicts excluded (missing output or `success=False`) | 1.0 (passthrough) |

# Session affinity (multiple workers)

Sessions are stored in-memory per process. For multiple workers you must use **client-side session affinity** so all requests for a session hit the same worker:

1. Run each worker on a different port (e.g. `http://host:8001`, `http://host:8002`).
2. In the resources server config, set `worker_urls` to the list of worker base URLs:
   ```yaml
   worker_urls:
     - "http://host:8001"
     - "http://host:8002"
   ```
3. The GDPVal agent passes `affinity_key=session_id` on every resources server call; `ServerClient` hashes the key to choose the URL. All calls for the same session then go to the same worker.

If you use a single worker (`num_workers: 1`), you can omit `worker_urls`.

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?

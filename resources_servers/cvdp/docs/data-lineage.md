# CVDP Dataset Creation

This documents how the Gym version of the CVDP dataset was created.
This step does NOT need to be repeated to run evals.

## Step 1 — Export prompts from CVDP

Use CVDP's built-in `local_export` mode to generate the exact prompts CVDP would send to a model:

```bash
cd /path/to/cvdp_benchmark
python run_benchmark.py \
    -f <dataset>.jsonl \
    --model local_export \
    --prompts-responses-file <output_prompts>.jsonl \
    --llm \
    --prefix export_run
```

This produces a JSONL with `{id, prompt, system, user}` per entry.

## Step 2 — Convert to NeMo-Gym format

```bash
python resources_servers/cvdp/scripts/convert_to_gym.py \
    --prompts  <prompts_from_step1>.jsonl \
    --dataset  <original_cvdp_dataset>.jsonl \
    --output   resources_servers/cvdp/data/<output>.jsonl
```

Each output row has `responses_create_params` (system + user prompts) and `verifier_metadata`
(harness files, target files, category, difficulty) needed by the resource server.

## Gym JSONL schema

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "<rtl-task system prompt>"},
      {"role": "user",   "content": "<problem spec + optional context>"}
    ]
  },
  "verifier_metadata": {
    "task_id": "cvdp_copilot_...",
    "categories": ["cid003", "medium"],
    "difficulty": "medium",
    "target_files": ["rtl/foo.sv"],
    "harness_files": {
      "docker-compose.yml": "...",
      "src/.env": "...",
      "src/test_foo.py": "..."
    }
  }
}
```

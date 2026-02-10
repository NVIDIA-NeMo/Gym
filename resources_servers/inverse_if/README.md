# Inverse IF Environment

Evaluates model responses on the **Inverse IF** (Instruction Following) benchmark using a per-task LLM judge. This benchmark assesses whether models can precisely follow complex, often counter-intuitive formatting and content constraints.

## Quick Start

```bash
# 1. Run unit tests
ng_test +entrypoint=resources_servers/inverse_if

# 2. Start servers (in terminal 1)
config_paths="resources_servers/inverse_if/configs/inverse_if.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# 3. Collect rollouts on example data (in terminal 2)
ng_collect_rollouts \
  +agent_name=inverse_if_simple_agent \
  +input_jsonl_fpath=resources_servers/inverse_if/data/example.jsonl \
  +output_jsonl_fpath=/tmp/inverse_if_rollouts.jsonl
```

## Overview

Each Inverse IF task contains:
- **Prompt**: A single-turn user instruction with intricate formatting/content requirements
- **Reference response**: A gold-standard response that perfectly follows the instructions
- **Rubric**: 3–10 binary criteria for evaluating compliance
- **Judge template & system prompt**: Per-task prompts for the LLM judge

Task taxonomies include:
- **Question Correction** — recognise and reject flawed premises
- **Mid-turn Instruction Modification** — follow revised instructions mid-prompt
- **Deliberately Incorrect Answers** — produce intentionally wrong output on request
- **Counter-Conventional Formatting** — apply unusual formatting constraints
- **Intentional Textual Flaws** — introduce deliberate misspellings or errors
- **Counterfactual Answering** — answer based on hypothetical, non-real premises

The environment:
1. Feeds the prompt to the policy model (single turn)
2. Retrieves the generated response (excluding thinking/reasoning blocks)
3. For each rubric criterion, queries the LLM judge with the task's own template
4. The judge returns a structured `{"result": "PASS"/"FAIL"}` JSON verdict
5. Aggregates per-criterion scores using a configurable method (mean, min, all, etc.)

### Key Difference from MultiChallenge

| Aspect | MultiChallenge | Inverse IF |
|--------|---------------|------------|
| Turns | Multi-turn conversation | Single-turn prompt |
| Judge template | Global (in config) | **Per-task** (in data) |
| Judge output | `[[YES]]` / `[[NO]]` | `{"result": "PASS"/"FAIL"}` |
| Reference answer | None | Gold response included |

## Data Preparation

### Option A: Use Example Data Only (Quick Testing)

The `data/example.jsonl` file contains 3 synthetic tasks ready to use:

```bash
ng_collect_rollouts \
  +agent_name=inverse_if_simple_agent \
  +input_jsonl_fpath=resources_servers/inverse_if/data/example.jsonl \
  +output_jsonl_fpath=/tmp/test_rollouts.jsonl
```

### Option B: Full Dataset Setup

> **Important**: Run the preprocessing script **before launching training jobs**.
> The preprocessed JSONL file must exist in `data/` for the training pipeline to work.

1. **Preprocess to JSONL format**:
   ```bash
   cd resources_servers/inverse_if
   python dataset_preprocess.py
   ```

   This reads 1,000 JSON files from the raw data directory and outputs:
   - `data/inverse_if.jsonl` (1,000 tasks)

   ```bash
   # Custom input/output paths
   python dataset_preprocess.py \
     --data-dir /path/to/raw/json/files \
     --output-dir ./data \
     --output-name inverse_if.jsonl
   ```

2. **Run on full dataset**:
   ```bash
   ng_collect_rollouts \
     +agent_name=inverse_if_simple_agent \
     +input_jsonl_fpath=resources_servers/inverse_if/data/inverse_if.jsonl \
     +output_jsonl_fpath=/tmp/inverse_if_rollouts.jsonl
   ```

## Testing

### Unit Tests

```bash
# Run all unit tests
ng_test +entrypoint=resources_servers/inverse_if

# Or run directly with pytest
cd resources_servers/inverse_if
source .venv/bin/activate
pytest -v
```

Tests cover:
- Verdict extraction (JSON parsing, fallbacks)
- Rubric key normalisation (criteria, criteria1, rule, question, …)
- Score aggregation (mean, min, max, all, any)

### End-to-End Sanity Test

1. **Start servers**:
   ```bash
   config_paths="resources_servers/inverse_if/configs/inverse_if.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
   ng_run "+config_paths=[${config_paths}]"
   ```

2. **In another terminal, run on example data**:
   ```bash
   ng_collect_rollouts \
     +agent_name=inverse_if_simple_agent \
     +input_jsonl_fpath=resources_servers/inverse_if/data/example.jsonl \
     +output_jsonl_fpath=/tmp/inverse_if_rollouts.jsonl \
     +limit=3
   ```

3. **View results**:
   ```bash
   cat /tmp/inverse_if_rollouts.jsonl | python -c "
   import json, sys
   for line in sys.stdin:
       d = json.loads(line)
       print(f\"Reward: {d.get('reward')} | Passed: {d.get('num_passed')}/{d.get('num_total')}\")
   "
   ```

## Configuration

### Basic Setup

```yaml
inverse_if:
  resources_servers:
    inverse_if:
      entrypoint: app.py

      judge_model_server:
        type: responses_api_models
        name: policy_model

      judge_responses_create_params:
        input: []
        max_output_tokens: 512
        temperature: 0.0

      aggregation_mode: mean
```

### Aggregation Modes

| Mode | Description |
|------|-------------|
| `mean` | Average of all criterion scores |
| `min` | Minimum score (strictest) |
| `max` | Maximum score (most lenient) |
| `all` | All criteria must pass (binary: 0 or 1) |
| `any` | Any criterion passes (binary: 0 or 1) |
| `weighted` | Weighted average (falls back to mean for this dataset) |

## Data Format

### Raw JSON Format (Input)

Each task JSON file contains:

```json
{
  "metadata": {
    "task_id": 70258,
    "domain": "Business, Finance, Industry",
    "use_case": "Analysis & Critique",
    "l1_taxonomy": "Counter-Conventional Formatting"
  },
  "messages": [
    {"role": "prompt", "content": "Critique corporate tax avoidance..."},
    {"role": "response", "content": "\"Corporate?|#XAT?|#..."},
    {"role": "response_reference", "content": "[{\"id\": \"C1\", ...}]"},
    {"role": "judge_prompt_template", "content": "## Question\n{prompt}\n..."},
    {"role": "judge_system_prompt", "content": "Your role is..."}
  ],
  "rubrics": [
    {"id": "C1", "criteria": "Is the response a single continuous string..."},
    {"id": "C2", "criteria": "Does the response start with an opening double quote..."}
  ],
  "model_responses": [...]
}
```

### Preprocessed JSONL Format (Output)

Each line contains:

```json
{
  "uuid": "70258",
  "task_id": 70258,
  "responses_create_params": {
    "input": [{"role": "user", "content": "Critique corporate tax avoidance..."}]
  },
  "rubric": [
    {"id": "C1", "criteria": "Is the response a single continuous string..."}
  ],
  "reference_response": "\"Corporate?|#XAT?|#...",
  "prompt": "Critique corporate tax avoidance...",
  "judge_prompt_template": "## Question\n{prompt}\n...",
  "judge_system_prompt": "Your role is...",
  "metadata": {...}
}
```

Key transformations:
- Inconsistent rubric keys normalised to `{"id", "criteria"}`
- Per-task judge template and system prompt extracted from messages
- `responses_create_params` wraps input for `ng_collect_rollouts`
- `model_responses` are discarded

## File Structure

```
inverse_if/
├── app.py                   # Main server implementation
├── dataset_preprocess.py    # JSON → JSONL converter
├── requirements.txt         # Dependencies (-e nemo-gym[dev])
├── README.md                # This file
├── .gitignore               # Excludes data from git
├── configs/
│   └── inverse_if.yaml      # Server + agent configuration
├── data/
│   ├── example.jsonl        # Example data (3 tasks, committed)
│   ├── inverse_if.jsonl     # Preprocessed (generated, ignored)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_inverse_if.py
```

## API Endpoints

- `POST /verify` — Evaluate a model response against the per-criterion rubric
- `POST /seed_session` — Initialise a new session

### Verify Response

```json
{
  "reward": 0.75,
  "prompt": "Critique corporate tax avoidance...",
  "generated_response": "...",
  "reference_response": "\"Corporate?|#XAT?|#...",
  "rubric_evaluations": [
    {
      "criterion_id": "C1",
      "criteria": "Is the response a single continuous string...",
      "verdict": "PASS",
      "explanation": "The response contains no spaces.",
      "score": 1.0
    }
  ],
  "num_passed": 3,
  "num_total": 4,
  "aggregation_mode": "mean"
}
```

---

**Note**: The default raw data path is hardcoded in `dataset_preprocess.py`:
```
/lustre/fsw/portfolios/llmservice/users/mfathi/data/inverse_if
```
Use `--data-dir` to specify a different location.

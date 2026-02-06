# MultiChallenge Environment

Evaluates model responses on the **MultiChallenge** benchmark using an LLM judge. This benchmark assesses multi-turn conversation quality through rubric-based evaluation.

## Quick Start

```bash
# 1. Run unit tests
ng_test +entrypoint=resources_servers/multichallenge

# 2. Start servers (in terminal 1)
config_paths="resources_servers/multichallenge/configs/multichallenge.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# 3. Collect rollouts on example data (in terminal 2)
ng_collect_rollouts \
  +agent_name=multichallenge_simple_agent \
  +input_jsonl_fpath=resources_servers/multichallenge/data/example.jsonl \
  +output_jsonl_fpath=/tmp/multichallenge_rollouts.jsonl
```

## Overview

Each MultiChallenge task contains:
- **Conversation context**: A multi-turn dialogue between user and assistant
- **Rubric**: A set of yes/no questions evaluating the final response quality
- **Metadata**: Task information including topic, challenge type, and persona

The environment:
1. Feeds the conversation context to the policy model
2. Retrieves the final response (excluding thinking/reasoning blocks)
3. Evaluates each rubric question using an LLM judge
4. Aggregates scores using a configurable method (mean, min, all, etc.)

## Data Preparation

### Option A: Use Example Data Only (Quick Testing)

The `data/example.jsonl` file contains 3 synthetic tasks ready to use:

```bash
# No preprocessing needed - just run
ng_collect_rollouts \
  +agent_name=multichallenge_simple_agent \
  +input_jsonl_fpath=resources_servers/multichallenge/data/example.jsonl \
  +output_jsonl_fpath=/tmp/test_rollouts.jsonl
```

### Option B: Full Dataset Setup

> **Important**: Run the preprocessing script **before launching training jobs**. 
> The preprocessed JSONL files must exist in `data/` for the training pipeline to work.

1. **Preprocess to JSONL format**:
   ```bash
   # Run from the multichallenge directory
   cd resources_servers/multichallenge
   python dataset_preprocess.py
   ```
   
   This reads from the raw data directory and outputs:
   - `data/advanced.jsonl` (994 tasks)
   - `data/vanilla.jsonl` (1023 tasks)

   The script supports two input modes:
   - `--mode jsonl` (default): Reads pre-compiled `{split}.jsonl` files
   - `--mode json-dir`: Reads individual `{split}/*.json` files from directories

   ```bash
   # Custom input/output paths
   python dataset_preprocess.py \
     --data-dir /path/to/raw/data \
     --output-dir ./data \
     --splits advanced vanilla
   ```

2. **Run on full dataset**:
   ```bash
   ng_collect_rollouts \
     +agent_name=multichallenge_simple_agent \
     +input_jsonl_fpath=resources_servers/multichallenge/data/advanced.jsonl \
     +output_jsonl_fpath=/tmp/advanced_rollouts.jsonl
   ```

## Testing

### Unit Tests

```bash
# Run all unit tests
ng_test +entrypoint=resources_servers/multichallenge

# Or run directly with pytest for more detail
cd resources_servers/multichallenge
source .venv/bin/activate
pytest -v
```

Tests cover:
- Verdict extraction (`[[YES]]`/`[[NO]]`)
- Context building (excluding thinking messages)
- Score aggregation (mean, min, max, all, any, weighted)

### End-to-End Sanity Test

1. **Start servers**:
   ```bash
   config_paths="resources_servers/multichallenge/configs/multichallenge.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
   ng_run "+config_paths=[${config_paths}]"
   ```

2. **In another terminal, run on example data**:
   ```bash
   ng_collect_rollouts \
     +agent_name=multichallenge_simple_agent \
     +input_jsonl_fpath=resources_servers/multichallenge/data/example.jsonl \
     +output_jsonl_fpath=/tmp/multichallenge_rollouts.jsonl \
     +limit=3
   ```

3. **View results**:
   ```bash
   cat /tmp/multichallenge_rollouts.jsonl | python -c "
   import json, sys
   for line in sys.stdin:
       d = json.loads(line)
       print(f\"Reward: {d.get('reward')} | Passed: {d.get('num_passed')}/{d.get('num_total')}\")
   "
   ```

## Configuration

### Basic Setup

```yaml
multichallenge:
  resources_servers:
    multichallenge:
      entrypoint: app.py
      
      # Judge model configuration
      judge_model_server:
        type: responses_api_models
        name: policy_model  # or a dedicated judge model
      
      # Judge request parameters
      judge_responses_create_params:
        input: []
        max_output_tokens: 512
        temperature: 0.0
      
      # Score aggregation: mean | min | max | all | any | weighted
      aggregation_mode: mean
```

### Aggregation Modes

| Mode | Description |
|------|-------------|
| `mean` | Average of all rubric scores |
| `min` | Minimum score (strictest) |
| `max` | Maximum score (most lenient) |
| `all` | All items must pass (binary: 0 or 1) |
| `any` | Any item passes (binary: 0 or 1) |
| `weighted` | Weighted average using rubric item weights |

### Custom Judge Prompt

You can customize the judge prompt template:

```yaml
judge_prompt_template: |-
  You are evaluating whether a model's response meets a specific criterion.

  CONVERSATION CONTEXT:
  {context}

  MODEL'S FINAL RESPONSE:
  {response}

  EVALUATION QUESTION:
  {question}

  EXPECTED ANSWER: {pass_criteria}

  Respond with [[YES]] or [[NO]].
```

Placeholders:
- `{context}`: Full conversation history
- `{response}`: The model's final response
- `{question}`: The rubric evaluation question
- `{pass_criteria}`: Expected answer (usually "YES")

## Data Format

### Raw JSON Format (Input)

Each task file contains:

```json
{
  "metadata": {
    "taskId": 12345,
    "topic": "Education & Research",
    "challenge": "Inference Memory"
  },
  "system": "Optional system prompt",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "thinking", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "rubric": [
    {
      "question": "Did the model correctly remember X?",
      "pass_criteria": "YES"
    }
  ]
}
```

### Preprocessed JSONL Format (Output)

Each line contains:

```json
{
  "uuid": "12345",
  "task_id": 12345,
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}]
  },
  "rubric": [...],
  "context": "[USER]: ...\n\n[ASSISTANT]: ...",
  "metadata": {...}
}
```

Key transformations:
- `thinking` role messages are excluded from input
- `context` is pre-formatted for the LLM judge
- `responses_create_params` wraps input for `ng_collect_rollouts`

## File Structure

```
multichallenge/
├── app.py                   # Main server implementation
├── dataset_preprocess.py    # JSON → JSONL converter
├── requirements.txt         # Dependencies (-e nemo-gym[dev])
├── README.md                # This file
├── .gitignore               # Excludes data from git
├── configs/
│   └── multichallenge.yaml  # Server + agent configuration
├── data/
│   ├── example.jsonl        # Example data (3 tasks, committed)
│   ├── advanced/            # Raw JSON files (ignored)
│   ├── vanilla/             # Raw JSON files (ignored)
│   ├── advanced.jsonl       # Preprocessed (generated, ignored)
│   ├── vanilla.jsonl        # Preprocessed (generated, ignored)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_multichallenge.py
```

## API Endpoints

- `POST /verify` - Evaluate a model response against the rubric
- `POST /seed_session` - Initialize a new session

### Verify Response

```json
{
  "reward": 0.75,
  "generated_response": "...",
  "rubric_evaluations": [
    {
      "question": "...",
      "pass_criteria": "YES",
      "verdict": "YES",
      "score": 1.0
    }
  ],
  "num_passed": 3,
  "num_total": 4,
  "aggregation_mode": "mean"
}
```

## Example Rubric Evaluation

Given a conversation about travel planning where the user mentioned a seafood allergy:

```json
{
  "question": "Did the model correctly remember that the user is allergic to seafood and avoid recommending seafood dishes?",
  "pass_criteria": "YES"
}
```

The LLM judge analyzes the model's response and returns `[[YES]]` or `[[NO]]`.

---

**Note**: The default raw data path is hardcoded in `dataset_preprocess.py`:
```
/lustre/fsw/portfolios/llmservice/users/mfathi/data/multichallenge
```
Update `DEFAULT_RAW_DATA_DIR` in the script or use `--data-dir` to specify a different location.

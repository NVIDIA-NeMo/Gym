# MultiChallenge Data Directory

This directory contains the MultiChallenge benchmark dataset.

## Quick Start

```bash
# Option A: Use example data only (no setup needed)
# The example.jsonl file is ready to use for testing

# Option B: Full dataset setup
# 1. Copy raw data
cp -r /path/to/multichallenge/advanced ./advanced
cp -r /path/to/multichallenge/vanilla ./vanilla

# 2. Preprocess to JSONL (run from parent directory)
cd ..
python dataset_preprocess.py
```

## Directory Structure

```
data/
├── example.jsonl       # Example dataset (3 tasks, committed to git)
├── advanced/           # Raw JSON task files (113 tasks, ignored)
│   └── *.json
├── vanilla/            # Raw JSON task files (111 tasks, ignored)
│   └── *.json
├── advanced.jsonl      # Preprocessed dataset (generated, ignored)
├── vanilla.jsonl       # Preprocessed dataset (generated, ignored)
├── .gitignore          # Excludes data files from git
└── README.md           # This file
```

## Example Dataset

The `example.jsonl` file contains 3 synthetic tasks for quick testing:

| # | Challenge | Rubric Items | Tests |
|---|-----------|--------------|-------|
| 1 | Memory Retention | 2 | Peanut allergy recall, name usage |
| 2 | Preference Update | 3 | Pescatarian diet, preference correction |
| 3 | Context Tracking | 2 | Presentation intro, climate change topic |

**Usage:**
```bash
ng_collect_rollouts \
  +agent_name=multichallenge_simple_agent \
  +input_jsonl_fpath=resources_servers/multichallenge/data/example.jsonl \
  +output_jsonl_fpath=/tmp/test_rollouts.jsonl
```

## Raw JSON Format

Each task JSON file contains:

```json
{
  "metadata": {
    "taskId": 12345,
    "topic": "Travel & Transportation",
    "challenge": "Inference Memory",
    "persona": "..."
  },
  "system": "Optional system prompt",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "thinking", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "rubric": [
    {
      "question": "Did the model correctly remember X?",
      "pass_criteria": "YES"
    }
  ],
  "ground_truth_answer": "...",
  "model_responses": [...]  // Ignored by this environment
}
```

## Preprocessed JSONL Format

Each line in the JSONL file:

```json
{
  "uuid": "12345",
  "task_id": 12345,
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  },
  "rubric": [...],
  "context": "[USER]: ...\n\n[ASSISTANT]: ...",
  "metadata": {...}
}
```

**Key transformations:**
- `thinking` role messages are **excluded** from `responses_create_params.input`
- `context` is a pre-formatted string for the LLM judge (also excludes thinking)
- `responses_create_params` wrapper is required by `ng_collect_rollouts`
- `metadata` preserves full original data for reference

## Regenerating JSONL Files

If you modify the raw data or preprocessing logic:

```bash
python dataset_preprocess.py --data-dir ./data --splits advanced vanilla
```

**Options:**
- `--data-dir`: Directory containing split subdirectories (default: `./data`)
- `--output-dir`: Where to write JSONL files (default: same as data-dir)
- `--splits`: Which splits to process (default: `advanced vanilla`)

## Git Ignored Files

The following are excluded from version control:
- `advanced/` and `vanilla/` directories (raw data)
- `advanced.jsonl` and `vanilla.jsonl` (preprocessed data)

The `example.jsonl` file **is committed** for testing purposes.

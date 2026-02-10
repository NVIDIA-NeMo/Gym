# Inverse IF Data Directory

This directory contains the Inverse IF benchmark dataset.

## Quick Start

```bash
# Option A: Use example data only (no setup needed)
# The example.jsonl file is ready to use for testing

# Option B: Full dataset setup — run from parent directory
cd ..
python dataset_preprocess.py
```

## Directory Structure

```
data/
├── example.jsonl       # Example dataset (3 tasks, committed to git)
├── inverse_if.jsonl    # Preprocessed dataset (generated, ignored)
├── .gitignore          # Excludes generated data from git
└── README.md           # This file
```

## Example Dataset

The `example.jsonl` file contains 3 synthetic tasks for quick testing:

| # | Taxonomy | Criteria | Tests |
|---|----------|----------|-------|
| 1 | Counter-Conventional Formatting | 3 | Sentence count, first word, punctuation |
| 2 | Question Correction | 3 | Reject options, identify Paris, avoid endorsing |
| 3 | Intentional Textual Flaws | 3 | Language count, letter swapping, numbered list |

**Usage:**
```bash
ng_collect_rollouts \
  +agent_name=inverse_if_simple_agent \
  +input_jsonl_fpath=resources_servers/inverse_if/data/example.jsonl \
  +output_jsonl_fpath=/tmp/test_rollouts.jsonl
```

## Raw JSON Format

Each task JSON file (1,000 total in the full dataset) contains:

```json
{
  "metadata": {
    "task_id": 70258,
    "domain": "Business, Finance, Industry",
    "l1_taxonomy": "Counter-Conventional Formatting"
  },
  "messages": [
    {"role": "prompt", "content": "..."},
    {"role": "response", "content": "..."},
    {"role": "response_reference", "content": "[{\"id\": \"C1\", ...}]"},
    {"role": "judge_prompt_template", "content": "..."},
    {"role": "judge_system_prompt", "content": "..."}
  ],
  "rubrics": [
    {"id": "C1", "criteria": "..."}
  ],
  "model_responses": [...]
}
```

## Preprocessed JSONL Format

Each line in the JSONL file:

```json
{
  "uuid": "70258",
  "task_id": 70258,
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}]
  },
  "rubric": [{"id": "C1", "criteria": "..."}],
  "reference_response": "...",
  "prompt": "...",
  "judge_prompt_template": "...",
  "judge_system_prompt": "...",
  "metadata": {...}
}
```

**Key transformations:**
- Inconsistent rubric keys normalised to `{"id", "criteria"}`
- Per-task judge template and system prompt extracted from messages
- `responses_create_params` wrapper required by `ng_collect_rollouts`
- `model_responses` discarded

## Regenerating JSONL Files

```bash
python dataset_preprocess.py --data-dir /path/to/raw/data --output-dir ./data
```

**Options:**
- `--data-dir`: Directory containing raw JSON files (default: hardcoded path)
- `--output-dir`: Where to write JSONL files (default: `./data`)
- `--output-name`: Output filename (default: `inverse_if.jsonl`)

## Git Ignored Files

The following are excluded from version control:
- `inverse_if.jsonl` (preprocessed data)

The `example.jsonl` file **is committed** for testing purposes.

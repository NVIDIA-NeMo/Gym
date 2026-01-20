# Multi-Pass LLM Judge Environment

A flexible, multi-dimensional LLM-as-judge environment for NeMo-Gym that evaluates model outputs using multiple distinct judge passes, each with its own prompt template and scoring criteria.

## Overview

Unlike single-pass judges that provide a binary correct/incorrect signal, this environment:

- **Evaluates multiple dimensions**: Each pass can assess different aspects (correctness, reasoning quality, clarity, etc.)
- **Uses different prompts per pass**: Customize judge prompts for each evaluation criterion
- **Supports flexible scoring modes**: Binary, numeric, or regex-based scoring
- **Aggregates scores**: Combine pass scores via weighted sum, min, max, mean, or logical operators

This is ideal for:
- Training models to produce high-quality, well-reasoned responses
- Multi-objective RLHF with different reward components
- Ablating which aspects of response quality matter most

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Multi-Pass LLM Judge                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model Response ───┬──► Pass 1 (Correctness) ──► Score 1 ──┐       │
│                    │                                        │       │
│                    ├──► Pass 2 (Reasoning)   ──► Score 2 ──┼──► Aggregate ──► Reward
│                    │                                        │       │
│                    └──► Pass 3 (Clarity)     ──► Score 3 ──┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Configure Your Model API

Edit `env.yaml` in the project root with your model endpoint:

```yaml
policy_base_url: "https://api.openai.com/v1"
policy_api_key: "your-api-key"
policy_model_name: "gpt-4o-mini"
```

Or use a local vLLM model:

```yaml
policy_base_url: "http://localhost:8000/v1"
policy_api_key: "none"
policy_model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

### 2. Start the Servers

```bash
# Using OpenAI-compatible API
config_paths="resources_servers/multi_pass_llm_judge/configs/multi_pass_llm_judge.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]"
```

### 3. Collect Rollouts

In a new terminal:

```bash
ng_collect_rollouts \
    +agent_name=multi_pass_llm_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/multi_pass_llm_judge/data/example.jsonl \
    +output_jsonl_fpath=results/multi_pass_rollouts.jsonl \
    +limit=5
```

### 4. View Results

```bash
ng_viewer +jsonl_fpath=results/multi_pass_rollouts.jsonl
```

## Configuration Guide

### Judge Passes

Each pass in `judge_passes` can have:

| Field | Description | Default |
|-------|-------------|---------|
| `name` | Identifier for logging | Required |
| `weight` | Weight for aggregation | `1.0` |
| `prompt_template` | Judge prompt with `{question}`, `{expected_answer}`, `{generated_answer}` placeholders | Required |
| `system_message` | Optional system message | `null` |
| `scoring_mode` | `binary`, `numeric`, or `regex` | `binary` |
| `judge_model_server` | **Optional** model server override for this pass | Uses global |
| `responses_create_params` | **Optional** request params override for this pass | Uses global |

### Per-Pass Model Configuration

Each pass can use a **different model** for evaluation. This enables ensemble judging:

```yaml
judge_passes:
  # Pass 1: Uses a large model for correctness (high accuracy)
  - name: correctness
    weight: 0.6
    judge_model_server:
      type: responses_api_models
      name: large_judge_model  # e.g., GPT-4
    responses_create_params:
      input: []
      max_output_tokens: 512
      temperature: 0.0
    scoring_mode: binary
    ...
    
  # Pass 2: Uses a smaller/faster model for style (efficiency)
  - name: style
    weight: 0.4
    judge_model_server:
      type: responses_api_models
      name: small_judge_model  # e.g., GPT-3.5
    responses_create_params:
      input: []
      max_output_tokens: 128
      temperature: 0.0
    scoring_mode: numeric
    ...
```

To use a different model, uncomment the `judge_model_server` block in any pass.

### Scoring Modes

#### Binary Mode
Look for success/failure labels in judge output:

```yaml
scoring_mode: binary
success_label: "[[CORRECT]]"
failure_label: "[[INCORRECT]]"
```

#### Numeric Mode
Extract a numeric score using regex:

```yaml
scoring_mode: numeric
numeric_regex: "Score:\\s*(\\d+(?:\\.\\d+)?)"  # Extracts "8" from "Score: 8/10"
numeric_max: 10.0  # Normalize to 0-1
```

#### Regex Mode
Match patterns for different score levels:

```yaml
scoring_mode: regex
regex_patterns:
  - pattern: "\\[\\[EXCELLENT\\]\\]"
    score: 1.0
  - pattern: "\\[\\[GOOD\\]\\]"
    score: 0.75
  - pattern: "\\[\\[PARTIAL\\]\\]"
    score: 0.5
regex_default_score: 0.0
```

### Aggregation Modes

| Mode | Description |
|------|-------------|
| `weighted_sum` | Weighted average of all pass scores |
| `min` | Minimum score across all passes |
| `max` | Maximum score across all passes |
| `mean` | Simple average of all pass scores |
| `all` | 1.0 only if ALL passes score 1.0 |
| `any` | 1.0 if ANY pass scores 1.0 |

## Customization Examples

### Two-Pass Correctness + Helpfulness

```yaml
judge_passes:
  - name: correctness
    weight: 0.7
    scoring_mode: binary
    success_label: "[[CORRECT]]"
    failure_label: "[[WRONG]]"
    prompt_template: |-
      Is this answer factually correct?
      Question: {question}
      Expected: {expected_answer}
      Candidate: {generated_answer}
      Output [[CORRECT]] or [[WRONG]]
      
  - name: helpfulness
    weight: 0.3
    scoring_mode: numeric
    numeric_regex: "(\\d+)/10"
    numeric_max: 10.0
    prompt_template: |-
      Rate how helpful this response is (0-10):
      Question: {question}
      Response: {generated_answer}
      Score: X/10

aggregation_mode: weighted_sum
```

### Strict All-Must-Pass

```yaml
judge_passes:
  - name: accurate
    scoring_mode: binary
    ...
  - name: safe
    scoring_mode: binary
    ...
  - name: relevant
    scoring_mode: binary
    ...

aggregation_mode: all  # Only reward if ALL passes succeed
```

### Using Custom Metadata

Your dataset can include custom fields:

```json
{
  "responses_create_params": {"input": [...]},
  "question": "...",
  "expected_answer": "...",
  "metadata": {
    "difficulty": "hard",
    "topic": "physics"
  }
}
```

Use them in prompts:

```yaml
prompt_template: |-
  This is a {metadata.difficulty} question about {metadata.topic}.
  Question: {question}
  ...
```

## Data Format

Each line in your JSONL dataset should have:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "Your question here"}]
  },
  "question": "The question text",
  "expected_answer": "The expected/reference answer",
  "metadata": {
    "optional": "custom fields"
  }
}
```

## Included Datasets

| Dataset | File | Description |
|---------|------|-------------|
| `example` | `data/example.jsonl` | 5 science QA examples for testing |
| `toy_qa` | `data/toy_qa.jsonl` | 20 general science QA questions |

## Files

```
multi_pass_llm_judge/
├── app.py                         # Server implementation
├── configs/
│   └── multi_pass_llm_judge.yaml  # Configuration (supports per-pass models)
├── data/
│   ├── example.jsonl              # 5 test examples
│   └── toy_qa.jsonl               # 20 training examples
├── tests/
│   └── test_app.py                # Unit tests
├── requirements.txt
└── README.md
```

## Licensing

- Code: Apache 2.0
- Data: Apache 2.0 (synthetic examples)

## Dependencies

- nemo_gym: Apache 2.0

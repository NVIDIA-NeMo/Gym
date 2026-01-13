(env-llm-as-judge)=

# LLM-as-a-Judge Verification

Use large language models to verify and score agent responses.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
30-45 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`/get-started/detailed-setup`
- Access to a judge model (OpenAI, vLLM, or other)

:::

::::

---

## What is LLM-as-a-Judge?

LLM-as-a-Judge uses a separate LLM to evaluate model outputs:

- Flexible evaluation criteria via natural language prompts
- Works for open-ended tasks without fixed answers
- Scales without hand-crafted rules

```text
Agent Response → Judge Model → Verdict (equal/not equal) → Reward
```

## When to Use

Use LLM verification when:

- Tasks have subjective or open-ended answers
- Multiple valid solutions exist
- Hand-crafted rules are impractical
- You need semantic similarity checking

## Quick Start

### 1. Configure the Judge

Create a configuration file referencing the `equivalence_llm_judge` resources server:

```yaml
equivalence_llm_judge:
  resources_servers:
    equivalence_llm_judge:
      entrypoint: app.py
      judge_model_server:
        type: responses_api_models
        name: judge_model
      judge_responses_create_params:
        input: []
      judge_prompt_template: |
        Compare the following answers:
        Question: {question}
        Expected: {expected_answer}
        Generated: {generated_answer}
        
        Output [[A=B]] if equivalent, [[A!=B]] if not.
```

### 2. Start the Servers

```bash
config_paths="resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[$config_paths]"
```

### 3. Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=equivalence_llm_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/equivalence_llm_judge/data/example.jsonl \
    +output_jsonl_fpath=data/rollouts.jsonl
```

## Configuration Reference

### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `judge_model_server` | ref | required | Model server to use as judge |
| `judge_prompt_template` | str | required | Prompt with `{question}`, `{expected_answer}`, `{generated_answer}` |
| `judge_system_message` | str | null | Optional system message for the judge |
| `judge_equal_label` | str | `[[A=B]]` | Token indicating answers are equivalent |
| `judge_not_equal_label` | str | `[[A!=B]]` | Token indicating answers differ |

### Answer Extraction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `question_extract_regex` | str | null | Regex to extract question from user message |
| `response_extract_regex` | str | null | Regex to extract answer from assistant message |

### Per-Record Regex (OpenQA Support)

These options enable mixed datasets with different answer formats:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_per_record_regex` | bool | true | Use `template_metadata.output_regex` per record |
| `extraction_length_threshold` | int | 120 | Skip regex for long answers (use full generation) |
| `check_full_generation_on_fail` | bool | true | Retry with full generation on regex failure |
| `reward_if_full_generation_succeeds` | float | 0.5 | Reward when full generation rescue succeeds |

### Reliability Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `check_twice_swap` | bool | false | Run second pass with swapped answers |
| `reward_if_swap_fails` | float | 0.0 | Reward if swap check disagrees |

## Prompt Engineering

### Equivalence Checking

A well-structured judge prompt should:

1. Define the grading criteria clearly
2. Provide examples of equivalent and non-equivalent answers
3. Specify the exact output format

Example prompt for STEM grading:

```text
You are a meticulous STEM grader. Compare a candidate answer to a GOLD reference.

Rules:
- Treat GOLD as authoritative for what counts as correct.
- Accept mathematically equivalent transformations.
- All essential parts must match for "equivalent".

Output (at the end):
- If equivalent: [[A=B]] they are equivalent
- If not equivalent: [[A!=B]] they are not equivalent

QUESTION: {question}
GOLD: {expected_answer}
CANDIDATE: {generated_answer}
```

::::{dropdown} Few-Shot Examples
:icon: code

Include examples in your prompt to improve judge consistency:

```text
===== Example 1 (equivalent) =====
QUESTION: State Avogadro's constant (include units).
GOLD: 6.022 × 10^23 mol^-1
CANDIDATE: 6.022e23 per mole.

[[A=B]] they are equivalent

===== Example 2 (not equivalent) =====
QUESTION: State the first law of thermodynamics.
GOLD: ΔU = Q − W
CANDIDATE: ΔU = Q + W

[[A!=B]] they are not equivalent
```

::::

## Reliability Features

::::{dropdown} Swap Check (Positional Bias)
:icon: sync

LLM judges can exhibit positional bias—preferring the first or second answer. Enable swap checking to detect this:

```yaml
check_twice_swap: true
reward_if_swap_fails: 0.0
```

When enabled:

1. First pass: Compare expected vs generated
2. If equal, second pass: Compare generated vs expected (swapped)
3. Reward 1.0 only if both passes agree

::::

::::{dropdown} Full Generation Rescue
:icon: rocket

When regex extraction fails, the server can retry with the full generation:

```yaml
check_full_generation_on_fail: true
reward_if_full_generation_succeeds: 0.5
```

This helps recover from extraction failures while giving partial credit.

::::

## Input Data Format

### Required Fields

```json
{
  "expected_answer": "The correct answer text",
  "messages": [
    {"role": "user", "content": "Your question here"}
  ]
}
```

### Optional Metadata

For per-record regex extraction (OpenQA support):

```json
{
  "expected_answer": "42",
  "template_metadata": {
    "output_regex": "\\[ANSWER\\]\\s*(.+?)\\s*\\[/ANSWER\\]"
  },
  "messages": [...]
}
```

## Example Dataset

An example dataset is available on Hugging Face:

```bash
# Download the dataset
huggingface-cli download nvidia/Nemotron-RL-knowledge-openqa \
    --local-dir resources_servers/equivalence_llm_judge/data/
```

See `resources_servers/equivalence_llm_judge/` for the complete implementation.

## Next Steps

- Scale to multi-step tasks with {doc}`multi-step`
- Add multi-turn conversations with {doc}`multi-turn`
- Train models with {ref}`training-nemo-rl-grpo-index`

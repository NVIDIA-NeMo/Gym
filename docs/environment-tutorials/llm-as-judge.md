(env-llm-as-judge)=

# LLM-as-a-Judge Verification

Use an LLM to compare model-generated answers against expected answers when exact string matching fails.

:::{card}

**Goal**: Use an LLM judge to verify semantic equivalence when exact matching isn't enough.

^^^

**In this tutorial, you will**:

1. Configure the equivalence LLM judge resources server
2. Set up judge prompts for your domain
3. Collect rollouts with LLM-based verification

:::

:::{button-ref} creating-training-environment
:color: secondary
:outline:
:ref-type: doc

← Previous: Creating a Training Environment
:::

---

## When to Use LLM-as-Judge

Use this approach when:

- Multiple valid phrasings exist ("Paris" vs "The capital is Paris")
- Semantic equivalence matters more than exact match
- Answers require domain knowledge to verify (math, science, code)

**Verification flow**:

```text
Model Response → Extract Answer → Judge LLM → Verdict → Reward
                                                         ├─ 1.0 (equivalent)
                                                         ├─ 0.5 (partial credit)
                                                         └─ 0.0 (not equivalent)
```

## Quick Start

### 1. Configure the Judge

```yaml
# my_judge.yaml
equivalence_llm_judge:
  resources_servers:
    equivalence_llm_judge:
      entrypoint: app.py
      judge_model_server:
        type: responses_api_models
        name: policy_model
      judge_responses_create_params:
        input: []
        temperature: 0  # Deterministic judging
      judge_prompt_template: |
        Compare the candidate answer to the gold reference.
        
        QUESTION: {question}
        GOLD: {expected_answer}
        CANDIDATE: {generated_answer}
        
        Output [[A=B]] if equivalent, [[A!=B]] if not.
```

**Placeholders**: `{question}` (from user message), `{expected_answer}` (from data), `{generated_answer}` (from model output)

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

**Output** (`data/rollouts.jsonl`):
```text
{"reward": 1.0, "expected_answer": "darwin", "judge_evaluations": [...]}
```

## Input Data Format

:::::{tab-set}

::::{tab-item} Basic Format

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Who proposed evolution by natural selection?"}
    ]
  },
  "expected_answer": "darwin"
}
```

::::

::::{tab-item} With Regex Extraction

For structured outputs, extract the answer before judging:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Put your answer in \\boxed{}.\n\nWhat is 2+2?"}
    ]
  },
  "expected_answer": "4",
  "template_metadata": {
    "output_regex": "\\\\boxed\\{(.*?)\\}"
  }
}
```

::::

:::::

## Configuration Reference

### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `judge_model_server` | ModelServerRef | required | Model server used as judge |
| `judge_prompt_template` | str | required | Prompt with `{question}`, `{expected_answer}`, `{generated_answer}` |
| `judge_system_message` | str | null | System message for judge |
| `judge_equal_label` | str | `[[A=B]]` | Verdict token for equivalent |
| `judge_not_equal_label` | str | `[[A!=B]]` | Verdict token for different |

### Answer Extraction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `question_extract_regex` | str | null | Regex to extract question from user message |
| `response_extract_regex` | str | null | Global regex to extract answer from response |
| `use_per_record_regex` | bool | true | Use `template_metadata.output_regex` per record |
| `extraction_length_threshold` | int | 120 | Skip regex for answers longer than this |

### Reliability Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `check_twice_swap` | bool | false | Run second pass with swapped answers (bias detection) |
| `reward_if_swap_fails` | float | 0.0 | Reward when swap check disagrees |
| `check_full_generation_on_fail` | bool | true | Retry with full output on regex failure |
| `reward_if_full_generation_succeeds` | float | 0.5 | Partial reward for rescue success |

## Reliability Features

### Swap Check (Bias Detection)

LLM judges can favor the first or second position. Detect this with swap checking:

```yaml
check_twice_swap: true
reward_if_swap_fails: 0.0
```

1. First pass: GOLD vs CANDIDATE → equal
2. Second pass: CANDIDATE vs GOLD → must also be equal
3. Reward 1.0 only if both agree

### Full Generation Rescue

When regex extraction fails, retry with full output for partial credit:

```yaml
check_full_generation_on_fail: true
reward_if_full_generation_succeeds: 0.5
```

## Judge Prompt Examples

::::{dropdown} STEM Grading Prompt
:icon: beaker

```text
===== System role =====
You are a meticulous STEM grader. Compare a candidate answer to a GOLD 
reference and decide strict equivalence.

Grading priorities:
1) Factual equivalence (accept algebraically equivalent formulations)
2) Completeness (all essential parts must match)

Rules:
- GOLD is authoritative
- Accept mathematically identical transformations
- Multi-part: all parts must match

Output:
- If equivalent: [[A=B]] they are equivalent
- If not equivalent: [[A!=B]] they are not equivalent

===== Example (equivalent) =====
GOLD: 6.022 × 10^23 mol^-1
CANDIDATE: 6.022e23 per mole
[[A=B]] they are equivalent

===== Example (not equivalent) =====
GOLD: ΔU = Q − W
CANDIDATE: ΔU = Q + W
[[A!=B]] they are not equivalent

===== Inputs =====
QUESTION: {question}
GOLD: {expected_answer}
CANDIDATE: {generated_answer}
```

::::

::::{dropdown} Bash Command Equivalence
:icon: terminal

```text
You are a Bash command grader.

Determine if the candidate command is functionally equivalent to GOLD:
1. Does it achieve the same outcome?
2. Are differences purely syntactic?

Output [[A=B]] if equivalent, [[A!=B]] if not.
```

::::

## Limitations

| Consideration | Mitigation |
|---------------|------------|
| **Cost** | Each verification = 1 API call. Budget for scale. |
| **Latency** | Synchronous. Consider batching for throughput. |
| **Non-determinism** | Set `temperature: 0` for consistency. |
| **Judge errors** | Enable `check_twice_swap` for bias detection. |

**Alternatives for specific domains**:

| Domain | Better approach |
|--------|-----------------|
| Single-token answers | Exact string match |
| Math expressions | `math_with_judge` (deterministic first) |
| Code correctness | `code_gen` (test execution) |

## Complete Example

::::{dropdown} Full Config + Sample Data
:icon: file-code

**Config** (`my_llm_judge.yaml`):

```yaml
equivalence_llm_judge:
  resources_servers:
    equivalence_llm_judge:
      entrypoint: app.py
      judge_model_server:
        type: responses_api_models
        name: policy_model
      judge_responses_create_params:
        input: []
        temperature: 0
      judge_prompt_template: |
        QUESTION: {question}
        GOLD: {expected_answer}
        CANDIDATE: {generated_answer}
        Output [[A=B]] if equivalent, [[A!=B]] if not.
      check_twice_swap: true
      
equivalence_llm_judge_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: equivalence_llm_judge
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: data/questions.jsonl
```

**Data** (`data/questions.jsonl`):

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Capital of France?"}]}, "expected_answer": "Paris"}
{"responses_create_params": {"input": [{"role": "user", "content": "Who wrote Hamlet?"}]}, "expected_answer": "Shakespeare"}
```

**Run**:

```bash
ng_run "+config_paths=[my_llm_judge.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# In another terminal
ng_collect_rollouts +agent_name=equivalence_llm_judge_simple_agent \
    +input_jsonl_fpath=data/questions.jsonl \
    +output_jsonl_fpath=data/rollouts.jsonl
```

::::

## Datasets

| Dataset | Use case |
|---------|----------|
| [nvidia/Nemotron-RL-knowledge-openQA](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-openqa) | Knowledge QA with mixed formats |
| [nvidia/Nemotron-RL-math-OpenMathReasoning](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning) | Math (use with `math_with_judge`) |

## Judge Model Selection

Tested with **Gemma3-27B-it**. Larger models provide more reliable judgments for nuanced equivalence. Verify your judge model's license permits your use case.

## Next Steps

- {doc}`multi-step` — Sequential tool calling
- {doc}`multi-turn` — Conversational environments
- {ref}`training-nemo-rl-grpo-index` — Train with verified rewards

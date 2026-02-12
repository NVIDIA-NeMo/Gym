# COBOL Compiler Benchmark

COBOL compilation and execution benchmark for NeMo-Gym. 499 problems from MultiPL-E (HumanEval + MBPP) adapted for COBOL with GnuCOBOL.

## Prerequisites

GnuCOBOL (`cobc`) must be installed:

```bash
# Linux (Ubuntu/Debian)
apt-get install gnucobol

# macOS
brew install gnucobol
```

Verify: `cobc --version`

## Configure Model Endpoint

Create/edit `env.yaml` in the NeMo-Gym project root:

```yaml
policy_base_url: http://localhost:8000/v1   # vLLM, OpenAI, etc.
policy_api_key: your-key-here
policy_model_name: your-model-name
```

**Note:** Use `vllm_model` (not `openai_model`) if your endpoint serves `/v1/chat/completions`. The `openai_model` server calls `/v1/responses` which most endpoints don't support.

## Running

### 1. Start servers

```bash
# Single-turn (simple agent) — use vllm_model for chat completions endpoints
ng_run "+config_paths=[resources_servers/cobol_compiler/configs/cobol_compiler.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Multi-turn with error correction (eval agent)
ng_run "+config_paths=[resources_servers/cobol_compiler/configs/cobol_compiler.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" +config=cobol_compiler_eval_agent

# If your endpoint supports the OpenAI Responses API (/v1/responses), use openai_model instead:
# ng_run "+config_paths=[resources_servers/cobol_compiler/configs/cobol_compiler.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

### 2. Collect rollouts

Reasoning models may exhaust their token budget before outputting code. Set `max_output_tokens` to ensure enough space for the COBOL response. Use `temperature: 1.0, top_p: 1.0` for diversity across repeats (recommended by NVIDIA for Nemotron reasoning models).

```bash
# Quick test on 5 examples (single attempt each)
ng_collect_rollouts \
    +agent_name=cobol_compiler_simple_agent \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/example.jsonl \
    +output_jsonl_fpath=results/cobol_rollouts.jsonl \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0, top_p: 1.0}"

# Full benchmark — pass@5 (499 tasks × 5 repeats = 2,495 rollouts)
ng_collect_rollouts \
    +agent_name=cobol_compiler_simple_agent \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/cobol_multipl_eval.jsonl \
    +output_jsonl_fpath=results/cobol_rollouts_full.jsonl \
    +num_repeats=5 \
    +num_samples_in_parallel=5 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0, top_p: 1.0}"

# Use +limit=20 to run on a subset first
```

### 3. Compute pass rates

```bash
ng_profile \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/cobol_multipl_eval.jsonl \
    +rollouts_jsonl_fpath=results/cobol_rollouts_full.jsonl \
    +output_jsonl_fpath=results/cobol_profiled_full.jsonl \
    +pass_threshold=1.0
```

Per-task output includes: `avg_reward`, `pass_rate`, `std_reward`, `min_reward`, `max_reward`, `total_samples`.

**Computing pass@1 and pass@5 from profiled output:**

- **pass@1** = mean of `avg_reward` across all tasks (fraction of individual attempts that pass)
- **pass@5** = fraction of tasks where `max_reward >= 1.0` (at least 1 of 5 attempts passed)

```bash
python3 -c "
import json
tasks = [json.loads(l) for l in open('results/cobol_profiled_full.jsonl')]
n = len(tasks)
pass1 = sum(t['avg_reward'] for t in tasks) / n
pass5 = sum(1 for t in tasks if t['max_reward'] >= 1.0) / n
print(f'Tasks: {n}')
print(f'pass@1: {pass1:.3f}  ({pass1*100:.1f}%)')
print(f'pass@5: {pass5:.3f}  ({sum(1 for t in tasks if t[\"max_reward\"] >= 1.0)}/{n} tasks)')
"
```

### 4. View results

```bash
# Aggregate summary (averages all numeric fields across tasks)
python scripts/print_aggregate_results.py +jsonl_fpath=results/cobol_profiled.jsonl

# Interactive Gradio UI
ng_viewer +jsonl_fpath=results/cobol_rollouts.jsonl
```

## Unit Tests

```bash
ng_test +entrypoint=resources_servers/cobol_compiler
```

## Data Conversion

To regenerate the JSONL dataset from the DomainForge source:

```bash
python scripts/convert_dataset.py \
    --input ~/projects/domainforge/datasets/cobol_multipl_eval.json \
    --output data/cobol_multipl_eval.jsonl \
    --example-output data/example.jsonl
```

## License

Apache 2.0

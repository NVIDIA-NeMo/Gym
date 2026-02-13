# Description

COBOL compilation and execution benchmark. 499 problems from MultiPL-E (HumanEval + MBPP) adapted for COBOL with GnuCOBOL. The model generates COBOL code, which is compiled and tested against stdin/stdout test cases. Reward is 1.0 if all tests pass, 0.0 otherwise.

Includes a multi-turn eval agent (`cobol_eval_agent`) that feeds compilation errors and test failures back to the model for iterative correction.

## Prerequisites

GnuCOBOL (`cobc`) must be installed:

```bash
# Linux (Ubuntu/Debian)
apt-get install gnucobol

# macOS
brew install gnucobol
```

Verify: `cobc --version`

## Example Usage

Configure your model endpoint in `env.yaml` at the NeMo-Gym project root:

```yaml
policy_base_url: http://localhost:8000/v1   # vLLM, OpenAI-compatible, etc.
policy_api_key: your-key-here
policy_model_name: your-model-name
```

```bash
# Start servers
ng_run "+config_paths=[resources_servers/cobol_compiler/configs/cobol_compiler.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Quick test on 5 examples
ng_collect_rollouts \
    +agent_name=cobol_compiler_simple_agent \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/example.jsonl \
    +output_jsonl_fpath=results/cobol_rollouts.jsonl \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Full benchmark (499 tasks x 5 repeats)
ng_collect_rollouts \
    +agent_name=cobol_compiler_simple_agent \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/cobol_multipl_eval.jsonl \
    +output_jsonl_fpath=results/cobol_rollouts_full.jsonl \
    +num_repeats=5 \
    +num_samples_in_parallel=5 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Compute per-task pass rates
ng_profile \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/cobol_multipl_eval.jsonl \
    +rollouts_jsonl_fpath=results/cobol_rollouts_full.jsonl \
    +output_jsonl_fpath=results/cobol_profiled.jsonl \
    +pass_threshold=1.0

# View results
ng_viewer +jsonl_fpath=results/cobol_rollouts_full.jsonl
```

Use `openai_model` instead of `vllm_model` if your endpoint supports the OpenAI Responses API (`/v1/responses`).

## System Prompts

Three system prompt tiers are provided in `prompts/` for different evaluation needs:

| Prompt | Description |
|--------|-------------|
| `cobol_minimal.txt` | HumanEval-style baseline — no hand-holding, most comparable to published benchmarks |
| `cobol_basic.txt` | Coding standards, program structure, I/O patterns |
| `cobol_comprehensive.txt` | Full reference with output formatting, error avoidance, pre-submission checklist |

The included JSONL data uses `cobol_basic.txt`. To regenerate with a different prompt:

```bash
python scripts/convert_dataset.py \
    --input /path/to/cobol_multipl_eval.json \
    --output data/cobol_multipl_eval.jsonl \
    --system-prompt prompts/cobol_minimal.txt \
    --example-output data/example.jsonl
```

## Unit Tests

```bash
ng_test +entrypoint=resources_servers/cobol_compiler
```

## Licensing Information

Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0

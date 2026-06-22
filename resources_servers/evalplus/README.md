# EvalPlus Resources Server

### Overview
Verifies Python function-completion benchmarks (HumanEval+, MBPP+) by running
the model's extracted code against the EvalPlus base + plus test inputs.
Returns two named scores per task:

- `passing_base_tests` — completion passes the base HumanEval / MBPP tests
- `passing_plus_tests` — completion ALSO passes the EvalPlus extra tests

Mirrors NeMo-Skills' `eval_evalplus` evaluator
(`nemo_skills/evaluation/evaluator/code.py`), which delegates to
`evalplus.evaluate.evaluate(...)`.

The dataset is selected via `dataset: humaneval | mbpp` in the server
config — both flow through the same `evalplus.evaluate.check_correctness`
entry point, so adding MBPP+ as a separate benchmark requires only a new
benchmark dir (no server changes).

### Input schema

- `responses_create_params`: OpenAI Responses create params with the
  user prompt (function signature + docstring + completion instructions).
- `verifier_metadata` (required):
  - `task_id` (required): string in EvalPlus's task-id space
    (e.g., `HumanEval/0`, `Mbpp/2`). The server looks up base + plus
    tests via `evalplus.data.get_human_eval_plus()` /
    `get_mbpp_plus()` at startup.

### Code extraction

The verifier uses **last-fence + strict-mode** extraction (matching
Skills' `preprocess_code`): picks the LAST fenced code block (preferring
``` ```python ``` over a generic ``` ``` ```), and returns empty if no
matching closing fence is found. `<think>...</think>` reasoning preambles
are NOT stripped here — start vLLM with `--reasoning-parser <name>` so
the extractor sees already-clean output.

### Reward

`reward = 1.0` iff `passing_plus_tests` (the strict verdict). Both
verdicts are returned in the verify response so the metrics layer can
compute pass@k for each separately.

### Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/evalplus/configs/evalplus.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=evalplus_simple_agent \
    +input_jsonl_fpath=resources_servers/evalplus/data/example.jsonl \
    +output_jsonl_fpath=results/evalplus_rollouts.jsonl \
    +num_repeats=1
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for
Nemotron-3) so `<think>…</think>` is stripped before the verifier sees
the model output.

## Licensing information
Apache 2.0

# Code FIM Resources Server

### Overview
Verifies Python code-infilling completions against the
[HumanEval-Infilling](https://github.com/openai/human-eval-infilling)
test suite (Bavarian et al., 2022). For each task the server splices

```
prefix + completion + suffix + "\n" + test + "\n" + check(entry_point)
```

and runs it in a sandboxed subprocess via
`human_eval_infilling.execution.check_correctness`.

Mirrors NeMo-Skills' `eval_human_eval_infilling` evaluator
(`nemo_skills/evaluation/evaluator/code.py`), which delegates to
`human_eval_infilling.evaluate.evaluate(...)`.

The split is selected via `split: single_line | multi_line | random_span | random_span_light`
in the server config. All splits share the same per-task schema and the
same `check_correctness` entry point, so adding a benchmark for a
different split requires only a new benchmark dir (no server changes).

### Input schema

- `responses_create_params`: OpenAI Responses create params with the
  user prompt (FIM template — prefix + suffix in fenced code blocks).
- `verifier_metadata` (required):
  - `task_id` (required): string in HumanEval-Infilling's task-id space
    (e.g., `SingleLineInfilling/HumanEval/0/L0`,
    `MultiLineInfilling/HumanEval/3/L4-L6`,
    `RandomSpanInfilling/HumanEval/12/0`). The server looks up the
    `prompt` (prefix), `suffix`, `entry_point` and `test` for this
    task at startup via `human_eval_infilling.data.read_problems(split)`.

### Code extraction

The verifier uses **last-fence + strict-mode** extraction (matching
Skills' `preprocess_code` with `strip_whitespace=False`): picks the LAST
fenced code block (preferring ` ```python ` over a generic ` ``` `),
returns empty if no closing fence is found, and does NOT strip
surrounding whitespace inside the fence (indentation is significant for
infill). `<think>...</think>` reasoning preambles are stripped.

After extraction the completion is post-processed (Skills behavior):

  1. Drop one leading `\n` (LLMs emit ` ```python\n<fill>``` `).
  2. Trim any prefix-overlap from the head of the completion.
  3. Trim any suffix-overlap from the tail of the completion.

### Reward

`reward = 1.0` iff the spliced program runs without error within the
configured `timeout`, else `0.0`. The `accuracy` score is identical to
the reward; pass@k / majority@k are computed by the metrics layer.

### Example usage

```bash
# Running servers
gym env start \
    --model-type vllm_model \
    --resources-server code_fim

# Collecting rollouts (5-example smoke test)
gym eval run --no-serve \
    --agent code_fim_simple_agent \
    --input resources_servers/code_fim/data/example.jsonl \
    --output results/code_fim_rollouts.jsonl \
    --num-repeats 1
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for
Nemotron-3) so `<think>…</think>` is stripped before the verifier sees
the model output. The extractor also strips `<think>...</think>` as a
fallback when the parser is not configured.

## Licensing information
Apache 2.0 (the upstream `human-eval-infilling` library is released
under the MIT license).

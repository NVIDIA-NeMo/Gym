# ChemBench

For preparation and evaluation through `gym eval`, use the [ChemBench benchmark integration](../../benchmarks/chembench/README.md).

Single-turn chemistry evaluation using Gym's `simple_agent`. The resources server grades multiple-choice questions (including multiple correct choices) and numeric answers with binary rewards.

## Data

Source: [jablonkagroup/ChemBench](https://huggingface.co/datasets/jablonkagroup/ChemBench), converted by `benchmarks/chembench/prepare.py`.

The prepared snapshot contains 1,785 tasks across eight topics: 1,542 MCQ and 243 numeric questions. The human subset contains 115 tasks. Chemical preference questions are excluded. Options retain source order; no system prompt is added.

Prompt preparation removes `\ce{...}`, `\pu{...}`, and `$...$` wrappers while
preserving their contents, matching the default cleanup in the pinned upstream
[prompter](https://github.com/lamalab-org/chembench/blob/45f8bad062fe552810c52be3a328d5da8597ed30/src/chembench/prompter.py#L177).
This applies to questions and answer options; chemical entity markers such as
`[START_SMILES]` / `[END_SMILES]` are also removed. Re-run preparation to update
previously generated datasets; importing an existing Gym JSONL preserves its
prompts as supplied.

Five source examples are committed in `data/example.jsonl`. Their `data/example_rollouts.jsonl` was refreshed on 2026-10-06 with the cleaned prompts and a 131,072-token output limit; all five responses completed without truncation. Full datasets are gitignored. Each row contains `responses_create_params.input` and `verifier_metadata`, which holds `question_type`, `expected_answer`, task UUID, topic, and other source metadata. Gold answers are not sent to the model.

## Automatic benchmark preparation

Use `gym eval prepare --benchmark chembench` from the Gym checkout.
The benchmark folder now downloads and converts its pinned upstream sources.
No external evaluator installation, converted local input, or manual data copying is required.
See the [benchmark README](../../benchmarks/chembench/README.md) for the current workflow.

## Grading

- MCQ: the predicted set of uppercase letters must equal the complete expected set. Order and duplicates do not matter; missing or extra choices score zero.
- Numeric: after extraction to finite Python floats, require `abs(predicted - target) < threshold`. Use the optional `relative_tolerance` metadata value directly as an absolute threshold; when absent or null, use `0.01 * target`. This matches the final `all_correct` overwrite in upstream [prompter.py](https://github.com/lamalab-org/chembench/blob/45f8bad062fe552810c52be3a328d5da8597ed30/src/chembench/prompter.py#L292), rather than the intermediate exact-equality metric.
- Upstream compatibility includes strict `<` boundaries and signed default thresholds: exact answers for zero/negative targets score zero unless a positive custom threshold is supplied. Explicit tolerance zero also rejects exact answers. These behaviors intentionally follow the pinned upstream implementation.
- Extraction: use the last complete `[ANSWER]...[/ANSWER]` block. `[ANS]`, `<ANSWER>`, and `<ANS>` variants are also accepted. Without tags, the entire response must be a bare letter list or supported numeric expression.
- Reasoning items and `<think>` / `<thinking>` blocks are excluded. Empty, refused, or unparseable answers score zero.
- Tagged answers with extra text fall back to the last uppercase letter list (MCQ) or the last numeric expression. For example, `[ANSWER]ANSWER is A[/ANSWER]` yields `A`, and `[ANSWER]5 units[/ANSWER]` yields `5`. Multi-select lists retain all choices with commas, whitespace, or `and`, including surrounding punctuation: `A, C.`, `(A, C)`, and `A and C` all yield `A, C`.
- Numeric parsing supports signs, decimals, `e` notation, fractions such as `1/2`, and powers of ten such as `3.5 × 10^-3` or `2 * 10^3`. Unsupported arithmetic and malformed expressions are rejected as a whole, rather than extracting their denominator or exponent. Units are not converted. Comma-separated numbers, NaN, infinity, and division by zero are rejected.

This is a deterministic extraction variant, not an exact reproduction of upstream extraction. Upstream uses an LLM fallback. Gym searches extra text only inside the last answer block, taking the last letter list or supported numeric expression when exact parsing fails. Untagged explanations are not searched. Outputs include `predicted_answer` and `no_answer` for inspection.

## Run

Use the model YAML shown in the [benchmark README](../../benchmarks/chembench/README.md).
The first command runs servers in the foreground; run the second in another
terminal with the same Gym environment active.

```bash
gym env start --config benchmarks/chembench/config.yaml \
    --model-type vllm_model --config /absolute/path/to/model.yaml

gym eval run --no-serve --agent chembench_simple_agent \
    --input resources_servers/chembench/data/example.jsonl \
    --output results/chembench/example_rollouts.jsonl \
    --num-repeats 1 --concurrency 4 --max-output-tokens 131072 --temperature 0
```

The `vllm_model` adapter also supports compatible hosted Chat Completions endpoints. For a native Responses endpoint, use `--model-type openai_model` with a matching model config; remove the `vllm_model` block from the example YAML. The complete prepared input is `benchmarks/chembench/data/test.jsonl`. The preparer does not emit a separate human-subset file; filter `verifier_metadata.in_human_subset` when that subset is needed.

## Validation

```bash
gym env test --resources-server chembench
gym dataset collate \
    --config resources_servers/chembench/configs/chembench.yaml \
    --output-dir /tmp/chembench-example --mode example_validation
```

The verifier tests cover exact grading, multi-select errors, scientific notation, malformed outputs, reasoning removal, refusal, schema validation, and the committed examples. `verified: false` remains set until full benchmark baselining and review.

## Licensing

Gym integration code: Apache-2.0. ChemBench data and source prompts: MIT, as declared by the dataset card. The upstream notice is included in [LICENSE-ChemBench](LICENSE-ChemBench). Dependency: `nemo-gym` (Apache-2.0).

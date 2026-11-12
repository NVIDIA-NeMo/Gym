# Long transduction

Long transduction measures how faithfully a model transforms long structured inputs.
The benchmark uses a single-turn `simple_agent` and a deterministic, CPU-only verifier.
No judge model or external tool is used for scoring.

## Tasks and scoring

| Family | Task types | Reward |
| --- | --- | --- |
| Arithmetic | `unnumbered_streaming_sum`, `streaming_sum`, `shuffled_streaming_sum` | Fraction of expressions with the correct numerical answer |
| UUID sorting | `unnumbered_uuid_sort`, `streaming_uuid_sort`, `shuffled_streaming_uuid_sort` | Fraction of lines whose entire sorted token list is correct |
| Variable expansion | `unnumbered_var_expand`, `streaming_var_expand`, `shuffled_streaming_var_expand` | Fraction of lines with the correct resolved words |
| CSV permutation | `csv_permutation_homogeneous`, `csv_permutation_heterogeneous` | Fraction of correct cells after row and column permutation |
| CSV lookup | `csv_kv_lookup` | Fraction of correctly resolved adjective/noun cells |

Rewards are fractional values in `[0, 1]`. Missing answers count as incorrect.
Numbered tasks match by index, so out-of-order answers can receive full credit even
though the prompt requests ascending order. Unnumbered tasks match by position.
Extra output beyond the expected items is not penalized. Arithmetic reward checks
the answer separately from expression copying; `item_scores` records copying,
answer correctness, and self-consistency. CSV scoring ignores the row/column label
text and compares cells positionally. Variable expansion ignores case and normalizes
whitespace. These tolerances are part of the scoring definition.

By default, `strip_reasoning: true` removes text preceding the last `</think>` or
Harmony final-channel marker. An unfinished reasoning block produces an empty
answer. Set `++long_transduction.resources_servers.long_transduction.strip_reasoning=false`
when starting the example server to disable this behavior. For the benchmark server,
use the `long_transduction_resources_server` instance name instead.

Each rollout retains its task type (`type`) and approximate input-token budget
(`target_tokens`). Aggregate metrics report accuracy by type, difficulty, context
length, type plus difficulty, and type plus context length, plus `overall_accuracy`.
For example, `target_tokens_2048` and `type_streaming_sum_target_tokens_2048` each
contain `accuracy` and `n` (rollout count). Rows without `target_tokens` still
contribute to the existing metrics but are excluded from context-length groups.
Aggregation weights each rollout equally, not each input cell or expression.
Difficulty-only groups can combine different task families.

Arithmetic aggregate accuracy requires **both correct expression copying and a
correct answer** for each item, matching the historical gym-evals plots. This rule
applies to every accuracy group and `overall_accuracy`. The stored `reward` and
`answer_correct` fields remain answer-only diagnostics, so generic reward summaries
can differ from benchmark accuracy. Legacy arithmetic rows with missing or empty
`item_scores` fall back to answer-only accuracy, as in gym-evals. Other task families
use their existing answer-correctness scores.

## Prepare the benchmark

Run commands from the Gym repository root with its environment activated:

```bash
gym eval prepare --benchmark long_transduction
```

The dataset config declares `tiktoken` and `wonderwords` as preparation dependencies;
Gym installs them using `uv` in the active Python environment. The first tokenizer
use may download the `cl100k_base` encoding. Preparation writes
`benchmarks/long_transduction/data/long_transduction.jsonl` and reports progress for
each family, context size, and sample. Existing data is reused. To regenerate it:

```bash
gym eval prepare --benchmark long_transduction \
  +use_cached_prepared_benchmarks=false +prepare_script_args.force=true
```

The default generator targets 2K, 4K, 8K, 16K, 32K, and 64K input tokens, with five
samples per difficulty. These are approximate `cl100k_base` prompt lengths, not
model-specific context guarantees. Reserve room for the answer and chat template
when selecting the serving model's context limit. Some variable-pool sizes may not
fit a short budget and are skipped with a progress message. Arithmetic generation
uses Python's random state; preserve the generated JSONL when comparing models.
Other task families use deterministic local seeds. Full generated datasets are
local artifacts and should not be committed.

## Run the five examples

The committed examples already contain `responses_create_params.input` and verifier
metadata. To collect fresh responses with an OpenAI-compatible endpoint, start Gym
in one terminal:

```bash
gym env start \
  --config resources_servers/long_transduction/configs/long_transduction.yaml \
  --model-type vllm_model \
  +policy_base_url=http://127.0.0.1:8000/v1 \
  +policy_api_key=dummy +policy_model_name=YOUR_MODEL
```

Then collect in another terminal:

```bash
gym eval run --no-serve --agent long_transduction_simple_agent \
  --input resources_servers/long_transduction/data/example.jsonl \
  --output results/long-transduction/example_rollouts.jsonl \
  --num-repeats 1 --max-output-tokens 4096 --temperature 0
```

## Run all context lengths together

Start servers using `--config benchmarks/long_transduction/config.yaml` and your
model settings, then collect the entire generated dataset in one run:

```bash
gym eval run --no-serve --agent long_transduction_agent \
  --input benchmarks/long_transduction/data/long_transduction.jsonl \
  --output results/long-transduction/rollouts.jsonl \
  --num-repeats 1 --temperature 0 \
  +prompt_config=benchmarks/prompts/generic/default.yaml
```

The prompt template materializes each row's `question` as the user message.
Newly generated rows set `responses_create_params.max_output_tokens` to
`target_tokens * 3 // 2`, matching the historical gym-evals output budget for each
length. A global `--max-output-tokens` overrides these per-row budgets. Older cached
datasets retain their contents; use the regeneration command above to generate
these budgets, or supply an explicit global limit when evaluating an older dataset.
Preserve the original dataset for comparisons because regenerating changes arithmetic tasks.

All lengths share `rollouts.jsonl` and `rollouts_aggregate_metrics.json`; the latter
includes the context-length and task-type groups. No per-length files or external
aggregation script are needed. Arithmetic accuracy uses the same joint expression
and answer correctness rule as the historical gym-evals plots.

## Validation and example provenance

```bash
gym dataset collate \
  --config resources_servers/long_transduction/configs/long_transduction.yaml \
  --output-dir /tmp/long-transduction-examples --mode example_validation
gym env test --resources-server long_transduction +should_validate_data=true
```

The tests cover all 12 verifier branches, partial rewards, missing output, reasoning
formats, metric aggregation, configuration routing, and the preparation API. The
small generation test also checks generated reference answers against each scorer;
it requires the preparation dependencies. All scoring tests run locally without an
inference endpoint, GPU, or Ray cluster.

`data/example_rollouts.jsonl` contains real responses from `Qwen3.5-35B-A3B` collected
on July 8, 2026, selected from
`gym-evals/long-transduction/results/qwen3-5-35b-a3b/rollouts.jsonl`.
Selection takes the first completed 2048-target-token row for each of these five
types: shuffled UUID sorting, numbered variable expansion, unnumbered arithmetic,
heterogeneous CSV permutation, and CSV lookup. Selection does not filter by reward.
Their original task indices are 101, 214, 45, 157, and 160 (rollout index 0).
The corresponding requests form `data/example.jsonl`. The original response IDs,
timestamps, token usage, and output text are retained. Scores were recomputed with
this verifier and checked against the recorded scores. These are historical model
responses re-scored locally, not newly collected responses or a complete baseline.
`test_committed_real_rollouts_reproduce_rewards` repeats this check without a model.
`data/example_metrics.json` is generated by Gym's example-data validation.

Keep `verified: false` until a complete benchmark baseline has been reviewed.

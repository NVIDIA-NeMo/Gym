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

Aggregate metrics report accuracy by task type, difficulty, and their combination,
plus `overall_accuracy`. Aggregation weights each rollout equally, not each input
cell or expression. Compare context lengths separately; difficulty-only groups can
combine different task families and do not replace the type-specific groups.

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

For the full benchmark, use `--config benchmarks/long_transduction/config.yaml`
when starting servers, `--agent long_transduction_agent`, and the generated JSONL
as input. The benchmark config materializes each row's `question` as the user message.
Use an output-token limit large enough for the selected context sizes and tasks.

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

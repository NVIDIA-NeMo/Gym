# Indic AlpacaEval

Gym implementation of
[AI4Bharat Indic AlpacaEval](https://huggingface.co/datasets/ai4bharat/indic-alpaca-eval),
a translated instruction-following benchmark evaluated with an LLM judge.

## What it tests

The benchmark evaluates 805 instructions per language across Bengali, Gujarati,
Hindi, Kannada, Marathi, Malayalam, Nepali, Odia, Punjabi, Tamil, Telugu, and
Urdu: **9,660 tasks across 12 languages**.

Each response is evaluated independently on four dimensions:

- **Instruction following:** adherence to the request and its constraints.
- **Correctness:** factual accuracy and sound reasoning in the final answer.
- **Usefulness:** relevance, completeness, and practical value.
- **Language quality:** fluency, clarity, and appropriate use of the requested language.

The judge assigns integer scores from 1 to 5 for each dimension and a separate
holistic overall score. It evaluates the final answer after reasoning is
removed. The overall score is assigned independently of the dimension scores.

This is a custom pointwise protocol. Its scores are not official AlpacaEval
pairwise or length-controlled win rates. See the
[resources server](../../resources_servers/alpaca_pointwise_judge/README.md)
for the rubric, judge configuration, and scoring details.

## Data

Data is loaded from
[ai4bharat/indic-alpaca-eval](https://huggingface.co/datasets/ai4bharat/indic-alpaca-eval/tree/94f35aa27454ddcfbf1128b8c8b61093ab284163)
at revision `94f35aa27454ddcfbf1128b8c8b61093ab284163`.
The source contains 805 rows with the original English instruction and
translation columns for 14 languages. This integration selects 12:
`bn gu hi kn mr ml ne or pa ta te ur`. Preparation also accepts `ka`, `mar`,
and `mal` as aliases for `kn`, `mr`, and `ml`.

[prepare.py](prepare.py) validates 805 unique source IDs, ordered positions
0–804, and nonempty translations for every selected language. Translations
remain attached to their original source row. English is excluded from
evaluation. The published dataset does not expose per-row translation-review
status; preparation records that status as unknown and applies no review filter.

Preparation writes `data/indic_alpaca_eval_benchmark.jsonl`, one JSONL shard
per language, `data/smoke.jsonl`, and a provenance manifest. Tasks retain stable
source identifiers, language metadata, and expected coverage. Language shards
share an evaluation-set ID for aggregation.

The dataset declares CC-BY-NC-4.0. Benchmark text is loaded at runtime and
generated files are ignored by Git. The repository includes only synthetic
Apache-2.0 examples.

## Example usage

Run from the repository root with Gym and [requirements.txt](requirements.txt)
installed. Configure the
[judge endpoint](../../resources_servers/alpaca_pointwise_judge/README.md#configuring-the-judge)
and supply a served candidate model:

```bash
gym eval prepare --benchmark indic_alpaca_eval

gym eval run --benchmark indic_alpaca_eval \
  --model-type vllm_model --model YOUR_MODEL \
  --model-url http://127.0.0.1:8000/v1 --model-api-key EMPTY \
  --output results/indic_alpaca_eval/rollouts.jsonl \
  --max-output-tokens 4096
```

The example overrides the prepared 120,000-token output budget. Choose a
budget that fits the candidate model's context window and record it with the
results. See [usage.md](usage.md) for local data, language selection, sampling
settings, the Gemma recipe, language jobs, and smoke validation.

## Metrics

Scores retain the 1–5 scale. Repeated responses are averaged within each task,
then tasks within each language, then languages with equal weight.

- `alpaca_pointwise/macro_mean_overall_score`: holistic score across languages.
- `alpaca_pointwise/macro_mean_<dimension>`: each rubric dimension across languages.
- `alpaca_pointwise/<lang>/mean_overall_score`: holistic score for one language.
- `alpaca_pointwise/<lang>/mean_<dimension>`: each rubric dimension for one language.
- `alpaca_pointwise/complete`: whether all expected tasks have valid judgments.

Macro scores are emitted only for complete evaluations. A language shard
reports local scores; aggregate all expected shards to obtain multilingual
scores. Gym reward is `(overall_score - 1) / 4`, on a 0–1 scale.

# Alpaca pointwise judge

LLM-judge resources server for
[Indic AlpacaEval](../../benchmarks/indic_alpaca_eval/README.md). It evaluates
each candidate response independently using Gym's `SimpleResourcesServer`,
`simple_agent`, model adapters, and rollout collection and aggregation.

## What it does

The judge receives the translated instruction, expected language, and final
answer. Structured reasoning and tagged thinking are removed before judging.
Model identity, English source text, and translation-review flags are excluded
from the judge prompt. Explicit output-language instructions take precedence
over the dataset language.

The [anchored rubric](prompts/alpaca_pointwise_v1.yaml) adapts AlpacaEval's
response-quality criteria to pointwise ratings:

| Score | What it measures |
| --- | --- |
| `instruction_following` | Adherence to the instruction and its constraints. |
| `correctness` | Factual accuracy and sound reasoning in the final answer. |
| `usefulness` | Relevance, completeness, and practical value. |
| `language_quality` | Fluency, clarity, and appropriate use of the requested language. |
| `overall_score` | Holistic quality, assigned independently of the four dimensions. |

All scores are integers from 1 to 5. The structured judgment also includes a
concise rationale and detected response language. Empty answers and refusals
are evaluated with the same rubric. These custom pointwise scores are not
official AlpacaEval pairwise or length-controlled win rates.

Requests use `nemo_gym.judge.call_judge` through Gym's `openai_model` server.
Malformed or truncated judgments receive up to three attempts. Exhausted
judge failures raise `JudgeError`; Gym preserves the candidate in its
failed-measurement sidecar for retry.

## Data schema

Each input row contains `responses_create_params.input` for the candidate and
`verifier_metadata` for scoring and coverage. The metadata includes the
question, source row ID and index, language code and name, evaluation-set ID,
expected per-language counts, and translation-review status. The published
AI4Bharat dataset has no per-row review flags: `judge_pass_stage` is
`not_provided` and `human_evaluation_pending` is `null`. See
[task_data.py](task_data.py) for validation rules.

The [example data](data/example.jsonl) contains five synthetic arithmetic
tasks in Bengali, Hindi, Kannada, Tamil, and Urdu. Benchmark preparation loads
translated instructions from
[ai4bharat/indic-alpaca-eval](https://huggingface.co/datasets/ai4bharat/indic-alpaca-eval)
at runtime. Code and synthetic examples are Apache-2.0; the external dataset
declares CC-BY-NC-4.0.

## Example usage

From the repository root, configure the judge and start Gym with a served
candidate model:

```bash
gym env start --resources-server alpaca_pointwise_judge \
  --model-type vllm_model --model YOUR_MODEL \
  --model-url http://127.0.0.1:8000/v1 --model-api-key EMPTY
```

In another terminal, collect the example tasks against those servers:

```bash
gym eval run --no-serve \
  --agent alpaca_pointwise_judge_simple_agent \
  --input resources_servers/alpaca_pointwise_judge/data/example.jsonl \
  --output results/alpaca_pointwise_judge/example/rollouts.jsonl \
  --max-output-tokens 4096 --num-repeats 1
```

See the [benchmark usage guide](../../benchmarks/indic_alpaca_eval/usage.md)
for dataset preparation, sampling settings, and language jobs.

## Configuring the judge

[configs/alpaca_pointwise_judge.yaml](configs/alpaca_pointwise_judge.yaml)
reads three environment variables:

- `INDIC_ALPACA_JUDGE_BASE_URL`: OpenAI-compatible endpoint base URL.
- `INDIC_ALPACA_JUDGE_MODEL`: judge model identifier.
- `INDIC_ALPACA_JUDGE_API_KEY`: endpoint credential.

The endpoint must support Chat Completions with JSON-schema structured output
and the configured reasoning effort. Defaults are temperature 0, low reasoning
effort, 4,096 maximum output tokens, and one concurrent judge request per model
server process. Override provider parameters and the budget in the Gym config
to match the evaluation protocol. The endpoint and judge model are explicitly
configured by the caller.

## Metrics

Gym reward is `(overall_score - 1) / 4`, on a 0–1 scale. Quality metrics use
the original 1–5 scores. Repeats are averaged within each task, tasks are
averaged within each language, and macro scores weight languages equally.
The holistic score is an independent judgment, not the arithmetic mean of the
four rubric dimensions.

- `alpaca_pointwise/macro_mean_overall_score`: headline multilingual score.
- `alpaca_pointwise/macro_mean_<dimension>`: multilingual score for each rubric dimension.
- `alpaca_pointwise/<lang>/mean_overall_score`: per-language holistic score.
- `alpaca_pointwise/<lang>/mean_<dimension>`: per-language rubric score.
- `alpaca_pointwise/<lang>/standard_error`: standard error across task-level overall scores.
- `alpaca_pointwise/{expected,judged,complete}`: total task coverage.

Per-language metrics also report expected and judged task counts,
completeness, invalid judgments, empty final answers, and judged tasks
explicitly flagged for translation review. An unknown review status does not
count as a flagged task. Macro scores are emitted only when every expected
language has all expected tasks judged and no invalid judgments. Mixed
evaluation sets, duplicate task identities, and inconsistent coverage are
rejected. A language shard reports local scores while retaining the shared
multilingual coverage requirement.

## Example rollouts

The checked-in [rollouts](data/example_rollouts.jsonl) contain five completed
synthetic tasks generated through Gym with `gcp/google/gemini-3.8-flash` as both
candidate and judge. Both used temperature 0 and low reasoning effort. The
candidate output budget was 8,192 tokens; the judge budget was 65,536 tokens.
All five final answers are nonempty and all judgments are valid on the first
attempt. These examples demonstrate integration, not full benchmark quality.

Validate the included evidence with:

```bash
python -m benchmarks.indic_alpaca_eval.check_run \
  --inputs resources_servers/alpaca_pointwise_judge/data/example.jsonl \
  --rollouts resources_servers/alpaca_pointwise_judge/data/example_rollouts.jsonl \
  --max-output-tokens 8192 --smoke
```

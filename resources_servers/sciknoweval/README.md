# SciKnowEval V2

For preparation and evaluation through `gym eval`, use the [SciKnowEval V2 benchmark integration](../../benchmarks/sciknoweval/README.md).

Gym integration of [SciKnowEval V2](https://huggingface.co/datasets/hicai-zju/SciKnowEval). The dataset has 28,392 questions across Biology, Chemistry, Material, and Physics and cognitive levels L1-L5.

| Answer family | Tasks | Reward |
|---|---:|---|
| Multiple choice | 18,670 | Exact selected-letter match |
| True/false | 3,228 | Yes/No match; true/false synonyms accepted |
| Equation filling | 300 | Gold equation is a substring of the extracted answer |
| Relation extraction | 1,364 | Task-specific LLM judge |
| Open-ended QA | 4,830 | Task-specific LLM judge |

## Data and preparation

Use `gym eval prepare --benchmark sciknoweval` from the Gym checkout.
The benchmark folder now downloads and converts its pinned upstream sources.
No external evaluator installation, converted local input, or manual data copying is required.
See the [benchmark README](../../benchmarks/sciknoweval/README.md) for the current workflow.

Generation receives only the instruction and question. Gold labels, original instruction, domain/level provenance, and rendered judge rubrics are in `verifier_metadata`. Preparation removes restrictive no-explanation clauses from original instructions and requests a final `Answer:` line for structured answers; this integration preserves those prepared prompts.

## Scoring details

MCQ extraction uses final-answer phrase / boxed / `Answer:` precedence. True/false, equation, and relation extraction are implemented in the bundled `grading.py`. Think/thinking blocks and separate reasoning items are excluded from answer grading.

MCQ extraction accepts `Answer: (B)` and `Answer: Option B` as well as bare labels after `Answer:`. Existing final-answer/boxed precedence is retained. True/false extraction includes negation, so `Answer: Not true` maps to `No`.

Equation normalization (arrows, spacing, and optional state symbols) is exposed as `symbolic_correct_normalized`; it does not replace the strict reward.

Judge scores use these normalizations:

- `Rating: 1` through `Rating: 5`: `(rating - 1) / 4`.
- Yes/No rubric: 1 / 0.
- Agreement options `(A)` through `(E)`: 0.5, 0.75, 1, 0.25, 0.

These fractional scores are the reward for judged tasks. Relations are extracted before judging; open-ended answers use the full final response text. Empty or unextractable answers receive zero without a judge call. Unparseable judge verdicts receive zero with `judge_parse_ok=false`; judge service failures use Gym's judge-failure sidecar instead of counting as wrong answers. The judge response is retained for audit.

Metrics include `overall_score`, scores by level/domain/answer type, and `level_macro_score`. Overall and per-level scores weight questions equally; the level macro score weights levels equally.

Per-level aliases `mean/L1` through `mean/L5` appear at the end of the opening
`mean/*` block in `agent_metrics` and in `key_metrics`. Only levels with scored
questions are emitted: an L3/L5 run reports `mean/L3` and `mean/L5`. These equal
the corresponding `by_level/L*/score` values; repeats are averaged within each
question before averaging questions within the level.

## Run

The benchmark config uses the policy model as judge. To use a separate
Responses-compatible judge, append `--config resources_servers/sciknoweval/configs/judge_model.yaml`
and supply `judge_base_url`, `judge_api_key`, and `judge_model_name` in your private
model configuration. A Chat Completions judge can instead use Gym's `vllm_model`
adapter. Apply judge/model overlays after the benchmark or resource config.

Use the model YAML shown in the [benchmark README](../../benchmarks/sciknoweval/README.md).
The first command runs servers in the foreground; run the second in another
terminal with the same Gym environment active.

```bash
gym env start --config benchmarks/sciknoweval/config.yaml \
    --model-type vllm_model --config /absolute/path/to/model.yaml

gym eval run --no-serve --agent sciknoweval_simple_agent \
    --input resources_servers/sciknoweval/data/example.jsonl \
    --output results/sciknoweval/example_rollouts.jsonl \
    --num-repeats 1 --concurrency 4 --max-output-tokens 16384 --temperature 0
```

The standalone resource config leaves the judge unset and can be used without one only for MCQ, true/false, or filling inputs. `judge_max_concurrency` defaults to 32. The benchmark config sets `judge_responses_create_params` to 16,384 output tokens and temperature 0; the standalone server default is 2,048 tokens.

## Validation and licensing

Run `gym env test --resources-server sciknoweval`. Tests cover each answer family and judge scale, exact prompt assembly, missing answers, judge failures, schema validation, and aggregate weighting. `verified: false` remains until model baselining and review.

Integration and grading code: Apache-2.0. Source dataset: MIT, as declared by the [dataset card](https://huggingface.co/datasets/hicai-zju/SciKnowEval). Dataset reference revision: `92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2`. Dependency: Gym (Apache-2.0).

Judge rubrics are downloaded at preparation time from pinned upstream source and checked by SHA-256. The source repository has no standalone code license notice; its rubrics are not vendored or relicensed here.

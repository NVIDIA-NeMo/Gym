# Alpaca pointwise judge

A native `SimpleResourcesServer` for the custom Indic AlpacaEval pointwise
protocol. Each answer receives integer 1–5 ratings for instruction following,
correctness, usefulness, language quality, and a separate holistic overall score.
The holistic score is not an average of the four dimensions.

The anchored rubric lives in
[prompts/alpaca_pointwise_v1.yaml](prompts/alpaca_pointwise_v1.yaml).
It adapts AlpacaEval's response quality, relevance, accuracy, and helpfulness
criteria to independent scoring. These are **custom scores**, not official
AlpacaEval pairwise or length-controlled win rates.

## Reused Gym components

- `SimpleResourcesServer`, `BaseVerifyRequest`, and `BaseVerifyResponse` provide
  the verification endpoint and stateless re-verification.
- `nemo_gym.judge.call_judge` and `ServerClient` route requests through the
  existing `openai_model` server. Transport and exhausted parse failures raise
  `JudgeError`; Gym preserves the candidate in its failure sidecar for retry.
- `load_prompt_config` / `fill_prompt` load the native YAML prompt.
- `simple_agent` generates a response using the existing `vllm_model` or
  another Gym policy server; `legacy_agent` exposes the environment interface.
- Native rollout collection and aggregation write responses, judgments,
  coverage, and metrics. Arena's pairwise battle/Elo implementation does not
  apply to this independent scoring protocol.

The judge sees only the translated instruction, expected language, and final
answer. Model identity, English source, review flags, and structured or tagged
reasoning are excluded. Explicit output-language instructions override the
dataset language. Empty answers and refusals use the same rubric. Strict JSON
validation requires all scores, a concise rationale, and detected language;
malformed/truncated judgments get up to three attempts.

## Metrics

`reward = (overall_score - 1) / 4` is a graded Gym reward.
The headline metric is `alpaca_pointwise/macro_mean_overall_score` on the
original 1–5 scale. The four `macro_mean_<dimension>` metrics also average
equally across languages. They are emitted only when every expected language has all
expected tasks judged. Per-language metrics include overall and dimension
means, standard error across task scores, judged/expected counts, empty final
answers, and translation-review-pending counts. Repeats are averaged within
each task before averaging tasks. Generic pass@k metrics are not the headline
for this graded protocol.

Prepared rows carry an evaluation-set ID and expected counts. Aggregation
rejects mixed sets, duplicate task identities, and inconsistent coverage.
A language shard reports its local scores but remains incomplete until the
other expected shards are aggregated. Gym records failed/masked measurements
separately; retry these before reporting a complete benchmark.

## Configuration and data

[configs/alpaca_pointwise_judge.yaml](configs/alpaca_pointwise_judge.yaml)
reads the endpoint, model, and key from `INDIC_ALPACA_JUDGE_BASE_URL`,
`INDIC_ALPACA_JUDGE_MODEL`, and `INDIC_ALPACA_JUDGE_API_KEY`. The reusable
server defaults to temperature 0, low reasoning effort, 4,096 output tokens,
and one concurrent judge request per model-server process. Override the
budget and provider parameters for the evaluation protocol being reproduced.
No credential is stored here.

The included five example rollouts were generated through Gym with
`gcp/google/gemini-3.8-flash` as both candidate and judge. The candidate used
8,192 maximum output tokens, temperature 0, and low reasoning; the judge used
65,536 maximum output tokens, temperature 0, and low reasoning. All five
synthetic arithmetic tasks (Bengali, Hindi, Kannada, Tamil, Urdu) completed
with nonempty final answers and valid judgments. This is integration smoke
evidence, not a model quality comparison or a full AlpacaEval score.

See [the Indic benchmark](../../benchmarks/indic_alpaca_eval/README.md) for
preparation and Gemma execution. Five synthetic arithmetic examples are included
for schema validation; benchmark text is downloaded or read locally at runtime.

Code and synthetic examples: Apache-2.0. Dependency: NeMo Gym (Apache-2.0).
The external Indic AlpacaEval dataset declares CC-BY-NC-4.0 and is not vendored.

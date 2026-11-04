# Chartography resources server

Professional chart reading: one chart image and one question per row, graded against a golden answer by the
benchmark's own judge. Numeric parts of a golden answer carry an acceptable range, for example
`Head: 34.5 m (acceptable range 33.5 to 35.5 m)`. This is the answer format of the
[Chartography](https://github.com/surge-ai/chartography) benchmark (Surge AI): a response passes only when every
part of the golden answer is matched.

## Grading

Mirrors [Surge AI's own scorer](https://github.com/surge-ai/chartography/blob/3f1bf837232d3918c2cc35c6e2418c9e70d2a57b/src/chartography/scorer.py)
(`src/chartography/scorer.py`, see `scorer.py`):

- One judge call per response with the upstream prompt (`prompt_templates/judge.txt`, unchanged; a test checks
  its hash) and the upstream JSON schema: a `scores` array with one 0/1 entry per criterion, each with the
  extracted answer and a rationale. No temperature or token limit is set, as upstream.
- The judge sees the question, the golden answer and the response's final answer. It never sees the reasoning
  (separate `reasoning` items when the policy server runs with a reasoning parser) or the chart.
- Strict parsing (markdown fences stripped, schema validated, score coerced to Decimal) with up to
  `judge_max_attempts` attempts (default 5, upstream `MAX_RETRIES=4`). A criterion passes iff its score,
  rounded half-up to four decimals, is at least 1.

`reward_mode`:

- `strict_pass` (default): 1 when every criterion passes, else 0. This is the benchmark metric.
- `mean_criteria`: the fraction of criteria that pass (partial credit for multi-part answers).

Every verify response also reports `all_pass`, `mean_criterion_pass`, the per-criterion verdicts and the judge's
raw output. A row without a question or golden answer is masked (`mask_sample`,
`failure_kind: chartography:invalid_reference`), since it cannot be measured.

Two deliberate divergences from upstream, as in `gdp_pdf`:

- **An empty answer scores 0 without a judge call.** Upstream sends it to the judge, which scores it 0.
- **Unparseable judge output fails the row.** Upstream marks it "unscored"; here `JudgeError` routes the row to
  the failures sidecar, so it is excluded from the metrics rather than scored 0.

The judge model is whatever you configure. The benchmark's leaderboard standard is Gemini 3.5 Flash with ten
runs per task.

## Data format

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": [
    {"type": "input_text", "text": "<question>"},
    {"type": "input_image", "image_url": "data:image/png;base64,<chart>", "detail": "auto"}
  ]}]},
  "verifier_metadata": {
    "task_id": "<id>",
    "domain": "<chart or question type, used for per-domain metrics>",
    "prompt": "<question>",
    "golden_answer": "<golden answer, one part per line>"
  }
}
```

`data/example.jsonl` holds five synthetic charts drawn from known values (`make_example_data.py`; needs
matplotlib). The Chartography benchmark itself is not included and is for evaluation only.

## Judge

`configs/judge_model_openai.yaml` points at any OpenAI-compatible endpoint with JSON-schema structured output.
Set `judge_base_url`, `judge_api_key` and `judge_model_name` in `env.yaml`.

## Running

```bash
gym env test +entrypoint=resources_servers/chartography   # unit tests, judge mocked

gym env start --config resources_servers/chartography/configs/chartography.yaml --model-type openai_model

gym eval run --no-serve \
  --agent chartography_simple_agent \
  --input resources_servers/chartography/data/example.jsonl \
  --output results/example_rollouts.jsonl \
  --num-repeats 1
```

The policy model must accept image inputs.

## Licensing

Code: Apache 2.0. Example data: Apache 2.0 (synthetic charts rendered by `make_example_data.py`).

# Chartography resources server

Professional chart reading: one chart image and one question per row, graded against a reference answer by
an LLM judge. Numeric parts of a reference answer carry an acceptable range, for example
`Head: 34.5 m (acceptable range 33.5 to 35.5 m)`. This is the answer format of the
[Chartography](https://github.com/surge-ai/chartography) benchmark (Surge AI), and the grading follows its
protocol: a response passes only when every part of the reference answer is matched.

## Grading

- One judge call per response, with the prompt in `prompt_templates/judge.txt` and a JSON schema that asks
  for one verdict per reference part (`correct` true or false, the matching piece of the answer, a reason).
- The judge sees the question, the reference answer and the response's final answer. It never sees the
  reasoning (separate `reasoning` items when the policy server runs with a reasoning parser) or the chart.
- A numeric part is correct when the answer lies inside its acceptable range (endpoints included); a part
  without a range must match at the requested precision. Labels must name the same item; lists must have
  exactly the reference items (in order when the reference says so).
- Judge output that does not parse is retried up to `judge_max_attempts` times (default 5); after that the
  row goes to the failures sidecar instead of scoring 0.

`reward_mode`:

- `strict_pass` (default): 1 when every part is correct, else 0. This is the benchmark metric.
- `mean_criteria`: the fraction of correct parts (partial credit for multi-part answers).

Every verify response also reports `all_pass`, `mean_criterion_pass`, the per-part verdicts and the judge's
raw output. An empty answer scores 0. A row without a question or reference answer is masked
(`mask_sample`, `failure_kind: chartography:invalid_reference`), since it cannot be measured.

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
    "golden_answer": "<reference answer, one part per line>"
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

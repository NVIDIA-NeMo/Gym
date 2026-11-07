# ui_generation

Single-turn front-end UI generation. The policy gets a request for a web UI (for example a
Next.js page, a React component or a plain HTML page, each in one file) and answers in
Markdown with the code in fenced blocks. An LLM judge reads only that code and scores
prompt adherence and quality from 0 to 4, using the task's reference answer as a
guideline. The reward is the quality score divided by 4.

- Integration profile: `custom-gym-verifier` (`simple_agent`, one model call)
- Scorer: [`resources_servers/ui_generation`](../../resources_servers/ui_generation/README.md)
  (judge, verdict schema, reward, failure handling)

## Data

The environment ships only 5 synthetic example tasks written for Gym (two Next.js, two
React, one HTML) in `resources_servers/ui_generation/data/example.jsonl`, the five rows
Gym's data check requires, with the `example_metrics.json` and `example_rollouts.jsonl`
that check also requires.

To train or evaluate on your own tasks, write them in the input format and convert them.

### Input format

One JSON object per line, in either shape:

```json
{"id": "weather-card", "category": "react", "prompt": "Build a weather card component that shows a city, its temperature and an icon for the conditions.", "reference_answer": "```tsx\nexport default function WeatherCard() { ... }\n```"}
{"id": "faq-page", "category": "html", "messages": [{"role": "user", "content": "Make an FAQ page whose answers expand when their question is clicked."}, {"role": "assistant", "content": "```html\n<!doctype html>\n...\n```"}]}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `prompt` | string | one of `prompt`/`messages` | The user request |
| `messages` | list of `{"role": "system"\|"user"\|"assistant", "content": string}` | one of `prompt`/`messages` | The conversation; a final assistant message is the reference answer unless `reference_answer` is given |
| `reference_answer` | string | yes, unless `messages` ends with an assistant message | A good answer, usually Markdown with fenced code; shown to the judge only |
| `id` | string | no | Unique task identifier |
| `category` | string | no | Task label, such as the target stack; metrics report it |

Prompt messages need nonempty string content and at least one user message. Lines with
a duplicate `id`, no reference answer or an unknown role are refused. `--id-field` and
`--category-field` read the id and label from other fields, including nested ones
(`--category-field meta.stack`); `--limit N` converts only the first N tasks.

```bash
python environments/ui_generation/prepare.py --input my_tasks.jsonl \
    --output resources_servers/ui_generation/data/train.jsonl
```

`data/*train.jsonl` and `data/*validation.jsonl` are gitignored. To use a converted file,
pass it to `gym eval run --input`, or add it as a `train`/`validation` dataset (with its
`license`) to `config.yaml`.

## Task contract

- Model-visible input: the task's prompt messages, unchanged, with no added system
  prompt. No tools.
- The reference answer goes to `verifier_metadata.reference_answer` only.
- The verifier reads the policy's last assistant message, without reasoning.
- Nothing is built, executed or rendered: the judge reads the code.

## Run

Put the policy and judge endpoints in `env.yaml` at the repository root:

```yaml
policy_base_url: https://<openai-compatible-endpoint>/v1
policy_api_key: ${oc.env:POLICY_API_KEY}
policy_model_name: <policy model>
judge_base_url: https://<openai-compatible-endpoint>/v1
judge_api_key: ${oc.env:JUDGE_API_KEY}
judge_model_name: <judge model>
```

```bash
# Terminal 1: start the servers
gym env start --environment ui_generation --model-type vllm_model

# Terminal 2: collect rollouts from the examples
gym eval run --no-serve --environment ui_generation --model-type vllm_model --agent ui_generation_agent \
    --input resources_servers/ui_generation/data/example.jsonl \
    --output results/ui_generation_example/rollouts.jsonl \
    --num-repeats 1 --concurrency 2 --temperature 1.0 --top-p 1.0 --max-output-tokens 65536
```

## Scoring and run conditions

- Reward in [0, 1], higher is better: `quality / 4` by default; `reward_score` selects
  `prompt_adherence`, `mean` or `min` instead. Both raw scores are reported per rollout
  and averaged in the aggregate metrics, overall and per category
  (`react/mean/quality_score`, ...).
- Cut-off or empty policy answers score 0 without a judge call. A failed judge call or a
  judge reply without a schema-valid verdict masks the row (`judge_failed`).
- Use a judge from a different model family than the policy: a judge tends to prefer
  answers of its own family.
- Determinism: stochastic (sampled policy and judge).

## Licensing information

Apache 2.0, including the synthetic tasks in `resources_servers/ui_generation/data/`.

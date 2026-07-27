# SECQUE

[SECQUE](https://huggingface.co/datasets/nogabenyoash/SecQue) is an open-book
financial analysis benchmark with 565 expert-written questions grounded in
10-K and 10-Q excerpts. This recipe reuses Gym's
[`equivalence_llm_judge`](../../resources_servers/equivalence_llm_judge/README.md)
resource server with a finance-specific judge prompt.

The relevant excerpt for each question is supplied in the prompt, so this
measures reasoning over the correct context rather than retrieval.

The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Verification

The judge in `prompts/judge.txt` rates each answer against the reference on a
0-2 scale, measuring correctness rather than completeness and tolerating
non-essential rounding differences:

| Rating | Meaning | Reward |
|--------|---------|--------|
| `[[2]]` | Correct | 1.0 |
| `[[1]]` | Partially correct | 0.0 |
| `[[0]]` | Does not match the reference | 0.0 |

The judge sees the model's full generation, not a regex-extracted span.

## Configure a model

Create `env.yaml` in the Gym repository root. For example, to use GPT-5 mini:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: ${oc.env:OPENAI_API_KEY}
policy_model_name: gpt-5-mini
```

Then export your API key:

```bash
export OPENAI_API_KEY=...
```

### Grading model

By default the `policy_model` also grades its own answers. Add
`judge_model_name` to grade with a different model instead:

```yaml
judge_model_name: gpt-5
```

Add `judge_base_url` and `judge_api_key` as well if the judge is served from a
different endpoint. Fixing the judge to one model keeps scores comparable when
you evaluate several policy models against each other.

## Prepare

Run from the Gym repository root:

```bash
gym eval prepare --benchmark secque
```

This downloads the single 565-row Hugging Face split and writes
`benchmarks/secque/data/secque_benchmark.jsonl`.

## Start servers

```bash
gym env start \
  --benchmark secque \
  --model-type openai_model
```

## Collect rollouts

In another terminal:

```bash
gym eval run --no-serve \
  --agent secque_simple_agent \
  --input benchmarks/secque/data/secque_benchmark.jsonl \
  --output results/secque_rollouts.jsonl \
  --prompt-config benchmarks/secque/prompts/default.yaml \
  --num-repeats 1
```

Use `--limit 1` for a quick end-to-end check.

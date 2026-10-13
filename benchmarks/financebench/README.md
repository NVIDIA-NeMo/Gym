# FinanceBench

[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) is an
open-book financial question-answering benchmark. The public dataset contains
150 annotated questions, reference answers, and evidence from SEC filings. This
recipe reuses Gym's
[`equivalence_llm_judge`](../../resources_servers/equivalence_llm_judge/README.md)
resource server with a finance-specific judge prompt.

The gold evidence for each question is supplied in the prompt, so this measures
reasoning over the correct excerpt rather than retrieval. Scores are not
comparable to FinanceBench results that require the model to find the evidence
itself.

The public dataset is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

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
gym eval prepare --benchmark financebench
```

This downloads the public 150-question Hugging Face dataset and writes
`benchmarks/financebench/data/financebench_benchmark.jsonl`.

## Run

Either form below works.

A single command starts the servers, collects rollouts, and shuts the servers
down. It resolves the agent, the data file, and the prompt config from
`config.yaml`:

```bash
gym eval run \
  --benchmark financebench \
  --split benchmark \
  --model-type openai_model \
  --output results/financebench/rollouts.jsonl
```

To reuse one server across several runs, start it in one terminal:

```bash
gym env start \
  --benchmark financebench \
  --model-type openai_model
```

and collect against it from another. This form takes no benchmark config, so
the agent, input, and prompt config are named explicitly:

```bash
gym eval run --no-serve \
  --agent financebench_simple_agent \
  --input benchmarks/financebench/data/financebench_benchmark.jsonl \
  --output results/financebench/rollouts.jsonl \
  --prompt-config benchmarks/financebench/prompts/default.yaml \
  --num-repeats 1
```

Use `--limit 1` for a quick end-to-end check.

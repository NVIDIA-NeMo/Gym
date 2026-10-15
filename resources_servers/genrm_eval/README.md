# GenRM evaluation

Evaluate a GenRM model against chat preferences. `simple_agent` calls the model;
`genrm_eval` resource server scores its ranking.

## Inputs

Preparation puts dataset records into Gym's input format and creates the
comparisons required by each chat benchmark.

```json
{
  "id": "pair-1",
  "responses_create_params": {
    "input": [{"role": "user", "content": "What is 1 + 1?"}],
    "metadata": {"response_1": "2", "response_2": "3"}
  },
  "verifier_metadata": {"pair_id": "pair-1", "source": "dataset-1", "label": "A>B"}
}
```

The GenRM model's chat template contains evaluation instructions for inference;
see the [Nemotron GenRM quick-start guide](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM#quick-start-guide)
for an example. Gym adds no extra prompt (`prompt_config: null`).

`genrm_model` converts candidate metadata into `response_1`/`response_2` messages.
Labels stay in verifier metadata. HS3/HS4 retain preference strength; scoring uses direction only.
JudgeBench and RM-Bench supply direction labels without strength.

The five examples reuse public JudgeBench inputs and saved GenRM outputs.
Direction-only labels use 3/4 for the generic preference scorer. Run and model
identifiers are anonymized; generated answers and reasoning are unchanged.

## Prepare and evaluate

Supported benchmarks: JudgeBench, RM-Bench, HS3, and HS4. Prepare their inputs:

```bash
uv run python -m benchmarks.genrm_judgebench.prepare  # gpt split; --split claude also supported
uv run python -m benchmarks.genrm_rmbench.prepare
uv run --extra mlflow python -m benchmarks.genrm_hs3.prepare
uv run --extra mlflow python -m benchmarks.genrm_hs4.prepare
```

HS3 and HS4 download preprocessed tasks from private GitLab and require registry
credentials in `env.yaml`. See each benchmark's README for sources.

For example, to evaluate a GenRM checkpoint on JudgeBench:

```bash
gym eval run \
  --split benchmark \
  --config benchmarks/genrm_judgebench/config.yaml \
  --config resources_servers/genrm_eval/configs/policy_model.yaml \
  --model /path/to/genrm-hf-checkpoint \
  --output results/genrm_judgebench.jsonl \
  --num-repeats 1 \
  --concurrency 32
```

Defaults: temperature 0.6, top-p 0.95, 16,384 generated tokens (reasoning plus
final response; excludes input). Override tensor parallelism and context length
through `policy_model.responses_api_models.genrm_model.vllm_serve_kwargs`.
The model config defaults to eight GPUs. JudgeBench requires both candidate orders;
incomplete pairs count as incorrect and are reported separately.

## Token metrics

`response_tokens` and `reasoning_tokens` use tiktoken's `gpt-4o` encoding.
Gym aggregates these counts and includes medians in key metrics.
The encoding is loaded at server startup; offline runs require a populated tiktoken cache.
Recompute metrics from saved outputs on CPU:

```bash
OPENBLAS_NUM_THREADS=1 gym eval reverify \
  --config benchmarks/genrm_judgebench/config.yaml \
  --config resources_servers/genrm_eval/configs/reverify.yaml \
  --inputs /path/to/full_materialized_inputs.jsonl \
  --rollouts /path/to/full.jsonl \
  --output /path/to/full_reverified.jsonl
```

## Scoring

| Benchmark | How the score is calculated |
|---|---|
| JudgeBench | Percentage of pairs with a positive sum across both orders: correct +1, incorrect -1, unparseable 0. Missing orders count as incorrect. |
| RM-Bench | Average of the chat, math, code, and safety domain scores, with equal weight per domain; missing rankings receive 0.5. |
| HS3 / HS4 | Percentage of records with the correct preference; missing rankings count as incorrect. |

Rankings must be integers 1–6; invalid values (including 3.5) count as unparseable.
The standalone config uses generic preference accuracy; benchmark configs select
their own `scoring_module` and `metric_key`.

RM-Bench includes `domain_coverage` alongside its headline score and reports per-domain counts. With coverage below 1,
the score is incomplete: absent domains contribute zero to the four-domain average.

# DeepSWE in Gym

Runs the pinned 113-task DeepSWE v1.1 benchmark with Pier and OpenSandbox.
Agent and verifier sandboxes deny network access by default.

## Prepare

```bash
uv run python benchmarks/deepswe/prepare.py
```

## Run

```bash
export OPENSANDBOX_API_KEY=...
export POLICY_API_KEY=...

ng_e2e_collect_rollouts \
  '+config_paths=[benchmarks/deepswe/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]' \
  ++input_jsonl_fpath=benchmarks/deepswe/data/deepswe_benchmark.jsonl \
  ++output_jsonl_fpath=results/deepswe.jsonl \
  ++max_samples=113 \
  ++policy_base_url=https://inference-api.nvidia.com/v1 \
  '++policy_api_key=${oc.env:POLICY_API_KEY}' \
  ++policy_model_name=openai/openai/openai/gpt-5.5
```

See <https://deepswe.datacurve.ai/run> for the upstream Pier workflow.

## Result

| Model | Raw avg@3 | Infra-adjusted avg@3 | Published |
| --- | ---: | ---: | ---: |
| GPT-5.5 `xhigh` | 65.19% | 66.27% | 67% +/- 6% |
| Gemini 3.6 Flash `high` | 38.94% | 39.30% | 49% +/- 5% |
| Opus 5 `max` | Incomplete | Incomplete | 74% +/- 4% |

One GPT infrastructure failure was rerun and passed. The adjusted scores
exclude four unresolved GPT and three unresolved Gemini infrastructure
failures. Opus stopped after the inference key reached its budget limit.

See the five [inputs](data/example.jsonl) and
[rollouts](data/example_rollouts.jsonl).

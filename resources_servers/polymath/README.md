# PolyMath resources server

Verifies multilingual math answers from the [PolyMath benchmark](https://huggingface.co/datasets/Qwen/PolyMath)
and reports difficulty-weighted, per-language metrics.

This server subclasses `math_with_judge` — verification is the same
`math-verify` symbolic check (with optional LLM-judge fallback). The
PolyMath additions are at the metric-aggregation layer:

- **Difficulty-weighted aggregation.** PolyMath rows carry a
  per-question `weight` field (low=1, medium=2, high=4, top=8). The
  server emits `pass@k/<score>_weighted`,
  `pass@1[avg-of-k]/<score>_weighted`, and
  `majority@k/<score>_weighted` alongside the unweighted metrics.
  Mirrors NeMo Skills' `WeightedMathMetrics`.
- **Per-language stratification.** Tasks are also grouped by
  `language` (the JSONL field), producing
  `<lang>/pass@k/<score>` keys — equivalent to Skills'
  pipeline-level `subset_for_metrics` fan-out.

## Per-row JSONL fields the server reads

| Field | Used by | Default |
|-------|---------|---------|
| `question` | `verify()` (judge prompt) | required |
| `expected_answer` | `verify()` (math-verify gold) | required |
| `weight` | weighted metric aggregation | 1.0 (matches Skills) |
| `language` | per-language subset metrics | omitted from per-language keys |

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/polymath/configs/polymath.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts (5-example smoke test)

```bash
ng_collect_rollouts \
    +agent_name=polymath_simple_agent \
    +input_jsonl_fpath=resources_servers/polymath/data/example.jsonl \
    +output_jsonl_fpath=results/polymath_rollouts.jsonl \
    +num_repeats=1
```

The server's vLLM endpoint should be started with
`--reasoning-parser deepseek_r1` (or the parser matching your model)
so `<think>…</think>` is stripped before `\boxed{…}` extraction —
otherwise truncated reasoning rollouts will report spurious `no_answer`
mismatches versus a NeMo Skills baseline.

## Licensing information

Code: Apache 2.0

Data: PolyMath (`Qwen/PolyMath`) — see the upstream repo for the dataset
license; not redistributed by this server.

Dependencies:
- nemo_gym: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)

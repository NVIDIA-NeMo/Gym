# HealthBench Professional

Clinical conversations scored with the shared `healthbench_rubric` verifier.

- Dataset: [OpenAI healthbench_professional](https://huggingface.co/datasets/openai/healthbench-professional), MIT.
- Dataset revision: pinned in `prepare.py` and `manifest.yaml`; `--revision` explicitly selects another revision.
- Policy input: original conversation roles and contents, without a wrapper instruction.
- Default judge: gpt-5.4-2026-03-05, Responses API, low reasoning effort, 4096 output tokens. Override `judge_base_url`, `judge_api_key`, or `judge_model_name` through Gym configuration.
- Default repeats: 1. Smoke runs explicitly use one repeat.
- Certification metric: `overall_score_length_adjusted` in the aggregate's `key_metrics`, on a 0–1 scale.

The row's `overall_score` and `overall_score_length_adjusted` preserve raw scoring values, including negatives and length bonuses. Gym's `reward` is bounded to [0, 1] for the manifest contract. **Do not compare `mean/reward` with published HealthBench scores.** Benchmark headline metrics clamp the mean of raw scores, matching simple-evals. Empty final answers receive zero without a length bonus. Masked judge failures are excluded from score aggregation.

## Prepare and validate

```bash
gym eval prepare --benchmark healthbench_professional
gym env validate healthbench_professional
gym env test healthbench_professional
```

Prepared prompts are declared with `prompt_source: prepared`. Regenerate data after upgrading from the earlier flattened-conversation format. Full benchmark data stays out of Git; committed smoke examples are synthetic.

## Nemotron Lightning smoke

Make `NVIDIA_API_KEY` available in the shell, then run:

```bash
bash benchmarks/healthbench/scripts/run-smoke.sh healthbench_professional
```

The script uses the hosted `inference_provider` model server, temperature 0.7, 8192 output tokens, eight tasks, one repeat, and two parallel rollouts. It saves a redacted resolved config and uses one absolute capture folder per run. This checks runtime behavior; the smoke sampling settings and small sample do not establish published-score parity.

The smoke uses separate GPT-4.1/GPT-5.4 judge instances through the NVIDIA gateway. Its deployment aliases are not pinned snapshots, so record them and do not claim published-score parity. For reference evaluations, use the benchmark config's pinned judges with an appropriate endpoint and key. Local `gym env start` plus `gym eval run --no-serve` must receive `++observability_enabled=true` and the same absolute `++model_call_capture_dir` in both commands. Slurm `gym eval submit` enables capture by default.

## Onboarding evidence

See [the shared onboarding checklist](../healthbench/onboarding.md). This workload remains experimental and `verified: false` until full-run score parity and exit-check evidence have been reviewed. A Professional result uses an unofficial external implementation; OpenAI reports its own internal implementation.

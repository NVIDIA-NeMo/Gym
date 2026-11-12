# HealthBench

Clinical conversations scored with the shared `healthbench_rubric` verifier.

- Dataset: [OpenAI healthbench](https://huggingface.co/datasets/openai/healthbench), MIT.
- Dataset revision: pinned in `prepare.py` and `manifest.yaml`; `--revision` explicitly selects another revision.
- Policy input: original conversation roles and contents, without a wrapper instruction.
- Default judge: gpt-4.1-2025-04-14, Chat Completions, temperature 0.5, 2048 output tokens. Override `judge_base_url`, `judge_api_key`, or `judge_model_name` through Gym configuration.
- Default repeats: 1. Smoke runs explicitly use one repeat.
- Certification metric: `overall_score` in the aggregate's `key_metrics`, on a 0–1 scale.

The row's `overall_score` and `overall_score_length_adjusted` preserve raw scoring values, including negatives and length bonuses. Gym's `reward` is bounded to [0, 1] for the manifest contract. **Do not compare `mean/reward` with published HealthBench scores.** Benchmark headline metrics clamp the mean of raw scores, matching simple-evals. Empty final answers receive zero without a length bonus. Masked judge failures are excluded from score aggregation.

## Prepare and validate

```bash
gym eval prepare --benchmark healthbench
gym env validate healthbench
gym env test healthbench
```

Prepared prompts are declared with `prompt_source: prepared`. Regenerate data after upgrading from the earlier flattened-conversation format. Full benchmark data stays out of Git; committed smoke examples are synthetic.

## Full, Consensus, and Hard

The default Full (`oss_eval`) evaluation covers the overall HealthBench set, including the cases represented in Consensus and Hard, and reports one overall score. Prepare and run Full when you want that overall result. Prepare and run Consensus or Hard separately only when you also want their individual scores; the Full run does not automatically report them.

All three evaluations use the same `healthbench` entry and verifier. The manifest's default canonical split is `oss_eval`; the variant configs select the separate Consensus and Hard evaluations.

| Protocol | Additional config | Prepared file under `data/` |
| --- | --- | --- |
| Full (`oss_eval`) | None | `healthbench_benchmark.jsonl` |
| Consensus | `configs/consensus.yaml` | `healthbench_consensus_benchmark.jsonl` |
| Hard | `configs/hard.yaml` | `healthbench_hard_benchmark.jsonl` |

Prepare the dataset for each score you want before evaluating it. Gym evaluation collates existing prepared files; it does not run the benchmark's download script automatically.

```bash
# Prepare Full for the overall HealthBench score.
gym eval prepare --benchmark healthbench

# Optional: prepare these for separate Consensus and Hard scores.
gym eval prepare --benchmark healthbench --config benchmarks/healthbench/configs/consensus.yaml
gym eval prepare --benchmark healthbench --config benchmarks/healthbench/configs/hard.yaml
```

Use the same variant config for evaluation. Here `model.yaml` is your policy and judge configuration, and each evaluation has its own absolute capture directory:

```bash
# Run Full for the overall HealthBench score.
gym eval run --benchmark healthbench --config model.yaml \
  --split benchmark --output results/healthbench/full/rollouts.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/healthbench/full/model-calls"

# Optional: run Consensus for its separate score.
gym eval run --benchmark healthbench --config model.yaml \
  --config benchmarks/healthbench/configs/consensus.yaml \
  --split benchmark --output results/healthbench/consensus/rollouts.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/healthbench/consensus/model-calls"

# Optional: run Hard for its separate score.
gym eval run --benchmark healthbench --config model.yaml \
  --config benchmarks/healthbench/configs/hard.yaml \
  --split benchmark --output results/healthbench/hard/rollouts.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/healthbench/hard/model-calls"
```

Read `key_metrics.overall_score` from each agent entry in that run's `rollouts_aggregate_metrics.json`. The Full run gives the overall HealthBench score; the optional Consensus and Hard runs give their respective separate scores. Each value is on a [0, 1] scale; multiply by 100 for a percentage. Check scored coverage and failures alongside it.

The resolved config records `healthbench_subset`, the preparation arguments and the selected dataset path. Each prepared row also records `metadata.healthbench_subset`. Save the resolved config with your run (the smoke helper does this; `gym eval submit` also saves it).

Direct script use also selects a separate default destination:

```bash
python benchmarks/healthbench/prepare.py --subset consensus
python benchmarks/healthbench/prepare.py --subset hard
```

## Nemotron Lightning smoke

Make `NVIDIA_API_KEY` available in the shell, then run:

```bash
bash benchmarks/healthbench/scripts/run-smoke.sh healthbench
```

The script uses the hosted `inference_provider` model server, temperature 0.7, 8192 output tokens, eight tasks, one repeat, and two parallel rollouts. It saves a redacted resolved config and uses one absolute capture folder per run. This checks runtime behavior; the smoke sampling settings and small sample do not establish published-score parity.

The smoke uses separate GPT-4.1/GPT-5.4 judge instances through the NVIDIA gateway. Its deployment aliases are not pinned snapshots, so record them and do not claim published-score parity. For reference evaluations, use the benchmark config's pinned judges with an appropriate endpoint and key. Local `gym env start` plus `gym eval run --no-serve` must receive `++observability_enabled=true` and the same absolute `++model_call_capture_dir` in both commands. Slurm `gym eval submit` enables capture by default.

## Onboarding evidence

See [the shared onboarding checklist](onboarding.md). This workload remains experimental and `verified: false` until full-run score parity and exit-check evidence have been reviewed.

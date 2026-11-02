# Litmus-Bench

[Litmus-Bench](https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.2)
evaluates direct-answer molecular-property and ADME reasoning with the reusable
[`litmus_agent`](../../resources_servers/litmus_agent/README.md) verifier.

The published `validation.jsonl` contains 2,056 held-out questions. The combined
benchmark retains every row; subset configs select rows by `question_family`:

| Subset | Question families | Questions | Config |
| --- | --- | ---: | --- |
| Tier 1/2 | `tier1_rdkit`, `tier2_rdkit` | 1,000 | `config_tier12.yaml` |
| ADME direct | `tier5_admet_direct` | 598 | `config_adme_direct.yaml` |
| ADME analogue | `tier5_admet_analog` | 358 | `config_adme_analogue.yaml` |
| ADME comparison | `tier5_admet_comparison` | 100 | `config_adme_comparison.yaml` |

Prompts already specify the answer format. Preparation preserves each row's
`output_regex`, `answer_type`, optional `match` rule, `question_license`,
`question_family`, and `metadata` provenance. It removes only `agent_ref` so Gym
can assign the configured agent. No rows are dropped or answers rounded.
The resource-server configs register all 47,118 questions in the published
`train.jsonl`, including both molecular-property and ADME questions.

## Prepare

```bash
gym eval prepare --benchmark litmus-bench

# Optional individual subsets:
gym eval prepare --config benchmarks/litmus-bench/config_tier12.yaml
gym eval prepare --config benchmarks/litmus-bench/config_adme_direct.yaml
gym eval prepare --config benchmarks/litmus-bench/config_adme_analogue.yaml
gym eval prepare --config benchmarks/litmus-bench/config_adme_comparison.yaml
```

Preparation downloads the held-out file, verifies its SHA-256, row count, schema,
and question-family counts, then writes gitignored datasets under `data/`.
A `.source.json` sidecar records the source and prepared-content fingerprints;
Gym rebuilds cached artifacts when these do not match. Re-run preparation after
updating the checkout, before evaluating an existing local benchmark file.

For an exact local copy, pass `+prepare_script_args.source_dir=/path/to/release`;
that directory must contain the published `validation.jsonl`. The same checks
apply. Source selection and content pins live in `prepare.py`.

## Evaluate

```bash
gym eval run \
  --benchmark litmus-bench \
  --model-type vllm_model \
  --split benchmark \
  --output results/litmus-bench.jsonl
```

All benchmark configs use five rollouts per question. The combined and Tier 1/2
configs allow up to 131,072 output tokens; ADME configs default to 8,192. The
model context window also limits the effective generation budget. Override
`--max-output-tokens` as needed. Dataset repeats and `--num-repeats` multiply,
so do not also pass `--num-repeats 5` for these benchmark splits.

## Examples

Ten-question samples from the published held-out data are committed under
`data/examples/`: five Tier 1 plus five Tier 2 questions, and ten per ADME
category, taking the first rows in source order. All examples are direct-answer
questions. The resource server's
`data/example.jsonl` contains one question from each of the five families.

Start the resource server and model bridge, then pass a bundled example as
explicit input with `--no-serve`. For NVIDIA's inference gateway:

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY first}"

gym env start \
  --resources-server litmus_agent \
  --model-type vllm_model \
  --model-url https://inference-api.nvidia.com \
  --model azure/openai/gpt-5.5 \
  '++policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  ++policy_model.responses_api_models.vllm_model.extra_body.reasoning_effort=medium
```

In another terminal:

```bash
gym eval run --no-serve \
  --agent litmus_agent_agent \
  --input benchmarks/litmus-bench/data/examples/litmus-tier12_example.jsonl \
  --output results/litmus-tier12-example.jsonl \
  --num-repeats 1 \
  --concurrency 8 \
  --max-output-tokens 8192
```

For ADME examples, use
`data/examples/adme-tier5-<subset>_example.jsonl`, where `<subset>` is `direct`,
`analogue`, or `comparison`. Give each run its own output file. Explicit input
files use `--num-repeats` directly; increase it for repeated-rollout profiling.

Reward profiling consumes the materialized inputs written beside the rollout:

```bash
gym eval profile \
  --inputs results/litmus-tier12-example_materialized_inputs.jsonl \
  --rollouts results/litmus-tier12-example.jsonl
```

## License

Tier 1/2 rows declare `CC-BY-SA-4.0`; ADME rows declare `OpenMDW-1.1`.
The combined benchmark and training configs record both as
`CC-BY-SA-4.0 AND LicenseRef-OpenMDW-1.1`. The `LicenseRef` spelling represents
the dataset's OpenMDW license in Gym's SPDX-expression field; each source row's
`question_license` is retained unchanged. See the linked dataset card for terms.

# ACMG variant classification

## Measurement contract

**Source.** [`nvidia/acmg_variant_benchmark`](https://huggingface.co/datasets/nvidia/acmg_variant_benchmark)
(private, NVIDIA-authored; card license `other`), revision `59fb412e7e39e8b209f431fce522458587ca8fec`. The benchmark
folders (`data/`, `rules/`, `prompts/`) are byte-identical to the earlier revision `f656c889` used for the team's
SFT and RL evaluations. The dataset is 395 real variants classified by ClinGen variant curation expert panels, each
rewritten as an evidence packet with identifiers, citations and criterion codes blocked. Answer keys are
**provisional** (0 of 395 confirmed by a variant scientist). Benchmark folders carry a canary string and must never
enter training corpora; the dataset's `training/` folder is separate training data and is not used here.

**Prepared task count: 790** = 395 cases x 2 arms. `with_rules` gives the model the gene's ClinGen specification
(rule document); `without_rules` does not. The arms are always reported separately and never pooled.

**Task contract.** One fresh context per case and arm, text only, no tools or browsing. Rows carry a fully built
`responses_create_params.input`:

- system: `prompts/system_prompt.md`, followed in the with-rules arm by two newlines and the case's own
  `rules/<rule_document>` (both stripped of surrounding whitespace);
- user: the evidence packet as compact JSON with sorted keys (`json.dumps(packet, sort_keys=True, separators=(",", ":"))`).

The answer key (`label`, `label_scoring`, `accepted_labels`, `required`, `acceptable`, `combining`) and case
metadata live in `verifier_metadata` and never reach the model. The model reasons in prose and ends with a
`CRITERIA:` line (code and strength pairs) and a `CLASSIFICATION:` line (P, LP, VUS, LB or B).

**Scoring.** The resources server runs the dataset's own standalone scorer (`training/reward.py`, vendored byte for
byte with its SHA-256 pinned and tested) on the final assistant text only; inline `<think>` blocks are removed first.
Gym's per-rollout `reward` is the team's composite from the grading recipe (sdg-pipelines
`recipes/acmg-grading/composite.py`): `gate x (0.5 x label_closeness + 0.5 x evidence_score)`, where the gate is 1
only when a classification was given and follows from the response's own criteria, label closeness gives half
credit one step away on B-LB-VUS-LP-P, and the evidence score is the scorer's evidence-point closeness
`max(0, 1 - distance / 8)`. For the 141 cases whose label is `excluded` the reward is `gate x evidence_score`. The
dataset's default scalar, `0.5 x label + 0.3 x criterion F1 + 0.2 x self-consistency`, is kept as `official_reward`.
Headline metrics are per arm, as the dataset's baseline report defines them:

| Metric | Definition |
|---|---|
| `label_accuracy` | Share of the 254 label-scorable cases whose classification is in `accepted_labels` |
| `criterion_precision` / `recall` / `f1` | Pooled over responses: returned (code, strength) pairs that are acceptable; required codes returned at an accepted strength |
| `exact_criteria` | Share of responses returning every required criterion and nothing unacceptable |
| `self_consistency` | Share of responses whose label follows from their own criteria (rule document's combining rules with rules, any standard system without) |
| `output_failure_rate` | Share of responses with no classification |
| `reward`, `gate`, `label_closeness` | Means of the composite reward, its self-consistency gate and the label closeness over scorable cases |
| `official_reward`, `evidence_score` | Means of the dataset's scalar and of its evidence-point closeness score |

**Run conditions.** The team's held-out protocol: temperature 1.0, top-p 0.95, thinking on, no output-token cap
beyond the context limit, 4 repeats (`num_repeats: 4`; report each repeat and their mean, not pass@k). The dataset's
published baselines used one response per prompt at temperature 0, medium reasoning effort and a 32,768-token cap.

## Reference scores

Dataset baselines (`results/baseline_summary.json`), label accuracy on the 254 scorable cases:

| Model | With rules | Without rules | Criterion F1 with / without | Official reward with / without |
|---|---:|---:|---|---|
| nemotron-3-ultra | 92.9% | 59.8% | 91.3% / 53.2% | 0.936 / 0.632 |
| gpt-5.6-sol | 92.5% | 74.0% | 93.3% / 68.0% | 0.942 / 0.769 |

Replaying the 1,580 published responses of those two models through this verifier reproduces every per-response
prediction, label verdict, criteria count, exactness, self-consistency and output-failure flag, and every published
aggregate. The official-reward column is computed from that replay.

## Prepare and validate

The dataset is private: put the Hugging Face token in `env.yaml` as `hf_token` (or export `HF_TOKEN`). From the
Gym checkout with its development environment active:

```bash
gym eval prepare --benchmark acmg
gym env validate acmg
gym env test acmg
gym env test --resources-server acmg
```

Preparation downloads the pinned revision's benchmark folders into `benchmarks/acmg/data/source/`, checks every file
against the dataset's `CHECKSUMS.json` and the hashes pinned in `prepare.py`, and atomically writes
`benchmarks/acmg/data/acmg_benchmark.jsonl` (790 rows, SHA-256
`63a07224509dd943bd6d0b91d357d29c8acf835b1c238b119bd25d588809f2ce`). Everything under `data/` is gitignored. To
reuse an existing snapshot: `python -m benchmarks.acmg.prepare --source /path/to/snapshot`.

## Evaluate

```bash
gym env start --benchmark acmg --model-type vllm_model
gym eval run --no-serve --benchmark acmg --model-type vllm_model \
    --temperature 1.0 --top-p 0.95 --num-repeats 4 \
    --output results/acmg/rollouts.jsonl
```

Read `acmg/with_rules/*` and `acmg/without_rules/*` from the agent metrics. Label accuracy, criterion F1,
self-consistency, the composite reward and the official reward are the key metrics; `label_scorable` and `self_consistency_assessable` are
their denominators.

## Limitations

- Keys are provisional until expert confirmation; `confirmation` and `version_status` are carried per row so results
  can be broken down by them.
- The source interpretations are public, so the benchmark must be run without tools or browsing.
- The dataset's lookup arm (specification lookup tools) is not implemented here.

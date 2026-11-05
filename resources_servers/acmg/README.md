# ACMG Resources Server

Deterministic verifier for ACMG/AMP variant classification, used by the
[`acmg` benchmark](../../benchmarks/acmg/README.md). The model receives an evidence packet for a real variant
curated by a ClinGen expert panel and must return the ACMG/AMP criteria it applies and a classification. The
verifier is pure Python with no LLM judge.

- **Tasks**: 790 = 395 variants x 2 arms. `with_rules` gives the model the gene's ClinGen rule document,
  `without_rules` does not. Text only, no tools or browsing; one fresh context per task.
- **Reward**: deterministic, in [0, 1]; combines how close the predicted label is to the reference with how close the
  predicted criteria are, and is zero when the classification does not follow from the model's own criteria.
- **Metrics**: `acmg/<arm>/reward`, `label_accuracy`, `criterion_f1` and `self_consistency` are the key numbers.

## Prepare benchmark data

```bash
gym eval prepare --benchmark acmg
```

Downloads the pinned dataset revision and writes `benchmarks/acmg/data/acmg_benchmark.jsonl` (790 rows).

## Running servers

```bash
gym env start --benchmark acmg --model-type vllm_model
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in `env.yaml` (or passed as CLI overrides).

## Collect rollouts

```bash
gym eval run --no-serve --benchmark acmg --model-type vllm_model \
    --output results/acmg/rollouts.jsonl
```

Rows already carry their `responses_create_params.input`, so do not pass `--prompt-config`.

## Metrics

Metrics are reported both pooled over all tasks (`mean/reward` and friends) and separately per arm as
`acmg/<arm>/<metric>` for `with_rules` and `without_rules`. Self-consistency is checked against the gene's rule
document in the with-rules arm and against the standard ACMG/AMP combining rules in the without-rules arm.

| Metric | Definition |
|---|---|
| `reward` | Mean reward |
| `label_accuracy` | Share of the label-scored cases whose classification is correct |
| `criterion_precision` / `recall` / `f1` | Over (criterion, strength) pairs: the model's predicted pairs against the expert's accepted ones for precision, the expert's required ones for recall. Counted over all responses in the arm |
| `exact_criteria` | Share of responses returning every required criterion and nothing unacceptable |
| `self_consistency` | Share of responses whose classification follows from their own criteria |
| `output_failure_rate` | Share of responses with no classification |
| `gate`, `label_closeness`, `evidence_score` | Means of the reward's three terms |
| `label_scorable`, `self_consistency_assessable` | Denominators of `label_accuracy` and `self_consistency` |

### Reward

The model generates a `CRITERIA:` line (code and strength pairs) and a `CLASSIFICATION:` line
(P, LP, VUS, LB or B). The verifier reads the final assistant text (`<think>`/`<thinking>` blocks removed), parses
those two lines (handles cases like markdown, `PM2_Supporting`, `PM2 (Supporting)`, full label names or a JSON
answer) and computes

```text
reward = gate x (0.5 x label_closeness + 0.5 x evidence_score)
```

- `gate` is 1 when the model's classification follows from its own predicted criteria, else 0.
- `label_closeness` is 1 for a correctly predicted label, 0.5 when the prediction is one step from the correct
  label on B-LB-VUS-LP-P, else 0.
- `evidence_score` measures how close the predicted criteria are to the reference. Start a penalty at 0 and compare
  the model's criteria with the expert's:
  - the model left out one of the expert's required criteria: penalty += that criterion's points
  - the model used a criterion the expert does not accept: penalty += 2 x that criterion's points
  - the model used the right criterion at the wrong strength: penalty += the point difference between the two
    strengths

  Score = 1 - penalty / 8, floored at 0. Points by strength are the usual Supporting 1, Moderate 2, Strong 4,
  Very Strong 8.
- For the cases with no reference label, the reward is `gate x evidence_score`.

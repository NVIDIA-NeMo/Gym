## Environment contribution

### Use case and eval contract

- Target user/workflow and unit of success:
- Primary metric and verifier semantics:
- Eval dataset name/version, source, license, and held-out method:
- Coverage matrix (task family, difficulty, horizon/context, priority and edge slices):

### Eval evidence

- [ ] Five-row smoke data, example metrics, and example rollouts are included.
- [ ] Representative eval rows have stable task IDs, provenance, split, difficulty, and slice labels.
- [ ] `gym dataset validate-eval --manifest <path>` passes.
- [ ] Known-good solutions reach intended success; empty/degenerate solutions stay at the floor.
- [ ] Identical seeds/actions reproduce task state and reward.
- [ ] Weak, target Nemotron, and strong reference baselines use the same frozen task set.
- [ ] Repeated rollout coverage, aggregate metrics, and per-slice metrics are reported with denominators.
- [ ] Priority slices, anomalies, and passing/failing repeats received manual review.

Commands and exact model/agent/sampling configuration:

```text

```

| Model | Tasks | Repeats | Coverage | Primary metric | pass@k | Consistency | Error/truncation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | | | | | | | |

Reviewed task/rollout IDs and dispositions:

### Decision and known gaps

- [ ] Eval-ready; no training claim is required for this PR.
- [ ] Eval/environment repair is required before training.
- [ ] Eval evidence supports an isolated training experiment; link curves and held-out before/after results below.

Known limitations, excluded slices, data risks, verifier risks, and follow-up work:

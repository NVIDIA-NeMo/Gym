---
name: nemo-gym-repeat-planning
description: Plan fixed benchmark or suite repeat counts from complete-pass SDs with gym eval plan-repeats, including an existing baseline or baseline investment across candidate checkpoints. Use for prospective repeat budgets, margin or variance sensitivity, and interpreting planning validation; use score-triage for unexplained observed score gaps.
---

# NeMo Gym Repeat Planning

Use [Plan Repeats](../../../fern/versions/latest/pages/evaluation/plan-repeats.mdx) as the configuration and interpretation reference. Run the installed command's help when the checkout differs from that page.

## Establish the planning question

- Choose one `aggregation_level` per file: one model × benchmark, selected benchmarks of one model, or a suite average. Every listed entry has equal aggregate weight; additional metrics share their benchmark's weight.
- Collect complete-pass SDs in normalized 0–100 score points, with higher scores better. A within-pass task SD or an existing mean's standard error is the wrong input. Keep evidence of the metric, population, source runs, and transformation used to obtain each SD beside the planning artifact.
- For multiple metrics in one benchmark, require the cell-level SD of the equal metric average within each complete repeat. Compute that average before estimating its SD, as in suite comparison. Individual metric SDs are descriptive inputs and cannot determine the composite SD. Omitted cell-level candidate SD defaults to the baseline composite SD.
- Use `baseline.mode: existing` when its listed counts are already collected. Use `optimize` to plan a baseline once plus `candidate_experiments` distinct candidate checkpoints, default one. Remove existing-count fields in optimization mode.
- Establish margin, expected true candidate-minus-baseline change, target power, and significance level before inspecting comparison outcomes. The defaults are a 1-point margin, zero true change, 80% analytic power, and one-sided alpha 0.05; these are editable planning choices.
- Keep prepared copies fixed with each cell's `repeat_step`. Use a common-unit `repeat_cost` when real cost data exist, or retain unit costs. Do not confuse distinct candidate checkpoints with repeat counts.

## Run and inspect

```bash
gym eval plan-repeats --plan planning.yaml --output-dir results/repeat-planning
```

Read both the human report and full-precision JSON. Check the selected entries, resolved SDs, weights, new versus existing passes, counts per candidate, all-candidate cost, and the search method. The exact single-benchmark search and suite local search have different optimality claims.

Read analytic power alongside simulated power and its Monte Carlo interval. Simulations retain fixed supplied variance forecasts; agreement does not establish that those forecasts are accurate. Alpha 0.05 corresponds to a one-sided 95% lower bound, or the lower endpoint of a two-sided 90% interval. Set `--confidence-level 0.90` in either final comparison command for this alpha.

The simulation redraws both arms in hypothetical normal experiments even with existing baseline counts; it is not conditional power given the collected baseline observations. Evaluate composite, model, and suite claims with `compare-suite` using the same metric selection, bounds, directions, and entries. A benchmark claim uses a one-cell manifest; a model claim uses its per-model aggregate. `compare` reports separate raw-unit metric intervals; a one-point normalized margin uses threshold −0.01 for a single higher-is-better 0–1 reward metric.

For sensitivity, change one of margin, candidate SD, expected change, target power, or candidate experiment count at a time. Re-run the command; do not scale an old answer by a rule of thumb. Compare cost and counts with the original plan, and explain when changing margin or suite membership changes the quality claim.

## Keep planning separate from execution

This workflow writes planning artifacts. It does not authorize model spending, cluster submission, baseline regeneration, or an automatic repeat-until-significant loop. A planned experiment that remains inconclusive is a valid outcome; changing its stopping rule needs a separately justified design.

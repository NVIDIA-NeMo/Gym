# genrm_rewardbench2

Prepare [allenai/reward-bench-2](https://huggingface.co/datasets/allenai/reward-bench-2):

```bash
uv run python -m benchmarks.genrm_rewardbench2.prepare
```

The `test` split has 1,865 records across six subsets: Factuality, Focus, Math, Precise IF,
Safety, and Ties. A scalar reward model scores a record's four completions independently and is
correct when the chosen one scores highest. A GenRM judge compares two responses at a time, so
each record is expanded into pairwise comparisons, and each comparison is evaluated in both
response orders:

| Comparison | Records | Comparisons | Rows |
|---|---|---|---|
| Chosen vs rejected, best-of-4 subsets | 1,763 | 5,289 | 10,578 |
| Chosen vs rejected, Ties | 102 | 5,739 | 11,478 |
| Chosen vs chosen, Ties | 51 | 2,298 | 4,596 |
| **Total** | **1,865** | **13,326** | **26,652** |

The chosen-versus-chosen comparisons exist only for Ties, whose score depends on how far apart
the judge places two equally correct answers. Source revision:
`7ff08853b0d5686e79b13fda8677024f566a104a`.

RewardBench 2 record IDs repeat across subsets, so task IDs are qualified:
`<subset>:<id>:<chosen index>:<rejected index>` for chosen versus rejected and
`<subset>:<id>:c<index>:c<index>` for chosen versus chosen, plus `-flipped` for the reversed
order. `label` is `A>B`/`B>A` when the first listed response should win and `A=B` when neither
should; `flipped` records which order a row holds.

## Scoring

A scalar reward model's score for a completion does not depend on presentation order. A pairwise
judge cannot be order-free, so each comparison is judged in both orders and the two rankings are
averaged in a first-response frame, mirroring the flipped order through `7 - ranking`. The
comparison's **margin** is `3.5 - mean`, positive when the judge prefers the first response. A
position bias that pulls both orders the same way cancels in that average; a sign-only vote would
not cancel it. If only one order parsed, it decides alone.

### The five best-of-4 subsets

A comparison is a strict win at a positive margin, a strict loss at a negative one, and a tie at
exactly zero - which happens when the judge returns the same ranking both ways, so it expressed
no net preference at all. A record scores 0 if the chosen response loses any comparison, and
otherwise `1 / (1 + tied comparisons)`:

| Comparisons tied | Credit | Paper |
|---|---|---|
| 0 | 1.0 | correct response rated highest |
| 1 | 0.5 | "two-way ties as partial credit of 0.5" |
| 2 | 0.33 | "three-way ties as 0.33" |
| 3 | 0.25 | "four-way ties as 0.25 (random)" |

That is the paper's ladder exactly (appendix I): credit equals what a random tie-break among the
responses sharing the top score would earn.

### Ties

The Ties subset tests whether a model keeps equally valid answers close together while still
ranking them above wrong ones, so it is scored by the published composite rather than the ladder.
Both of its statistics are *differences* of rewards, which pairwise margins estimate directly:

- `worst_correct - best_incorrect` is the smallest chosen-versus-rejected margin, since
  `min(r_c) - max(r_i)` equals `min(r_c - r_i)` over the pairs. A record is **accurate** when it
  is positive.
- `best_correct - worst_correct` is the largest absolute chosen-versus-chosen margin. This is why
  Ties records carry those extra comparisons; without them the correct-answer spread is unmeasured.

Each `tied:N` record pairs with the `ref:N` record holding one of the same correct answers, and
the subset score weights accuracy at 60% and margin structure at 40%, as upstream does:

```
0.30 * tied_accuracy + 0.30 * reference_accuracy
    + 0.20 * correctness_preferred        (correct-incorrect gap exceeds the correct-answer spread)
    + 0.20 * correctness_preferred_hard   (as above, using the harder of the tied and ref gaps)
    + 0.01 * margin_score                 (tanh tie-break)
```

The components are reported as `metrics/ties_*`. Those weights total 1.01 and the tanh term is
signed, so like upstream the subset ranges over roughly `[-0.01, 1.01]` rather than `[0, 1]`, and
`metrics/overall` inherits a 0.0017 margin at each end.

### Reporting

The headline `metrics/overall` follows the paper: "Scores are first measured per-domain, and the
final score is an unweighted average across all six domains"
([arXiv:2506.01937](https://arxiv.org/abs/2506.01937), section 3). Absent subsets contribute
zero, so `metrics/subset_coverage` below 1 means the score is incomplete.

Records with missing comparisons or missing orders score 0 and are reported as
`incomplete_records` and `incomplete_comparisons`. `tied_comparisons` counts order-dependent
verdicts, a direct read on the judge's position bias and ranking granularity. Per-subset `counts`
report records at full credit (`correct`) and on tie credit (`partial`); for Ties, `correct`
counts accurate records.

## Deviations from the paper

The paper evaluates generative models two ways, neither of them pairwise: a *rankings* prompt
holding all four completions at once, or *ratings* that score each completion separately on a
1-10 scale. A GenRM judge takes exactly two responses and returns a 1-6 preference, so this
benchmark decomposes each record into pairwise comparisons instead. The per-record criterion
survives the translation exactly - "selecting the correct response from 4 completions", and for
Ties "all valid correct answers being scored higher than all incorrect answers", both hold iff
the chosen response wins every pairwise comparison.

The reference harness judges each datapoint once, with the correct response shuffled into a
random position. Both orders are judged here instead, at twice the inference cost, because
averaging them is the closest a pairwise judge gets to an order-free score.

Margins here are quantized: a mean of two integer rankings takes eleven values from `-2.5` to
`2.5`, where a scalar reward model produces a continuous margin. The Ties comparisons are
therefore coarser than upstream's, though both are scale-free ratios of two margins in the same
units.

See [GenRM evaluation](../../resources_servers/genrm_eval/README.md) for evaluation commands and scoring.

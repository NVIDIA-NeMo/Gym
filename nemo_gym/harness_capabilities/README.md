# Harness trajectory evidence

Check P0 evidence from a collected evaluation:

```bash
python scripts/check_harness_conformance.py \
    --bundle results/my-harness/rollouts.jsonl \
    --output results/my-harness/evidence
```

Use `matrix --harness NAME=PATH` to compare multiple harnesses. The checker uses
TE-1–TE-9 and the `gym-p0/v1` profile. Reports are `evidence_summary.json`,
`evidence_results.jsonl`, and `evidence_report.md`.

See [Harness Conformance](../../fern/versions/latest/pages/observability/harness-conformance.mdx)
for contracts, applicability, matrix usage, producer onboarding, and limitations.

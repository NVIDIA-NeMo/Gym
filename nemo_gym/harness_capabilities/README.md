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

Unit tests use hand-authored contracts in `tests/unit_tests/harness_capabilities/synthetic.py`;
they validate the checker independently of harness implementations. To measure
actual harnesses and regenerate the commit-pinned documentation table, use
`python -m scripts.harness_conformance.table --commit <full-sha> --output <new-directory>`.
See the [runner guide](../../scripts/harness_conformance/README.md) for prerequisites and test gates.

See [Harness Conformance](../../fern/versions/latest/pages/observability/harness-conformance.mdx)
for contracts, applicability, matrix usage, producer onboarding, and limitations.

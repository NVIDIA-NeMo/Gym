# SWE-Next rows to Harbor tasks

Convert frozen SWE-Next rows using their existing prompts, images and grading implementation:

```bash
python scripts/swe_rows_to_harbor/convert.py rows.jsonl out/swe-tasks \
    --source swe_next --dataset-name nemo-gym/swe-tasks
```

Flat and typed rows are supported. Other sources fail unless `--skip-unsupported` is
explicit. `conversion.json` records converted/skipped rows, source identity and verifier
files. Set Harbor's `verifier_completion_key: evaluation_completed` and
`inconclusive_verification_retries: 1` to distinguish inconclusive verification from a
measured failure. The generated tasks use separate verifier sandboxes and preserve
hidden tests, patch capture, and the SWE-Next expected-status grader.

See the [matched training guide](../../fern/versions/latest/pages/training-tutorials/swe-matched-training.mdx)
for preparation, interpreter archives, timeout matching and execution limits.

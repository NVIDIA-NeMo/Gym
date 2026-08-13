# Internal GitLab report adapter

`collect_downstream_junit.py` is the GitLab-specific final report hop for Gym CPU CI. The required
`gym-cpu-ci` bridge remains the authoritative NeMo CI verdict. After it reaches a terminal state,
the report job downloads the receipt-bound Generic Package written by the trusted NeMo parent
collector and republishes its XML with `artifacts:reports:junit`.

The package coordinate is unique to the originating pipeline and source:

```text
nemo-gym-ci-junit / pipeline-<CI_PIPELINE_ID>-<CI_COMMIT_SHA> / reports.zip
```

The ZIP contains `receipt.json` plus only `junit/**/*.xml`. The receipt binds the exact Gym
project, pipeline, MR, and source SHA to the NeMo parent, collector job, and selected child pipeline.
It also declares the sorted path, size, and SHA-256 of every report. The receiver rejects stale or
mismatched identities, extra or duplicate archive members, unsafe paths, non-regular files,
oversized content, hash/size mismatches, malformed XML, and DTD/entity declarations before it
atomically publishes any output. A valid zero-report receipt represents a downstream failure that
occurred before pytest; the separately required bridge still carries that failure.

The receiver uses only the current Gym `CI_JOB_TOKEN` to read Gym's own package registry. It does
not query NeMo pipeline metadata and never accepts `RO_API_TOKEN`, `GITLAB_API_TOKEN`, or another
persistent credential. The trusted NeMo job uploads with its ephemeral job token after an explicit
owner-approved NeMo-to-Gym job-token allowlist entry. Successful mailbox cleanup must occur only
after GitLab has uploaded the Gym JUnit artifact; until that post-artifact behavior is live-proven,
an age-bounded package retention policy owns cleanup.

Focused validation runs with:

```bash
uv run pytest tests/unit_tests/test_collect_downstream_junit.py
```

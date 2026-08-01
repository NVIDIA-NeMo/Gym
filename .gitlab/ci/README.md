# Internal GitLab report adapter

`collect_downstream_junit.py` is the GitLab-specific second report hop for Gym CPU CI. It runs in
the originating `dl/nemo/gym` pipeline after the required `gym-cpu-ci` bridge reaches a terminal
state. The bridge remains the authoritative NeMo CI verdict; the collector only transports native
JUnit data and is independently required so transport failures are visible.

The collector resolves the bridge relationship from the immutable Gym pipeline ID, uses the
returned NeMo project and pipeline IDs, and selects the newest terminal
`nemo_gym_collect_junit` job attempt. It never looks up a pipeline or artifact by branch, ref, or
job name alone. The GitLab jobs endpoint is called without `include_retried=true`, and duplicate
job records are resolved to the greatest immutable job ID.

Partial reports from a failed or canceled NeMo parent are republished when its collector uploaded
JUnit. A successful NeMo collector with no JUnit is accepted as the legitimate pre-pytest case and
produces an empty Gym report directory. A failed/canceled collector without JUnit, an API or
artifact error, malformed archive, unsafe path, malformed XML, DTD/entity declaration, or JUnit
outside the expected `gym-junit/` artifact subtree fails the Gym collector without replacing an
existing output directory.

Authentication is intentionally split:

- Metadata calls use masked `RO_API_TOKEN`, then `GITLAB_API_TOKEN`, when either is configured with
  `read_api` access to both Gym and NeMo CI. Otherwise they try `CI_JOB_TOKEN` and fail with an
  actionable error if GitLab 17.4.6 rejects job-token bridge/job enumeration.
- Artifact downloads always use the originating Gym `CI_JOB_TOKEN`. The NeMo CI project must
  authorize that token to read the selected job artifact. This code does not add or widen a
  job-token allowlist; that remains an explicit project-owner action if live validation proves it
  necessary.
- Authentication headers are removed on cross-origin redirects to signed object-storage URLs.

Focused validation runs with
`uv run pytest tests/unit_tests/test_collect_downstream_junit.py`.

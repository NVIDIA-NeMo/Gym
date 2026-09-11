# Description

A resources server that hosts a published Blobfish enterprise-operations suite in-process: per-rollout
worlds built by the task package's own runtime, `POST /<tool>` dispatch, and `/verify` through the
package's own deterministic verifier (no LLM judge). Each suite is a separate environment entry under
`environments/blobfish_*` that points this server at its task packages through `suite_dir`; the default
instance in `configs/blobfish_enterprise_ops.yaml` serves the first contributed suite.

Task packages are executable code and are pinned by SHA-256 (`suite_manifest.json` next to each
environment); the server refuses packages that differ from the reviewed ones.

Reward modes (`reward_mode` in the server config): `fraction` (criterion fraction, the default),
`strict` (all criteria), and `gated` (fraction times the earned share of the suite's outcome
families, measured per suite by perturbing each oracle write, so process without outcomes is not
rewarded). `strict_pass`, `criterion_fraction` and `outcome_fraction` are always reported.

# Example usage

See the environment README (`environments/blobfish_*/README.md`) for the download, run and rollout
commands.

## Tests

```bash
pytest resources_servers/blobfish_enterprise_ops/tests
gym env test <environment>   # onboarding verifier cases: full, zero, malformed, determinism
```

`tests/fixtures/mini_suite` is a self-contained task package, so the seed, call and verify path and
the verifier fixture (`tests/verifier_cases.jsonl`) run in CI without any download; a downloaded
suite is exercised when its `BLOBFISH_<SUITE>_DIR` is set.

# Licensing

Server code: Apache-2.0, Copyright (c) 2026 Blobfish AI (vendored; see VENDORING.md). Task data:
Creative Commons Attribution 4.0 International.

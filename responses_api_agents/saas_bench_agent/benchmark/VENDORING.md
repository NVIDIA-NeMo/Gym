# SaaS-Bench vendoring

Upstream: [https://github.com/UniPat-AI/SaaS-Bench](https://github.com/UniPat-AI/SaaS-Bench), commit `48c22deb18b98c0ed78f81e7f3be82bc162de2c8`, before v1.1.
Runtime source uses Apache-2.0; task/graders/media use CC-BY-4.0 (see tasks/LICENSE).
NVIDIA modifications carry the Apache-2.0 modifications block; original content keeps its license.
NVIDIA-authored files carry the standard NVIDIA header without attributing their authorship upstream.
`provenance.json` distinguishes upstream imports from NVIDIA-authored extensions and binds shipped bytes.

The external data manifest pins 323 files including 95 byte-identical upstream graders.
They are copied during prepare and validated again at server startup. They must not overwrite the 11 modified graders.
The 4 Drupal reader fixes were added after the original review's seven changed tasks, hence 95 rather than 99 external graders.

## Modified task verifiers

These changes existed before this review correction. They are retained and disclosed, not claimed equivalent to public upstream scores.

| Task | Changes relative to the public pinned release |
|---|---|
| `agriculture_003` | Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged. |
| `agriculture_011` | Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged. |
| `agriculture_016` | Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged. |
| `agriculture_031` | Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged. |
| `business_031` | Use deployed contact DISPLAY_NAME and journal PUBLISHED_AT schema for offboarding checks. |
| `business_032` | Normalize NULL debit/credit database values before numeric comparisons. |
| `business_051` | Use deployed journal schema; require postings in one published journal; report database faults as verifier errors. |
| `business_052` | Require exactly 24 percent voucher discount (instead of upstream approximate 10.04 or 24); use SALES_ESTIMATES and search matching estimates. |
| `business_143` | Use MANUAL_JOURNALS_ENTRIES and PUBLISHED_AT schema instead of upstream singular table and STATUS. |
| `software_008` | Correct literal f-string construction in the title-check diagnostic (no check-weight change). |
| `software_025` | Use JWT authorization for the deployed Taiga API instead of Token authorization. |

## Runtime and harness adaptations

- Native browser driver: provider configuration, screenshot evidence, and explicit actor versus runtime fault tracking.
- Native slot driver: rootless Enroot support, service readiness and deployment network/UID handling.
- Verifier runner: bounded retries while live apps remain available, completion SCORE requirement, readiness checks and judge fault channel.
- NVIDIA extensions: rootless app/network/UID helpers, OpenCode integration and deployment scripts listed as nvidia-authored in provenance.

## Harness migration boundary

This integration currently owns episodes through legacy_agent and a benchmark-specific OpenCode worker. It is not interchangeable with stock opencode_agent scores: model protocol repair, turn budgets and browser tools belong to this adapter.
A recommended follow-up is a resources server owning per-session app seed/verify/close, browser tools that open apps by name, single_agent_turn plus stock opencode_agent, with browser-use retained as the native baseline. Seed observations cannot be appended to stock OpenCode as a second user message; app-name tools must expose the URLs. Sandbox-provider migration can follow separately. No migration completion is claimed here.

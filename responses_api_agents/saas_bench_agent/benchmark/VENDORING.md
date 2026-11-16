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
| `business_052` | Require exactly 24 percent voucher discount (instead of upstream approximate 10.04 or 24); use SALES_ESTIMATES and search matching estimates. Check 15 requires exact line items and a total within 0.01 (upstream: 3 or more lines and total within 1.0). |
| `business_143` | Use MANUAL_JOURNALS_ENTRIES and PUBLISHED_AT schema instead of upstream singular table and STATUS. |
| `software_008` | Use JWT instead of Token in the Baserow Authorization header. |
| `software_025` | Use JWT instead of Token in the Baserow Authorization header of baserow_api_get. |

## Runtime and harness adaptations

- Native browser driver: provider configuration, screenshot evidence, and explicit actor versus runtime fault tracking.
- Pretix template: new events enable the built-in manual payment provider by default. Upstream new events start with no payment provider enabled. This can make go-live checks easier; scores are not claimed equivalent to the public upstream release.
- Native slot driver: rootless Enroot support, service readiness and deployment network/UID handling.
- Verifier runner: bounded retries while live apps remain available, completion SCORE requirement, readiness checks and judge fault channel.
- NVIDIA extensions: rootless app/network/UID helpers, OpenCode integration and deployment scripts listed as nvidia-authored in provenance.

## Harness migration boundary

Both evaluation settings use the shipped legacy_agent relay and the native SaaS agent's /run. The trusted native worker couples app provisioning, harness execution and native final-state verification; the agent applies outcome policy and aggregates metrics. This adapter supports OpenCode and browser-use evaluation, without claiming interchangeable stock-harness scores or multi-harness training support.

A separate follow-up will move app seed/verify/close, browser tools, action budgets and grading into a Resources Server and select the shipped single_agent_turn_legacy Environment Server. OpenCode will use stock opencode_agent over MCP; browser-use will retain its baseline protocol through a session adapter. Apps remain Resources-owned live state, separate from the harness sandbox. Each harness requires session, model-rollout and evidence validation; RL support also requires token-capture and GRPO smokes. Existing frozen campaigns retain their original source pins and scores.

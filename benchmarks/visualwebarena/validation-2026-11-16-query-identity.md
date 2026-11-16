# Snapshot query resource identity — 2026-11-16 UTC / 2026-11-17 CST

The shared snapshot URL matcher sorted all query `(key, value)` pairs. A request for `/item?item=first&item=second&mode=edit` could therefore accept a redirect to `/item?mode=edit&item=second&item=first`. An application that selects the first repeated value has changed from object `first` to object `second`. The scorer could award the second object's `saved` text to the first object's task.

This was reproduced on the unmodified published parent `f28e5396558d608337c643a9477b4ad5f0b5a028`. The pair-sorting matcher also predates that parent and is present on main; this report does not attribute its introduction to the preceding semantic repair. PR #2918 already changes this shared snapshot module for its configured VisualWebArena grading path. This focused follow-up preserves each key's ordered value list while continuing to accept reordering between distinct keys. Duplicate multiplicity, changed values and blank values remain significant. Existing origin/path checks and bounded site canonicalization remain in place.

The complete new regression comparison is in [all 22 cases, all nine columns](validation-2026-11-16-query-identity-cases.csv). The file retains every exact test ID, both verdicts, both durations, both class names and the old failure type/message. It is an actual before/after comparison, not an inferred old verdict. The original negative-control log records the imported module/adapter authority check and a byte comparison with the published parent before editing product source.

## Regression coverage and results

| Coverage | Cases | Observed behavior |
| --- | ---: | --- |
| Direct resource-identity controls | 8 | Reordering distinct keys and interleaving them between repeated values succeeds; reversing one key's values, dropping a duplicate, changing a value, blanking a value or adding a blank flag changes identity |
| Sync/async × HTML/image × same/wrong resource | 8 | Real collector/context functions retain healthy navigation metadata; a wrong after-page yields conclusive absent target content and a wrong baseline raises an infrastructure error; every auxiliary page closes |
| Native sync, native async, public evaluator × same/wrong resource | 6 | Normal public adaptation creates the shared-target plan; real snapshot construction refuses retained `saved` text from the other resource, while distinct-key reordering still scores 1 |

| Check | Result |
| --- | --- |
| Unmodified f28 parent, before product edit | 8 failed counterexamples, 14 passing controls; 0 errors/skips |
| Fixed query module | 22 passed; 0 errors/skips |
| Query + existing navigation + previous exact/semantic modules | 177 passed; 0 errors/skips; root Python environment has no Playwright installed |
| Official complete browser-component command and shipped data | 1074 actual test cases passed; 0 failures/errors/skips; data validated |
| Selected shared web/agent/dataset/public API/import/Fern checks | 2101 actual test cases passed; 0 failures/errors/skips |

All JUnit counts above enumerate `<testcase>` elements rather than trusting an aggregate emitted by a nested test runner. The full old/new XML and command output are retained in local run `2026-11-16-pr2918-query-identity-followup`. The 22 new cases mix URL identity, capture and scoring checks; they are not all described as scoring-only cases. The existing 651 VisualWebArena scoring regressions remain covered by the component suite.

Executed test commands from the Gym checkout:

```bash
.venv/bin/python -m pytest resources_servers/webarena_browser/tests/test_snapshot_query_identity.py -q --no-cov -o log_cli=false --junitxml=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-merge-followup/query-before.xml

.venv/bin/python -m pytest resources_servers/webarena_browser/tests/test_snapshot_query_identity.py resources_servers/webarena_browser/tests/test_snapshot_navigation.py resources_servers/webarena_browser/tests/test_visualwebarena_semantic_context.py resources_servers/webarena_browser/tests/test_visualwebarena_exact_evidence.py -q --no-cov -o log_cli=false --junitxml=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-query-identity-followup/junit/targeted.xml

PLAYWRIGHT_BROWSERS_PATH=/tmp/vwa-pr2918-main-rebase-browser-cache PATH=/tmp/vwa-pr2918-checkin-uv-cache/archive-v0/iB5P_KDaZXSbMV8R/uv-0.13.4.data/scripts:$PATH UV_CACHE_DIR=/tmp/vwa-pr2918-checkin-uv-cache GYM_CI_JUNIT_DIR=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-query-identity-followup/junit .venv/bin/gym env test +entrypoint=resources_servers/webarena_browser +should_validate_data=true +uv_cache_dir=/tmp/vwa-pr2918-checkin-uv-cache

.venv/bin/python -m pytest tests/unit_tests/web tests/unit_tests/test_public_api.py tests/unit_tests/test_public_api_facades.py tests/unit_tests/test_import_linter_coverage.py tests/unit_tests/data/test_task_data.py tests/unit_tests/tooling/test_migrate_imports.py tests/unit_tests/test_fern_docs_links.py benchmarks/visualwebarena/tests benchmarks/webarena/tests responses_api_agents/web_agent/tests -q --no-cov -o log_cli=false --junitxml=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-query-identity-followup/junit/core.xml
```

The first command ran under a wrapper that asserted source bytes and import authority. The original log and XML are preserved byte-for-byte in the new run's `logs/before.log` and `junit/before.xml`; their original paths also remain intact.

## Real-model and real-browser verification

Two fresh, synthetic HTTP sites use server-side state visible to the capture collector's new browser tabs. Their application selects the first `item` query value. Before Save, object `first` displays `old`. A real model click commits server state, and an actual HTTP 302 redirects every subsequent request for the configured target to the scenario's final URL, which returns HTTP 200. The two cases differ only in whether that redirect preserves repeated-value order. The evaluator sees the real final status and URL. Screenshots visibly show object identity as well as the actual `saved` text, including the other object's text in the negative case.

| Scenario | Requested query | Actual final query | Actual selected object | Actual displayed value | Captured target value | Reward | Valid / unmasked | Policy calls | Judge calls | Model turns | Environment steps |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| reordered-keys | `item=first&item=second&mode=edit` | `mode=edit&item=first&item=second` | `first` | `saved` | saved | 1.0 | yes / yes | 2 | 0 | 2 | 2 |
| reversed-values | `item=first&item=second&mode=edit` | `mode=edit&item=second&item=first` | `second` | `saved` | empty string (off target) | 0.0 | yes / yes | 2 | 0 | 2 | 2 |

Both cases passed a separate actual-browser/cross-tab preflight with zero model/judge calls before the model smoke. The model then made four real policy calls in total and no judge calls. Both resulting samples are valid and unmasked, with no infrastructure failure. The negative 0 is the configured verifier's decision after the real model changed the selected object, not a browser failure or judge outage. Both real model traces are retained in full, including model requests/responses, parsed calls, all action arguments and agent results.

Policy endpoint: `https://inference-api.nvidia.com/v1`; model: `us/azure/openai/gpt-4.1`. This is the established GPT-4.1 synthetic qualification transport. It does not run the historical Nano Omni checkpoint. Tokenizer and chat template are provider-managed; no local tokenizer/template override is applied. Gym's unchanged `nano_omni_toolcall` protocol and `parse_nano_omni_tool_calls` parser handle the model calls; reasoning parser is `null`, token-ID capture is disabled. Policy temperature is 0, max output tokens 1024, maximum steps 5, image history 3, parse retries 0, model-turn retries 0. `top_p`, seed and stop are not sent. The Azure history bridge inserts explicit tool acknowledgements; it does not rewrite action arguments or replace model calls. The negative case's requested 2-second wait was honored. No semantic or image judge is required for `exact_match: saved`.

The in-process fixture transport uses the actual WebAgent loop, converter, parser, real Chromium mouse actions, HTTP server, native captures, snapshot context and public verifier. It implements seed/step/evaluate/close for these two fixtures; it does not start the production resource-server process, PyAutoGUI, remote sandbox provider or maintained site stack. The `_config()` verifier configuration retains the production headed requirement; the separately declared fixture Chromium is headless with viewport 640×480 and device scale 1. The full two resolved configurations below retain both values explicitly. Python is 3.13.14; Playwright 1.55.0 uses installed Chromium build 1187. No dependency/lock changes are part of this follow-up.

All six model-observation frames are retained. The three distinct images were visually inspected and show first/old, first/saved and second/saved. Full SHA-256 values identify the other three duplicates in `frame-inventory.json`. The local source-identity record includes the exact snapshot-module bytes and SHA-256 used for the model smoke.

One setup failure is retained: the first zero-API preflight selected Playwright's uninstalled separate headless-shell executable by default. It failed before browser launch and before any model/judge call. Explicitly selecting the already-installed Chromium executable fixed the fixture; the first script and full failure log remain unchanged. There were no rejected/scored model attempts or model-result reruns in this follow-up.

Executed browser/model commands:

```bash
PLAYWRIGHT_BROWSERS_PATH=/tmp/vwa-pr2918-main-rebase-browser-cache PYTHONPATH=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/checkouts/pr2918-review-fixes-20261116 resources_servers/webarena_browser/.venv/bin/python /Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-query-identity-followup/real-model-query-smoke.py --preflight

PLAYWRIGHT_BROWSERS_PATH=/tmp/vwa-pr2918-main-rebase-browser-cache PYTHONPATH=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/checkouts/pr2918-review-fixes-20261116 resources_servers/webarena_browser/.venv/bin/python /Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-query-identity-followup/real-model-query-smoke.py
```

The model credential stays at its existing remote location and is used in place through the existing SSH connection. No key is copied into the evidence or repository. No denied historical worker configuration, raw bundle or remote rollout frame is transferred by this follow-up.

## Full resolved configuration for each real-model case

These are the complete saved YAML values, including each actual loopback port, task/plan, every agent default and the distinct resource-verifier/fixture browser settings. They are not a reconstruction of the historical benchmark configuration.

### reordered-keys

```yaml
scope: synthetic query-identity capture smoke; not maintained-population evaluation
endpoint: https://inference-api.nvidia.com/v1
model: us/azure/openai/gpt-4.1
tokenizer: provider managed
chat_template: provider managed
policy_protocol: nano_omni_toolcall
tool_parser: parse_nano_omni_tool_calls
reasoning_parser: null
judge:
  called: false
  reason: native exact_match only
sampling:
  temperature: 0
  max_output_tokens: 1024
  top_p: not sent
  seed: not sent
  stop: not sent
agent:
  host: localhost
  port: 8001
  num_workers: null
  uvicorn_reuse_port: null
  entrypoint: app.py
  domain: null
  name: web_agent
  session_close_timeout_seconds: 60.0
  session_close_retry_window_seconds: 300.0
  skip_verification: false
  skip_verification_reward: 0.0
  tool_servers: []
  token_id_capture: false
  tool_accesses: []
  resources_server:
    type: resources_servers
    name: browser
  environment_server: null
  model_server:
    type: responses_api_models
    name: policy
  policy_protocol: nano_omni_toolcall
  max_steps: 5
  max_parse_retries: 0
  nano_omni_max_tool_calls: 8
  nano_omni_max_computer_actions: 20
  nano_omni_max_scroll_amount: 50
  nano_omni_retry_invalid_tool_calls: true
  nano_omni_parse_retry_feedback: false
  nano_omni_parse_retry_temperature: null
  nano_omni_parse_retry_delay_secs: 0.0
  repeated_action_warning_threshold: 0
  repeated_action_window: 5
  max_consecutive_execution_failures: 3
  model_turn_max_retries: 0
  model_retry_delay_secs: 1.0
  max_image_history: 3
  qwen_fold_size: 10
  qwen_history_n: 100
  qwen_coordinate_type: relative
  qwen_thinking: true
  judge_max_screenshots: 3
  task_image_root: null
  max_task_image_bytes: 26214400
  visual_observation_text: full_axtree
  redact_old_visual_observations: false
  resources_request_timeout_secs: 180.0
  seed_request_timeout_secs: 1800.0
  seed_retry_initial_delay_secs: 2.0
  seed_retry_max_delay_secs: 30.0
  model_request_timeout_secs: 600.0
  judge_request_timeout_secs: 300.0
  close_request_timeout_secs: 30.0
  run_timeout_secs: 1800.0
task:
  benchmark: visualwebarena
  task_id: '0'
  intent: Click Save once to submit the preset state, then terminate after the update.
    The button center is at pixel (200,120), normalized (0.3125,0.25).
  start_urls:
  - http://127.0.0.1:64357/item?item=first&item=second&mode=edit
  sites: []
  input_images: []
  runtime_profile: visual_browser
  observation_profile: screenshot
  action_profile: computer_use
  verifier_profile: visualwebarena_classic
  auth_profile: null
  seed: 0
  task_kwargs:
    collision_plan:
      snapshot_adapters:
        program_html:
          targets:
          - key: 5c15f84c27bedba5
            target:
              url: http://127.0.0.1:64357/item?item=first&item=second&mode=edit
              locator: document.querySelector('#status').textContent
              required_contents:
                exact_match: saved
      target_overrides: {}
  original_metadata:
    id: query-reordered-keys-0
    ques: Click Save once to submit the preset state, then terminate after the update.
      The button center is at pixel (200,120), normalized (0.3125,0.25).
    web:
    - http://127.0.0.1:64357/item?item=first&item=second&mode=edit
    eval:
      eval_types:
      - program_html
      program_html:
      - url: http://127.0.0.1:64357/item?item=first&item=second&mode=edit
        locator: document.querySelector('#status').textContent
        required_contents:
          exact_match: saved
resource_verifier_config:
  host: localhost
  port: 8010
  num_workers: 1
  uvicorn_reuse_port: null
  entrypoint: app.py
  domain: agent
  name: webarena
  expose_tools_over_mcp: false
  session_close_retry_window_seconds: 300.0
  max_sessions: 1
  artifact_dir: cache/webarena-browser/artifacts
  inline_screenshots: true
  max_evidence_screenshots: 200
  session_ttl_seconds: 3600
  reaper_interval_seconds: 60
  browser_session_provider:
    local_process: {}
  browser_session_options: {}
  browser_lease_ttl_seconds: 900
  browser_acquire_timeout_seconds: 300.0
  browser_release_timeout_seconds: 60.0
  browser_heartbeat_interval_seconds: 60.0
  browser_heartbeat_timeout_seconds: 30.0
  browser_heartbeat_failure_limit: 3
  require_auth: false
  auth_token_env: VISUAL_BROWSER_RESOURCES_TOKEN
  site_pool_mode: unmanaged
  max_sessions_per_site: 4
  allowed_benchmarks:
  - webarena
  - visualwebarena
  browser_auto_install: true
  headless: false
  viewport_width: 1920
  viewport_height: 1080
  action_delay_seconds: 2.0
  default_timeout_ms: 45000
  terminate_on_action_error: true
  max_tool_calls: 8
  max_computer_actions: 20
  max_scroll_amount: 50
  scroll_timeout_seconds: 30.0
  action_limits:
    max_wait_seconds: 30.0
    max_step_wait_seconds: null
  record_video: false
  browser_channel: null
  task_image_root: null
  max_task_image_bytes: 26214400
browser:
  headless: true
  viewport:
  - 640
  - 480
  device_scale_factor: 1
history_bridge: explicit Azure tool acknowledgements; no argument rewriting
transport: existing remote credential used in place
```

### reversed-values

```yaml
scope: synthetic query-identity capture smoke; not maintained-population evaluation
endpoint: https://inference-api.nvidia.com/v1
model: us/azure/openai/gpt-4.1
tokenizer: provider managed
chat_template: provider managed
policy_protocol: nano_omni_toolcall
tool_parser: parse_nano_omni_tool_calls
reasoning_parser: null
judge:
  called: false
  reason: native exact_match only
sampling:
  temperature: 0
  max_output_tokens: 1024
  top_p: not sent
  seed: not sent
  stop: not sent
agent:
  host: localhost
  port: 8001
  num_workers: null
  uvicorn_reuse_port: null
  entrypoint: app.py
  domain: null
  name: web_agent
  session_close_timeout_seconds: 60.0
  session_close_retry_window_seconds: 300.0
  skip_verification: false
  skip_verification_reward: 0.0
  tool_servers: []
  token_id_capture: false
  tool_accesses: []
  resources_server:
    type: resources_servers
    name: browser
  environment_server: null
  model_server:
    type: responses_api_models
    name: policy
  policy_protocol: nano_omni_toolcall
  max_steps: 5
  max_parse_retries: 0
  nano_omni_max_tool_calls: 8
  nano_omni_max_computer_actions: 20
  nano_omni_max_scroll_amount: 50
  nano_omni_retry_invalid_tool_calls: true
  nano_omni_parse_retry_feedback: false
  nano_omni_parse_retry_temperature: null
  nano_omni_parse_retry_delay_secs: 0.0
  repeated_action_warning_threshold: 0
  repeated_action_window: 5
  max_consecutive_execution_failures: 3
  model_turn_max_retries: 0
  model_retry_delay_secs: 1.0
  max_image_history: 3
  qwen_fold_size: 10
  qwen_history_n: 100
  qwen_coordinate_type: relative
  qwen_thinking: true
  judge_max_screenshots: 3
  task_image_root: null
  max_task_image_bytes: 26214400
  visual_observation_text: full_axtree
  redact_old_visual_observations: false
  resources_request_timeout_secs: 180.0
  seed_request_timeout_secs: 1800.0
  seed_retry_initial_delay_secs: 2.0
  seed_retry_max_delay_secs: 30.0
  model_request_timeout_secs: 600.0
  judge_request_timeout_secs: 300.0
  close_request_timeout_secs: 30.0
  run_timeout_secs: 1800.0
task:
  benchmark: visualwebarena
  task_id: '0'
  intent: Click Save once to submit the preset state, then terminate after the update.
    The button center is at pixel (200,120), normalized (0.3125,0.25).
  start_urls:
  - http://127.0.0.1:64410/item?item=first&item=second&mode=edit
  sites: []
  input_images: []
  runtime_profile: visual_browser
  observation_profile: screenshot
  action_profile: computer_use
  verifier_profile: visualwebarena_classic
  auth_profile: null
  seed: 0
  task_kwargs:
    collision_plan:
      snapshot_adapters:
        program_html:
          targets:
          - key: 47f8a981a8b37c4c
            target:
              url: http://127.0.0.1:64410/item?item=first&item=second&mode=edit
              locator: document.querySelector('#status').textContent
              required_contents:
                exact_match: saved
      target_overrides: {}
  original_metadata:
    id: query-reversed-values-0
    ques: Click Save once to submit the preset state, then terminate after the update.
      The button center is at pixel (200,120), normalized (0.3125,0.25).
    web:
    - http://127.0.0.1:64410/item?item=first&item=second&mode=edit
    eval:
      eval_types:
      - program_html
      program_html:
      - url: http://127.0.0.1:64410/item?item=first&item=second&mode=edit
        locator: document.querySelector('#status').textContent
        required_contents:
          exact_match: saved
resource_verifier_config:
  host: localhost
  port: 8010
  num_workers: 1
  uvicorn_reuse_port: null
  entrypoint: app.py
  domain: agent
  name: webarena
  expose_tools_over_mcp: false
  session_close_retry_window_seconds: 300.0
  max_sessions: 1
  artifact_dir: cache/webarena-browser/artifacts
  inline_screenshots: true
  max_evidence_screenshots: 200
  session_ttl_seconds: 3600
  reaper_interval_seconds: 60
  browser_session_provider:
    local_process: {}
  browser_session_options: {}
  browser_lease_ttl_seconds: 900
  browser_acquire_timeout_seconds: 300.0
  browser_release_timeout_seconds: 60.0
  browser_heartbeat_interval_seconds: 60.0
  browser_heartbeat_timeout_seconds: 30.0
  browser_heartbeat_failure_limit: 3
  require_auth: false
  auth_token_env: VISUAL_BROWSER_RESOURCES_TOKEN
  site_pool_mode: unmanaged
  max_sessions_per_site: 4
  allowed_benchmarks:
  - webarena
  - visualwebarena
  browser_auto_install: true
  headless: false
  viewport_width: 1920
  viewport_height: 1080
  action_delay_seconds: 2.0
  default_timeout_ms: 45000
  terminate_on_action_error: true
  max_tool_calls: 8
  max_computer_actions: 20
  max_scroll_amount: 50
  scroll_timeout_seconds: 30.0
  action_limits:
    max_wait_seconds: 30.0
    max_step_wait_seconds: null
  record_video: false
  browser_channel: null
  task_image_root: null
  max_task_image_bytes: 26214400
browser:
  headless: true
  viewport:
  - 640
  - 480
  device_scale_factor: 1
history_bridge: explicit Azure tool acknowledgements; no argument rewriting
transport: existing remote credential used in place
```

## Scope and remaining qualification

This changes configured planned-snapshot resource identity. No effect on the maintained 908-task population has been demonstrated or measured. There is no new canonical 908 score, site-state parity claim or replacement for either historical Nano Omni result. The maintained dataset, historical run configurations and benchmark population are unchanged. Reference/protocol qualification, run-bound historical environment evidence and earlier raw-sharing decisions retain their previous recorded status.

The preceding [semantic-context report](validation-2026-11-16-semantic-context.md), [task-568 report](validation-2026-11-16-exact-match.md), [second review](validation-2026-11-16-followup.md), [first review](validation-2026-11-16.md) and their complete CSVs remain in place. Public-source attribution and protected CI are separate maintainer decisions; local tests and this smoke do not constitute review approval or a CI trust grant.

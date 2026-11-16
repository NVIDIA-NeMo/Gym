# VisualWebArena complete semantic context — 2026-11-16

This repairs [P2 discussion 4585934668](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4585934668) on reviewed source `83f21c8a711c30612da843106fcd707e768c6858`. The reviewer independently verified the previous task-568 repair, all 60 comparison verdicts, and resolved that P1.

## Trigger and repair

A configured planned target requires `fuzzy_match: saved`, or `must_include: [saved]` using its semantic fallback. Changing the actual value from `not` to `not\npersisted` leaves the negation in the observed target. The real snapshot producer emits line delta `persisted`; token freshness also produces `persisted`. The previous scorer could award 1 from those projections even when the same matcher rejected the complete observed target. Its later full-value score could not reduce that earlier maximum.

The scorer now evaluates and caches all requirements against the complete observed target before awarding any freshness credit. A rejected full target ends that record's evaluation. Fragments only establish freshness, and their contribution is bounded by the cached full-target score. The existing strict predicates, literal matching, semantic judge prompts and label parsing are reused. New matching copies and paraphrases remain eligible; unchanged or moved matching content gains no freshness.

This changes configured planned-semantic grading. No canonical 908-score impact has been demonstrated or measured. Dataset preparation, task population, model protocols, dependencies and shared scanner policy are unchanged. There is no new maintained-population run, reference-parity claim or replacement of either historical Nano Omni score.

## Complete regression comparison

[All 30 old/new cases](validation-2026-11-16-semantic-cases.csv) retain the original nine columns, every test ID, both verdicts, both durations, both class names and the original failure type/message. Each transition below runs for both matcher kinds through native sync, native async and the public evaluator: 5 × 2 × 3 = 30 cases.

| Before | Actual after | Expected reward |
| --- | --- | ---: |
| `not` | `not\npersisted` | 0 |
| `not` | `not persisted` | 0 |
| `not` | `persisted` | 1 |
| `persisted` | `persisted\npersisted` | 1 |
| `persisted\nother` | `other\npersisted` | 0 |

These are configured synthetic tasks. Normal public adaptation of two shared-target records creates the plan; no manual collision keys are supplied. Actual snapshot construction retains the complete observed value and independently proves that the multiline counterexample's line delta omits `not`. Only browser capture and semantic API responses are controlled in these CPU tests. The deterministic matcher rejects retained negation, making the six counterexamples independent of a hosted judge's paraphrase preferences.

| Check | Actual result |
| --- | --- |
| Reviewed source, before editing the scorer | 6 failed counterexamples, 24 passing controls, zero errors/skips; scorer bytes checked against `83f21c8a7` |
| Fixed new module | 30 passed, zero errors/skips |
| New module plus the preceding exact/numeric/semantic module | 90 passed, zero errors/skips; no Playwright installed in the root environment |
| Full browser component and shipped-data validation | 1052 passed, zero errors/skips |
| Selected shared web/agent/preparation/public API/import/Fern checks | 2101 passed, zero errors/skips |

The first shared-suite attempt ran in a sandbox that prohibited local socket binding: 2090 passed, six port-allocation failures and five test-server setup errors. The identical suite passed with loopback permission; the initial full log and XML remain intact. This is separate from scorer assertions.

Executed commands, from the Gym checkout:

```bash
.venv/bin/python -m pytest resources_servers/webarena_browser/tests/test_visualwebarena_semantic_context.py -q --no-cov -o log_cli=false --junitxml=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-semantic-context-followup/junit/before.xml

.venv/bin/python -m pytest resources_servers/webarena_browser/tests/test_visualwebarena_semantic_context.py resources_servers/webarena_browser/tests/test_visualwebarena_exact_evidence.py -q --no-cov -o log_cli=false --junitxml=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-semantic-context-followup/junit/targeted.xml

PLAYWRIGHT_BROWSERS_PATH=/tmp/vwa-pr2918-main-rebase-browser-cache PATH=/tmp/vwa-pr2918-checkin-uv-cache/archive-v0/iB5P_KDaZXSbMV8R/uv-0.13.4.data/scripts:$PATH UV_CACHE_DIR=/tmp/vwa-pr2918-checkin-uv-cache GYM_CI_JUNIT_DIR=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-semantic-context-followup/junit .venv/bin/gym env test +entrypoint=resources_servers/webarena_browser +should_validate_data=true +uv_cache_dir=/tmp/vwa-pr2918-checkin-uv-cache

.venv/bin/python -m pytest tests/unit_tests/web tests/unit_tests/test_public_api.py tests/unit_tests/test_public_api_facades.py tests/unit_tests/test_import_linter_coverage.py tests/unit_tests/data/test_task_data.py tests/unit_tests/tooling/test_migrate_imports.py tests/unit_tests/test_fern_docs_links.py benchmarks/visualwebarena/tests benchmarks/webarena/tests responses_api_agents/web_agent/tests -q --no-cov -o log_cli=false --junitxml=/Users/jepeng/workspace/rl/my-rl/webarena/visualwebarena/runs/2026-11-16-pr2918-semantic-context-followup/junit/core-final.xml
```

The first command was executed before changing the scorer, after checking its bytes against the reviewed commit. Full JUnit XML and command outputs are retained under local run `2026-11-16-pr2918-semantic-context-followup`; the CSV has no omitted cases or columns.

## Real policy and semantic-judge probes

The local fixture uses real WebAgent/ChatCompletionsConverter, the Nano Omni tool parser, Chromium, normal public task adaptation, native snapshot capture, native semantic prompts/JSON label parsing and the public verifier. A thin in-process fixture transport implements click/wait/terminate. Its Save button persists a preset value in browser storage so the verifier's fresh tabs see the committed state. This exercises semantic context and freshness, separately from production headed-site deployment and maintained task performance.

| Setting | Actual value |
| --- | --- |
| Policy and judge model / returned model ID | `us/azure/openai/gpt-4.1` / `us/azure/openai/gpt-4.1` |
| Endpoint | `https://inference-api.nvidia.com/v1` |
| Tokenizer / template | Provider managed; no local override |
| Policy protocol / tool parser | `nano_omni_toolcall` / Gym `parse_nano_omni_tool_calls`; arguments preserved |
| Policy reasoning parser | None; converter `uses_reasoning_parser=False` |
| Policy sampling | Temperature 0; 1024 output tokens; top-p, seed and stop not sent |
| Policy limits | 5 steps, 3 recent screenshots, 0 parse retries |
| Judge protocol / parser | Native `_llm_fuzzy_match` or `_llm_must_include` prompts; native `_parse_judge_json_label`; correct → 1, incorrect/partially correct → 0 |
| Judge sampling | Temperature 0; 768 output tokens; top-p 1.0; seed and stop not sent |
| History transport | Explicit Azure tool acknowledgements; screenshots remain user observations |
| API credential transport | Existing remote credential used in place; only HTTPS transport delegated over the existing SSH connection; local presence marker satisfies evaluator configuration validation |
| Browser fixture | Playwright 1.55.0 / matching Chromium 1187; headless 640 × 480, device scale factor 1; fresh context for each scenario |
| Runtime | Python 3.13.14; Pillow 12.3.0; NumPy 2.5.4; scikit-image 0.26.0; httpx 0.28.1; Pydantic 2.14.0; OpenAI 2.45.0 |
| Resources | No GPU allocation; no production site deployment or reset |

The table retains every completed model probe, including the unqualified persistence attempt. Each row used two policy calls, two model turns and two environment steps; every native verifier outcome was valid and unmasked, with no infrastructure failure. `\n` denotes an actual newline.

| Attempt | Scenario | Captured before | Captured after | Line delta | Reward | Judge calls | Disposition |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| r2 | fallback-retained-negation | `not` | `not` | *none* | 0.0 | 0 | Unqualified fixture persistence; no mutation reached the snapshot |
| r2 | fuzzy-new-copy | `persisted` | `persisted` | *none* | 0.0 | 0 | Unqualified fixture persistence; no mutation reached the snapshot |
| r2 | fuzzy-retained-negation | `not` | `not` | *none* | 0.0 | 0 | Unqualified fixture persistence; no mutation reached the snapshot |
| r3 | fallback-retained-negation | `not` | `not\npersisted` | `persisted` | 0.0 | 1 | Required control passed |
| r3 | fuzzy-new-copy | `persisted` | `persisted\npersisted` | `persisted` | 0.0 | 1 | Retained valid 0: native full-value judgement was partially correct; initial expected-1 assertion failed |
| r3 | fuzzy-retained-negation | `not` | `not\npersisted` | `persisted` | 0.0 | 1 | Required control passed |
| r4 | fallback-moved-content | `saved\nother` | `other\nsaved` | `other` | 0.0 | 0 | Required control passed |
| r4 | fallback-new-paraphrase | `saved` | `saved\nsuccessfully persisted` | `successfully persisted` | 1.0 | 1 | Required control passed |
| r4 | fuzzy-new-saved-copy | `saved` | `saved\nsaved` | `saved` | 1.0 | 3 | Required control passed |

The five required persistent-state controls passed across attempts r3/r4. Those attempts also retain the separate `persisted\npersisted` probe: the real judge called it only partially equivalent to `saved`, so the native rule correctly produced 0. That observation was neither rerun nor replaced. A distinct explicit `saved\nsaved` control establishes new-copy credit without assuming a hosted judge will accept the first paraphrase. Its successful score is a separate scenario.

All seven native judge calls from r3/r4 are retained below, including baseline/freshness calls and full reasoning. There were 12 policy calls and 7 judge calls across these six persistent-state probes. The earlier r2 attempt consumed another 6 policy calls and zero judge calls. Attempt r1 failed configuration validation before any model/judge call. Before r3, all five originally planned transitions passed actual cross-tab capture preflight with zero API calls. The r3 assertion on the ambiguous probe ended that attempt; r4 executed only the remaining controls and the separately named explicit-copy control.

| Attempt | Scenario | Request index within attempt | Actual prediction sent to native judge | Label | Full judge reasoning |
| --- | --- | ---: | --- | --- | --- |
| r3 | fuzzy-retained-negation | 2 | `not\npersisted` | incorrect | The student's answer 'not persisted' does not convey the same meaning as 'saved'. 'Saved' implies that something has been stored or kept, while 'not persisted' suggests it has not been stored. The meanings are opposite. |
| r3 | fallback-retained-negation | 5 | `not\npersisted` | incorrect | The student's answer does not mention 'saved', which is the only required item. |
| r3 | fuzzy-new-copy | 8 | `persisted\npersisted` | partially correct | The student's answer 'persisted' is similar in meaning to 'saved', but they are not always semantically equivalent. 'Persisted' implies data is stored for a long duration, while 'saved' can refer to storing data temporarily or permanently. The answer captures part of the meaning but not all. |
| r4 | fuzzy-new-saved-copy | 2 | `saved\nsaved` | correct | The student's answer 'saved saved' contains the reference answer 'saved' twice, which does not alter the meaning and is semantically equivalent to the reference answer. |
| r4 | fuzzy-new-saved-copy | 3 | `saved` | correct | The student's answer 'saved' is exactly the same as the reference answer, demonstrating semantic equivalence. |
| r4 | fuzzy-new-saved-copy | 4 | `saved` | correct | The student's answer 'saved' is exactly the same as the reference answer, demonstrating semantic equivalence. |
| r4 | fallback-new-paraphrase | 7 | `successfully persisted` | correct | The student's answer 'successfully persisted' semantically contains the required information 'saved', as 'persisted' means saved and 'successfully' indicates the action was completed. |

All 18 r3/r4 frames are retained, with full file sizes and hashes in the local frame inventory. The nine distinct images were visually inspected; duplicate frames were matched by full SHA-256. Raw policy requests/responses, all native judge prompts/responses, actual per-case agent/task/browser configuration, snapshots and verifier outputs remain in the same local run record. No raw bundle, historical configuration or original remote frame was transferred or uploaded in this follow-up.

Qualification failures are preserved: r1 attempted a headed-resource configuration with `headless=True` and failed validation before a model call; the fixture then used the established evaluator configuration and a separately declared headless browser. r2 failed to persist DOM-only updates across the snapshot collector's new tabs, so its three zero scores are not semantic-fix evidence. Browser storage and cross-tab preflight repaired that fixture without changing product configuration constraints. The r3 native-judge partial-equivalence result and its failed initial expectation also remain in full.

The preceding reference/protocol, run-bound production health/resolved-configuration and original historical-frame gaps remain open. The separately requested raw-sharing authorizations are still pending. WebArena's original 812-task/step7000 scope is unchanged.

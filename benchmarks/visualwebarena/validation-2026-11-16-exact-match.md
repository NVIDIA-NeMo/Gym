# VisualWebArena observed-value repair — 2026-11-16

This fixes [P1 discussion 4584578204](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4584578204) on reviewed source `435c297d652e288a5ee7d14ddd6c260ea53cec4d`. The reviewer verified and resolved the four preceding findings; this separate repair addresses a regression introduced by token-based freshness handling.

## Trigger and corrected behavior

Canonical runtime/source task 568 requires the comment exactly `It has 3 $%@`. With baseline `unrelated` and observed comment `It has 3 $%@ unrelated`, token subtraction produced the required string and incorrectly awarded 1. An old-word prefix had the same false success. The complete observed comment fails the exact criterion.

Strict `exact_match`, `required_values` and `must_exclude` predicates now grade the complete observed target before any freshness credit. Exact and numeric requirements gain credit when the actual full value improves over the baseline. They are never applied to a token-subtracted reconstruction. Semantic freshness uses new fragments as an additional gate, and the observed added text must independently satisfy the semantic matcher; removing an old negation cannot manufacture a fuzzy success. Identical new/observed text reuses its score.

This preserves valid exact replacement, valid deletion/correction, normalization already allowed by the reference matcher, new semantic confirmations, newly added copies and moved-content/no-op rejection. It also prevents old-token removal from turning a nonnumeric observation into a passing number.

## Regression evidence

The [complete 60-case comparison](validation-2026-11-16-exact-cases.csv) retains every case, old/new verdict, old/new duration, class name and original failure type/message. Full JUnit XML, original failures and all command logs remain under workspace run `2026-11-16-pr2918-exact-match-followup`.

| Check | Result |
| --- | --- |
| Exact reviewed source with asserted import authority | 60 cases: 12 failed counterexamples, 48 passing controls, zero errors/skips |
| Fixed source | All 60 passed |
| Core environment without Playwright installed | All 60 passed; the new module has no browser SDK import requirement |
| Full browser component and shipped-data validation | 1022 passed, zero skipped; includes the previous 561 scoring regressions and all 60 new cases |
| Shared web/agent/preparation/public-API/import/Fern checks | 2101 passed, zero skipped |
| Managed locks / all-files hooks / full PR hygiene | 304 locks passed; hooks, Guardwords and secret checks passed |

The canonical tests call the normal public adapter on actual source tasks 568 and 569, whose common target produces the snapshot plan. They supply no manual collision keys. Both canonical target predicates are graded through the real snapshot-context producer and native sync/async/public evaluator paths. Separate normal adaptation of the complete hash-verified 908-row source confirms task 568's two targets are planned. Numeric and semantic-negation cases are explicitly synthetic controls. The two complete canonical records are retained in a dedicated YAML fixture; only occurrences of the public Reddit target URL carry explanatory inline secret-scan allowlists. Shared scanner policy and the baseline remain byte-for-byte unchanged from reviewed source `435c297d`.

| Observed transition | Correct reward |
| --- | ---: |
| `unrelated` → `It has 3 $%@` | 1 |
| `unrelated` → `It has 3 $%@ unrelated` | 0 |
| `unrelated` → `unrelated It has 3 $%@` | 0 |
| `old` → `It has 3 $%@ newly invalid` | 0 |
| Required exact comment unchanged | 0 |
| Required exact comment → required comment plus suffix | 0 |
| Wrong digit → required exact comment | 1 |
| Invalid prefix removed, leaving the required exact comment | 1 |
| A full value containing two copies of the exact answer | 0 |
| Existing permitted case/outer-whitespace normalization | 1 |

The first full-suite attempt retained 1008 passes and two setup errors from Chromium launch timeouts (180 seconds each). Neither error entered the scorer. The final complete suite was rerun after the final source change without a concurrent model/browser smoke; the original error logs remain preserved rather than being relabeled as passing tests.

Fixture provenance: the complete records in `resources_servers/webarena_browser/tests/fixtures/visualwebarena_exact_tasks.yaml` come from [`visualwebarena.jsonl`](https://github.com/jayl940712/webarena_benchmarks/blob/6a2977939b157b0ab9de7799bb089c721f1ac115/visualwebarena.jsonl) at revision `6a2977939b157b0ab9de7799bb089c721f1ac115` (Apache-2.0). The fixture and its test header retain this attribution.

## Separate real-model strict-format smoke

Two synthetic browser scenarios use the real Gym agent, native tool parser, actual Chromium before/after capture and public verifier. The fixture's Save button submits a preset comment. This validates grading of actual browser state, not the model's visual counting ability on maintained task 568.

| Setting | Actual value |
| --- | --- |
| Model / endpoint | `us/azure/openai/gpt-4.1` through `https://inference-api.nvidia.com/v1` |
| Tokenizer / template | Provider managed; no local override |
| Policy protocol / parser | `nano_omni_toolcall` / Gym `parse_nano_omni_tool_calls`; model arguments unchanged |
| Reasoning parser | None; converter `uses_reasoning_parser=False` |
| History transport | Existing Azure tool acknowledgements; screenshots remain user observations |
| Sampling / output limit | Temperature 0, 1024 output tokens; top-p, seed and stop not sent |
| Agent limits | 5 steps, 3 recent screenshots, 0 parse retries |
| Browser | Playwright 1.55.0 / matching Chromium 1187; headless, 640 × 480, device scale factor 1 |
| Runtime | Python 3.13.14; Pillow 12.3.0; NumPy 2.5.4; scikit-image 0.26.0; httpx 0.28.1; Pydantic 2.14.0; OpenAI 2.45.0 |
| Judge / GPUs | No judge call for these exact-rule scenarios; no GPU allocation |

| Scenario | Actual observed comment | Reward | Masked | Model turns | Environment steps |
| --- | --- | ---: | --- | ---: | ---: |
| Valid format | `It has 3 $%@` | 1.0 | false | 2 | 2 |
| Invalid retained suffix | `It has 3 $%@ unrelated` | 0.0 | false | 2 | 2 |

The raw policy requests/responses, native tool arguments, screenshots, observed values, configuration and verifier outcomes remain in the local run record. Attachment sharing is still awaiting the separate authorization already requested; no denied historical configuration, original remote frame or raw historical rollout was transferred. The synthetic smoke does not establish a corrected-head 908-task score or reference parity. Previously documented PIC/reference/protocol, full-run health/configuration, observability and original-frame gaps remain open.

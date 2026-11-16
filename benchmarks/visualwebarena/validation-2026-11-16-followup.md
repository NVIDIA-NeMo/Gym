# VisualWebArena second review follow-up — 2026-11-16

This addresses the four findings and two smaller follow-ups in [review 5917644521](https://github.com/NVIDIA-NeMo/Gym/pull/2918#pullrequestreview-5917644521), which reviewed `33b1924bfea1f10416a6faaa60e649431c91f440`. The changes correct scoring and complete the benchmark declaration. They do not establish a new full-population score or reference parity.

## Fixes and acceptance evidence

| Review item | Change | Evidence |
| --- | --- | --- |
| [Combined predicates, task 622](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4583933100) | Apply every independently supplied predicate kind with AND; keep OR inside each alternative group. Snapshot exclusions also apply to the entire final target, including old content. | Canonical runtime 622/source `visualwebarena-624` and runtime 623/source `visualwebarena-625`; every forbidden subscription, missing-include and success controls, both mapping insertion orders, sync/async/public entrypoints; planned final-target exclusions too. |
| [First-root image capture](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4583933110) | Both snapshot collectors enumerate every matching root and descendant, deduplicate overlapping roots/URLs and preserve the wishlist/Reddit selector fallbacks. | Actual HTTP/Chromium old-first/new-second captures; matching/nonmatching additions, unchanged/deletion/reorder controls, direct images, wishlist fallback and nested roots; native sync/async/public scoring. |
| [Explicit empty image delta](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4583933120) | Distinguish an absent `delta_image_urls` field from an explicitly empty list. Deletion/reorder cannot reuse an old matching image. Retain compatibility with older records lacking the field. | Real snapshot producer → validated context → native sync/async/public scorer; old red match retained while blue is deleted/reordered scores zero. |
| [Missing manifest](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4583933128) | Add the typed manifest with identity, license, binary reward, canonical test split, composition mirror, `prompt_source: agent` and `experimental: true`. | `gym eval prepare --benchmark visualwebarena` verified the pinned 908 rows and referenced local images; `gym env validate visualwebarena` resolves the declared components and validates all 908 rows with the same source-root setting. |
| Semantic freshness follow-up | Remove old/moved token occurrences while retaining the actual new fragments for semantic scoring. A new paraphrase can satisfy a planned mutation even when the baseline already matches. | Literal/no-op/reorder/escaped-baseline controls plus new semantic matches for both `must_include` and `fuzzy_match`; real semantic judge accepted `successfully persisted` as a new saved confirmation. |
| Baseline-count correction | Report parsed baseline records separately from raw text occurrences, retaining all original bytes and metadata. | Parsed main → feature records: **1280 → 1282**. Raw `hashed_secret` occurrences: **1332 → 1334**, because main already contains duplicate JSON object keys. Exactly two source-lock records are added; removing that block reproduces main byte-for-byte. |

The [complete 908-row inventory](validation-2026-11-16-population-inventory.csv) preserves source/runtime identity, evaluator types, predicate groups, planned target/query counts and every image selector. It identifies both combined-predicate tasks (622 and 623), nine tasks with planned image queries, and three fuzzy-DOM tasks (535, 537, 538) with zero planned fuzzy targets. Synthetic planned fuzzy tests therefore demonstrate the helper contract; they do not prove an observed 908-score change.

The [complete new-case comparison](validation-2026-11-16-followup-cases.csv) retains each case, both verdicts, both measured durations, both pytest class names and original failure type/message. Complete original JUnit XML, failure tracebacks and command outputs remain in workspace run `2026-11-16-pr2918-review-followup-r2` and the internal review bundle. The previous [386-case comparison](validation-2026-11-16-cases.csv) and [first fix report](validation-2026-11-16.md) remain intact.

## Validation

| Check | Result |
| --- | --- |
| New cases on exact reviewed source, with asserted scorer/snapshot/adapter import origins | 175 cases: 122 failing counterexamples, 53 passing controls |
| New cases after fixes | 175 passed |
| New cases plus previous 386 regressions | 561 passed; old source: 122 failed, 439 passed |
| Full browser component suite and shipped example-data validation | 962 passed, zero skipped; examples validated |
| Shared web/environment-manifest/catalog/onboarding/agent/preparation/API/Fern suites | 2784 passed, zero skipped |
| Managed dependency locks | 304 checked, zero failed |
| All-files hooks | Passed |
| Named benchmark preparation and validation | 908 rows; image roots and composition aligned |

## Representative model and native-judge smoke

These are four synthetic interoperability cases using actual `WebAgent`, `ChatCompletionsConverter`, native tool parsing, Chromium snapshots, native VisualWebArena scoring, two real VQA calls and one real semantic-judge call. The in-process fixture transport handles browser actions and serves deterministic PNG bytes at the image HTTP-client boundary. The separate regression suite tests actual local HTTP image fetching. This smoke does not exercise the headed PyAutoGUI driver or maintained site deployment.

| Configuration | Actual value |
| --- | --- |
| Policy and judge endpoint/model | NVIDIA inference API; `us/azure/openai/gpt-4.1`; all 11 responses report this model |
| Tokenizer / chat template | Provider managed; no local override or available local hash |
| Policy protocol | `nano_omni_toolcall` |
| Native tools | `navigate`, `computer`, `tabs_create`, `tabs_focus`, `terminate` |
| Executed tools / argument handling | `computer`, `terminate`; model arguments used unchanged |
| Tool parser | Gym `parse_nano_omni_tool_calls` |
| Reasoning parser | None; converter `uses_reasoning_parser=False` |
| Service history bridge | Explicit Azure tool acknowledgements; subsequent browser screenshots remain user observations |
| Temperature / top-p / seed / stop | 0 / not sent / not sent / not sent |
| Output cap / steps / recent screenshots / parse retries | 1024 tokens / 5 / 3 / 0 |
| Browser | Playwright 1.55.0, matching Chromium 1187, headless, 640 × 480, device scale factor 1 |
| Runtime | Python 3.13.14; Pillow 12.3.0; NumPy 2.5.4; scikit-image 0.26.0; httpx 0.28.1; Pydantic 2.14.0; OpenAI SDK 2.45.0 |
| Credential handling | Credential remains on the authenticated login node; the local preflight uses an opaque remote-credential reference and the transport authenticates in place |
| GPU allocation / formal benchmark requests | None / none |

| Scenario | Before → after | Policy calls | Judge calls | Model turns | Environment steps | Reward | Masked |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Image addition | Old blue first → blue first and new red second | 2 | 1 | 2 | 2 | 1.0 | false |
| Deletion only | Old red and blue → old red retained | 2 | 0 | 2 | 2 | 0.0 | false |
| New nonmatch | Old blue → blue and new green | 2 | 1 | 2 | 2 | 0.0 | false |
| Semantic addition | Existing `saved` → `saved` plus `successfully persisted` | 2 | 1 | 2 | 2 | 1.0 | false |

All eight policy requests, three judge requests, four rollout rows, original screenshots, original before/after captures, actions, raw judge responses and verifier decisions are retained. The post-click image and semantic screenshots were visually inspected. Every scenario has its actual `resolved-config.yaml`; a root config includes all four case configurations. These are execution configs for this smoke, not retrospective replacements for a historical 908-task config.

## Native health and observability accounting

The unmodified command `gym eval health-check <smoke-run> --workers 1 --json` produced a real `quality_summary.json`. It reports **0 healthy / 0 unhealthy / 4 unobserved**, with no ignored checks. It observes four scored rows and no typed timeout, but has no expected-rollout inventory or standard invocation captures. Native failed/missing/not-started counts are consequently unavailable, not assumed zero. The standalone retained HTTP transcripts independently contain all 11 successful policy/judge calls; all four agent rows have `mask_sample=false` and zero action-execution failures.

| Native health check | Evaluated | Unobserved | Ignored | Reported issues |
| --- | ---: | ---: | ---: | ---: |
| `check_execution_error` | 4 | 0 | 0 | 0 |
| `record_unreadable` | 4 | 0 | 0 | 0 |
| `rollout_duplicate_identity` | 4 | 0 | 0 | 0 |
| `rollout_missing_agent_turns` | 0 | 4 | 0 | 0 |
| `rollout_ended_on_failed_model_call` | 0 | 4 | 0 | 0 |
| `agent_turn_hollow` | 0 | 4 | 0 | 0 |
| `model_call_zero_completion_tokens` | 0 | 4 | 0 | 0 |
| `model_call_missing_token_counts` | 0 | 4 | 0 | 0 |
| `trajectory_capture_mismatch` | 0 | 4 | 0 | 0 |
| `rollout_token_count_mismatch` | 0 | 4 | 0 | 0 |
| `model_call_runaway_generation` | 0 | 4 | 0 | 0 |
| `rollout_tool_call_failed` | 0 | 4 | 0 | 0 |
| `task_consistently_unhealthy` | 0 | 4 | 0 | 0 |
| `task_no_successful_model_calls` | 0 | 4 | 0 | 0 |
| `run_excessive_failures` | 0 | 1 | 0 | 0 |

The informational `inspect_harness_conformance.py --bundle <smoke-run>/rollouts.jsonl --output <dir>` result is `not_fulfilled`, with a completed checker and all four records retained. `WebAgent` is absent from `HARNESSES` at this source revision. No P0 pass or registration is claimed; the benchmark PIC must disposition the observability gap.

## Onboarding status and remaining decisions

The supplied onboarding guide was read at Config Management RFC revision `581654131aa142d70a77b4c5fc2f704c1f6894c6`. Required checks, proposed thresholds and later release stages remain distinct.

| Check | Evidence and status |
| --- | --- |
| EC-1 | Native smoke health counts and every original outcome are inspectable. Historical full-population native health artifacts were not found in the run root or nine checked worker locations; selected-valid summary counts do not replace them. The 0.1% value is proposed in the supplied guide and is not asserted as a newly adopted threshold. |
| EC-2 | Informational for existing non-P0 `WebAgent`; checker result and missing invocation evidence are disclosed. PIC disposition remains open. |
| EC-3 | Open: agree the current frontier model/reference, tolerance and comparable data/site-state/prompt/action/judge/sampling/retry tuple, then obtain evidence applicable to these repairs. The synthetic smoke establishes neither maintained-population parity nor a new score. Existing historical artifacts were checked before commissioning the small image/semantic smoke. |
| EC-4 | Complete actual smoke configs and current benchmark composition are inspectable. Nine original historical worker input configs were located; they are not saved execution-resolved configs. Their contents were not transferred because automatic approval rejected that private-data transfer. The historical execution-resolved-config gap remains open. |
| EC-5 | One existing-harness pairing; no second harness is introduced or required by this patch. |
| EC-6 | N/A for this non-P0 pairing. The informational evidence checker is not a scripted P0 qualification report. |
| EC-7 / config handoff | No VisualWebArena binding was found among all 141 main benchmark bindings checked, and no open MR matched the VisualWebArena search. A separate draft binding/leaf and its resolve/dry-run handoff need Frontier Eval/PIC coordination. Early draft pins may use the full PR SHA; main/tag is required before leaving draft. Proposed Evergreen checks and later SWQA certification/release are not represented as completed or universal initial-merge gates. |

The prior 231/908 and 219/908 results retain their recorded scorer revisions and tuned Nano Omni `iter_0004622` checkpoint. No new maintained-population run, reference agreement, complete original-frame certification, review approval, CI trust authorization or merge is claimed. Automation-owner review remains required for the two existing `.secrets.baseline` source-lock entries.

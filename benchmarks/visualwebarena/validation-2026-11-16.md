# VisualWebArena review fixes — 2026-11-16

The seven blocking findings and four risks in [review 5900114752](https://github.com/NVIDIA-NeMo/Gym/pull/2918#pullrequestreview-5900114752) are addressed in this revision. The VisualWebArena commits were rebased onto main `b8fe13f676849b05d875ca66c8446236710108e6` before the fixes. The complete [386-case comparison](validation-2026-11-16-cases.csv) preserves every case, both verdicts, both measured durations, both pytest class names, and each original failure type/message. Complete original JUnit XML, failure tracebacks, command logs, patches, model requests/responses and synthetic screenshots are retained in the workspace run record `2026-11-16-pr2918-review-fixes`; no rows or raw reports were discarded.

## Findings and resulting behavior

| Review discussion | Finding | Implemented behavior | Regression coverage |
| --- | --- | --- | --- |
| [4570318665](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318665) | Planned empty DOM, order and review deltas fell back to existing live state | `None` denotes an unplanned target/adapter/SKU; a planned empty list scores zero and cannot fall through. Review text still uses `detail`. | Unchanged, new matching/nonmatching, missing review fields, unrelated targets/SKUs, sync/async/public paths |
| [4570318672](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318672) | A changed line could repeat an existing match | Compare baseline and full content and require fresh matching content; count increases allow a newly added copy. Preserve VisualWebArena's semantic judge fallbacks. | Existing match plus unrelated edit, escaped baseline, unchanged content, added duplicate, empty-to-match, semantic fallback control |
| [4570318678](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318678) | Failed DOM/helper reads became fabricated empty evidence | Propagate selector, preparation, browser and helper failures. Optional missing nodes remain valid empty observations. Harden both Python and JavaScript helper boundaries. | Empty exact-match vs disconnect; invalid selector/preparation; nine helper families; actual Chromium empty-page reads |
| [4570318683](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318683) | Per-tab errors became valid misses | Retain candidate errors and fail an inconclusive target unless another tab proves success. | Error alone, error/miss in both orders, all misses, error/match in both orders; browser and judge errors |
| [4570318692](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318692) | Missing image evidence became a valid zero | Preserve image/browser/transport/judge errors unless an available candidate proves success. A candidate HTTP 404/410 or successfully fetched corrupt image remains a task miss. Missing/corrupt reference images remain evaluation errors. Planned image captures are authoritative. | Live and captured URLs; timeout, browser disconnect, HTTP 401/503, 404/410, corrupt data, empty selection, real matching/nonmatching PNGs, match plus unavailable candidate in either order; sync/async/public paths |
| [4570318697](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318697) | DOM fuzzy alternatives used AND | Combine `|OR|` alternatives with the maximum score; independent targets and evaluator types retain AND. Preserve errors unless an alternative proves success. | `[1,0]`, `[0,1]`, `[0,0]`, single alternatives, error/match, error/miss, independent failing target/evaluator |
| [4570318703](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318703) | N/A fallback overwrote an earlier failed predicate | Compute the unachievable-answer predicate separately, then multiply it into the accumulated score. | Both predicate orders, failed/satisfied exclusions, literal N/A without a judge call, empty/incorrect answers |
| [4570318708](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318708) | Lambda numeric zero was treated as absence | Preserve numeric values including zero; empty/None/false selections remain misses. | Zero passing/failing inequalities, positive number, empty list and None; sync/async/public paths |
| [4570318711](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318711) | DOM judge fallbacks were absent from preflight | Require credentials for VisualWebArena DOM `must_include`/`fuzzy_match` before capture/policy work. Classic WebArena remains rule-only for DOM matching. | Both benchmarks, five DOM predicates, credential rejection before any capture |
| [4570318716](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318716) | Initial environment generation was missing from setup | Document dataset preparation followed by the preparation script that writes private `env.yaml`; retain source-root overrides and explicit `--force-env`. | Private mode, repeated generation preserves customized content, intentional forced replacement restores generated content |
| [4570318721](https://github.com/NVIDIA-NeMo/Gym/pull/2918#discussion_r4570318721) | Imported scorer lacked required attribution | Add the original revision/path/license header, VisualWebArena file mapping and vendored-component entry. Independently verify the pinned original source hash. Correct historical AST-audit wording. | Original file SHA-256 matches `PROVENANCE.md`; original header has no separate copyright notice; full Apache-2.0 root license remains in `SOURCE_LICENSE` |

The additional review note is addressed by adding VisualWebArena's 908-task row, reference-image-root requirement and evaluator description to the Fern environment catalog, alongside the existing 812-task WebArena row.

## Validation

| Check | Result |
| --- | --- |
| Final regression matrix on reviewed head `52444c112b5e9eadcaa9e4a19bca478be79a435c` | 386 cases: 174 failed, 212 passed, zero errors/skips; scorer and adapter import paths asserted before pytest startup |
| Same regression matrix after fixes | 386 passed, zero failures/errors/skips |
| Official browser component suite | 787 passed, zero failures/errors/skips; includes real Chromium tests and all 386 new cases |
| Shipped browser example data | Validated by `gym env test +should_validate_data=true` |
| Shared Web, web agent, both benchmark preparations, task data, public API/import contracts and Fern links | 2101 passed, zero skips |
| Managed component locks | 304 checked, zero failed; no lock updates |
| All-files pre-commit | Passed |
| Fern check | Zero errors; one warning: redirect checking requires Fern authentication, which was not configured |
| Rebase secret-baseline preservation | Parsed records: 1280 on main → 1282 with exactly two VisualWebArena source-lock additions. Raw literal `hashed_secret` occurrences: 1332 → 1334 because main already contains duplicate JSON object keys. Removing only the VWA block reproduces every original main byte, including metadata and duplicate keys. |

The negative-control run required explicit source imports before pytest startup: the component's editable-install/plugin setup initially selected the fixed source. That incorrectly green attempt is retained but excluded from the comparison. The subsequent runs assert the resolved old scorer path and reproduce the defects.

## Separate real-model smoke

This smoke exercises the real Gym `WebAgent`, `ChatCompletionsConverter`, native tool parser, headless Chromium fixture, before/after captures and public `WebArenaTaskEvaluator`. The synthetic fixture uses a small in-process transport for click/wait/terminate actions. It does not exercise the headed PyAutoGUI driver, the maintained site deployment or the Nano Omni checkpoint. Every policy tool argument was used unchanged. The successful post-click screenshot was inspected and showed `saved`.

| Configuration | Value |
| --- | --- |
| Policy service/model | NVIDIA inference API; `us/azure/openai/gpt-4.1`, also reported by the response |
| Tokenizer / chat template | Provider managed; no local tokenizer/template override or local hash is available |
| Policy protocol / tools | `nano_omni_toolcall`; Gym's five native tools |
| Tool parser | Gym `parse_nano_omni_tool_calls`; no argument repair |
| Reasoning parser | None; `ChatCompletionsConverter(uses_reasoning_parser=False)` |
| Service history bridge | Explicit tool acknowledgements required by Azure between assistant tool calls and the next browser observation; screenshots remain user observations |
| Temperature | 0 |
| Top-p / seed / stop | Not sent; provider defaults |
| Output limit / step limit / recent images | 1024 output tokens / 5 steps / 3 recent screenshots |
| Parse retries | 0 |
| Browser | Playwright 1.55.0; matching Chromium 1187, headless; 640 × 480, device scale factor 1 |
| Runtime | Python 3.13.14; Pillow 12.3.0; NumPy 2.5.4; scikit-image 0.26.0; httpx 0.28.1; Pydantic 2.14.0; OpenAI SDK 2.45.0 |
| Judge during smoke | DOM exact-match rules only; no judge request. The same existing API credential was used in place for the separate policy requests. |
| GPU allocation | None |

| Synthetic scenario | Model turns | Environment steps | Reward | Masked | Failure kind |
| --- | ---: | ---: | ---: | --- | --- |
| New save: blank status, actual model click, then terminate | 2 | 2 | 1.0 | false | null |
| Planned no-op: pre-existing saved status, model terminates without changing it | 1 | 1 | 0.0 | false | null |
| Closed live browser before final DOM evaluation; unplanned final-state path | 2 | 2 | 0.0 | true | `web:evaluator_infrastructure_error` |

Earlier fixture qualifications are retained separately: implicit browser-context creation was unsuitable for auxiliary snapshot tabs; the Responses adapter needed an explicit model identity; Azure rejected Nano-style unacknowledged tool history; and closing a planned capture context tested snapshot creation rather than the intended final-DOM path. Those attempts are not counted as passing smoke cases. The final smoke uses an explicit browser context, persistent synthetic site state, the documented service bridge and a distinct unplanned final-DOM fault case.

## Reproduction commands

From a locked development checkout with the matching Chromium bundle installed:

```bash
gym env test +entrypoint=resources_servers/webarena_browser +should_validate_data=true

python -m pytest tests/unit_tests/web tests/unit_tests/test_public_api.py tests/unit_tests/test_public_api_facades.py tests/unit_tests/test_import_linter_coverage.py tests/unit_tests/data/test_task_data.py tests/unit_tests/tooling/test_migrate_imports.py tests/unit_tests/test_fern_docs_links.py benchmarks/visualwebarena/tests benchmarks/webarena/tests responses_api_agents/web_agent/tests -q --no-cov -o log_cli=false

python scripts/component_locks.py --jobs 4
pre-commit run --all-files
cd fern
npm run check
```

The raw run record retains the exact expanded paths, cache/Chromium environment variables, all invocations including failed qualifications, both complete patches, JUnit XML and the standalone smoke runner. Provider credentials and large captures are excluded from Git.

## Benchmark-result boundary

The existing Nano Omni `iter_0004622` checkpoint and full benchmark configuration are unchanged and remain fully documented in [the 2026-11-12 report](validation-2026-11-12.md). Its historical **231/908 (25.44052863436123%)** result measures scorer source `f8031c55e638491f4d0b39890714d553ed1fc09e`, not these fixes. The earlier **219/908 (24.118942731277532%)** result also remains historical. Neither result is relabeled or certified as a corrected-scorer score. A new comparable 908-task score requires a separately qualified complete run; no maintained task was rerun for this code-fix smoke.

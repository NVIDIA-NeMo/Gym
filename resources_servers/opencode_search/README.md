# search

Verifier for judge-graded QA (BrowseComp-style): the answer is free text, so grading is an LLM
judge call rather than test execution.

Two differences from `resources_servers/swebench`, which this server is otherwise shaped like:

* `seed_session` creates ONE general-purpose sandbox from a fixed `sandbox_image`. These tasks ship
  no repo, so there is no per-instance image to derive, no conda env to activate and no git history
  to scrub.
* `verify` creates no eval sandbox and runs no tests. It takes the agent's last assistant message
  and grades it with the simple-evals BrowseComp grader.

## Grading semantics

Reward is 1.0 for a `correct: yes` verdict, else 0.0. Three cases are deliberately distinguished:

| case | outcome |
| --- | --- |
| agent produced no answer | reward 0.0, no judge call spent |
| judge answered but the verdict could not be parsed | retried; a parse that never succeeds is a judge failure |
| judge never answered (endpoint down) | `JudgeError` → `judge_failsafe` → failures sidecar, not a fabricated zero |

The verdict parser takes the LAST `correct: yes|no` match: reasoning judges echo the rubric inside
their think block, and a first-match parse grades the echo. A reply that echoes the template instead
of grading counts as unparsed.

## Config

Needs `sandbox_image`, `sandbox_provider`/`sandbox_config`, and a `judge_model_server` pointing at a
model-server block. See `configs/search.yaml`; wired up in `benchmarks/search/opencode.yaml`.

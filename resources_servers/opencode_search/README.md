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
| judge answered but the verdict could not be parsed | retried; a parse that never succeeds is masked (`mask_sample`, `failure_kind: judge_unparseable`), not a measured zero |
| judge never answered (endpoint down) | `JudgeError` → `judge_failsafe` → failures sidecar, not a fabricated zero |

The verdict parser takes the LAST `correct: yes|no` match: reasoning judges echo the rubric inside
their think block, and a first-match parse grades the echo. A reply that echoes the template instead
of grading counts as unparsed.

Every verify response carries `evaluation_completed`: true when grading reached a result (a verdict, or
an empty answer), false when the judge gave no verdict or never answered and when the session was
lost. A rollout that ended on its budget keeps its judged reward and is masked, with the ending's
`failure_kind`; the ending (`max_iteration`, `context_window`, `agent_timed_out`) arrives as
`opencode_terminal_reason` in the verify body, which `opencode_sandboxed_agent` sends. Gym's aggregate
metrics leave masked rows out of the quality scores; the reward stays on each row.

## Sessions

The server implements the session state hooks, so the base class serves `/seed_session` and
`/close_session`. A session holds one sandbox, started from `sandbox_image` and `sandbox_config`;
grading never reads it.

* **Episode seed (Environment Server).** `/seed_session` takes a `ResourcesSeedSessionRequest` and keys
  the session by its `resources_session_id`. With `sandbox_workdir` set, the seed starts the sandbox,
  creates that directory in it, and returns `sandbox_access` (an `operate` descriptor and the
  workdir) for a session harness to borrow; only this server stops the sandbox. Without
  `sandbox_workdir` the seed is refused with 422. A seed whose `task_data` lacks `question` or
  `ground_truth` is refused with 422 and starts nothing; an ID bound to another episode or task, or
  one already closed, gets 409; repeating the seed returns the same sandbox.
* **Verify.** `/verify` stops the sandbox before it calls the judge. A session is verified once: an
  identical retry replays the verdict, and a retry with another body gets 409. After an attempt that
  reached no verdict, an identical retry is graded again. A verify after the session's close is
  masked (`failure_kind: session_lost`). A verify with no seed, such as `gym eval reverify`, is
  graded, since the grade reads only the request.
* **Close.** `/close_session` stops the sandbox by ID, with or without the session cookie, and fences
  the ID: a seed for it gets 409 for `session_close_retry_window_seconds`. A close whose stop fails
  raises and keeps the sandbox, so a retried close stops it. A session that is never closed keeps its
  sandbox until the server shuts down, which stops every sandbox still running, or until the
  provider reclaims it after `sandbox_config.ttl_s`; keep `ttl_s` above the episode deadline plus
  grading and cleanup.
* **Agent-run seed, kept for compatibility.** An agent's own `/run` seeds with the dataset row and its
  session cookie and gets `{"sandbox_handle": ...}` back, the sandbox it attaches to. That sandbox is
  started exactly as before episode sessions existed (the image's working directory, no task
  label), and `/verify` stops it, since this path sends no close.

Supported harnesses: `opencode_sandboxed_agent` (`allowed_agents`), on the agent-run seed. It serves
no agent sessions, so its configs (`benchmarks/browsecomp/configs/browsecomp_opencode.yaml` and
`configs/opencode_search_example.yaml`) declare a `legacy_agent` relay in front of it. The episode seed
lends its sandbox to any session harness that accepts `sandbox_access`; the server serves no tools of
its own, so the harness brings its tools. No shipped config pairs a session harness with this server
yet. With lending on, the server runs with one worker (`num_workers: 1`): several workers are not
tested with lent sandboxes.

The web-search cache proxy (`POST /mcp`, registered only when the cache is configured in
`configs/opencode_search.yaml`) is not part of a session: it reads no session cookie and keeps no
session state. It owns the `/mcp` route, so this server cannot also set `expose_tools_over_mcp`.

## Config

Needs `sandbox_image`, `sandbox_provider`/`sandbox_config`, and a `judge_model_server` pointing at a
model-server block. `sandbox_workdir` (default null) turns on lending to episode seeds; it must be an
absolute directory other than `/` and `/tmp`. See `configs/opencode_search.yaml`; wired up in
`benchmarks/browsecomp/configs/browsecomp_opencode.yaml`.

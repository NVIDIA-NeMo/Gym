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
| judge answered but the verdict could not be parsed | retried; a parse that never succeeds is masked (`mask_sample`, `failure_kind: judge_unparseable` unless the run ended on its budget), not a measured zero |
| judge never answered (endpoint down) | `JudgeError` → `judge_failsafe` → failures sidecar, not a fabricated zero |

The verdict parser takes the LAST `correct: yes|no` match: reasoning judges echo the rubric inside
their think block, and a first-match parse grades the echo. A reply that echoes the template instead
of grading counts as unparsed.

Both judge failures count toward `judge_max_consecutive_failures`: once that many rollouts in a row
get no verdict, verify answers 500 for that rollout and for each later one without a verdict, until a
verdict resets the count, rather than masking every remaining row. On the relay pairing that 500 fails
the agent's `/run`: without `route_failures_to_sidecar` the first one ends the run, and with it each
becomes a failures-sidecar row. On an episode session each becomes a non-terminal failure row, and
collection continues. Watch the failure rows for judge failures.

`inconclusive_verification_retries`, which `AGENTS.md` asks for when verification starts sandboxes,
does not apply: verify starts no sandbox of its own, and an episode's plugin log exists only in the
sandbox the agent ran in, which cannot be replaced. The judge's own attempts are the grading retry, and
a plugin log that cannot be read answers 500, so the episode fails ungraded and runs again on resume.
None of these outcomes is recorded as a measured 0.

Every verify response carries `evaluation_completed`: true when grading reached a result (a verdict, or
an empty answer), false when the judge gave no verdict or never answered and when the session was
lost. A rollout that ended on its budget keeps its judged reward and is masked, with the ending's
`failure_kind`; when the judge gave it no verdict, it keeps that kind, with `judge_parsed` and
`evaluation_completed` false. Verify masks the endings `max_iteration`, `context_window` and
`agent_timed_out` when the verify body's `opencode_terminal_reason` names one. `opencode_sandboxed_agent`
sends only `max_iteration` or `context_window`, read from the turn-budget plugin's lines in its run
output; on an episode session with `plugin_log_path` set, verify reads them from the plugin log
instead (see Sessions). Gym's aggregate metrics leave masked rows out of the quality scores; the
reward stays on each row.

A verify that grades a saved answer again (`gym eval reverify`, which `REVERIFY_MODE: STATELESS` allows,
or recovery re-judging a row whose judge failed) cannot mask a budget ending, on either pairing: its
body carries no `opencode_terminal_reason`, and an episode's ending was only in its sandbox's plugin
log. A row that ended on its budget comes back from such a verify graded and unmasked.

## Sessions

The server implements the session state hooks, so the base class serves `/seed_session` and
`/close_session`. A session holds one sandbox, started from `sandbox_image` and `sandbox_config`.
Grading reads nothing from it but the plugin log of an episode session, when `plugin_log_path` is set.

* **Episode seed (Environment Server).** `/seed_session` takes a `ResourcesSeedSessionRequest` and keys
  the session by its `resources_session_id`. With `sandbox_workdir` set, the seed starts the sandbox
  with the files of `episode_sandbox_files` and the environment of `episode_sandbox_env`, creates that
  directory in it, and returns `sandbox_access` (an `operate` descriptor and the workdir) for a session
  harness to borrow; only this server stops the sandbox. Without `sandbox_workdir` the seed is refused
  with 422. A seed whose `task_data` lacks `question` or `ground_truth` is refused with 422 and starts
  nothing; an ID bound to another episode or task, or one already closed, gets 409; repeating the seed
  returns the same sandbox.
* **Verify.** `/verify` stops the sandbox before it calls the judge. A session is verified once: an
  identical retry replays the verdict, and a retry with another body gets 409. After an attempt that
  reached no verdict, an identical retry is graded again. A verify after the session's close is
  masked (`failure_kind: session_lost`). A verify with no seed, such as `gym eval reverify`, is
  graded, since the grade reads only the request. With `plugin_log_path` set, an Environment
  Server's verify (it sends `x-ng-checkpoint-close-follows: 1`) for a session this server holds no
  sandbox or earlier verify for, as after a restart, is masked as `session_lost` instead: its plugin
  log went with the sandbox.
* **Plugin log.** With `plugin_log_path` set, an episode's verify first reads that file from the sandbox,
  base64-encoded so that no provider's output framing can split a line: the receipts plugin
  (`opencode_bundle/nemo_gym_receipts.js`) copies the OpenCode plugins' lines there.
  Verify takes the run's ending from the turn-budget plugin's lines, as `opencode_sandboxed_agent` reads
  its run output, and returns the log with it (`opencode_terminal_reason`, `opencode_plugins_loaded`,
  `opencode_plugin_log`). A log without the sentinel plugin's line, or without a line of
  `required_plugin_markers`, is refused with 422 and not graded. A log that cannot be read answers 500,
  which an Environment Server treats as retryable. Without `plugin_log_path`, verify reads and refuses
  nothing, so a harness that brings no OpenCode plugins is graded on its answer alone.
* **Close.** `/close_session` stops the sandbox by ID, with or without the session cookie, and fences
  the ID: a seed for it gets 409 for `session_close_retry_window_seconds`. A close whose stop fails
  raises and keeps the sandbox, so a retried close stops it. A session that is never closed keeps its
  sandbox until the server shuts down, which stops every sandbox still running, or until the
  provider reclaims it after `sandbox_config.ttl_s`. Keep `ttl_s` above the episode deadline plus the
  cleanup after it (the agent's close, then this server's, each within `cleanup_timeout_seconds`):
  verify stops the sandbox before it calls the judge, so grading needs no margin.
* **Agent-run seed, kept for compatibility.** An agent's own `/run` seeds with the dataset row and its
  session cookie and gets `{"sandbox_handle": ...}` back, the sandbox it attaches to. That sandbox is
  started exactly as before episode sessions existed (the image's working directory, no task
  label), and `/verify` stops it, since this path sends no close.

Supported harnesses, each in the `allowed_agents` of the config that pairs it:

* `opencode_agent`, on the episode seed. `benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml`
  runs it through `single_agent_turn_legacy` and stages the OpenCode plugins that the relay agent uploads.
  That config admits no other harness, on purpose: the model's tools are OpenCode's own, shaped by those
  plugins, and this server serves no tools, so another harness would run a different benchmark.
* `opencode_sandboxed_agent`, on the agent-run seed, as `configs/opencode_search.yaml` declares. It serves
  no agent sessions, so its configs (`benchmarks/browsecomp/configs/browsecomp_opencode.yaml` and
  `configs/opencode_search_example.yaml`) declare a `legacy_agent` relay in front of it.

The episode seed lends its sandbox to any session harness that accepts `sandbox_access`; the harness brings
its tools. With lending on, the server refuses `num_workers` above 1: lent sandboxes live in one worker's
memory, and several workers are not tested with them.

### What differs from the relay

The session pairing repeats the relay pairing's settings; `benchmarks/browsecomp/README.md` lists what the
model sees differently. On this server's side:

* **Plugins and their settings.** This server stages them in the lent sandbox (`episode_sandbox_files`,
  `episode_sandbox_env`). On the relay, `opencode_sandboxed_agent` uploads them and passes their settings
  to OpenCode.
* **How a run ended.** Verify reads it from the plugin log. On the relay, the agent reads its run output
  and sends the ending in the verify body. Neither agent sends an `agent_timed_out` ending, so neither
  pairing masks a run that ran out of time: both grade what it produced.
* **A run whose plugins did not load.** A log without the sentinel's line or a required line is refused
  with 422, which an Environment Server does not retry, and a log that cannot be read answers 500. The
  relay agent checks only the sentinel's line, and refuses a run whose output lacks it with 500.
* **Working directory.** The lent sandbox's is `sandbox_workdir`, which the seed creates; the agent-run
  sandbox keeps the image's.
* **Close.** An episode session ends with `/close_session`, which fences the ID and releases the sandbox if
  verify did not stop it, and a verify after it is masked as `session_lost`. The relay sends no close.

The web-search cache proxy (`POST /mcp`, registered only when the cache is configured in
`configs/opencode_search.yaml`) is not part of a session: it reads no session cookie and keeps no
session state. It owns the `/mcp` route, so this server cannot also set `expose_tools_over_mcp`.

## Config

Needs `sandbox_image`, `sandbox_provider`/`sandbox_config`, and a `judge_model_server` pointing at a
model-server block. Episode sessions use these settings, each off by default:

* `sandbox_workdir` turns on lending to episode seeds. It must be an absolute directory other than `/`
  and `/tmp`; the seed creates it.
* `episode_sandbox_files` maps absolute sandbox paths to files relative to the Gym root, read at startup
  and staged in each lent sandbox. Stage under `/tmp`, which every image has: a provider need not create
  parent directories.
* `episode_sandbox_env` is the lent sandbox's environment, over `sandbox_config.env`; an empty value sets
  nothing. Supply secrets from an overlay. With `exa_cache_root` set, `NEMO_GYM_EXA_BASE_URL` defaults to
  this server's `/mcp` proxy, so the server needs an address the sandbox reaches; on loopback it refuses
  to start.
* `plugin_log_path` is the plugin log verify reads, and also sets `NEMO_GYM_PLUGIN_LOG` in the lent
  sandbox. `required_plugin_markers` lists the lines the log must hold besides the sentinel's.

See `configs/opencode_search.yaml`; wired up in `benchmarks/browsecomp/configs/browsecomp_opencode.yaml`
(relay) and `benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml` (episode sessions).

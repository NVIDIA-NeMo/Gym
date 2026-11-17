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

Reward is 1.0 for a `correct: yes` verdict, else 0.0. Four cases are deliberately distinguished:

| case | outcome |
| --- | --- |
| agent produced no answer | reward 0.0, no judge call spent |
| judge answered but the verdict could not be parsed | retried; a parse that never succeeds scores 0.0, unmasked, as a 0 counted against the evaluation rather than the model (`failure_kind: judge_unparseable`), or is masked with `unparseable_verdict_policy: mask`, which training sets (see [Training](#training)); a run that ended on its budget keeps that ending's kind and mask either way |
| judge never answered (endpoint down) | `JudgeError` → `judge_failsafe` → failures sidecar, not a fabricated zero |
| an episode's plugin log is missing, unsafe to read or lacks a required line | reward 0.0, no judge call spent, counted as the model's 0 (see [Plugin log](#sessions)) |

The verdict parser takes the LAST `correct: yes|no` match: reasoning judges echo the rubric inside
their think block, and a first-match parse grades the echo. A reply whose extracted answer is exactly
the template's own line for it ("The final exact answer extracted from the [response]. Put the
extracted answer as 'None' ...") repeated the template instead of grading, and counts as unparsed,
unless the graded answer holds that line itself. An extracted answer that only quotes part of that
line, or adds to it, is graded on its verdict.

A verdict that never parses scores the row 0 by default (`unparseable_verdict_policy: zero`, for
evaluation): `reward` 0.0, `mask_sample` false, `evaluation_completed` and `judge_parsed` false,
`failure_kind: judge_unparseable`, a `failure_reason` that gives the attempts, and the applied outcome
`failure_outcome_action: count_as_zero` with `failure_outcome_attribution: evaluation`. Gym counts it in
the score, as a wrong answer was counted before, reports it under `judge_unparseable` in the coverage's
`counted_zero_kinds`, and counts it as an evaluation-side fault in the acceptance gate, so a judge that
often gives no verdict fails the gate rather than lowering the score unseen. Gym does not judge such a
row again; an identical retry of its verify asks the judge again.

Each of the `judge_max_attempts` attempts (4 by default) is one `call_judge`, which sends the request
up to three times on a timeout, a lost connection or a 408, 429 or 5xx reply, with at most 2 s and then
4 s of backoff. With the defaults (`judge_timeout_seconds`, 300 s for each request of the first attempt;
`judge_retry_timeout_seconds`, 120 s for each request of a retry; `judge_retry_delay_seconds`, 30 s
between attempts), a judge that never answers holds a rollout for
(3 × 300 + 6) + 3 × (3 × 120 + 6) + 3 × 30 = 2,094 s when each request is sent once. Gym's HTTP client
also sends a timed-out or refused request again until `global_aiohttp_internal_retry_budget_seconds`
(600 s by default) is spent, which stretches that to about two hours.

Both judge failures count toward `judge_max_consecutive_failures` (default 25, counted by each worker; 0
turns it off), under either `unparseable_verdict_policy`: once that many rollouts in a row are left
without a verdict, verify answers 500 rather than scoring 0 or masking every remaining row, and a verdict
starts the count again. Changed: a reply without a parseable verdict now counts too; before, only a judge
that never answered counted, and a reply without a verdict scored the row 0, as a wrong answer, with no
`failure_kind`. The server does not refuse seeds after a trip: each later rollout whose judge also fails
runs its agent and then answers 500, until a verdict resets the count. On the relay pairing that 500
fails the agent's `/run`: with `route_failures_to_sidecar` off (the default) the first one ends the run,
and with it on each becomes a failures-sidecar row. On an episode session each becomes a non-terminal
failure row, and collection continues. Watch the failure rows for judge failures.

`inconclusive_verification_retries`, which `AGENTS.md` asks for when verification starts sandboxes,
does not apply: verify starts no sandbox of its own, and an episode's plugin log exists only in the
sandbox the agent ran in, which cannot be replaced. The judge's own attempts are the grading retry, and
a plugin log that cannot be read answers 500, so the episode fails ungraded and runs again on resume. A
plugin log that was read, or found missing or unsafe to read, is a result about the run, and one that
fails the check scores 0 (see Sessions).

Every verify response carries `evaluation_completed`: true when grading reached a result (a verdict, or
an empty answer), false when the judge gave no verdict or never answered, when the session was lost and
when an episode was scored 0 for its plugin log. A rollout that ended on its budget keeps its judged
reward and is masked, with the ending's `failure_kind`; when the judge gave it no verdict, it keeps that
kind, with `judge_parsed` and `evaluation_completed` false. Verify masks the endings `max_iteration`,
`context_window` and `agent_timed_out` when the verify body's `opencode_terminal_reason` names one.
`opencode_sandboxed_agent` sends only `max_iteration` or `context_window`, read from the turn-budget
plugin's lines in its run output; on an episode session with `plugin_log_path` set, verify reads them
from the plugin log instead (see Sessions). Gym's aggregate metrics leave masked rows out of the quality
scores; the reward stays on each row. A budget ending is the model's outcome, so its own mask stays on
the row: it does not set `instance_config.mask_sample`, and a training run keeps its reward (see
[Training](#training)).

A verify that grades a saved answer again (`gym eval reverify`, which `REVERIFY_MODE: STATELESS` allows,
or recovery re-judging a row whose judge failed) cannot mask a budget ending, on either pairing: its
body carries no `opencode_terminal_reason`, and an episode's ending was only in its sandbox's plugin
log. A row that ended on its budget comes back from such a verify graded and unmasked. For the same
reason it cannot score an episode 0 for its plugin log: `gym eval reverify` grades such a row's saved
answer. Recovery never re-judges one.

Gym judges a masked `judge_unparseable` row (`unparseable_verdict_policy: mask`) again from its saved
answer, on either pairing, at the end of the collection and when the run resumes, and replaces it once
the judge gives a verdict. A row scored 0 under the default policy is settled, and no re-judge picks it
up, as none picked up the 0 such a row scored before.

### Training

Training keeps the relay pairing. A rollout the judge gave no verdict says nothing about the model, so
a training run masks it rather than train on a 0: set `unparseable_verdict_policy: mask` on this server.
In NeMo RL's environment config:

```yaml
env:
  nemo_gym:
    search_browsecomp_resources_server:
      resources_servers:
        opencode_search:
          unparseable_verdict_policy: mask
```

or `++search_browsecomp_resources_server.resources_servers.opencode_search.unparseable_verdict_policy=mask`
on a Gym rollout collection (both BrowseComp presets name the server instance
`search_browsecomp_resources_server`). Such a row then has `mask_sample` true,
`failure_kind: judge_unparseable`, a `failure_reason` that ends in "masked", and no applied outcome.
NeMo RL masks a sample's loss by its `instance_config.mask_sample`, not by the row's own `mask_sample`,
so a row whose reward is no outcome of the model also sets that field, over the request's other
`instance_config` fields, as `judge_failsafe` does for its rows: a judge without a verdict under this
policy, whatever the ending, and a lost session. A row whose judge never answered is masked the same way
by `judge_failsafe`. A run that leaves the default trains on a rollout without a verdict as a 0. A
rollout that ended on its budget (`max_iteration`, `context_window`, `agent_timed_out`), with a verdict
or without an answer, is the model's outcome: it is masked on its row only, which keeps it out of Gym's
quality metrics, and training keeps its reward. The judge's failures count toward
`judge_max_consecutive_failures` under either policy, so a dead judge still stops the run.

## Sessions

On the episode path, an Environment Server seeds this server, which starts the episode's sandbox and
lends it to the agent's session; the agent works in the sandbox; `/verify` reads the plugin log, copies
the session exports to this host when `episode_export_copy_dir` is set, stops the sandbox and grades the
answer; and `/close_session` fences the session. The server implements the session state hooks, so the
base class serves `/seed_session` and `/close_session`. A session holds one sandbox, started from
`sandbox_image` and `sandbox_config`. Grading reads nothing from it but the plugin log of an episode
session, when `plugin_log_path` is set. Sessions are not exported to checkpoints (`restart_only`): after
a crash an episode starts over from its input.

* **Episode seed (Environment Server).** `/seed_session` takes a `ResourcesSeedSessionRequest` and keys
  the session by its `resources_session_id`. With `sandbox_workdir` set, the seed starts the sandbox
  with the files of `episode_sandbox_files` and the environment of `episode_sandbox_env`, creates that
  directory in it, and returns `sandbox_access` (an `operate` descriptor and the workdir) for a session
  harness to borrow; only this server stops the sandbox. Without `sandbox_workdir` the seed is refused
  with 422. A seed whose `task_data` lacks `question` or `ground_truth` is refused with 422 and starts
  nothing; an ID bound to another episode or task, or one already closed, gets 409, and so does an
  episode seed for an agent run's session or for a session already verified; repeating the seed returns
  the same sandbox. A seed whose `mkdir` of the workdir exits non-zero is refused with 422, which an
  Environment Server does not retry: the image causes it (permission denied, a read-only root, a file in
  the way), so every episode would fail the same way. A sandbox that cannot start or be described, or a
  `mkdir` that does not complete (an error type, such as a timeout), answers 500, which an Environment
  Server retries, since a fresh sandbox may succeed. A seed that fails stops the sandbox it started.
* **Verify.** `/verify` stops the sandbox before it calls the judge. A session is verified once: an
  identical retry replays the verdict, and a retry with another body gets 409. After an attempt that
  reached no verdict, an identical retry is graded again. A verify after the session's close is
  masked (`failure_kind: session_lost`). A verify with no seed, such as `gym eval reverify`, is
  graded, since the grade reads only the request. With `plugin_log_path` set, an Environment
  Server's verify (it sends `x-ng-checkpoint-close-follows: 1`) for a session this server holds no
  sandbox or earlier verify for, as after a restart, is masked as `session_lost` instead: its plugin
  log went with the sandbox. Recovery keeps such a row masked; `gym eval resubmit` with `--recover`,
  `--rerun-masked-without-artifact` and a `--rerun-reason` runs it again, which counts as one failed
  attempt within its cap.
* **Plugin log.** With `plugin_log_path` set, an episode's verify first reads that file from the sandbox
  (60 s at most): the receipts plugin (`opencode_bundle/nemo_gym_receipts.js`) copies the OpenCode
  plugins' lines there. The model's tools can reach the file: it lies outside `sandbox_workdir`, but
  `/tmp`, where the BrowseComp preset keeps it, is writable to the agent in the same sandbox. So the read
  opens only a regular file that is not a symbolic link, and reads at most 4 MiB and one byte of it: a
  link, a FIFO, a directory or a device is reported and never opened, and the size of a file the model
  controls is never measured. It prints what it found, or `file:` and the bytes in base64, and then
  `:end`, so that no provider's output framing can split a line and output cut short is not taken for a
  shorter log. Verify takes the run's ending from the turn-budget plugin's lines in the whole log, as
  `opencode_sandboxed_agent` reads its run output, and returns the receipts with it
  (`opencode_terminal_reason`, `opencode_plugins_loaded`, `opencode_plugin_log`). The row carries at most
  1,048,576 characters of the log: a longer log keeps its two ends around a line that gives its length.

  An episode whose log fails the check is scored 0 without a judge call, as the model's 0, rather than
  refused: the model's tools can delete, replace, fill or edit the file, and a refused episode would be
  neither scored nor penalized. Its row has reward 0, `mask_sample` false, `evaluation_completed` false,
  the applied outcome `failure_outcome_action: count_as_zero` with `failure_outcome_attribution: model`,
  `opencode_plugins_loaded` false, no ending (so the log never masks it), and the `failure_kind`:

  - `opencode_search:plugin_log_missing`: no file at `plugin_log_path`;
  - `opencode_search:plugin_log_unsafe`: a symbolic link, not a regular file, a file the read cannot
    read, or one of more than 4 MiB;
  - `opencode_search:plugin_log_incomplete`: a log read whole that lacks the sentinel plugin's line or a
    line of `required_plugin_markers`, as an empty, cut, edited or garbled log does.

  Its `failure_reason` names what failed, and gives the bytes read (`log_bytes`) and the first and last
  600 characters of the log with line breaks escaped (`head`, `tail`). Gym counts the row in the score as
  a 0, reports it under its kind in the coverage's `counted_zero_kinds`, settles it, so a run that must
  complete still can, and never judges it again; training trains on it as a 0. An identical retry of the
  verify replays it. A plugin that did not load, or `required_plugin_markers` that do not match the run's
  plugins, fail the check on every episode, so verify answers 500 once
  `plugin_log_max_consecutive_failures` (25 by default; 0 turns it off) episodes in a row were scored 0
  for their log, and for each later one that fails until a log that passes resets the count. No other line
  is checked: a forged line, such as an appended `context_overflow` that masks a wrong answer, is not
  detected, as on the relay. A verify that grades a saved answer again cannot reproduce such a 0 (see
  Grading semantics).

  A log that cannot be read answers 500, which an Environment Server treats as retryable. Its failure
  kind is `provider_io_timeout` when the read timed out; `provider_request_failed` when it exited
  non-zero, reported another error type, or printed output cut short or malformed; or the provider's own
  kind when `exec` raised a typed failure. An untyped exception from `exec` carries no kind. An
  Environment Server records these kinds as the cause of the failed episode, in the failure row's
  `_ng_failure_events`. The model can still cause such a 500: by putting a FIFO at the path between the
  read's checks and its read, which holds the read until its timeout, or by removing `base64` or `sh` as
  root. Without `plugin_log_path`, verify reads nothing and scores nothing 0 for it, so a harness that
  brings no OpenCode plugins is graded on its answer alone.
* **Export copy.** With `episode_export_copy_dir` set, an episode's verify copies the session exports to
  this host after the plugin log read and before the sandbox stops, within 300 s. It copies the regular
  files directly in `episode_export_dir`, where the harness writes them (`opencode_agent`'s `export_dir`),
  to `<episode_export_copy_dir>/<capture key>/<task id>/attempt-<attempt>/`, which it empties first, so
  a retried episode never finds an earlier attempt's files there. The task ID gets `_` in place of each
  character other than an ASCII letter, a digit, `.`, `_` and `-`, and in place of each dot of an ID made
  only of dots. Links, directories, names that start with `.` (such as a partial file the harness has not
  renamed into place yet) and names with a character other than those are left out. The row carries
  `opencode_export_copy`: `status`, `dir` (that directory), `files` (each file's size in bytes, by name),
  `bytes` (their total) and `detail` (why nothing was copied). `status` is `copied`, with no files when
  the directory is empty; `absent` when the sandbox has no `episode_export_dir`; `oversize` when the
  files hold more than `episode_export_max_bytes`, and none is copied; or `failed` when the listing, a
  download or a file operation on this host failed, a copied file's size differs from the listing, or
  the copy ran past 300 s. Only a `copied` status leaves files on this host, and any other is logged.
  No copy changes the reward, `mask_sample` or `evaluation_completed`, and an identical retry replays
  the record. A row scored 0 for its plugin log carries its copy too. A verify that answers 500 once
  `judge_max_consecutive_failures` or `plugin_log_max_consecutive_failures` trips still makes the copy, but
  its reply has no `opencode_export_copy`, so its failure row does not name the files. A verify whose
  plugin log cannot be read (500) copies nothing, and neither does one masked as `session_lost`.
* **Close.** `/close_session` stops the sandbox by ID, with or without the session cookie, and fences
  the ID: a seed for it gets 409 for `session_close_retry_window_seconds`. A close whose stop fails
  raises and keeps the sandbox, so a retried close stops it. A session that is never closed keeps its
  sandbox until the server shuts down, which stops every sandbox still running, or until the
  provider reclaims it after `sandbox_config.ttl_s`. Keep `ttl_s` above the episode deadline plus the
  cleanup after it (the agent's close, then this server's, each within `cleanup_timeout_seconds`):
  verify stops the sandbox before it calls the judge, so grading needs no margin.

Supported harnesses, each in the `allowed_agents` of the config that pairs it:

* `opencode_agent`, on the episode seed. `benchmarks/browsecomp/configs/browsecomp_opencode.yaml` runs it
  through `single_agent_turn_legacy` and stages the OpenCode plugins that the relay agent uploads.
  That config admits no other harness, on purpose: the model's tools are OpenCode's own, shaped by those
  plugins, and this server serves no tools, so another harness would run a different benchmark. The
  pairing is for evaluation only, as `opencode_agent`'s session path is; training keeps the relay
  pairing. Run it with `++route_failures_to_sidecar=true`: a failed episode is a failure row either way,
  and the flag keeps a `/run` that fails outright from ending the run. The BrowseComp README lists how
  both pairings account for failures
  ([Failure accounting](../../benchmarks/browsecomp/README.md#failure-accounting)) and the session
  pairing's [time limits](../../benchmarks/browsecomp/README.md#time-limits).
* `opencode_sandboxed_agent`, on the agent-run seed, as `configs/opencode_search.yaml` declares. It serves
  no agent sessions, so its configs (`benchmarks/browsecomp/configs/browsecomp_opencode_legacy_agent.yaml`
  and `configs/opencode_search_example.yaml`) declare a `legacy_agent` relay in front of it.

`benchmarks/browsecomp/configs/browsecomp_opencode.yaml` was the relay pairing until the session pairing
took its name; the relay pairing is now `browsecomp_opencode_legacy_agent.yaml` beside it. A launch that
relies on the relay must switch to the new path: under the old one it gets the session pairing, which
does not compose without its OpenCode pins and judge and declares no `opencode_sandboxed_search_agent`.

The episode seed lends its sandbox to any session harness that accepts `sandbox_access`; the harness brings
its tools. With lending on, the server refuses `num_workers` above 1: lent sandboxes live in one worker's
memory, and several workers are not tested with them.

**Agent-run seed, kept for compatibility.** An agent's own `/run` seeds with the dataset row and its
session cookie and gets `{"sandbox_handle": ...}` back, the sandbox it attaches to. That sandbox is
started exactly as before episode sessions existed (the image's working directory, no task label), and
`/verify` stops it, since this path sends no close; `opencode_sandboxed_agent` also stops it when its run
ends. Without a verify, as when a run fails before grading, the server keeps it until it shuts down, or
the provider reclaims it after `ttl_s`. On this path:

* the first cookie seed in a server process logs that cookie seeds are deprecated;
* a seed whose cookie still holds a sandbox stops that sandbox first; if the stop fails, the seed answers
  500 and the earlier sandbox stays registered;
* the sandbox is registered before it starts, so a seed that fails or is cancelled while it starts
  stops it;
* every stop (at verify, on a repeated seed, at shutdown) is shielded from request cancellation and
  bounded at 120 s (`STOP_TIMEOUT_S`, as for lent sandboxes);
* a stop that fails at verify leaves the grade in place: it is logged and recorded on the verify
  response as a `session_release_failed` failure, and the sandbox stays registered until a later stop or
  shutdown.

The grading semantics above (verdict-less rows under `unparseable_verdict_policy`, the breaker's count,
`evaluation_completed`, the training mask) and verify-once with 409 apply on this path too.

A session gets no tools from this server: the BrowseComp session preset sets
`resources_tool_transports: []`, and OpenCode's web tools reach the web from the sandbox. The web-search
cache proxy (`POST /mcp`, registered only when the cache is configured in `configs/opencode_search.yaml`)
is not part of a session either: it reads no session cookie and keeps no session state, and it serves
every session's searches while the cache is on. So it keeps answering after a session closes, on
purpose: there is no per-session tool to refuse. It owns the `/mcp` route, so this server cannot also set
`expose_tools_over_mcp`.

### What differs from the relay

The session pairing repeats the relay pairing's settings; `benchmarks/browsecomp/README.md` lists what the
model sees differently. On this server's side:

* **Plugins and their settings.** This server stages them in the lent sandbox (`episode_sandbox_files`,
  `episode_sandbox_env`): each file is uploaded once, when the sandbox starts, and nothing reads it back.
  On the relay, `opencode_sandboxed_agent` uploads them, reads each back, trying up to 3 times, and passes
  their settings to OpenCode.
* **How a run ended.** Verify reads it from the plugin log. On the relay, the agent reads its run output
  and sends the ending in the verify body. Neither agent sends an `agent_timed_out` ending, so neither
  pairing masks a run that ran out of time: both grade what it produced.
* **A run whose plugins did not load.** A log that is missing, unsafe to read, or lacks the sentinel's
  line or a required line scores the episode 0 without a judge call, since the model's tools can reach the
  log, and a log that cannot be read answers 500. The relay agent checks only the sentinel's line, in its
  run output, and refuses a run whose output lacks it with 500.
* **Working directory.** The lent sandbox's is `sandbox_workdir`, which the seed creates; the agent-run
  sandbox keeps the image's.
* **OpenCode's exports.** With `episode_export_copy_dir` set, verify copies each session's exports to this
  host and reports the copy on the row (`opencode_export_copy`). On the relay, `opencode_sandboxed_agent`
  saves its run's export on the host itself (`opencode_results_fpath`).
* **Close.** An episode session ends with `/close_session`, which fences the ID and releases the sandbox if
  verify did not stop it, and a verify after it is masked as `session_lost`. The relay sends no close.

## Config

Needs `sandbox_image`, `sandbox_provider`/`sandbox_config`, and a `judge_model_server` pointing at a
model-server block. Episode sessions use these settings, which leave each episode feature off by default;
startup refuses a setting that cannot work:

* `sandbox_workdir` turns on lending to episode seeds; the seed creates it. It follows `opencode_agent`'s
  rules for a workdir: an absolute directory other than `/` and `/tmp`, without `..` or a NUL byte, and
  outside `/tmp/nemo-gym-opencode*`, where `opencode_agent` keeps its runtime and sessions.
* `episode_sandbox_files` maps absolute sandbox paths to files relative to the Gym root, read at startup
  and staged in each lent sandbox. A path must lie outside `sandbox_workdir`, where the model's tools
  could list and change the file, and outside `/tmp/nemo-gym-opencode*`. A source must name a regular
  file relative to the Gym root, without `..`: an absolute source or one with `..` fails startup, and
  with `sandbox_workdir` set, which makes the server read each source, so does a missing one or a
  directory. Stage under `/tmp`, which every image has: a provider need not create parent directories.
  Each file is uploaded once, when the sandbox starts (`SandboxSpec.files`); an upload that fails stops
  the sandbox, and the seed answers 500, a non-terminal episode failure that a resume retries. Nothing
  reads the files back or checks their digests in each episode; the BrowseComp scripted smoke checks
  them once. A file missing at the agent's seed is refused by `opencode_agent` with a terminal 422, and
  one that arrives but does not load scores its episode 0 at verify when the log then lacks the
  sentinel's line or a line of `required_plugin_markers`, until `plugin_log_max_consecutive_failures`
  episodes in a row make verify answer 500.
* `episode_sandbox_env` is the lent sandbox's environment, over `sandbox_config.env`; an empty value sets
  nothing. Supply secrets from an overlay. The search key is one `EXA_API_KEY`: the OpenCode plugins
  read only that key, and the lent sandbox's environment is fixed at startup, so no key can be picked
  from a pool for each row as the relay agent does. A lending server refuses an `EXA_API_KEYS` pool,
  here or in `sandbox_config.env`, at startup. With `exa_cache_root` set, `NEMO_GYM_EXA_BASE_URL`
  defaults to this server's `/mcp` proxy, so the server needs an address the sandbox reaches; on
  loopback it refuses to start.
* `plugin_log_path` is the plugin log verify reads, and also sets `NEMO_GYM_PLUGIN_LOG` in the lent
  sandbox. It must lie outside `sandbox_workdir`, the model's working directory. That keeps the log out of
  the way, not out of reach: `/tmp` is writable to the agent in the same sandbox, so verify scores an
  episode 0 when its log is missing, unsafe to read or lacks a line it must hold (see Plugin log under
  Sessions). `required_plugin_markers` lists the lines the log must hold besides the sentinel's.
  `plugin_log_max_consecutive_failures` (25; 0 turns it off) is how many episodes in a row may be scored
  0 for their log before verify answers 500.
* `episode_export_dir` is the sandbox directory the harness writes its session exports to. It follows
  `opencode_agent`'s rules for `export_dir`: absolute, not `/`, outside `/tmp/nemo-gym-opencode*`, and
  neither `sandbox_workdir`, inside it nor above it. `episode_export_copy_dir` turns on the export copy at
  verify: an absolute directory on this host, one per run, which the run's overlay sets; it needs
  `episode_export_dir`. `episode_export_max_bytes` (256 MiB) caps what one episode copies.

`configs/opencode_search.yaml` sets none of these, not even as `null`. `gym eval run` refuses to resume
across a changed semantic setting and counts a `null` as a value, so the relay preset, which resolves
none of them, still resumes a run started before these settings existed. See
`configs/opencode_search.yaml`; wired up in `benchmarks/browsecomp/configs/browsecomp_opencode.yaml`
(episode sessions) and `benchmarks/browsecomp/configs/browsecomp_opencode_legacy_agent.yaml` (relay). The
session preset takes `episode_export_dir` from `BROWSECOMP_OPENCODE_EXPORT_DIR`, the variable that also
sets the agent's `export_dir`, and leaves `episode_export_copy_dir` to a run's overlay.

`unparseable_verdict_policy` decides a rollout the judge gave no verdict, on either pairing: `zero` (the
default, for evaluation) or `mask` (for training; see [Training](#training)). Neither preset sets it.

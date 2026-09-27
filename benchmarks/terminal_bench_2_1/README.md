# terminal_bench_2_1

Terminal-Bench 2.1 with same-sandbox verification: the resources server starts one sandbox per
episode from the row's `docker_image`, the agent works in it, and `verify()` uploads the task's
`tests/` folder into that same sandbox, runs `bash /tests/test.sh` and reads
`/logs/verifier/reward.txt`.

Profiles:

- `terminal_bench_2_1/opencode`: OpenCode (`opencode_sandboxed_agent`).
- `terminal_bench_2_1/terminus_2`: Terminus 2 (`terminus_2_sandboxed_agent`).
- `terminal_bench_2_1/miniswe_in_sandbox`: mini-SWE 2.4.6 with the harness running INSIDE the
  sandbox (`miniswe_in_sandbox_agent`, see its README). The model is reached from the sandbox
  through `tb21_model_gateway_url`; `tb21_agent_max_timeout_sec` caps the row's agent budget and
  `tb21_agent_artifacts_dir` / `tb21_session_records_dir` select where runner records and session
  records go.

## Two seed/verify contracts

The resources server serves both contracts at once:

- **Handle contract** (OpenCode, Terminus): `/seed_session` returns `sandbox_handle`; `/verify`
  takes the row's task fields plus the agent's `response`.
- **Session contract** (in-sandbox mini-SWE, the TB4 shape): `/seed_session` also returns
  `session_id`, `sandbox_descriptor`, `sandbox_provider`, `instruction` (the row's user message),
  `task_id`, `agent_timeout_sec` (row field, default 28,800 s) and `user` (row `agent_user`, default
  the image identity); `/verify` takes `session_id`, `termination`, `agent_started`,
  `agent_timings`, `harness_metadata`, `responses_create_params` and `response`. The task fields are
  looked up from the seeded session.

Verification rules for the session contract:

- `termination.reason` `infrastructure_error` or `cancelled`, or `agent_started` false: the tests are
  NOT run; the sandbox is released and the response carries `evaluation_completed: false`,
  `mask_sample: true`, `failure_kind` `agent_run_error` / `cancelled`, `failure_reason` and
  `infrastructure_error` (the termination detail). Everything else (`completed`, `timeout`,
  `nonzero_exit`) is graded as it stands.
- Rows may carry `verifier_timeout_sec` (wall budget for `test.sh`, default the server's
  `evaluation_timeout`) and `verifier_env` (extra environment for `test.sh`, e.g. a task's own
  `VERIFIER_WALL_SEC`).
- A test run that leaves no `/logs/verifier/reward.txt` is `evaluation_completed: false` with
  `mask_sample: true` and `failure_kind: verifier_error` (reward stays 0.0 for compatibility).

Other server options: `prepare_apt_sources` (default true; turn off for self-contained images or a
deny-all network policy), `session_records_dir` (one small JSON per session, `phase` open/closed,
for campaign drivers that resume). The tests folder is shipped as ONE tar archive (one upload, one
exec) instead of one exec plus one upload per file.

Sandbox networking is a provider option: put
`sandbox_config.provider_options.network_policy: {defaultAction: deny, egress: [{action: allow,
target: <gateway ip>}]}` in the resources block to isolate tasks that declare `allow_internet =
false` while keeping the model gateway reachable.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

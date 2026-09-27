# mini-SWE in-sandbox agent

A mini-SWE agent server whose harness runs **inside** the resource-owned sandbox and calls the
Gym model server directly, following the OpenCode sandboxed agent's paradigm. It replaces the
per-step control-plane traffic of `miniswe_sandboxed_agent` (one sandbox exec API call per bash
command, one Gym model hop per model call, both from the Gym host) with a single long exec per
rollout and model calls that originate in the sandbox.

## Contract

Unchanged TB4 split contract. `/seed_session` gives the sandbox descriptor, provider
configuration, instruction, execution user and agent budget; this server reconnects, runs, and
posts `/verify` with the termination it observed (`session_id`, `termination`, `agent_started`,
`agent_timings`, `harness_metadata`, `responses_create_params`, `response`). Failed setup is an
unstarted infrastructure error (not graded); a started runner is graded even after a timeout.

## What runs where

Per rollout the server:

1. connects to the seeded sandbox and, as `agent.user`, checks `pwd`, `setsid` and a
   Python ≥ 3.9 (`python_executable`);
2. stages `/tmp/ng-miniswe-<session>/{miniswe_runner.py, vendor.zip, config.json}` and hands
   the directory to the task user when the sandbox default identity is root;
3. runs ONE exec as `agent.user` in the task workdir:
   `setsid --wait bash -c 'echo $$ >> /tmp/<session>.pids; exec python3 …/miniswe_runner.py --config …'`
   with `timeout_s = min(agent_timeout_sec, agent_max_timeout_sec)`;
4. downloads `trajectory.json`, `output_items.json`, `usages.json`, `result.json`, `runner.log`
   and builds the Gym response (native Responses items: assistant messages, `function_call`,
   `function_call_output`; usage summed when every call reported it).

`runner/miniswe_runner.py` is standard-library Python 3.9+ (the vendored MarkupSafe 3.0 needs 3.9) plus a vendored pure-Python Jinja2 and
MarkupSafe (built from this venv at startup, C speedups excluded). It re-implements
mini-swe-agent **2.4.6** `DefaultAgent` / `LocalEnvironment` / `actions_toolcall` semantics with
the pinned `mini.yaml` templates (`runner/mini_2_4_6.yaml`, copied verbatim): native `bash` tool
calls through `POST <gateway><capture prefix>/v1/responses`, format-error handling with the
three-strikes rule and the `OutputTokenLimitExceeded` mapping, `LimitsExceeded`/`TimeExceeded`,
`ContextWindowExceeded` on a context-overflow 400, `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
submission, per-command timeout with process-group kill and the same observation JSON (including
Jinja's `tojson` escaping and the long-output branch). Every executed command registers its
process group in `/tmp/<session>.pids`, which the resources server's quiesce step kills before
collection. Records are rewritten after every step, so a runner killed by the budget still leaves
a trajectory. Trajectory format: `mini-swe-agent-1.1`.

Termination mapping (same as the server-side harness): exec timeout or the runner's own
`TimeExceeded` → `timeout` (including a wall budget that runs out while a model call is in flight: the
pending call is dropped, the trajectory so far is graded); runner exit ≠ 0 → `infrastructure_error` (log tail in the detail);
`Submitted` → `completed`; any other exit status → `nonzero_exit` with the status as detail. On an
exec timeout the agent first SIGTERMs the runner's process groups (the runner saves a `TimeExceeded`
exit) and only then downloads the records. A command that hits the per-step timeout is observed as
`returncode -1` with `exception_info "Command timed out after N seconds."` (mini-swe's
LocalEnvironment mechanics; the wording follows the Gym harness, not upstream's exception text).

## Model access

OpenCode-style: the sandbox cannot reach the Gym host, so `model_gateway_url` must be a
sandbox-reachable origin that forwards to this Gym's `policy_model` server (for example a relay
on a login node in front of a reverse ssh tunnel). The runner appends Gym's rollout-capture prefix
(`/ng-rollout/<id>[/training-token-capture]`) when capture is on and sends `x-session-id`. For
`no-network` TB4 tasks set the resources server's `environment.agent_egress_allow` to that host;
the verifier stays deny-all.

## Configuration

`configs/miniswe_in_sandbox_agent.yaml`: `model_gateway_url` (required), `harness.step_limit`
(0 = unlimited), `harness.step_timeout_sec` (30), `harness.max_consecutive_format_errors` (3),
`harness.http_retries`/`http_timeout_sec`, `agent_max_timeout_sec`, `setup_timeout_sec`,
`runner_exit_margin_sec` (the runner stops itself this long before the exec budget so it exits
cleanly with a saved trajectory), `instruction_suffix` (appended to the task instruction),
`python_executable`, `remote_dir_prefix`, `artifacts_dir`, `provider_overrides` (deep-merged into the seeded
provider block for this agent's transport only — polling cadence; a `connection` override is rejected and
`background_exec` is always on). Benchmark profile:
`benchmarks/terminal_bench_4/miniswe_in_sandbox.yaml` (`++tb4_model_gateway_url=…`).

## Trust boundary

The runner and the model's commands share one uid, so the records it leaves (and therefore the response the
verifier and training see) are writable by the policy — the same boundary the OpenCode paradigm has. The agent
records cheap consistency checks (`harness_metadata.consistency`: every `function_call_output` matches a
`function_call`) and the staging identity (`harness_metadata.staging`); the server-side harness kept these in host
memory instead. Do not run the runner as root to "fix" this: the resources server's quiesce runs as `agent.user`.

## Not supported (yet)

Task MCP servers (`mcp_servers` in the seed → unstarted infrastructure error); images without
`python3` ≥ 3.9 or `setsid`; multimodal tool output; the `mode: confirm` interactive path.

## Terminal-Bench 2.1 resources (same-sandbox verification)

The agent also runs on `resources_servers/terminal_bench_2_1`, which serves the same
seed/verify session contract next to its original handle contract (profile
`benchmarks/terminal_bench_2_1/miniswe_in_sandbox.yaml`). Differences from TB4 that matter here:

- One sandbox per episode: the agent runs in it and the resources server then uploads the task's
  `tests/` and runs `bash /tests/test.sh` in the same sandbox, so the workspace is graded in place
  (no artifact declarations, no separate verifier image). Rows carry `task_name`, `docker_image`,
  `task_folder`, the instruction as the single user message, and optionally `agent_timeout_sec`,
  `agent_user`, `verifier_timeout_sec`, `verifier_env`, `rollout_id`.
- Identity: `seed.user` is the row's `agent_user` (absent for root images), so TB2 tasks run as the
  image default unless the row says otherwise. No `mcp_servers` / `skills_dir`.
- An `infrastructure_error` or `cancelled` termination, or an agent that never started, releases the
  sandbox WITHOUT running the tests (`mask_sample: true`, `failure_kind: agent_run_error`); every other
  termination is graded as it stands, as on TB4.
- Network: TB2.1 sandboxes are created without a network policy unless the resources block sets
  `sandbox_config.provider_options.network_policy`; for `allow_internet = false` tasks use a deny-all
  policy with one egress allow for the model gateway (the verifier shares the sandbox, so it sees the
  same policy).

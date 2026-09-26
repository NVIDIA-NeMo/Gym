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
   Python ≥ 3.8 (`python_executable`);
2. stages `/tmp/ng-miniswe-<session>/{miniswe_runner.py, vendor.zip, config.json}` and hands
   the directory to the task user when the sandbox default identity is root;
3. runs ONE exec as `agent.user` in the task workdir:
   `setsid --wait bash -c 'echo $$ >> /tmp/<session>.pids; exec python3 …/miniswe_runner.py --config …'`
   with `timeout_s = min(agent_timeout_sec, agent_max_timeout_sec)`;
4. downloads `trajectory.json`, `output_items.json`, `usages.json`, `result.json`, `runner.log`
   and builds the Gym response (native Responses items: assistant messages, `function_call`,
   `function_call_output`; usage summed when every call reported it).

`runner/miniswe_runner.py` is standard-library Python 3.8+ plus a vendored pure-Python Jinja2 and
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

Termination mapping (same as the server-side harness): exec timeout → `timeout`; runner exit ≠ 0
→ `infrastructure_error` (log tail in the detail); `Submitted` → `completed`; any other exit
status → `nonzero_exit` with the status as detail.

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
`python_executable`, `remote_dir_prefix`, `artifacts_dir`. Benchmark profile:
`benchmarks/terminal_bench_4/miniswe_in_sandbox.yaml` (`++tb4_model_gateway_url=…`).

## Not supported (yet)

Task MCP servers (`mcp_servers` in the seed → unstarted infrastructure error); images without
`python3` ≥ 3.8 or `setsid`; multimodal tool output; the `mode: confirm` interactive path.

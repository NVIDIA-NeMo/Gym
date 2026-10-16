# CaP-X resources server

A `GymnasiumServer` (driven by `gymnasium_agent`) that runs [CaP-X](https://github.com/capgym/cap-x) code-as-policy
episodes. See [`benchmarks/capx`](../../benchmarks/capx/README.md) for tasks, tiers, metrics, and run commands.

## Episode protocol

Each task row carries `capx_config` (a CaP-X YAML path relative to the CaP-X checkout) and `seed`.

1. `reset` builds the CaP-X environment from the config, resets it with `seed`, and returns the task prompt with its
   API documentation as the first user message. The system prompt comes from the dataset row.
2. The model's first response is executed as a program. A ```` ```python ```` fence is optional.
3. Single-turn configs end after that program runs.
4. Multi-turn configs (configs that define `multi_turn_prompt`) return the executed code, stdout, and stderr. The model
   then answers `REGENERATE` followed by new code, which is executed, or `FINISH`, which ends the episode. The episode
   is truncated after `max_code_blocks` programs (default 11, matching CaP-X) or when the simulator ends the episode.

Reward is `1.0` if the simulator reports the task completed at the end of the episode, otherwise `0.0`. The step
`info` also carries `task`, `tier`, `seed`, `sandbox_rc`, `env_reward`, `num_code_blocks`, `num_regenerations`,
`ended_by`, and the stderr tail.

Code extraction and REGENERATE/FINISH parsing match CaP-X's `capx.utils.launch_utils`. Unlike CaP-X, which re-sends
the original prompt plus the latest feedback on each turn, Gym keeps the whole conversation in context.

## Failure masking

The final step's `info` sets `mask_sample`, `failure_kind`, and `failure_reason`; `gymnasium_agent` copies them onto the
rollout. Masked samples are excluded from scores.

| Outcome | Masked | `failure_kind` |
|---|---|---|
| Program crashes, fails the task, or reaches `max_code_blocks` | no | none |
| Program runs past `step_timeout_s` (CaP-X scores trial timeouts as failures) | no | `capx:program_timeout` |
| Worker process exited or its pipe closed | yes | `session_lost` |
| CaP-X raised outside the generated program | yes | `capx:simulator_error` |
| Program could not connect to a local API server (port 8113-8117) | yes | `transport_unreachable` |
| Reset failed | not scored: `/reset` returns an error and Gym retries the rollout | |

An API server that answers with an error, for example for bad arguments, is the program's outcome and is not masked.

## Process model

CaP-X requires Python 3.10-3.12 and its simulator stack, so the server does not import it. `setup_capx.py` installs the
pinned CaP-X commit with its own uv venv (CaP-X's lockfile syncs only with uv 0.9.x, run through `uvx`). `worker.py`
runs under that venv and serves one environment over a JSON-lines pipe on private file descriptors, so generated code
cannot disturb the protocol. Generated code runs unsandboxed inside the worker, as in CaP-X.

- At most `max_workers` workers run at once. Each holds one episode at a time.
- Idle workers are reused, preferring one that already built the same config, because environment construction
  dominates reset time.
- A worker that exceeds `reset_timeout_s` or `step_timeout_s` is killed.
- The server advertises `supports_explicit_close`, so `gymnasium_agent` releases the worker when a rollout fails.
- CaP-X's API servers are started with its `full` launch profile, and readiness is checked on every port, because
  CaP-X's launcher reports ready even when one server failed.

## Configuration

| Field | Default | Description |
|---|---|---|
| `capx_root` | `$CAPX_ROOT`, else install into `.capx` | set-up CaP-X checkout |
| `capx_python` | `<capx_root>/.venv/bin/python` | CaP-X venv interpreter |
| `max_workers` | 4 | concurrent simulator processes |
| `max_episodes_per_worker` | 25 | episodes a worker serves before it is replaced, bounding memory on long runs |
| `reset_timeout_s` / `step_timeout_s` | 900 / 600 | per-operation budgets |
| `max_code_blocks` | 11 | multi-turn program limit |
| `worker_env` | `{MUJOCO_GL: egl}` | extra worker environment variables |
| `worker_log_dir` | none (inherit stderr) | per-worker simulator log directory |
| `launch_api_servers` | true | start CaP-X's API servers unless ports 8113-8117 already answer |
| `api_server_gpus` | all | GPU indices for the API servers, for example `'1'` |
| `api_server_timeout_s` | 1800 | API server start-up budget (first start downloads weights) |
| `api_server_log_dir` | `<capx_root>/api_server_logs` | per-server logs |

## Licensing

CaP-X code and task configs: MIT. This server: Apache-2.0.

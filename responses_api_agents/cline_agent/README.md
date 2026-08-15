# Cline Agent

Runs the [Cline](https://cline.bot) CLI headlessly (`cline --json`). Cline runs its own tools
internally, and its newline-delimited JSON event stream is parsed into Gym format and verified by
the resources server.

Minimal, meant to be extended, and currently eval-only: token IDs and logprobs are not wired up.

## Quick start

Cline must be on PATH (auto-installed on first start, or `npm install -g cline`). Set
`policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`; the model server started
by `--model-type` serves that backend and Cline calls the model server.

```bash
gym env start \
  --resources-server math_with_judge/math_with_judge_cline_agent \
  --model-type openai_model

gym eval run --no-serve --agent math_with_judge_cline_agent \
  --input resources_servers/math_with_judge/data/example.jsonl \
  --output cline_rollout.jsonl
```

Per request the agent creates an isolated run dir, runs one `cline --json --auto-approve true`, and
parses the streamed JSON events for the trajectory. The run dir holds a per-request Cline data
directory (`CLINE_DATA_DIR` and the settings, session, db, and team paths all point inside it), so
concurrent rollouts share no provider settings or session state and the user's `~/.cline` is never
touched or read. `CLINE_SESSION_BACKEND_MODE=local` keeps each run in its own process rather than a
shared hub daemon. The whole run dir is removed afterwards.

## SWE-bench

`anyswe_agent` runs this harness inside a SWE task image, extracts the repository patch, and grades
it — see [`anyswe_cline.yaml`](../anyswe_agent/configs/anyswe_cline.yaml) and the
[anyswe_agent README](../anyswe_agent/README.md):

```bash
export ANYSWE_CONTAINER_FORMATTER='registry.example.com/anyswe/swebench:{instance_id}'
python3 responses_api_agents/anyswe_agent/prepare.py --limit 5

gym env start \
  --config responses_api_agents/anyswe_agent/configs/anyswe_cline.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model

gym eval run --no-serve --agent anyswe_cline \
  --input responses_api_agents/anyswe_agent/data/swebench_verified.jsonl \
  --output results/anyswe_cline_rollouts.jsonl --limit 5
```

That config sets `repo_dir: /testbed` so Cline edits the checkout anyswe takes the patch from.
`setup_scripts/cline_agent_deps.sh` builds the portable runtime (`agent_runtime_source: auto`) when
the CLI is not already baked into the task image.

## Model server

With `model_server` set (the shipped default), Cline's model calls go to that Gym model server
rather than to a provider directly. That is what makes requests and responses show up in Gym's
capture, and it means one config runs against vLLM, OpenAI, or an inference provider by swapping
`--model-type`.

Cline reads its provider, key, model, and base URL from a provider settings file rather than from
run flags — `-k` overrides only the key, and no flag sets a base URL — so on this path the agent
runs one `cline auth openai-compatible --baseurl <model server> --modelid <model> --data-dir <run
dir>` before the task, writing those settings into the run's own data dir. `model` is therefore
required here, and is the bare name the model server serves:

```yaml
model_server: {type: responses_api_models, name: policy_model}
model: ${policy_model_name}
```

The provider is forced to `openai-compatible` on this path: `cline auth` accepts a base URL only for
the OpenAI and OpenAI-compatible providers. The API key written to the run's settings is the literal
`EMPTY` — Gym model servers do not check it, and the real credential (if any) lives on the model
server. During a `/run` the base URL carries the per-rollout `/ng-rollout/<id>` prefix, so captured
model calls are attributable to the rollout that made them.

### Calling a provider directly

Set `model_server: null` and name the provider yourself. Cline then uses whatever that provider is
already authenticated with (`cline auth` outside Gym, or `openai_base_url` / `openai_api_key` for
the OpenAI-compatible one), and no model calls are captured:

```yaml
model_server: null
provider: anthropic
model: claude-sonnet-4-5
```

## Config fields

- `model_server`: Gym model server Cline calls; `null` to use an already-configured provider
- `concurrency`: max simultaneous `run()` calls
- `command`: the Cline command, split on spaces so a multi-word launcher works (e.g. `npx cline`)
- `model`: the model id; required with a model server, `null` otherwise leaves the provider's own
  selection (`-m` is omitted)
- `provider`: provider id for `-P`; ignored with a model server (forced to `openai-compatible`)
- `openai_api_key` / `openai_base_url`: passed to the subprocess as `OPENAI_API_KEY` /
  `OPENAI_BASE_URL`; a model server takes precedence over both
- `env`: extra env vars for the subprocess
- `workspace_root`: where per-request run dirs are created and deleted
- `repo_dir`: optional persistent project dir to run in (default: ephemeral per-request dir)
- `system_prompt`: prepended to the user message
- `system_prompt_override`: passed as `-s`, which **replaces** Cline's built-in system prompt rather
  than adding to it, dropping the instructions its tools are described by. Prefer `system_prompt`
- `thinking`: `--thinking` level (`none|low|medium|high|xhigh`); `null` leaves the provider default
- `compaction`: `--compaction` mode (`agentic|basic|off`); `null` leaves Cline's default
- `retries`: `--retries`, max consecutive mistakes before Cline halts
- `timeout`: seconds for the `cline` call; also passed as `--timeout` so Cline winds the session
  down itself before the subprocess is killed
- `setup_timeout`: seconds for the one-off `cline auth` call
- `extra_args`: extra flags appended to the `cline` invocation
- `command_permissions`: JSON policy passed as `CLINE_COMMAND_PERMISSIONS`, restricting the shell
  commands Cline may run (e.g. `{allow: ["python3 *"], deny: ["sudo *"]}`); empty means unrestricted
- `cline_version`: `cline` npm version installed on a clean machine (shipped pinned to `3.0.55`;
  the event parser was validated against it, so treat a bump as a deliberate change: raise it,
  re-run the tests and a live eval, then commit). `null` installs `@latest`

See `configs/cline_agent.yaml`.

## Trajectory mapping

| Cline `--json` record | Gym item |
|---|---|
| `agent_event` `content_start` / `content_end` (`text`) | assistant message |
| `agent_event` `content_start` (`tool`) | `function_call` |
| `agent_event` `content_end` (`tool`) | `function_call_output` (its `error` when the tool failed) |
| `agent_event` `content_*` (`reasoning`) | `<think>` block on that turn's message |
| `agent_event` `usage`, `run_result` | token usage |
| `hook_event` | ignored (duplicates tool boundaries the agent events already carry) |

Cline closes a turn's reasoning *after* its text, so the think block is assembled from the reasoning
streamed so far rather than from the trailing `content_end`, which would otherwise attach it to the
next turn. `run_result`'s `aggregateUsage` is authoritative when present; a run killed on timeout
never prints it, so the per-turn `usage` totals are used and `timed_out` is recorded.

## Limitations

- Eval-only: no token IDs or logprobs, so this harness is not usable for training as-is.
- One prompt per request. Multi-turn conversation input is reduced to the last user message (plus
  any system message), matching the other CLI agent harnesses.
- A prompt with no whitespace is padded with a trailing space: Cline decides a bare argument was
  quoted by looking for whitespace in it, and rejects a single-word prompt as
  `Unknown command or unquoted prompt` even after `--`.

# Examples

## `simple_agent.py`: run an agent on tasks in one command

[`simple_agent.py`](simple_agent.py) runs the usual two-step flow as a single command. It starts
the servers (`gym env start`) in the background, waits until they and the model endpoint are ready,
collects rollouts (`gym eval run --no-serve`), then stops every server it started. It stops them on
success, on failure, and on Ctrl-C.

Run it with any Python environment that has NeMo Gym installed. `examples/` is not part of the
installed package, so run the file by its path in a checkout (or a copy of it).

Each server runs in its own virtualenv (`<server dir>/.venv`), which Gym sets up with
[uv](https://docs.astral.sh/uv/). By default the script reuses virtualenvs that already exist
(`--skip-venv-if-present`, i.e. `+skip_venv_if_present=true`), so only the first run of a server
needs `uv`. Pass `--no-skip-venv-if-present` to set them all up again, e.g. after dependencies change.

### Run a coding agent on a repository

By default the script serves the [`codex_tools`](../resources_servers/codex_tools/README.md) coding
environment. The agent works on the git repository you run the script from (for a directory that
is not a git repository, see below):

```bash
# policy_base_url and policy_api_key come from env.yaml (see "Configure your model" in the top-level README)

cd /path/to/target/repo
python /path/to/Gym/examples/simple_agent.py \
    --model-type openai_model --model gpt-6-sol \
    --prompt "The tests in tests/ fail. Fix the bug in the library code, then run the tests." \
    --check-command "python -m pytest -q tests" \
    --output /tmp/rollouts.jsonl
```

- `--prompt` is the task. The agent gets Codex-style tools (`exec_command`, `write_stdin`,
  `apply_patch`, `update_plan`) and works in its own git worktree of the repository, so the
  repository itself is not modified. Outside a git repository, the agent edits the directory itself
  and the result has no diff.
- `--check-command` (optional) runs in that worktree after the agent finishes. Exit status 0 gives
  reward 1.0; without a check the reward is 0.0.
- While the agent works, its items are printed as they happen: the prompt, each model response
  (text, tool calls, reasoning), and each tool result (see [Watching the agent](#watching-the-agent)).
- The result, including the agent's diff, is in the `--output` file (see
  [Input and output](#input-and-output)).

For several tasks, write an input file with
[`make_tasks.py`](../resources_servers/codex_tools/make_tasks.py) and pass it with `--input`
instead of `--prompt`:

```bash
python /path/to/Gym/resources_servers/codex_tools/make_tasks.py \
    --prompt "Fix the failing test in tests/test_parser.py" \
    --check-command "python -m pytest -q tests/test_parser.py" -o /tmp/tasks.jsonl
```

### Other environments

Choose another environment with `--resources-server`, `--environment`, `--benchmark`, or
`--config`. With any of these, `codex_tools` is not served. This is the
[Quick Start](../README.md#-quick-start) evaluation as one command, run from the Gym checkout with
the model configured in `env.yaml` as described there:

```bash
python examples/simple_agent.py \
    --resources-server mcqa \
    --model-type openai_model \
    --input resources_servers/mcqa/data/example.jsonl \
    --output results/mcqa_rollouts.jsonl \
    --limit 5
```

`--agent` is optional when the served configuration has exactly one agent (here
`mcqa_simple_agent`).

### Input and output

`--input` is a JSONL file of tasks, one per line. Each line holds the model request in
`responses_create_params` (the instructions and prompt in `input`, the tool definitions in `tools`)
and environment-specific task settings. For `codex_tools` these are in `verifier_metadata`:
`check_command`, and optionally `repo_path` and `base_ref`. Each environment's `data/example.jsonl`
shows its format. `--prompt` writes a one-task file for you, `<output stem>_task.jsonl`.

`--output` is the rollouts file, one line per rollout. Each line has the full transcript
(`response.output`) and the `reward`. For `codex_tools` it also has `diff`, `check_exit_code`,
`check_output`, and the agent's `plan`. Other files are written next to it:

| File | Contents |
|---|---|
| `<output stem>_aggregate_metrics.json` | Summary metrics, also printed at the end |
| `<output stem>_materialized_inputs.jsonl` | The tasks exactly as sent |
| `<output stem>_failures.jsonl` | Attempts that failed to produce a rollout (agent errors), one row each |
| `<output stem>_servers.log` | Server output |
| `<output stem>_task.jsonl` | The task written for `--prompt` |
| `<output stem>_items.log` / `_items.jsonl` | The `--echo` output |
| `quality_summary.json`, `rollout_verdicts.jsonl` | Rollout health-check results |

### Watching the agent

`--echo pretty` prints `simple_agent`'s items as they happen, one readable block per item headed
`[simple_agent:<step>] <kind>`. `--echo json` prints each item exactly as it arrives, in the Responses
API format, one JSON object per line. With `json`, the collection's own output (progress and
metrics) goes to stderr, so stdout contains only the items. The echo is also saved next to the
output as `<output stem>_items.log` or `<output stem>_items.jsonl`.

`--echo` defaults to `pretty` with `--prompt` and to `off` otherwise. It works for any environment
that uses `simple_agent`. See the [agent's README](../responses_api_agents/simple_agent/README.md)
for the underlying settings.

### Arguments

The script accepts the same flags as `gym eval run` and passes each one to the command that uses it:

| Arguments | Passed to |
|---|---|
| Flags that `gym env start` also accepts: `--config`, `--resources-server`, `--model-type`, `--model`, `--model-url`, `--model-api-key`, `--environment`, `--benchmark`, `--agent-type`, ... | `gym env start` |
| All other `gym eval run` flags: `--input`, `--output`, `--agent`, `--limit`, `--num-repeats`, `--concurrency`, sampling and health-check flags, ... | `gym eval run --no-serve` |
| `-v`/`--verbose`, `--search-dir`, and Hydra overrides (`+key=value`) | both |
| Everything after `--` | `gym eval run --no-serve`, unchanged |

`--output` is required, and so is either `--input` or `--prompt`. The script has these options of
its own:

- `--prompt TEXT`: a task for the `codex_tools` coding agent, in place of `--input`.
- `--check-command COMMAND`: with `--prompt`, the command that decides the reward.
- `--echo off|pretty|json`: print the agent's items as they happen (default: `pretty` with
  `--prompt`, otherwise `off`).
- `--harness-cwd DIR`: the repository a coding agent works on (default: the current directory).
- `--gym-workdir DIR`: Gym's working directory for non-editable installs (default below).
- `--startup-timeout SECONDS`: how long to wait for the servers (default: 1800).
- `--skip-venv-if-present` / `--no-skip-venv-if-present`: reuse existing per-server virtualenvs
  (default), or set them all up again with `uv`. Missing virtualenvs are always set up, which needs `uv`.

### Directories

The script keeps the repository an agent works on separate from Gym's own files:

- **Harness cwd.** The current directory, or `--harness-cwd`. It is passed to the servers as
  `+harness_cwd=<path>`, which `codex_tools` uses as its default repository. An explicit
  `+harness_cwd=...` or `+codex_tools_repo_path=...` takes precedence.
- **Gym's working directory.** This is where Gym creates server virtualenvs, caches, and results.
  It is the Gym checkout for an editable install. Otherwise it is `--gym-workdir`, by default
  `~/.cache/nemo_gym/simple_agent`, so Gym does not write into the repository you run from.

Relative paths behave as they would for `gym` run in the current directory. `--config` and
`--input` are looked up there first and then among Gym's built-in environments. `--output` is
relative to the current directory.

### Exit code and shutdown

- The exit code is the collection's exit code. It is 1 if the servers fail to start, and the end of
  the server log is printed. It is 130 if the run is interrupted.
- On Ctrl-C (or SIGTERM), the collection gets a chance to finish its interrupt handling, then the
  servers are stopped. A second Ctrl-C does not skip the shutdown.
- Each run uses its own free head-server port, so several runs can execute at the same time.

### The same flow in two steps

To keep servers running between collections, run the two commands yourself:

```bash
gym env start --resources-server mcqa --model-type openai_model

# In another terminal, once the servers are ready:
gym eval run --no-serve --agent mcqa_simple_agent \
    --input resources_servers/mcqa/data/example.jsonl --output results/mcqa_rollouts.jsonl --limit 5
```

For `codex_tools`, add `+harness_cwd=$PWD` (or `+codex_tools_repo_path=...`) to `gym env start`.

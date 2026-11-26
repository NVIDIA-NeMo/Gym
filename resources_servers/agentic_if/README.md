# Description

Measures whether a coding agent follows instructions while it completes a real repository (Scale-SWE) or terminal
(Terminal-Bench 2.1) task in OpenCode. This resources server owns each episode under `single_agent_turn`:

1. **Seed.** It checks the sealed row, borrows the task's sandbox through the `scale_swe` or `terminal_bench_2_1`
   episode session, checks the agent user and working directory, and stages the row's workspace files. For a row that
   continues a recorded session, it also restores that session's workspace and stages its export. It then stages the
   OpenCode plugins in `opencode_bundle/`, and any `opencode_runtime_files`, and lends the sandbox to `opencode_agent`.
2. **Agent turn.** `opencode_agent` runs OpenCode in that sandbox with the plugins. A profile plugin applies the row's
   tool argument maps, read-before-edit guards and planted tool results, and writes receipts. A boundary plugin
   applies the row's system prompt, tool selection and tool descriptions, and records every model call.
3. **Verify.** It reads the plugins' receipts and call records and OpenCode's session exports from the sandbox, builds
   the session evidence, and has the task backend verify the task. It saves the episode under `artifacts_dir` and grades
   the row's instruction checks with `if_grading/`, whose judge settings are in `if_grading/data/judge.yaml`.
4. **Close.** It releases the backend session, which stops the sandbox.

`configs/agentic_if.yaml` wires this server, its judge model server, `opencode_agent` and `single_agent_turn` for the
`example` split. `benchmarks/agentic_if` adds the benchmark's task set and states its measurement contract.

## Rows

Each row is sealed: `agentic_if` holds the run configuration (`task_variants/`) with receipts over the actor input,
the task, the configuration and the tool interfaces, and `task_id` is the row's content digest. Seed and verify refuse
a row whose receipts do not match.

A row whose instruction stops the work, so that the task's tests measure nothing of its run, is sealed with
`agentic_if.outcome_tracked: false`. Only the `task_and_if` reward reads it (see Training reward).

`data/example.jsonl` holds 5 public examples, 3 terminal and 2 repository tasks, each with a synthetic system prompt
and instruction checks that exercise the features the benchmark rows use. `data/feature_examples.jsonl` holds 4 more
for the opt-in features (see Feature examples).

## Tools

`agentic_if.tools` chooses and words OpenCode's built-in tools; a row never carries tool code:

- `tool_names` maps each selected built-in tool (`bash`, `read`, `write`, `edit`, `glob`, `grep`, `task`,
  `todowrite`) to the name the model sees. Every other tool is denied. Without it, OpenCode offers its own tools.
- `tool_descriptions` and `tool_parameter_descriptions` hold literal text, and `tool_interfaces` installs a
  catalog interface (argument names and schema, `task_variants/data/opencode_interfaces.json`), per built-in tool.
- `read_first` refuses, with the row's text, an edit or write of a file the model has not read or that changed
  since it read it.

A tool mapped to another name is offered under that name, and a call to that name runs the built-in tool:

```json
"tools": {
  "tool_names": {"bash": "shell", "read": "view_file", "edit": "edit", "write": "write"},
  "tool_descriptions": {
    "bash": "Run one shell command in the working directory.",
    "read": "Show a file with line numbers.",
    "edit": "Replace one exact piece of text in a file.",
    "write": "Write a whole file."
  }
}
```

- **Names.** Each name matches `[A-Za-z_][A-Za-z0-9_]{0,63}`, and the names the model sees are distinct (one may
  be another built-in tool's ID, when that tool is not offered under it).
- **What the model sees.** The alias in the tool definitions, in history (its own calls, a restored prefix's calls,
  and compaction calls), in `tool_choice`, in the message of a rejected call, and in OpenCode's texts that name a tool:
  argument errors and permission refusals, the edit error that suggests write, and the hint after a truncated output.
- **Calls.** A name whose lowercase is an alias (`Shell`) runs the tool, as OpenCode's own lowercase repair would. A
  renamed tool's built-in ID (`bash` here, in any capitals) is refused with the message of any unavailable tool.
- **Grading.** Checks and planted events name built-in tools. Evidence keeps the tool that ran (`tool`) and the name
  the model wrote (`model_tool`), which the judge sees; a restored prefix's calls show the row's alias. A
  continuation's `source_tools_sha256` covers the names, so a prefix is bound to the names it was recorded under.
- **Requirements.** OpenCode's own system prompt, tool descriptions and rejected-call text name the built-in tools,
  so a row that renames a tool needs a system message (sealing refuses the row otherwise), a description for every
  selected tool (the row's or its interface's), and no `unreceipted_rejected_calls` (seed refuses the row otherwise).
- **Not renamed.** The shell tool's timeout note calls it the "shell tool", and argument names in OpenCode's hints
  follow an interface only in `read`'s paging footer.

## Settings

- `policy_base_url`, `policy_api_key` and `policy_model_name`, in `env.yaml` or as overrides.
- `judge_api_key` for the judge model server, and `judge_base_url` to use another endpoint.
- The sandbox provider's settings, for example `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`.
- `agentic_if_artifacts_dir`, optional: an absolute directory for the saved episodes. It defaults to
  `<results_dir>/agentic_if`.
- `opencode_runtime_files`, optional: host files that seed copies into each task sandbox's bundle directory
  (`/tmp/nemo-gym-agentic-if/`), for `opencode_agent` to read by in-sandbox path; relative paths are under the Gym
  root. The benchmark stages OpenCode's ripgrep and plugin dependencies this way for sandboxes without internet access;
  see `benchmarks/agentic_if/README.md`.

## Run the examples

The terminal rows use the task folders that `benchmarks/terminal_bench_2_1/prepare.py` clones. The example rows route
by taskset, so collate them first, then collect against running servers. With a remote sandbox provider,
`use_absolute_ip=true` lets OpenCode in the sandbox reach Gym's model server.

```bash
python benchmarks/terminal_bench_2_1/prepare.py
gym dataset collate --resources-server agentic_if --mode example_validation --output-dir results/agentic_if_example_data
gym env start --resources-server agentic_if --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml +use_absolute_ip=true
# In a second shell, once the servers are ready:
gym eval run --no-serve --resources-server agentic_if --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml +use_absolute_ip=true \
    --input results/agentic_if_example_data/example.jsonl --output results/agentic_if_example.jsonl
```

## Results

- `reward`: the task's own verifier result. Instruction following does not change it, unless the server's
  `reward_mode` is `task_and_if` (see Training reward).
- `mean/if_all`: the share of graded rollouts that pass every applicable check.
- `mean/if_frac`: the share of applicable checks passed, averaged over graded rollouts.
- `mean/task_and_if`: the share of graded rollouts that pass both the task and every check.
- `if/graded`, `if/not_applicable` and `if/error` count rollouts; only graded rollouts enter the rates.
- `judge_calls` and `judge_usage`: each judge call's reply ID and token usage, and their totals. The judge's calls are
  not among the rollout's captured model calls, which belong to the agent.

An episode whose evidence could not be collected, or whose task verification failed for an infrastructure reason, is
masked (`mask_sample: true`, with `failure_kind` and `failure_reason`). An OpenCode run that times out or fills its
context window is scored as it stands (`agent_timeout`, `model_context_limit`), unless `cut_run_reward` is `zero`. A
check whose judge call failed on every attempt is recorded as `judge_failed`; `gym eval reverify --judge-failed-only`
grades it again from the saved episode, without the sandbox.

## Training reward

The default reward is the task verdict, which the benchmark reports. To train on instruction following, set
`reward_mode: task_and_if` on this resources server; `benchmarks/agentic_if/config.yaml` leaves it unset.

- `reward_mode`:
  - `task` (default): `reward` is the task verdict.
  - `task_and_if`: `reward` is the task verdict times the run's IF score, so a run earns reward only when it does
    the task and follows the instructions. A row sealed with `agentic_if.outcome_tracked: false` earns its IF score
    alone.
- `if_score`, with `task_and_if` only: `if_all` (default: 1 when every applicable check passed, else 0) or `if_frac`
  (the share of applicable checks passed).
- `cut_run_reward`, with either mode: `as_is` (default) scores a run that OpenCode's timeout or a full context window
  cut short as it stands, like any other run; `zero` gives it reward 0.0, unmasked, with its `failure_kind`
  (`agent_timeout` or `model_context_limit`).

Under `task_and_if`:

- A run without an IF score is masked with reward 0.0, whatever its task verdict: `agentic_if:if_not_applicable`
  when no check applied to it, `agentic_if:if_error` when a check could not be graded.
- A sample masked for another reason stays masked with its own `failure_kind`, and its reward is 0.0.
- With `as_is`, a cut run whose interrupted tail left its checks ungraded is masked as `agentic_if:if_error`;
  `zero` keeps it, at 0.0.

When any of these settings is not its default, `task_reward` keeps the task verdict beside `reward`, and
`mean/task_and_if` still counts the task verdict. Re-verification applies the server's current settings to the task
verdict saved with the episode.

For example, in a training run's config, on top of the example stack or the benchmark (or as overrides such as
`++agentic_if_resources.resources_servers.agentic_if.reward_mode=task_and_if`):

```yaml
agentic_if_resources:
  resources_servers:
    agentic_if:
      reward_mode: task_and_if
      cut_run_reward: zero  # optional
```

## Feature examples

`data/feature_examples.jsonl` holds 4 public rows that, together, exercise every opt-in feature: the check forms,
plants, continuation fields and tool renaming above, and a row whose outcome is not tracked. They are not part of the
`example` split. Each row's `agentic_if.provenance` names its example and lists, as a label only, what it exercises.
`tests/test_feature_examples.py` checks that they stay sealed, compile, grade their scripted sessions and still cover
every feature.

| Row (`feature_example`) | Task | Features |
| --- | --- | --- |
| `rename-tools-and-replace-plant` | Terminal-Bench 2.1 `cobol-modernization` | `tool_names` aliases, with a description per tool and a system message; a `replace` plant; an `after_event` `then` with `calls_include` (the arg spec `{field: command, writes: P, by: shell}`) and `calls_avoid`; `unquoted: true`; `parts: true` with `as_written: true`; length in `characters` |
| `append-plant-and-final-message` | Terminal-Bench 2.1 `raman-fitting` | an `append` plant followed by `then: no_calls`; `element_absent` `{end_before_call: P}` and `{element: any_code_block}`; a list of length bounds; a judge rule with `evidence: true` |
| `index-relative-continuation` | Scale-SWE `cthoyt_pystow_pr43` | a continuation with `workspace_diff_base: index` and a non-empty prefix diff; `calls` `mode: alone` with `writes_file`, which reads the change marks after the restored prefix; a judge rule from the continuation with `whole_run: true`; `marks: code_reference` with `mark_parts`; the ending `closing_sentence` |
| `status-only-continuation` | Terminal-Bench 2.1 `fix-git` | a continuation with `replay_timeouts: "mismatch"` and `unreceipted_rejected_calls`; `outcome_tracked: false`; `then: no_calls` from the continuation; the ending `{last_line: X}` |

The continuation prefixes are synthetic, in the form `opencode_sandboxed_agent` recorded: an OpenCode export with
per-call workspace diffs, and shell commands that seed replays.

To run them with the training reward, save an overlay that replaces the example dataset, for example as
`feature_examples.yaml`:

```yaml
config_paths:
  - resources_servers/agentic_if/configs/agentic_if.yaml

agentic_if_resources:
  resources_servers:
    agentic_if:
      reward_mode: task_and_if
      datasets:
        - name: feature_examples
          type: example
          jsonl_fpath: resources_servers/agentic_if/data/feature_examples.jsonl
          taskset: agentic_if:example
          num_repeats: 1
          license: TBD
```

Then collate and collect as in Run the examples, with the overlay in place of `--resources-server agentic_if`:

```bash
python benchmarks/terminal_bench_2_1/prepare.py
gym dataset collate --config feature_examples.yaml --mode example_validation \
    --output-dir results/agentic_if_feature_data
gym env start --config feature_examples.yaml --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml +use_absolute_ip=true
# In a second shell, once the servers are ready:
gym eval run --no-serve --config feature_examples.yaml --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml +use_absolute_ip=true \
    --input results/agentic_if_feature_data/example.jsonl --output results/agentic_if_feature_examples.jsonl
```

Each result's `reward` is then the task verdict times the run's IF score, `status-only-continuation` earns its IF
score alone, and `task_reward` keeps the task verdict. Collation keeps the dataset's metrics in
`data/feature_examples_metrics.json`.

# Licensing information

Code: Apache 2.0
Data: the instructions, system prompts and recorded prefixes of the example and feature-example rows are synthetic
(Apache 2.0). Their tasks come from Terminal-Bench 2.1 (Apache 2.0) and Scale-SWE (no declared license, as in
`scale_swe`).

Dependencies
- nemo_gym: Apache 2.0

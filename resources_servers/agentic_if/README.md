# Description

Measures whether a coding agent follows instructions while it completes a real repository (Scale-SWE) or terminal
(Terminal-Bench 2.1) task in OpenCode. This resources server owns each episode under `single_agent_turn`:

1. **Seed.** It checks the sealed row, borrows the task's sandbox through the `scale_swe` or `terminal_bench_2_1`
   typed session, checks the agent user and working directory, and stages the row's workspace files. For a row that
   continues a recorded session, it also restores that session's workspace and stages its export. It then stages the
   OpenCode plugins in `opencode_bundle/`, and any `opencode_runtime_files`, and lends the sandbox to `opencode_agent`.
2. **Agent turn.** `opencode_agent` runs OpenCode in that sandbox with the plugins. A profile plugin applies the row's
   tool argument maps, read-before-edit guards and planted tool failures, and writes receipts. A boundary plugin
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

`data/example.jsonl` holds 5 public examples, 3 terminal and 2 repository tasks, each with a synthetic system prompt
and instruction checks that exercise the features the benchmark rows use.

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

- `reward`: the task's own verifier result. Instruction following does not change it.
- `mean/if_all`: the share of graded rollouts that pass every applicable check.
- `mean/if_frac`: the share of applicable checks passed, averaged over graded rollouts.
- `mean/task_and_if`: the share of graded rollouts that pass both the task and every check.
- `if/graded`, `if/not_applicable` and `if/error` count rollouts; only graded rollouts enter the rates.
- `judge_calls` and `judge_usage`: each judge call's reply ID and token usage, and their totals. The judge's calls are
  not among the rollout's captured model calls, which belong to the agent.

An episode whose evidence could not be collected, or whose task verification failed for an infrastructure reason, is
masked (`mask_sample: true`, with `failure_kind` and `failure_reason`). An OpenCode run that times out or fills its
context window is scored. A check whose judge call failed on every attempt is recorded as `judge_failed`;
`gym eval reverify --judge-failed-only` grades it again from the saved episode, without the sandbox.

# Licensing information

Code: Apache 2.0
Data: the example instructions and system prompts are synthetic (Apache 2.0). Their tasks come from Terminal-Bench 2.1
(Apache 2.0) and Scale-SWE (no declared license, as in `scale_swe`).

Dependencies
- nemo_gym: Apache 2.0

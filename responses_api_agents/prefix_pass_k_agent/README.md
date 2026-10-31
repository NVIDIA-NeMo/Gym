# prefix_pass_k_agent

Agent for **prefix pass@K**: how often a candidate model finishes a task when it takes over at the decisive turn T of a
captured agent trajectory.

For each rollout the agent:

1. attaches to the task sandbox the resources server created (`/seed_session` returns its handle; `sandbox_provider`
   names a provider config such as `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`);
2. replays the trajectory's turns 1..T-1: the assistant turns verbatim, each command re-executed so the repository
   reaches the state the trajectory was in;
3. hands over to the candidate model for up to `max_forwards` turns;
4. returns, and the resources server grades the result with the benchmark's own verifier.

K rollouts of a row (`num_repeats`) give its mean reward (pass@1) and pass@K. The loop reimplements mini-swe-agent's,
on either of its wires: `backticks` or `function_calling` (see `responses_api_models/base_model`). The observations it
renders mirror the captured harness exactly, because the candidate reads them.

## Configuration

| field | default | notes |
|---|---|---|
| `wire` | `backticks` | must match the wire the trajectories were captured on |
| `max_forwards` | 3 | candidate turns after the handover |
| `step_timeout` | 60 | seconds per command |
| `interpreter` | none | e.g. `[bash, -c]`: run each command in a fresh interpreter, as mini-swe-agent does, so `BASH_ENV` is sourced |
| `exec_env` | UTF-8 locale | variables exported before every command |
| `commit_worktree` | false | commit the working tree before grading, for verifiers that grade committed work only (DeepSWE) |
| `record_git_state` | false | record whether HEAD moved and how many files are staged or untracked before grading |
| `uncommit_worktree` | false | before grading, fold commits, staged edits and new files back into unstaged edits, for verifiers that grade `git diff` (SWE-bench Verified) |
| `offline` | false | before the replay, make hostnames unresolvable in the sandbox, for traces captured with no network (SWE-bench Pro); grading runs in the resources server's own sandbox and stays online |
| `replay_through_target` | false | gold check, below |

## Validating a pool

Both checks take no model turns, so run them at `num_repeats` 1:

- **gold check** (`replay_through_target: true`): replays turns 1..T; every row must resolve.
- **contamination screen** (`max_forwards: 0`): replays turns 1..T-1; no row may resolve.

A pass@K of 0 is more often a broken replay than a weak model.

## Per-rollout diagnostics

Every rollout row also carries `forwards`, `unparsed_forwards`, `truncated_forwards` (unparsed forwards cut off at
`max_tokens`), `context_overflow`, `exec_errors` (a command the sandbox failed to run at all; it aborts the attempt and masks the sample),
`submitted`, and `prefix_observations_compared` / `prefix_observations_identical` (replay fidelity against the captured
observations).

Two fields make a rollout re-gradable without the model:

- `worktree_patch`: everything the rollout changed since its starting commit (commits, staged and unstaged edits, new
  files), as one binary patch taken before any grading-side git steps. A verifier change that only alters how the
  patch is collected can be re-run on this patch alone.
- `candidate_turns`: every forward's raw model output, unparseable ones included. With the row's prefix, the whole
  rollout replays in a fresh sandbox, for verifiers that need the environment rather than a patch.

## Usage

See `benchmarks/prefix_pass_k/README.md`: `verified.yaml` (SWE-bench Verified, backticks) and `deepswe.yaml`
(DeepSWE, function calling).

## Tests

```bash
gym env test +entrypoint=responses_api_agents/prefix_pass_k_agent
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0

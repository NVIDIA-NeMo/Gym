# Description

Codex-style coding tools for a minimal headless coding agent: `simple_agent` plus this resources
server gives a model `exec_command`, `write_stdin`, `apply_patch`, and `update_plan` on a real git
working tree.

Tool names, parameters, descriptions, outputs, and error texts follow
[Codex](https://github.com/openai/codex) (`codex-rs`, commit `44fe510ce3`):

| Tool | Upstream source | Notes |
|---|---|---|
| `exec_command` | `core/src/unified_exec`, `handlers/shell_spec.rs` | `<shell> -lc <cmd>`, pipes or PTY (`tty`), 250–30000 ms yield (default 10000), 10000-token head/tail truncation, Codex's output header and environment (`TERM=dumb`, `PAGER=cat`, `CODEX_CI=1`, ...), at most 64 sessions. |
| `write_stdin` | same | Writes to PTY sessions; pipe sessions only accept `\x03` (interrupt). Empty polls wait 5–300 s but return as soon as the process exits. |
| `apply_patch` | `apply-patch/` (parser, fuzzy matching, errors), `handlers/apply_patch.rs` | JSON function variant (`{"input": patch}`) because the freeform grammar tool needs custom-tool support that Gym agents and model servers lack. Every hunk is verified before anything is written. An `apply_patch` command is also on the shell `PATH`. |
| `update_plan` | `handlers/plan.rs` | Returns `Plan updated`; the latest plan is included in the verify response. |

`apply_patch.py` is checked against recorded results of the upstream `apply_patch` binary
(`tests/fixtures/apply_patch_upstream.json`); see `tests/test_apply_patch.py` to compare with a
live build or re-record.

Deliberate differences from Codex: there is no sandbox or approval flow (approval parameters are
omitted from the specs); `apply_patch` rejects paths outside the workspace; commands do not see
server environment variables matching `*KEY*`, `*SECRET*`, `*TOKEN*`, or `NEMO_GYM_*`
(`env_exclude_patterns`); `exec_command` does not intercept `apply_patch` invocations, which run
the `apply_patch` command instead.

## Workspaces and verification

`seed_session` creates a workspace for the rollout: a detached `git worktree` of `repo_path` at
`base_ref` (default), or the repository itself with `isolation: in_place` (one session at a time).
Worktrees start from a commit, so uncommitted changes in the source tree are not included.

A `repo_path` outside any git repository is edited in place (as with `isolation: in_place`,
whatever `isolation` says) and `verify` returns an empty `diff`; `check_command` still decides the
reward. The server logs a warning when this happens.

`repo_path` is, in order: `verifier_metadata.repo_path` of the row, `+codex_tools_repo_path=...`,
the harness cwd (`+harness_cwd`, which `examples/simple_agent.py` sets to the directory it runs from),
and otherwise `.`, which resolves against the server's own directory (the Gym checkout).

`verify` stops the session's processes and returns:

- `diff`: tracked and untracked changes against `base_ref`, honouring `.gitignore` and leaving out
  Python bytecode; staged in a temporary index, so the repository's own index is untouched. Empty
  when `repo_path` is not in a git repository.
- `resolved`, `check_exit_code`, `check_output`: when a `check_command` is set, it runs in the
  workspace and exit status 0 gives `reward` 1.0. Without one, `reward` is 0.0 and `resolved` is null.
- `plan`, and `workspace` for `in_place` sessions or with `keep_workspace: true`.

Worktrees are removed after `verify` (unless `keep_workspace`), after `session_idle_timeout_s`
without tool calls, and at server shutdown.

Per-row settings go in `verifier_metadata` (`repo_path`, `base_ref`, `check_command`) and override
the server config.

## Security

Commands run **without a sandbox**, with the resources server's user, filesystem, network, and
`PATH` (including its virtualenv). A worktree protects the source checkout from edits through
relative paths only. Use a trusted model and repository.

## Headless flow

Run [`examples/simple_agent.py`](../../examples/simple_agent.py) from the repository the agent should
work on, with any Python environment that has NeMo Gym installed. It starts the servers, collects
the rollouts, and shuts everything down in one command:

```bash
# policy_base_url and policy_api_key come from env.yaml (see "Configure your model" in the top-level README)

cd /path/to/target/repo
python /path/to/Gym/examples/simple_agent.py \
    --model-type openai_model --model gpt-6-sol \
    --prompt "Fix the failing test in tests/test_parser.py" \
    --check-command "pytest -q tests/test_parser.py" \
    --output /tmp/rollouts.jsonl
```

The script serves this environment by default. For several tasks, write an input file with
`make_tasks.py` (one JSONL row per task: instructions, Codex environment context, prompt, and tool
specs) and pass it with `--input` instead of `--prompt`.

Gym's own files (venvs, cache, results) stay out of the target repository. See
[`examples/README.md`](../../examples/README.md) for the script's options. The same flow in two
steps is `gym env start` with the servers' flags and `+harness_cwd=$PWD` (or
`+codex_tools_repo_path=...`), then `gym eval run --no-serve -a codex_tools_simple_agent -i ... -o ...`.

Each output row holds the full transcript (`response.output`), `diff`, check results, and
`reward`. Apply a diff to a checkout with `git apply`.

`data/example.jsonl` (written by `make_tasks.py` with no `--prompt`) has two tasks against this
repository, so run it from the Gym checkout: a code-search question and a small implementation with
tests. With `gpt-6-sol`, both were solved (reward 1.0).

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0

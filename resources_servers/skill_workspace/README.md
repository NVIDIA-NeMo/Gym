# skill_workspace

Ephemeral per-session workspace with bash + file-read tools. Built for grading
[agentskills.io](https://agentskills.io/specification) skills inside NeMo Gym,
but usable by any rollout that needs a scoped filesystem + shell.

Each `/seed_session` call creates a fresh tmpdir, optionally copies the skill's
`scripts/` and `references/` (gated by per-request flags), copies the fixtures
referenced by the scenario, and returns an `env_id`. Subsequent `/run_bash` and
`/read_file` calls are scoped to that tmpdir. `/close` removes it.

`SKILL.md` is **never** copied into the workspace. The skill specification
belongs in the system prompt (when the caller decides it should) — putting it
on disk let a "without-skill" rollout `cat SKILL.md` its way to the skill body.
We learned this the hard way: pre-fix, 100% of without-skill rollouts ran
`ls && cat SKILL.md` on turn 1.

## Endpoints

- `POST /seed_session` — `{skill_path, scenario_id, files, with_references?, with_scripts?}` → `{env_id}`
  - `with_references` (default `True`) — copy `<skill>/references/` into the workspace
  - `with_scripts` (default `True`) — copy `<skill>/scripts/` into the workspace
  - Both flags independent: `False`/`False` is the "fully blind" control that
    the skill-eval harness uses for its `blind` cell.
- `POST /run_bash` — `{env_id, cmd, timeout_seconds?}` → `{stdout, stderr, exit_code, truncated, timed_out}`
- `POST /read_file` — `{env_id, path}` → `{content, truncated}`
- `POST /close` — `{env_id}` → `{message, success}`

Output is capped at `output_cap_bytes` (default 50 KB combined). Bash timeout is
clamped to `[1, bash_timeout_hard_cap_seconds]` (default hard cap 120 s).
Concurrent bash processes are bounded by `max_concurrent_bash` (default 16).

## Isolation model

`tempfile.mkdtemp` per session. Path access is validated via `resolve()` +
`relative_to(workspace)` — absolute paths and `..` escapes are rejected.

`/run_bash` subprocesses run with a minimal env (see `_build_sandbox_env` in
`app.py`). `PATH` is restricted to `/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`
— the host's NeMo Gym `.venv` is **not** inherited, so rollouts cannot see host
`ng_*` CLIs, Ray state, or HF/MLflow credentials that Uvicorn's parent env
might carry. Only `HOME`, `USER`, `LANG`, `LC_ALL`, `SHELL`, `TERM`, `TMPDIR`,
and a workspace-scoped `PWD` pass through.

This is **not a container**. Skill authors are assumed to be trusted NVIDIA
contributors. If that assumption changes, add a container layer.

# Licensing information
Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0

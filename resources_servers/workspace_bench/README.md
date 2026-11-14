# Workspace-Bench resources server

Gives each task a sandbox with its input files in `/workspace/input`. After the agent runs, a separate judge sandbox
runs the pinned upstream agent-as-a-judge on `/workspace/output`. Reward is passed rubrics over total rubrics.

- Inputs are snapshotted before the agent runs, so the judge compares against the original files.
- Judge API failures go to the failures file instead of scoring zero.
- Set `artifact_root` to save each trial. Passing a saved `artifact_id` to `/verify` regrades it without rerunning
  the agent.

## Sessions

An Environment Server (`single_agent_turn_legacy` for the flat benchmark rows) seeds an episode session with the row's
`task_id` and `task_dir`. The seed starts the task sandbox with the inputs in `/workspace/input`, freezes a copy of
them on this server, and lends the sandbox to the agent through `sandbox_access` with the `operate` scope and
`/workspace` as the working directory. The server owns the sandbox; the agent only borrows it.

- `/verify` archives the sandbox's `input` and `output`, stops the task sandbox, and judges the archive against the
  frozen inputs. An identical retried verify gets the first verdict back, or a new judgment of the same capture when
  judging did not finish. A verify for a closed or expired session is masked (`mask_sample: true`) instead of
  scoring 0. What the agent did to `input` and `output` is its result: a removed folder grades as no output, and a
  file still changing while it is archived is graded as archived. Only a failure of the sandbox or of the archive
  fails the verify.
- The judge sandbox is stopped like the task sandbox: a cancelled verify still stops it, and a failed stop is retried
  by the close or at shutdown.
- `/close_session` stops the sandbox if verify did not, drops the frozen inputs, and fences the session ID, so a late
  seed gets 409. A failed stop raises and keeps the sandbox for the retried close.
- A session that is never closed expires after the sandbox's `ttl_s`, and shutdown stops every sandbox still open.

`harness_agent`'s `/run` still seeds with the row and gets the sandbox descriptor back; a second seed of the same
session gets 409, and its verify stops the sandbox, even when it fails before collecting the files. Two answers differ
from before: a verify with no live sandbox is masked instead of answering 400, and a task without a usable
`metadata.json` gets 422 instead of 500. The session behavior comes from `SandboxSessionResourcesServer` in
`resources_servers/swebench/sandbox_sessions.py`.

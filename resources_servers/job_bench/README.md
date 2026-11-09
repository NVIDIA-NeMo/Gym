# JobBench resources server

Creates a sandbox per task, collects `/workspace/output` after the agent runs, and grades it with the vendored upstream
Job-Bench judge (`vendor/judge.py`). Task reward is earned rubric weight divided by total rubric weight.

## Sessions

An Environment Server (`single_agent_turn_legacy` for the flat benchmark rows) seeds an episode session with the row's
`task_id`, `task_dir` and `rubrics_file`. The seed starts the task sandbox, unpacks the task folder into
`/workspace/task`, creates `/workspace/output`, and lends the sandbox to the agent through `sandbox_access` with the
`operate` scope and `/workspace` as the working directory. The server owns the sandbox; the agent only borrows it.

- `/verify` archives `/workspace/output`, stops the task sandbox, and grades the archive. An identical retried verify
  gets the first verdict back, or a regrade of the saved archive when grading did not finish. A verify for a closed
  or expired session is masked (`mask_sample: true`) instead of scoring 0. What the agent did to `/workspace/output`
  is its result: a removed folder grades as no output, and a file still changing while it is archived is graded as
  archived. Only a failure of the sandbox or of the archive fails the verify.
- `/close_session` stops the sandbox if verify did not, and fences the session ID, so a late seed gets 409. A failed
  stop raises and keeps the sandbox for the retried close.
- A session that is never closed expires after the sandbox's `ttl_s`, and shutdown stops every sandbox still open.

`harness_agent`'s `/run` still seeds with the row and gets the sandbox descriptor back; its verify stops the sandbox,
even when it fails before collecting the output. Two answers differ from before: a verify with no live sandbox is
masked instead of answering 400, and a task folder without `TASK_INSTRUCTIONS.txt` gets 422 instead of 500.
The session behavior comes from `SandboxSessionResourcesServer` in `resources_servers/swebench/sandbox_sessions.py`.

## Differences from upstream

Rubric prompts, weighted scoring, and the default judge (Grok 4.3) match upstream. Grading inputs differ in two ways,
so scores are not an exact reproduction of published numbers:

- The judge sees at most 120,000 bytes of extracted output text. Long files keep their beginning and end.
- `venv`, `.venv`, `node_modules`, `.git`, `.cache`, and `__pycache__` directories are not graded.

Judge API failures go to the failures file instead of being scored as zero.

Each judge call asks for 200,000 completion tokens, as upstream does. A judge served with a smaller
context window (for example a self-hosted 262k-context model) rejects every grading prompt longer than
the remainder; lower the budget with `judge_max_completion_tokens` (or `JOB_BENCH_JUDGE_MAX_COMPLETION_TOKENS`)
and keep it fixed across compared runs.

## Optional artifact retention

Set `artifact_root` to save each trial's request, rubrics, output archive, judge receipts, and result. Passing a saved
`artifact_id` to `/verify` regrades that trial without rerunning the agent.

# JobBench resources server

Creates a sandbox per task, collects `/workspace/output` after the agent runs, and grades it with the vendored upstream
Job-Bench judge (`vendor/judge.py`). Task reward is earned rubric weight divided by total rubric weight.

## Differences from upstream

Rubric prompts, weighted scoring, and the default judge (Grok 4.3) match upstream. Grading inputs differ in two ways,
so scores are not an exact reproduction of published numbers:

- The judge sees at most 120,000 bytes of extracted output text. Long files keep their beginning and end.
- `venv`, `.venv`, `node_modules`, `.git`, `.cache`, and `__pycache__` directories are not graded.

Judge API failures go to the failures file instead of being scored as zero.

## Optional artifact retention

Set `artifact_root` to save each trial's request, rubrics, output archive, judge receipts, and result. Passing a saved
`artifact_id` to `/verify` regrades that trial without rerunning the agent.

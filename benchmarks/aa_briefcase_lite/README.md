# AA-Briefcase-Lite

Public four-task example of Artificial Analysis's private AA-Briefcase evaluation.

This integration uses the existing NeMo Gym Stirrup wrapper. It intentionally omits web tools, requires an isolated execution provider, uses 500 turns, and exposes `finish` plus `abandon_task_finish`.

Prepare data from an explicitly pinned local checkout. A fresh user can create
that checkout and materialize its Git LFS objects with:

```bash
git clone https://huggingface.co/datasets/ArtificialAnalysis/AA-Briefcase-Lite /path/to/AA-Briefcase-Lite
git -C /path/to/AA-Briefcase-Lite checkout 4dec557b47d43867a1648c0974db1d8208c8b677
git -C /path/to/AA-Briefcase-Lite lfs pull
```

Then generate the four-row Gym input:

```bash
AA_BRIEFCASE_LITE_DATASET_DIR=/path/to/AA-Briefcase-Lite \
AA_BRIEFCASE_LITE_REVISION=4dec557b47d43867a1648c0974db1d8208c8b677 \
  python benchmarks/aa_briefcase_lite/prepare.py
```

The generated JSONL contains task execution metadata only. It deliberately excludes checks, rubrics, traceability records, source graphs, and judge prompts so grader-only information cannot enter the agent request.
It also records the checkout's absolute path, so
`data/aa_briefcase_lite.jsonl` is intentionally ignored and must be regenerated
for each installation. The preparation script rejects a different dataset
revision or missing referenced source files.

Set `AA_BRIEFCASE_CONTAINER_PATH` to an audited Apptainer image and
`PERSIST_DELIVERABLES_DIR` to an absolute shared-filesystem output path, then
run the normal `gym eval run --benchmark aa_briefcase_lite ...` command. The
configuration uses the shared Stirrup wrapper, a 500-turn limit, no web tool,
an Apptainer network namespace with no interfaces, an isolated writable
`/home/user`, and read-only `/home/user/shared` and `/home/user/week` inputs.

Generation is the safe default (`EXECUTE_ONLY=true`). The resources server can
subsequently judge cached artifacts with the 55 released binary checks and a
local, GDPval-style pairwise path for the eight AQ/P checks. Set
`EXECUTE_ONLY=false`, `JUDGE_ONLY=true`, and
`AA_BRIEFCASE_REWARD_MODE=binary|pairwise|all` for that second pass. Pairwise
uses configured public example submissions; AA has not released its production
pairwise prompt or private comparison graph, so pairwise and combined results
must be labeled local/unofficial.

AA-Briefcase-Lite is demonstrative and does not produce official AA-Briefcase Elo.

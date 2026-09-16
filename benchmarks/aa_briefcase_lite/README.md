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

Keep the dataset location available to both preparation and evaluation, then
generate the four-row Gym input through the normal benchmark command:

```bash
export AA_BRIEFCASE_LITE_DATASET_DIR=/path/to/AA-Briefcase-Lite
export AA_BRIEFCASE_LITE_REVISION=4dec557b47d43867a1648c0974db1d8208c8b677
gym eval prepare --benchmark aa_briefcase_lite
```

The generated JSONL contains task execution metadata only. It deliberately excludes checks, rubrics, traceability records, source graphs, and judge prompts so grader-only information cannot enter the agent request.
It also records the checkout's absolute path, so
`data/aa_briefcase_lite.jsonl` is intentionally ignored and must be regenerated
for each installation. The preparation script rejects a different dataset
revision or missing referenced source files.

Set `AA_BRIEFCASE_CONTAINER_PATH` to an audited Apptainer image and
`PERSIST_DELIVERABLES_DIR` to an absolute shared-filesystem output path. Configure
the policy credentials with Gym's normal `policy_api_key` setting, and set
`JUDGE_BASE_URL` and `JUDGE_API_KEY` for the judge endpoint. Then run:

```bash
gym eval run \
    --benchmark aa_briefcase_lite \
    --split benchmark \
    --model-type openai_model \
    --model YOUR_POLICY_MODEL \
    --model-url https://your-policy-endpoint/v1 \
    --output results/aa_briefcase_lite.jsonl
```

The configuration uses the shared Stirrup wrapper, a 500-turn limit, no web tool,
an Apptainer network namespace with no interfaces, an isolated writable
`/home/user`, and read-only `/home/user/shared` and `/home/user/week` inputs.

The default runs all four tasks and judges their deliverables with both the 55
released binary checks and the eight local analytical-quality/presentation
pairwise criteria (`AA_BRIEFCASE_REWARD_MODE` defaults to `all`, `execute_only: false`, `judge_only: false`).
Pairwise judging uses the public GPT-5.5 reference submission (`gpt-5-5`) with two
position-debiased trials per criterion. AA has not released its production
pairwise prompt or private comparison graph, so these pairwise and combined
results are local/unofficial.

For diagnostic runs, append native Gym YAML overrides to the run command and
choose a distinct `--output` path. To generate and persist deliverables without
judging:

```bash
++aa_briefcase_lite_stirrup_agent.responses_api_agents.stirrup_agent.execute_only=true
```

To judge existing deliverables without rerunning the policy, retain the same
`PERSIST_DELIVERABLES_DIR` and use:

```bash
++aa_briefcase_lite_stirrup_agent.responses_api_agents.stirrup_agent.execute_only=false \
++aa_briefcase_lite_stirrup_agent.responses_api_agents.stirrup_agent.judge_only=true \
++aa_briefcase_lite_stirrup_agent.responses_api_agents.stirrup_agent.rerun_incomplete=false
```

This rejudges the cached artifacts with both scoring paths. To isolate one path,
also append either
`++aa_briefcase_lite_resources_server.resources_servers.aa_briefcase_lite.reward_mode=binary`
or
`++aa_briefcase_lite_resources_server.resources_servers.aa_briefcase_lite.reward_mode=pairwise`.
Alternatively, set `EXECUTE_ONLY`, `JUDGE_ONLY`, or `AA_BRIEFCASE_REWARD_MODE`
environment variables. Explicit Gym config overrides take precedence.

Report `aa_lite/binary_pass_rate` and `aa_lite/pairwise_win_rate` separately,
alongside their counts and Gym's expected, scored, and missing rollout coverage.
Failed rollouts are recorded in the failures sidecar and may never reach the
resource aggregate, so `aa_lite/rows_total` alone does not establish completeness.
Invalid judge rows are excluded from aggregate scores. In `all` mode, an invalid
judgment in either path excludes the whole task from both aggregate scores.
The pairwise rate counts a tie
as half a win. In `all` mode, each task's scalar `reward` is the convenience mean
of its binary and pairwise scores; it is not an Elo calculation.

AA-Briefcase-Lite is demonstrative and does not produce official AA-Briefcase Elo.

Each judge-panel member uses its own model server because the `openai_model`
adapter fixes the upstream model. Override the three model names with
`JUDGE_GPT_MODEL`, `JUDGE_GEMINI_MODEL`, and `JUDGE_CLAUDE_MODEL` as needed.
Claude uses adaptive thinking with medium effort and a local 49,152-token output
limit in both binary and pairwise judging. Medium effort is a local deviation
from AA's published maximum-effort setting and is under validation.
GPT and Gemini inherit the local 4,096-token binary and 65,535-token pairwise
limits. AA's judge output budgets are not disclosed.

Binary judging uses the released AA prompts with an added JSON-format instruction
on the first call and up to two retries for invalid answers. Empty answers repeat
the same request; nonempty malformed answers receive a formatting reminder.
This is a local formatting
adaptation, not an exact reproduction of the unpublished AA judging harness.
SRT support lives in the shared GDPval file reader, so GDPval submissions that
contain SRT files also include their subtitle text in judge inputs.
LaTeX source and PDF content are shown separately; the harness does not compile
the source to verify that it produces the submitted PDF.

The 49,152-token Claude budget is under validation; it is not an AA-prescribed
limit or a demonstrated reliability fix.

For long judge requests, set `GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS=600` before
starting Gym. This controls the client timeout, not upstream service deadlines.
Binary requests permit two transport retries with SDK backoff. Pairwise requests
retain the shared sender's existing transport retry policy. Both paths
separately allow two retries for invalid answers;
pairwise formatting repair adds a short boxed-verdict reminder while preserving
the judge and submission positions. Valid judgments are not retried, and
exhausted requests remain evaluation errors. Binary token-limit completions are logged
without exposing response content.

# AA-Briefcase-Lite

Public four-task example of Artificial Analysis's private AA-Briefcase evaluation.

This integration runs the shared Stirrup agent in a sandbox that the resources server starts for each task. It
intentionally omits web tools, uses 500 turns, and exposes `finish` plus `abandon_task_finish`. The prepared rows carry
the rendered prompts and the two tool definitions. The resources server grades the files the agent submits with
`finish`; it also derives from the GDPVal server, so GDPVal's sessions, deliverable persistence and judge failure
handling apply.

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
It also records the checkout's revision, and
`data/aa_briefcase_lite.jsonl` is intentionally ignored and must be regenerated
for each installation. The preparation script rejects a different dataset
revision or missing referenced source files.

Set `AA_BRIEFCASE_SANDBOX_IMAGE` to an OCI image for AA's working environment and `PERSIST_DELIVERABLES_DIR` to an
absolute shared-filesystem output path. The image must run commands as the unprivileged `user` (UID 1000) with a
writable `/home/user`, and carry the document tooling AA's system prompt lists (LibreOffice, TeX Live, Python with
python-docx, python-pptx, openpyxl and PyMuPDF, and the others named there). The resources server uploads the checkout's
`source_files/shared` and `source_files/week` to `/home/user/shared` and `/home/user/week` at the start of each task.
AA's protocol runs without network access; the sandbox provider enforces that, so set the network policy of the sandbox
in `sandbox_config` for your provider. Configure
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

The configuration uses the shared Stirrup agent, a 500-turn limit and no web tool. The agent works in `/home/user`
with the shared and week inputs at `/home/user/shared` and `/home/user/week`.

The default runs all four tasks and judges their deliverables with both the 55
released binary checks and the eight local analytical-quality/presentation
pairwise criteria (`AA_BRIEFCASE_REWARD_MODE` defaults to `all`).
Pairwise judging uses the public GPT-5.5 reference submission (`gpt-5-5`) with two
position-debiased trials per criterion. AA has not released its production
pairwise prompt or private comparison graph, so these pairwise and combined
results are local/unofficial.

To generate and persist deliverables without judging, for example for a reference set, append
`++aa_briefcase_lite_resources_server.resources_servers.aa_briefcase_lite.judge_deliverables=false`. `/verify` still
persists each deliverable, then returns `reward: 0.0` and `mask_sample: true` with `failure_kind: gdpval:judging_disabled`.

To judge a finished run's persisted deliverables again, without the policy or a sandbox, use `gym eval reverify` on its
output, as described under "Deliverables and re-judging" in the GDPVal benchmark README. A judge failure is masked as
`judge_failed`, keeps the saved deliverable, and `gym eval reverify --judge-failed-only` judges it again. To isolate one
scoring path, append
`++aa_briefcase_lite_resources_server.resources_servers.aa_briefcase_lite.reward_mode=binary` or `...reward_mode=pairwise`,
or set `AA_BRIEFCASE_REWARD_MODE`. Explicit Gym config overrides take precedence.

Report `aa_lite/binary_pass_rate` and `aa_lite/pairwise_win_rate` separately (a rate is
omitted when its path did not run), alongside their counts and Gym's expected, scored, and missing rollout coverage.
Failed rollouts are recorded in the failures sidecar and may never reach the
resource aggregate, so `aa_lite/rows_total` alone does not establish completeness.
A judge that gives no usable answer to a check after the format retries is a judge
failure: the task is masked as `judge_failed` and `gym eval reverify --judge-failed-only`
judges it again, so the score never counts a check the judge did not answer. A failed
binary check masks the task. A pairwise reference whose judging failed, or that has no
submission for the task, is skipped and counted in `aa_lite/pairwise_references_skipped`;
the task is masked only when every reference failed (`judge_failed`) or none could be
judged (`verifier_error`). The pairwise rate counts a tie as half a win. In `all` mode, each task's scalar `reward` is the convenience mean
of its binary and pairwise scores; it is not an Elo calculation.

AA-Briefcase-Lite is demonstrative and does not produce official AA-Briefcase Elo.

Each judge-panel member uses its own model server because the `openai_model`
adapter fixes the upstream model. Override the three model names with
`JUDGE_GPT_MODEL`, `JUDGE_GEMINI_MODEL`, and `JUDGE_CLAUDE_MODEL` as needed.
Claude uses adaptive thinking with medium effort and a local 49,152-token output
limit in both binary and pairwise judging. Medium effort differs from AA's published
maximum-effort setting.
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

The 49,152-token Claude budget is a local setting, not an AA-prescribed limit.

For long judge requests, set `GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS=600` before
starting Gym. This controls the client timeout, not upstream service deadlines.
Binary and pairwise requests each permit two transport retries with SDK backoff; the
shared sender's own retry loop is off for pairwise requests, so attempts are not
multiplied. Both paths
separately allow two retries for invalid answers;
pairwise formatting repair adds a short boxed-verdict reminder while preserving
the judge and submission positions. Valid judgments are not retried, and
exhausted requests remain evaluation errors. Binary token-limit completions are logged
without exposing response content.

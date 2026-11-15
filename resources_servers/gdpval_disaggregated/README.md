# GDPVal disaggregated resources

This resources server stages each task in a sandbox, collects the OpenHands
submission, stops the sandbox, and judges the saved files in a separate process.
The rubric and judge credentials stay outside the agent workspace. It uses the
structured rubric scorer and media reader from `resources_servers/gdpval`.

Use it with `responses_api_agents/gdpval_sandboxed_agent/configs/training.yaml`
and the `legacy_agent` environment server supplied by that configuration.
`gdpval_train_resources` and `gdpval_val_resources` each require one worker because
their live sessions are process-local. Multiple named resources instances can
be selected by the agent's `resources_servers` configuration.

The resources server owns the task sandbox. The agent borrows its connection;
`/cleanup_session` and server shutdown retry failed releases. Completed verification
is cached within the live session, so a repeated request does not call the judge twice.

## Configuration and data

Configure a normal Gym judge model server named `gdpval_judge_model`, separate
from the policy model, and include its configuration when starting Gym. The
resources server uses `judge_model_server` to reach it; credentials and endpoint
selection belong in that model server's configuration. Older direct-judge
`judge_url`, `judge_model`, `JUDGE_API_KEY`, and endpoint-file settings are replaced
by this model server reference. `judge_create_params` forwards sampling and
reasoning options to the shared scorer.

Set these deployment values:

- `GDPVAL_SANDBOX_IMAGE`: a task image with the OpenHands runtime prerequisites and
  document tools described by the agent README.
- `GDPVAL_ARTIFACT_ROOT`: a writable directory for per-rollout artifacts and diagnostics.
- `GDPVAL_TRAIN_DATASET_PATH` / `GDPVAL_VAL_DATASET_PATH`: materialized task JSONL;
  the defaults use the five small synthetic examples in `data/tasks.jsonl`.
- `GDPVAL_DATA_ROOT`: a directory with `manifest.json` and
  `references/<reference_root_key>/<reference_file>`. The default is `data/`.

Each task row carries `task_id`, `prompt`, `reference_root_key`, `reference_files`,
and `rubric_json` or `rubric_pretty`. The Gym input row supplies the same task ID
in `responses_create_params.metadata.task_id`. `reference_files` may be a list
or a JSON-encoded list. Only the references and rendered task prompt reach
`/workspace`; the rubric stays with the resources server. The committed data is
synthetic and licensed under Apache-2.0; supply appropriately licensed GDPVal data
for actual training or evaluation.

`data/example_rollouts.jsonl` records five actual CodeAct/Docker integration runs
with deterministic `fixture-policy` and `fixture-judge` HTTP responses. They exercise
tool execution, artifact collection, rubric parsing, and cleanup. Their scores are
fixture expectations, not model-quality or benchmark measurements.

Startup uses the shared GDPVal helper to check and automatically provision
LibreOffice and its Java runtime on Linux. Set `libreoffice_command` to use a
container/wrapper instead; startup validates it with conversion probes in the
temporary directory and `artifact_root`, and refuses an unusable wrapper. If
default provisioning is unavailable, the server warns and continues accepting
text tasks; Office submissions are masked as `artifact_processing_failed`.
The judge defaults
to `images_and_text` with 144 DPI and a 120-page limit. Set its audio/video
capabilities explicitly with `judge_handles_audio` and `judge_handles_video`.

## Rewards and failures

A completed submission uses one structured rubric trial and two formatting
attempts. Finished empty submissions and known policy limits produce a valid zero
by default; `mask_policy_failures` can explicitly mask named policy failures.
Unreadable saved files are reported as `artifact_processing_failed`; unavailable
or invalid judges are `judge_unavailable`. Saved deliverables remain available
for inspection in either case. Unfinished agents and inconclusive measurements return
`mask_sample: true`, `reward_valid: false`, and `evaluation_completed: false`.
`agent_finished` is preserved for training masks, as are any compaction `responses`.
Queueing, retry delays, and judge attempts share one `judge_timeout` budget.

Artifacts include the staged task, original trajectory, deliverable manifest,
judge diagnostics, reward, and final rollout. Nested submission directories are
read with their relative directory labels. Unsafe paths and symlinks are rejected.
Extraction is bounded to 10,000 archive entries and 1 GiB; submissions are limited
to 200 deliverable files. Exceeding these limits masks the measurement.

This port uses current Gym media conversion instead of the former custom judge
transport. That changes media conversion details and removes its model-name-based
sampling overrides and archive-member rendering. Configure judge sampling explicitly.
Benchmark score parity and full training qualification have not been established.

## Tests

Run:

```bash
python -m pytest resources_servers/gdpval_disaggregated/tests
```

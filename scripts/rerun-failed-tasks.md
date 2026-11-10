# Rerun selected failures with their original benchmarks

`rerun_failed_tasks.py` selects original **task input rows** and invokes their original NeMo Gym agents and configured graders. It delegates tool execution, reference handling, grading, and model calls to those existing integrations. It does not construct prompts, generate rubrics, or implement a replacement scorer.

Use a private original dataset and the original benchmark checkout, configs, reference assets, and scorer services. A saved rollout is a baseline result: it can omit task fields required by the original grader and is not a substitute for the original task dataset. If those dependencies cannot be recovered, preparation must remain blocked until they are supplied. Pinning an available checkout does not establish that it was the historical evaluation version.

## Prepare the selection

Keep selection files, datasets, configuration containing credentials, and all generated artifacts outside Git. The utility rejects an output directory inside a Git worktree and never overwrites an existing output directory.

The selection JSONL contains one row per failed **attempt**, including repeated attempts at the same task:

```json
{"benchmark":"fictional_workflow","orig_trajectory_id":"failed-attempt-a","orig_task_index":7,"orig_rollout_index":2,"dataset_row_index":7,"expected_task_sha256":"<canonical-task-row-sha256>"}
```

`dataset_row_index` is the zero-based position in the original task dataset, not an index into saved rollouts. Its canonical digest is produced by the script's `canonical_sha256`: UTF-8 JSON with sorted keys, compact separators, `ensure_ascii=False`, and no nonfinite numbers. Verify this binding against original run provenance before preparing a selection. Original failure verdicts are selection evidence, not reference answers for fresh outputs.

The private routes JSON declares each original dataset and agent/configuration:

```json
{
  "schema_version": "original-task-rerun/routes/v1",
  "gym_revision": "<full-40-character-original-checkout-commit>",
  "policy_server_names": ["policy_model"],
  "benchmarks": {
    "fictional_workflow": {
      "dataset_path": "/private/original/tasks.jsonl",
      "dataset_sha256": "<original-dataset-file-sha256>",
      "config_paths": ["/private/original/benchmark.yaml"],
      "agent_name": "original_workflow_agent",
      "required_files": [{"path": "/private/original/evidence.txt", "sha256": "<original-asset-file-sha256>"}]
    }
  }
}
```

Paths may be absolute or relative to the routes file. A benchmark may specify its own `gym_revision`, but all jobs in one execution must match the supplied checkout. Config paths are merged in their listed order. Include any explicit, provenance-supported restorations of original runtime settings there, such as original sampling settings or authentic asset locations; record their origin separately. Do not substitute a different grader or infer missing historical settings. Original tracked code and imported configs are bound to the clean checkout revision; listed config files, datasets, and optional `required_files` assets are additionally bound to file hashes.

If an original integration modifies its local task files during startup, declare `runtime_assets` in its route: `[{"source_dir":"/private/original/tasks","config_key":"original_resource.resources_servers.original_backend.tasks_dir"}]`. Preparation pins the complete source tree, including file modes. Preflight reads that immutable source only. Execution copies it into a fresh per-job directory and changes the declared asset-location setting to the copy, so native setup can modify its own files without changing the reference source. Symlinked or special-file asset trees require a separate explicit integration.

```bash
python scripts/rerun_failed_tasks.py prepare \
  --selection /private/selection.jsonl \
  --routes /private/routes.json \
  --output-dir /private/prepared
```

Preparation performs no model calls. It checks the complete selection before writing, preserves all fields of each original task row, and emits a separate input JSONL per benchmark. It adds original-source identifiers, records and replaces Gym's two rollout bookkeeping indices, and supplies the declared original agent only when the task row has no agent reference. Conflicting identities or agent routes fail validation.

Each selected failed attempt becomes one input row. Repeated attempts share a stable task index and receive distinct rollout indices. Runs use `--num-repeats 1`; applying task-ID deduplication or another repeat multiplier would change the intended attempt count. The mapping sidecars preserve original and new indices even if an original verifier drops extra source metadata.

## Plan or execute the original integrations

Provide the checkpoint as a JSON configuration overlay, which is also valid YAML. Only explicitly allowlisted policy-server top-level keys are accepted. Prompt, tool, agent, and grader overrides are rejected. The file contents, including credentials, are never printed or copied into the run manifest.

```json
{
  "policy_model": {
    "responses_api_models": {
      "vllm_model": {
        "entrypoint": "app.py",
        "base_url": "http://localhost:8000/v1",
        "api_key": "unused",
        "model": "new-checkpoint"
      }
    }
  }
}
```

The backend and model-server configuration must be compatible with the original integrations. Their existing judge or reward-model services still need their original configuration and credentials. The utility does not create a judge or silently replace an unavailable service.

```bash
python scripts/rerun_failed_tasks.py run \
  --manifest /private/prepared/prepared_manifest.json \
  --gym-root /private/original-gym-checkout \
  --policy-config /private/checkpoint.json \
  --python /private/original-gym-checkout/.venv/bin/python \
  --output-dir /private/new-checkpoint-run
```

This prints a plan and performs local checks only. Add `--execute` to start the original benchmark processes. Before starting any servers, every job runs an offline worker preflight using the supplied Python: it verifies imports come from the pinned checkout, resolves original config/server references, and runs the original collector's subset preprocessor to check all selected attempt indices and counts. Dataset, row, input/mapping, config, asset, and optional `env.yaml` hashes are checked too. The target policy overlay takes precedence over the original environment's target model settings. Original credentials remain functional; config contents are not printed in receipts.

Execution calls the original `RunHelper.start`, `RolloutCollectionHelper.run_from_config`, and `RunHelper.shutdown` APIs directly. This preserves the explicit selected input file: some versions of the full `gym eval run` pipeline replace `--input` by collating configured datasets. This utility never invokes that full-dataset preparation path. Original tool caps and scorer settings are preserved. `--concurrency` controls per-job request concurrency; jobs execute sequentially to avoid conflicting benchmark server names. W&B uploads are disabled. Original custom collection drivers require a separate explicit integration and are rejected rather than bypassed.

Private stdout/stderr logs and `run_manifest.json` record execution. A failed preflight prevents every benchmark server from starting; a process failure stops subsequent jobs and leaves remaining jobs planned. The command returns a nonzero exit code for an unsuccessful explicit execution. Use a fresh output directory for another execution; no inference is triggered by preparation, planning, or summarization.

## Read the original results

```bash
python scripts/rerun_failed_tasks.py summarize \
  --run-manifest /private/new-checkpoint-run/run_manifest.json \
  --output-dir /private/new-checkpoint-summary
```

`matched_results.jsonl` joins each saved source identity to the complete, unchanged `original_gym_result`, including its original reward, masks, grader diagnostics, and errors. The summary reports missing, duplicate, unrecognized, masked, and error results separately. Raw recorded zero rewards remain distinguishable from zeros in usable graded results. `comparison_ready` is false for incomplete or technically unsuccessful collections; missing or masked outcomes are never filled with zero. This is metadata accounting, not a new grading pass. An original integration may itself report technical errors as zero; the original output remains intact for inspection.

This helper requires the original server/collection APIs and caller-supplied `_ng_task_index` support in the pinned runtime. It does not convert unsupported runtime APIs or claim historical grader equivalence when the original run configuration or service version is unavailable.

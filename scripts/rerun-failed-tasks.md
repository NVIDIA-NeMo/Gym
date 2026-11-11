# Rerun selected failures with their original benchmarks

`rerun_failed_tasks.py` selects original **task input rows** and invokes their original NeMo Gym agents and configured graders. It delegates tool execution, reference handling, grading, and model calls to those existing integrations. It does not construct prompts, generate rubrics, or implement a replacement scorer.

Use the original benchmark checkout, registrations, reference assets, and scorer services. Original datasets can be downloaded from their existing registrations and cached locally; no second copy needs to be uploaded or registered. A saved rollout is a baseline result: it can omit task fields required by the original grader and is not a substitute for the original task dataset. If those dependencies cannot be recovered, preparation must remain blocked until they are supplied. Pinning an available checkout does not establish that it was the historical evaluation version.

## Prepare the selection

Reviewed metadata-only selection indices may be committed, including the LHA selection below. Keep task datasets, raw rollouts, configuration containing credentials, and all generated artifacts outside Git. The utility rejects an output directory inside a Git worktree and never overwrites an existing output directory.

The selection JSONL contains one row per selected **attempt**, including repeated attempts at the same task:

```json
{"benchmark":"fictional_workflow","orig_trajectory_id":"failed-attempt-a","orig_task_index":7,"orig_rollout_index":2,"dataset_row_index":7,"expected_task_sha256":"<canonical-task-row-sha256>"}
```

`dataset_row_index` is the zero-based position in the original task dataset, not an index into saved rollouts. Its canonical digest is produced by the script's `canonical_sha256`: UTF-8 JSON with sorted keys, compact separators, `ensure_ascii=False`, and no nonfinite numbers. Verify this binding against original run provenance before preparing a selection. Original failure verdicts are selection evidence, not reference answers for fresh outputs.

## Included LHA Agent-IF selection

[`selections/lha-agentic-if-attempts.jsonl`](selections/lha-agentic-if-attempts.jsonl) contains all **1,494 recorded original attempts across 558 tasks** selected from the `nemotron3.5-super-vl-ga` rollouts. The original 988 zero-reward Agent-IF failure traces are included; the expansion adds the other recorded attempts for the same tasks. The expanded selection contains **1,174 zero-reward attempts and 320 successful attempts**, with their original attempt indices preserved.

| Recorded attempts per task | Unique tasks | Selected attempts |
|---|---:|---:|
| 1 | 90 | 90 |
| 3 | 468 | 1,404 |
| **Total** | **558** | **1,494** |

The 90 single-attempt tasks belong to `big_law_bench_agentic`, `dassault_enterprise`, `dassault_scaled`, `legal_agent_bench`, `palantir_fde`, `perplexity_facts`, and `perplexity_frames`. Their source files record only attempt 0. These source files leave 180 historical slots unrecorded relative to 558 × 3 attempts; the selection includes only the existing original attempts.

[`selections/lha-agentic-if-manifest.json`](selections/lha-agentic-if-manifest.json) records the index SHA-256, benchmark counts, task/attempt distribution, and original reward totals. The index contains only benchmark names, original task/attempt/trajectory identifiers, dataset row positions, and task-row hashes. Original prompts, model responses, references, and credentials remain in their existing sources.

The commands below use this checked-in selection directly. Supply another reviewed index with `--selection` to choose a different cohort. Task datasets and required assets still download from their original registrations through the supplied routes catalog.

## Download existing registered datasets

Use `download_failed_task_sources.py` for any number of benchmark routes. Each route identifies its original agent, dataset declaration and registry version. The helper uses the original Gym registry reader and `TrainDataProcessor`; it does not fetch a separately published failure dataset. Selection identities come from a reviewed metadata-only index, such as the checked-in LHA selection above. Original task datasets do not contain the historical checkpoint's reward or failure-analysis labels, so those selection identities must come from its saved results.

A portable download catalog contains config paths relative to the original Gym checkout, rather than local original-dataset locations:

```json
{
  "schema_version": "original-task-download/routes/v1",
  "gym_revision": "<full-40-character-original-checkout-commit>",
  "benchmarks": {
    "fictional_workflow": {
      "config_paths": ["resources_servers/fictional_backend/configs/original.yaml"],
      "agent_name": "original_workflow_agent",
      "dataset_name": "original_validation",
      "dataset_type": "validation",
      "registry_identifier": {
        "backend": "gitlab",
        "dataset_name": "original_workflow",
        "version": "0.0.1",
        "artifact_fpath": "validation.jsonl"
      }
    }
  }
}
```

The original declaration must match this registry identifier. GitLab and Hugging Face identifiers are supported through their existing native backends. Registry credentials come from the original runtime's configured environment, are held only in memory, and are not copied into catalog/worker/receipt files. Data preparation projects only registry settings; it does not require the policy model or grader service configuration to resolve.

```bash
python scripts/download_failed_task_sources.py prepare \
  --selection scripts/selections/lha-agentic-if-attempts.jsonl \
  --routes /private/original-downloads.json \
  --gym-root /path/to/original-gym-checkout \
  --python /path/to/original-gym-checkout/.venv/bin/python \
  --output-dir /private/failed-task-workspace
```

This command performs original dataset downloads, validates and caches complete task rows, binds the selected attempts by their full row digests, and creates `/private/failed-task-workspace/prepared/prepared_manifest.json`. References, rubrics, tools and unknown fields are preserved. Every selected attempt remains represented, including repeated attempts at the same task; native dataset repetition is normalized to one before the selection index supplies its intended multiplicity. Repeating this command with identical bindings revalidates and reuses the completed cache without new downloads. A failed or incomplete transfer is never published as a valid cache.

Original auxiliary assets can be downloaded through the same catalog's optional `registry_assets` list. Each entry supplies `name`, `registry_identifier`, the original artifact `sha256`, and its native configuration location `config_key`. Regular file assets become hash-pinned original location overrides. For an archive, add `archive: {"format":"tar.gz","strip_prefix":"original_tasks","select_prefixes":["task-a"]}`. Only declared regular files/directories are extracted after the complete archive hash is verified. Set `runtime_copy: true` when the original integration modifies task mirrors; the existing runner then supplies a fresh writable copy for each job.

A route may include explicit `config_overrides` as a JSON object for provenance-supported original sampling settings or service-variable bindings. Binding writes these to a private, hash-pinned overlay; it does not invent endpoints, credentials or grading rules. Actual model execution remains a separate step:

```bash
python scripts/rerun_failed_tasks.py run \
  --manifest /private/failed-task-workspace/prepared/prepared_manifest.json \
  --gym-root /path/to/original-gym-checkout \
  --policy-config /private/checkpoint.json \
  --python /path/to/original-gym-checkout/.venv/bin/python \
  --output-dir /private/checkpoint-run
```

Add `--execute` to the run command only when the checkpoint and original scorer services are ready. Preparation never starts a policy, agent or grader, and makes no inference calls. Source data and generated subsets stay in the local workspace; no new registry upload occurs. The lower-level `download` command prints a plan by default (`--execute` fetches data only), and `bind` can attach a selection to an already downloaded source manifest.

## Prepare from existing local task files

The private routes JSON below is an optional alternative when original task files are already available. The download workflow generates this local binding automatically:

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

Each selected attempt becomes one input row. Repeated attempts share a stable task index and receive distinct rollout indices. Runs use `--num-repeats 1`; applying task-ID deduplication or another repeat multiplier would change the intended attempt count. The mapping sidecars preserve original and new indices even if an original verifier drops extra source metadata.

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

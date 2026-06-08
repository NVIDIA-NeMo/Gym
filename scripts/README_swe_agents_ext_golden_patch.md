# SWE-Bench-Ext Golden-Patch Evaluation Scripts

These scripts run the non-scheduler parts of SWE-Bench-Ext golden-patch
validation through `swe_agents`. They intentionally do not submit, monitor, or
cancel cluster jobs.

## 1. Build the input JSONL

```bash
python scripts/swe_bench_ext_tasks_to_jsonl.py \
  --output "$ARTIFACT_DIR/tasks.jsonl" \
  "$TASK_DIR"
```

`$TASK_DIR` may be a directory containing task subdirectories, or an individual
task directory. The generated rows default to `dataset_name=swe-bench-ext` and
`agent_ref.name=swe_agents`.

## 2. Create a smoke input

```bash
head -n 1 "$ARTIFACT_DIR/tasks.jsonl" > "$ARTIFACT_DIR/tasks.one.jsonl"
```

## 3. Run golden-patch validation

Call this from the environment your launcher provides:

```bash
bash scripts/run_swe_agents_ext_golden_patch.sh \
  --input-jsonl "$ARTIFACT_DIR/tasks.one.jsonl" \
  --output-jsonl "$RUN_DIR/swe_agents_golden_patch_result_one.jsonl" \
  --task-image-root "$TASK_IMAGE_ROOT" \
  --concurrency 1 \
  --apptainer-memory-limit-mb 65536 \
  --test-timeout-seconds 1200
```

For a full run, pass the full `tasks.jsonl` and set the concurrency appropriate
for the allocated environment.

The runner sets `verify_golden_patch=true`, starts `ng_run`, waits for the Gym
head server, then calls `ng_collect_rollouts`. No real policy model is used.

## 4. Summarize results

```bash
python scripts/swe_agents_golden_patch_summary.py \
  --output-jsonl "$RUN_DIR/swe_agents_golden_patch_result.jsonl" \
  --expected-count "$EXPECTED_TASKS" \
  --show-settings
```

For failed samples and their eval logs:

```bash
python scripts/swe_agents_golden_patch_summary.py \
  --output-jsonl "$RUN_DIR/swe_agents_golden_patch_result.jsonl" \
  --show-failures
```

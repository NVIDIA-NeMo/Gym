# Inflight Reward Profiling

Scope: using `ng_collect_rollouts +inflight_reward_profile=True` and validating that the final collection-written profile matches `ng_reward_profile`.

## Contract

Inflight reward profiling is optional checkout behavior. When supported, collection writes task-level profile rows while rollouts complete, then rewrites the final profile through the canonical `RewardProfiler.profile_from_data(...)` path.

Expected final files:

- `rollouts.jsonl`: one completed rollout row per materialized input.
- `rollouts_materialized_inputs.jsonl`: repeated/materialized inputs with `_ng_task_index` and `_ng_rollout_index`.
- `rollouts_reward_profiling.jsonl`: one profile row per original task, same final structure as `ng_reward_profile`.

The live file can be partial while collection is running. Treat only the completed final file as canonical.

## Core CLI Shape

```bash
ng_collect_rollouts \
    +agent_name="__AGENT_NAME__" \
    +input_jsonl_fpath="__DATA_JSONL__" \
    +output_jsonl_fpath="__ROLLOUTS_JSONL__" \
    +num_samples_in_parallel="__NUM_SAMPLES_IN_PARALLEL__" \
    +num_repeats="__NUM_REPEATS__" \
    +resume_from_cache=False \
    +inflight_reward_profile=True
```

Use `+inflight_reward_profile_fpath="__PROFILE_JSONL__"` only when the default path is not wanted. The default is `<rollouts_stem>_reward_profiling.jsonl`.

After collection, this should match the offline path:

```bash
ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="__MATERIALIZED_JSONL__" \
    ++rollouts_jsonl_fpath="__ROLLOUTS_JSONL__"
```

## Current StructEval Smoke Wrapper

Current runnable internal example:

```text
/lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github/temp_my/scripts/260416_structeval/run_inflight_reward_profile_external.sh
```

It runs StructEval nonrenderable collection against an already-running OpenAI-compatible model endpoint. It starts Gym servers with `ng_run`, calls `ng_collect_rollouts` with `+inflight_reward_profile=True`, and can optionally compare the final inflight profile against a fresh `ng_reward_profile` run.

Important env vars:

- `BASE_URL` or `MODEL_PATH`: policy endpoint, for example `http://host:port/v1`.
- `MODEL_NAME` or `SERVED_MODEL_NAME`: served policy model name.
- `API_KEY` or `MODEL_API_KEY`: policy API key; defaults to `dummy`.
- `OUTPUT_BASE_DIR`: base directory for saved rollouts and logs.
- `OUTPUT_DIR`: exact output directory; defaults to `${OUTPUT_BASE_DIR}/${MODEL_NAME}`.
- `NUM_REPEATS`: rollouts per task; default `1`.
- `NUM_SAMPLES_IN_PARALLEL`: in-flight request concurrency; default `256`.
- `LIMIT`: optional smoke-test row limit.
- `STARTUP_SLEEP`: seconds to wait after `ng_run`; default `60`.
- `RESUME_FROM_CACHE`: default `False`.
- `INFLIGHT_REWARD_PROFILE_FPATH`: optional explicit profile path.
- `RUN_LOG_FPATH`: default `${OUTPUT_DIR}/inflight_reward_profile_run.log`.
- `VERIFY_OFFLINE_PROFILE`: default `false`; when `true`, snapshot final inflight output, rerun `ng_reward_profile`, and `cmp` the files.

No-offline smoke run:

```bash
NUM_REPEATS=4 \
BASE_URL=http://nvl72065-T12:10240/v1 \
MODEL_NAME=nemotron-3-ultra-test \
API_KEY=dummy \
LIMIT=20 \
NUM_SAMPLES_IN_PARALLEL=16 \
STARTUP_SLEEP=60 \
VERIFY_OFFLINE_PROFILE=false \
OUTPUT_BASE_DIR=/lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github/rollouts/structeval_nonrenderable_260416_inflight_reward_profile_smoke_no_offline \
bash /lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github/temp_my/scripts/260416_structeval/run_inflight_reward_profile_external.sh
```

Validation run:

```bash
NUM_REPEATS=4 \
BASE_URL=http://nvl72065-T12:10240/v1 \
MODEL_NAME=nemotron-3-ultra-test \
API_KEY=dummy \
LIMIT=20 \
NUM_SAMPLES_IN_PARALLEL=16 \
STARTUP_SLEEP=60 \
VERIFY_OFFLINE_PROFILE=true \
OUTPUT_BASE_DIR=/lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github/rollouts/structeval_nonrenderable_260416_inflight_reward_profile_smoke \
bash /lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github/temp_my/scripts/260416_structeval/run_inflight_reward_profile_external.sh
```

For `LIMIT=20` and `NUM_REPEATS=4`, expect:

- `rollouts.jsonl`: 80 lines
- `rollouts_materialized_inputs.jsonl`: 80 lines
- `rollouts_reward_profiling.jsonl`: 20 lines
- each profile row: `num_rollouts == 4`

When `VERIFY_OFFLINE_PROFILE=true`, the wrapper writes `rollouts_reward_profiling.inflight_final_snapshot.jsonl` before rerunning `ng_reward_profile`. A successful run prints:

```text
Verified: inflight final profile matches ng_reward_profile output.
```

## Manual Comparison

If a run skipped offline verification, compare later from a clean temporary directory so the offline command cannot overwrite the original profile unexpectedly:

```bash
tmpdir="$(mktemp -d /tmp/nemo-gym-profile-compare.XXXXXX)"
cp "__ROLLOUTS_JSONL__" "$tmpdir/rollouts.jsonl"
cp "__MATERIALIZED_JSONL__" "$tmpdir/rollouts_materialized_inputs.jsonl"

ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$tmpdir/rollouts_materialized_inputs.jsonl" \
    ++rollouts_jsonl_fpath="$tmpdir/rollouts.jsonl"

cmp -s "__PROFILE_JSONL__" "$tmpdir/rollouts_reward_profiling.jsonl"
```

Use structural comparison rather than byte comparison when comparing separate model runs. Values can differ between runs, but key order, value types, row counts, task ordering, and `rollout_infos` structure should match for the same code path.

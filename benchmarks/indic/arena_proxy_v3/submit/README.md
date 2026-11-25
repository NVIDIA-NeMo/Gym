# Submitting Indic Arena Proxy v3

Use Gym's Slurm executor for both stages. `generate.yaml` serves the policy on
one eight-GPU node, runs the reasoning preflight, then collects answers with
verification deferred. `judge.yaml` requests one CPU node and runs
`gym eval reverify` after the generation job succeeds. Gym renders the batch
scripts and owns service health checks, cleanup, submission records, and logs.

The configs use Gym's supported benchmark `command` field to retain the preflight
and run reverification. Both commands explicitly use `--resume`; automatic
`resumable` chains and submission-level exporters do not apply to command jobs.
The Gym commands still write rollout stores and aggregate metrics normally.

## Deployment settings

Run from a clean Gym checkout containing this integration, using the standard
Gym installation. Publish its commit before submission so both drivers can install
the same source revision. Configure these variables for your deployment:

| Variable | Meaning |
| --- | --- |
| `SLURM_ACCOUNT` | Your Slurm account |
| `SHARED_ROOT` | Absolute shared directory containing the checkpoint, prepared inputs, run outputs, and judge-key file; mounted at the same path in both containers |
| `RUN_DIR` | A fresh absolute directory under `SHARED_ROOT` |
| `MODEL_PATH`, `MODEL_KEY` | Checkpoint directory and served model name |
| `IMAGE` | Readable local vLLM 0.26.0 `.sqsh` image, matching the reference run |
| `GYM_DRIVER_IMAGE` | Driver container supporting Gym's pinned installation and component dependencies; use the same image for both stages |
| `GYM_GIT_REPO`, `GYM_GIT_REF` | Repository URL containing this integration and its published full commit SHA |
| `JUDGE_KEY_FILE` | Your readable, owner-only API-key file on shared storage |

For the current checkout, obtain the pin with `git rev-parse HEAD`. Keep the
checkpoint, images, source pin, prepared inputs, and settings unchanged during a
run or retry. Gym installs the pinned lockfile's dependencies in each driver.
If the deployment requires additional mounts, configure both `driver.mounts`
and `services.policy.mounts` as appropriate.

The partition defaults are `interactive` and `cpu_interactive`; override with
`GPU_PARTITION` and `CPU_PARTITION`. Sampling defaults remain temperature 1.0,
top-p 0.95, top-k −1, and seed 42. `TEMPERATURE`, `TOP_P`, `TOP_K`, `CONTEXT`,
`MAX_OUTPUT_TOKENS`, `TP`, `DP`, `GPU_UTIL`, and `CONCURRENCY` retain their generation
overrides; `JUDGE_CONCURRENCY` defaults to 12. `ARENA_JUDGE_BASE_URL` and
`ARENA_JUDGE_MODEL` retain the judge overrides.

The former `EP`, `PARSER`, `TOOL_PARSER`, `EAGER`, and `MAX_BATCHED_TOKENS` shell
switches are now explicit flags in `services.policy.extra_args`. Edit a deployment
profile or use Hydra overrides; these old variables are no longer read.
`services.policy.port` replaces `VLLM_PORT`, and `health_check.timeout_seconds`
replaces the startup-attempt count. Review the resolved config before submitting.

## Prepare and submit

Prepare once into the run directory using the registry configuration described
in the benchmark README. The source prompts and reference answers remain separate.

```bash
set -euo pipefail
set +x
umask 077
mkdir -p "$RUN_DIR"
python -m benchmarks.indic.arena_proxy_v3.prepare \
  --output-dir "$RUN_DIR/inputs" --languages bn gu hi kn ml mr ne or pa ta te ur
```

After preparation, the workers need neither registry nor Hugging Face credentials.
Only the judge-key **path** is submitted; the CPU driver checks ownership and
permissions and reads its value at execution time.

```bash
unset ARENA_JUDGE_API_KEY MLFLOW_TRACKING_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN
GENERATION_CONFIG=benchmarks/indic/arena_proxy_v3/submit/generate.yaml
gym eval submit --config "$GENERATION_CONFIG" --resolve-only
gym eval submit --config "$GENERATION_CONFIG" --dry-run
gym eval submit --config "$GENERATION_CONFIG" --json > "$RUN_DIR/generation-submission.json"

GENERATION_JOB_ID=$(python -c '
import json, sys
job = json.load(open(sys.argv[1]))["benchmarks"][0]
assert job["job_id"] and job["job_id"].isdecimal(), job.get("error")
print(job["job_id"])
' "$RUN_DIR/generation-submission.json")
GENERATION_DIR=$(python -c '
import json, sys
print(json.load(open(sys.argv[1]))["benchmarks"][0]["job_dir"])
' "$RUN_DIR/generation-submission.json")
export GENERATION_JOB_ID GENERATION_DIR

gym eval submit --config benchmarks/indic/arena_proxy_v3/submit/judge.yaml --dry-run
gym eval submit --config benchmarks/indic/arena_proxy_v3/submit/judge.yaml \
  --json > "$RUN_DIR/judge-submission.json"
JUDGE_DIR=$(python -c '
import json, sys
job = json.load(open(sys.argv[1]))["benchmarks"][0]
assert job["job_id"], job.get("error")
print(job["job_dir"])
' "$RUN_DIR/judge-submission.json")
export JUDGE_DIR
```

The judge config records `afterok:GENERATION_JOB_ID`, requests no GPU or policy
service, and cancels an unsatisfiable dependency if generation fails. It reads the
saved answers and materialized inputs from `GENERATION_DIR`. It also uses the
original prepared population and adjacent report for coverage and metrics.

## Outputs and retries

Each submission writes `gym-job.json`, `resolved-config.yaml`, generated job
scripts, and logs under `RUN_DIR/jobs`. `GENERATION_DIR/artifacts` contains
`rollouts.jsonl` and `rollouts_materialized_inputs.jsonl`. `JUDGE_DIR/artifacts`
contains the judged `rollouts.jsonl`, `rollouts_aggregate_metrics.json`, and the
formatted `metrics.json`. The benchmark's metric names and scoring are unchanged.

For an unchanged run, `gym eval resubmit GYM_JOB_ID generation` or
`gym eval resubmit GYM_JOB_ID judging` requeues the recorded command in the same
artifact directory. Use each submission record's `gym_job_id`, not its Slurm ID.
The explicit `--resume` retains completed outcomes. Generation resubmission gets
a new Slurm ID: cancel the obsolete dependent judge job and submit `judge.yaml`
again with that new `GENERATION_JOB_ID` and the same `GENERATION_DIR`. A judge-only
retry reuses the original successful generation. Use a fresh run directory when
inputs, checkpoints, or evaluation settings change.

The former `slurm/submit.sh`, `.sbatch` scripts, and `common.sh` are replaced by
these configs. For existing runs, the standard `gym eval reverify` command accepts
their saved rollout and materialized-input paths directly; set `INPUT_JSONL` to
that run's original prepared file so its metrics use the correct population.

## Five-sample Gemma profile

For `google/gemma-4-31B-it`, prepare a fresh run with
`--languages hi bn ta te ur --limit 1`, then use `submit/gemma31b.yaml` as
`GENERATION_CONFIG` in the commands above and set `JUDGE_CONCURRENCY=5`.
The profile fixes context 131,072, top-k 64, TP 8 / DP 1, Gemma reasoning and tool
parsers, and no expert parallelism. It keeps the same temperature, top-p, seed,
and 120,000-token output cap. These five rows are translations of one conversation;
they cannot produce the official Indic headline.

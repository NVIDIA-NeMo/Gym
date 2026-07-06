# BiomniBench-DA environment

[BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) data-analysis
tasks, run through the [Harbor Agent](../../responses_api_agents/harbor_agent) bridge
(Harbor manages the sandboxed environment + verifier; the Harbor agent bridges that to
NeMo Gym). Materialized task trees live under `data/` (gitignored — see `prepare.py`).

Each task gives the agent a data-analysis question and a data directory; the agent
writes `trace.md` (its analysis) and `answer.txt` (its final answer) inside the
container, and an OpenAI-compatible LLM judge scores the trace/answer against a
per-task rubric (upstream-faithful scoring, see `prepare.py`'s embedded
`llm_judge.py`).

Use Gym's venv from the repo root for all commands below.

## 1) Download and materialize the five-task example tree (docker profile)

The checked-in example set uses five small BiomniBench-DA tasks:
`da-1-3`, `da-1-4`, `da-10-1`, `da-10-3`, and `da-11-1`.
These tasks include singleton or otherwise uncovered task types, so a bare
materialization selects **zero** tasks. Pass `--include-singletons --include-uncovered` to keep them.

`prepare.py` downloads the upstream dataset from HuggingFace (`--download`, skipped
automatically if `--local-dir` already has data), then materializes Harbor task dirs
under `--output-dir` and writes `rollout_input.jsonl` there — the
`ng_collect_rollouts` input file (one row per task, `instance_id` form
`biomnibench_da::<task_name>`):

```bash
python environments/biomnibench_da/prepare.py \
  --download \
  --environment-type docker \
  --tasks da-1-3 da-1-4 da-10-1 da-10-3 da-11-1 \
  --include-singletons --include-uncovered \
  --output-dir environments/biomnibench_da/data/example \
  --overwrite
# -> data/example/rollout_input.jsonl  (5 rows, one per task)
```

Override the rollout-input path with `--rollout-input-fpath` if needed.

For HPC, use `--environment-type singularity` and
`--output-dir environments/biomnibench_da/data/example_singularity` instead.

See `python environments/biomnibench_da/prepare.py --help` for the full flag set
(train/test split controls, `--limit`, `--papers`, `--max-data-mb`, `--n-repeats`,
`--judge-model`, `--docker-image`, etc.) — the full (non-smoke) dataset is prepared
the same way, just without `--tasks`/`--include-singletons`/`--include-uncovered`.

## 2) Build (or verify) the shared runtime image

Docker profile tasks reference a prebuilt image (`[environment].docker_image` in each
`task.toml`), not a per-task Dockerfile build. The generated
`environment/docker-compose.yaml` will fail immediately if this image isn't present
locally, so build/pull it once up front:

```bash
bash environments/biomnibench_da/docker/build_biomnibench_runtime_image.sh
# Sanity check:
docker image inspect biomnibench-da-runtime:smoke
```

## 3) Export judge credentials

Each task's `[verifier.env]` in `task.toml` is resolved by **Harbor itself**
(`harbor.utils.env.resolve_env_vars`) against literal OS environment variables — this
is a separate mechanism from NeMo Gym's own `${...}` config interpolation in
`env.yaml`/config YAMLs, so these must be `export`ed in the shell that launches
`ng_run` (uppercase names, matching what's baked into `task.toml`):

```bash
export JUDGE_API_KEY=...     # same value as env.yaml's judge_api_key
export JUDGE_BASE_URL=...    # same value as env.yaml's judge_base_url
export JUDGE_MODEL=...       # same value as env.yaml's judge_model_name
```

## 4) Configure the policy model server

Use `responses_api_models/vllm_model/configs/vllm_model.yaml`, **not**
`vllm_model_for_training.yaml`, unless the policy model is a real self-hosted vLLM
server. `vllm_model_for_training.yaml` sets `return_token_id_information: true`,
which makes `app.py` inject a vLLM-specific `return_tokens_as_token_ids` sampling
param — remote gateway models (e.g. `azure/openai/gpt-5.5` via
`https://inference-api.nvidia.com/v1`) reject that param and the request fails with
an opaque `500`.

## 5) Launch Gym and collect the five example rollouts

`config.yaml` points `harbor_datasets.biomnibench_da.local_dataset_path` at
`environments/biomnibench_da/data/example`, so the five rows in
`environments/biomnibench_da/data/example.jsonl` resolve to the five materialized
task directories from step 1. If you materialize to a different `--output-dir`,
either update `config.yaml` or override
`+harbor_agent.responses_api_agents.harbor_agent.harbor_datasets.biomnibench_da.local_dataset_path`
when starting the server.

Recommended CLI:

```bash
gym env start --environment biomnibench_da --model-type vllm_model &
./scripts/wait_for_servers.sh $!

gym eval run --no-serve \
    --agent harbor_agent \
    --input environments/biomnibench_da/data/example.jsonl \
    --output ./example_rollout.jsonl \
    --concurrency 1
```

The legacy `ng_run` / `ng_collect_rollouts` commands are equivalent:

```bash
ng_run "+config_paths=[environments/biomnibench_da/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" &
./scripts/wait_for_servers.sh $!

ng_collect_rollouts +agent_name=harbor_agent \
  +input_jsonl_fpath=environments/biomnibench_da/data/example.jsonl \
  +output_jsonl_fpath=./example_rollout.jsonl \
  +num_samples_in_parallel=1
```

Rollout JSONL uses `instance_id` in the form `biomnibench_da::<task_name>` (for
example `biomnibench_da::da-1-3-r001`). For HPC/Singularity, swap in
`environments/biomnibench_da/config_singularity.yaml` and the
`example_singularity` dataset path.

**Important:** export the `JUDGE_*` vars (step 3) *in the same shell* before running
`ng_run` — Harbor resolves them from that process's OS environment when it launches
each task's container, not from wherever `ng_collect_rollouts` is later run from.

The checked-in `data/example_rollouts.jsonl` and
`data/example_rollout_aggregate_metrics.json` were generated from this five-row
example input. The `data/example/` materialized task tree is generated locally and
gitignored because its Docker bind mounts contain absolute host paths.

## Troubleshooting

- `**service "main" has neither an image nor a build context specified`**: the
materialized `environment/docker-compose.yaml` is stale (missing `image:`/mount
overrides) — re-run step 1 to regenerate it, and confirm the runtime image exists
(step 2).
- `**No such file or directory` from `tee .../logs/verifier/...` / trial fails with
`RewardFileNotFoundError` despite the judge printing a score**: same cause —
Harbor's Docker environment assumes `/logs/agent` and `/logs/verifier` are
bind-mounted from the trial dir (`harbor.models.trial.paths.EnvironmentPaths`);
regenerate the compose file (step 1) rather than hand-editing it.
- `**Error response from daemon: all predefined address pools have been fully subnetted`** when a trial's `docker compose up -d` tries to create a network: free
up unused Docker networks (`docker network prune`) or reduce concurrency.
- `**ValueError: Environment variable 'JUDGE_BASE_URL' not found in host environment`** (raised inside the trial by `harbor.utils.env.resolve_env_vars`,
visible in `harbor_agent/jobs/.../exception.txt`): the `harbor_agent` server
process — not the shell you're currently typing in — didn't have `JUDGE_*`
exported when it was started. Harbor resolves `[verifier.env]` from that server
process's own OS environment at verification time, so exporting the vars *after*
`ng_run` is already running (or in a different terminal) has no effect on it, even
if `echo $JUDGE_BASE_URL` shows it correctly in your current shell. Fix: kill the
running `ng_run`, `export JUDGE_API_KEY`/`JUDGE_BASE_URL`/`JUDGE_MODEL` in that
exact shell, then restart `ng_run` from there (step 3 must come first).
- `**ng_collect_rollouts` crashes with `aiohttp.client_exceptions.ClientResponseError: 404 ... /aggregate_metrics`** after "Computing aggregate metrics": harmless to your
data — the rollouts JSONL is fully written and closed *before* this step runs, so
nothing is lost. This was a real gap (now fixed) where `harbor_agent`'s
`setup_webserver()` didn't register `/aggregate_metrics`, unlike the
`SimpleResponsesAPIAgent` base class default. If you still hit this on an older
checkout, pass `+disable_aggregation=true` to `ng_collect_rollouts`/`gym eval run`
as a workaround, then run `gym eval aggregate` once all shards finish.

See the [Harbor Agent README](../../responses_api_agents/harbor_agent/README.md) for
details on the underlying Harbor bridge (custom agents/environments, NeMo RL
training notes, and rollout storage layout), which applies to any Harbor-backed
environment, not just BiomniBench-DA.

# Licensing information

Code: Apache 2.0
Data: see [phylobio/BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA)
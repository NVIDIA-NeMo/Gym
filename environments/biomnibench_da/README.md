# BiomniBench-DA environment

[BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) data-analysis
tasks as Harbor tasks. The [`harbor_tasks`](../../resources_servers/harbor_tasks/README.md)
Resources Server starts each task's sandbox and grades it with the task's verifier, Harbor's
Terminus-2 runs in that sandbox through
[`harbor_harness_agent`](../../responses_api_agents/harbor_harness_agent/README.md), and the
single-agent-turn Environment Server drives each episode. Materialized task trees live under
`data/` (gitignored — see `prepare.py`).

Each task gives the agent a data-analysis question and a data directory; the agent
writes `trace.md` (its analysis) and `answer.txt` (its final answer) inside the
container, and an OpenAI-compatible LLM judge scores the trace/answer against a
per-task rubric (upstream-faithful scoring, see `prepare.py`'s embedded
`llm_judge.py`).

Use Gym's venv from the repo root for all commands below.

## 1) Download and materialize an example task tree

The checked-in example set uses five representative BiomniBench-DA tasks:
`da-1-3`, `da-1-4`, `da-10-1`, `da-10-3`, and `da-11-1`.
Some of these tasks use multi-gigabyte data. The command below selects `da-10-1`
for a quick evaluation. The dataset is gated, so request access on HuggingFace and
authenticate with `HF_TOKEN` before downloading it.
These tasks include singleton or otherwise uncovered task types, so pass
`--include-singletons --include-uncovered` to keep them.

`prepare.py` downloads the requested task from HuggingFace, builds the shared runtime
image, then materializes the Harbor task directory under `--output-dir` and writes
`rollout_input.jsonl` there. This is the `gym eval run` input file, with one row per
task naming it by `harbor_dataset` (`biomnibench_da`) and `task_name`, and carrying its
instruction.

```bash
python environments/biomnibench_da/prepare.py \
  --download \
  --build-docker-image \
  --tasks da-10-1 \
  --include-singletons --include-uncovered \
  --output-dir environments/biomnibench_da/data/example \
  --overwrite
# -> data/example/rollout_input.jsonl  (1 row)
```

Override the rollout-input path with `--rollout-input-fpath` if needed.

The default `--environment-type sandbox` copies each task's data into `environment/data`;
`harbor_tasks` uploads it to `/app/data` after starting the sandbox, with any Gym sandbox
provider. `--environment-type docker` instead bind-mounts the data through a
`docker-compose.yaml`, for Harbor's own `harbor run`; `harbor_tasks` does not run
docker-compose tasks.

See `python environments/biomnibench_da/prepare.py --help` for the full flag set
(train/test split controls, `--limit`, `--papers`, `--max-data-mb`, `--n-repeats`,
`--judge-model`, `--docker-image`, etc.). The full dataset is prepared
the same way, just without `--tasks`/`--include-singletons`/`--include-uncovered`.

## 2) Build (or verify) the shared runtime image

Tasks reference a prebuilt image (`[environment].docker_image` in each `task.toml`), not a
per-task Dockerfile build. The `--build-docker-image` flag in step 1 builds it. If you omit
that flag, build or pull the image before evaluation.

## 3) Export judge credentials

Each task's `[verifier.env]` in `task.toml` is resolved by **Harbor's verifier**
(`harbor.utils.env.resolve_env_vars`) against the OS environment of the `harbor_tasks`
server process. This is separate from NeMo Gym's own `${...}` config interpolation, so
`export` these in the shell that launches Gym (uppercase names, matching what's baked
into `task.toml`):

```bash
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=...
export JUDGE_MODEL=...
```

## 4) Configure the policy model server

Create `env.yaml` in the repo root with the hosted policy endpoint:

```yaml
policy_base_url: https://your-policy-endpoint/v1
policy_api_key: your-policy-api-key
policy_model_name: your-policy-model
```

Use `responses_api_models/vllm_model/configs/vllm_model.yaml`, **not**
`vllm_model_for_training.yaml`, unless the policy model is a real self-hosted vLLM
server. `vllm_model_for_training.yaml` sets `return_token_id_information: true`,
which makes `app.py` inject a vLLM-specific `return_tokens_as_token_ids` sampling
param — remote gateway models (e.g. `azure/openai/gpt-5.5` via
`https://inference-api.nvidia.com/v1`) reject that param and the request fails with
an opaque `500`.

## 5) Launch Gym and collect the example rollout

`config.yaml` points `harbor_datasets.biomnibench_da.path` at
`environments/biomnibench_da/data/example`. If you materialize to a different
`--output-dir`, either update `config.yaml` or override
`++biomnibench_da_resources_server.resources_servers.harbor_tasks.harbor_datasets.biomnibench_da.path`
when starting the servers. Add a sandbox provider config: Docker below, or
`config_singularity.yaml`, which adds an Apptainer provider for HPC.

```bash
gym env start \
    --config environments/biomnibench_da/config.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --model-type vllm_model &
./scripts/wait_for_servers.sh $!

gym eval run --no-serve \
    --agent biomnibench_da_agent \
    --input environments/biomnibench_da/data/example/rollout_input.jsonl \
    --output ./example_rollout.jsonl \
    --concurrency 1
```

The tasks ask for no network access. Gym sandboxes do not enforce that, so `config.yaml`
sets `allow_unenforced_network_policy: true`; Harbor's own Docker environment enforced it.

**Important:** export the `JUDGE_*` vars from step 3 in the same shell before running
`gym env start`. Harbor's verifier resolves them from the `harbor_tasks` server's OS
environment, not from wherever `gym eval run` is later run from.

The checked-in `data/example_rollouts.jsonl` and `data/example_metrics.json` were
generated from the five-row example input with the earlier `harbor_agent` bridge, before
this environment moved to `harbor_tasks`. The `data/example/` materialized task tree is
generated locally and gitignored.

## Troubleshooting

- `**ValueError: Environment variable 'JUDGE_BASE_URL' not found in host environment`**
(raised by `harbor.utils.env.resolve_env_vars` and reported as `verifier_error` on the
rollout): the `harbor_tasks` server process didn't have `JUDGE_*` exported when it was
started. Stop the running Gym environment, export `JUDGE_API_KEY`/`JUDGE_BASE_URL`/`JUDGE_MODEL`
in that exact shell, then restart `gym env start` from there (step 3 must come first).
- `**Harbor task ... uses unsupported features: docker-compose environments`**: the
task tree was materialized with `--environment-type docker`. Re-run step 1 with the
default `sandbox` profile.

# Licensing information

Code: Apache 2.0
Data: see [phylobio/BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA)

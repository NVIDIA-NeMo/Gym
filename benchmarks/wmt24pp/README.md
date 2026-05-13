# WMT24++ Translation Benchmark

English to {de_DE, es_MX, fr_FR, it_IT, ja_JP} segment-level translation
from [`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp).

Verification is deterministic corpus-level BLEU (sacrebleu) per language
pair, with cross-pair aggregations `en->xx`, `xx->xx`, and `xx->{tgt}`.
Optionally augments with xCOMET-XXL neural QE scores when
`compute_comet: true` is set on the wmt_translation server.

See `resources_servers/wmt_translation/README.md` for the verifier
details and the Ray GPU-scheduled COMET path.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/wmt24pp/config.yaml]"
```

In addition to writing `data/wmt24pp_benchmark.jsonl`, the prepare step
pre-fetches the xCOMET-XXL checkpoint and its xlm-roberta-xxl tokenizer
into `HF_HOME` (when `unbabel-comet` is installed in the active env).
That keeps the resource server's Ray actors fully offline at runtime —
no HF Hub calls during `verify()`, no rate-limit retries.

## Running servers

The xCOMET-XXL actor pool requires the `extra_gpu` Ray resource, which
is only advertised on multi-node SLURM deployments via NeMo-Skills'
`get_ray_server_cmd` (see the SLURM block below). Local / single-node
runs disable COMET via Hydra override and rely on corpus-BLEU only;
xCOMET scoring still works end-to-end on the cluster path:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/wmt24pp/config.yaml"
ng_run "+config_paths=[$config_paths]" \
    "++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=false"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=wmt24pp_wmt_translation_simple_agent \
    +prompt_config=benchmarks/wmt24pp/prompts/default.yaml \
    +input_jsonl_fpath=benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl \
    +output_jsonl_fpath=results/wmt24pp_rollouts.jsonl \
    +num_repeats=4
```

## End-to-end reproduction on a SLURM cluster (via NeMo-Skills)

The commands above assume the Gym head and a Ray cluster have been
brought up manually. For a fully reproducible run on SLURM, use
NeMo-Skills' `ns nemo_gym_rollouts` CLI — it handles vLLM bring-up
with the right Ray topology (the `vllm_dp_ray` server type reserves a
hidden `extra_gpu` node for the streaming xCOMET-XXL actor pool),
placement-group setup, and Gym launch in one shot.

### One-time setup

```bash
# 1. Install NeMo-Skills (provides the `ns` CLI). Pinned to the SHA where
#    server_type=vllm_dp_ray landed on main; bump as new Skills releases tag.
pip install git+https://github.com/NVIDIA-NeMo/Skills.git@f57b1735b
```

#### Cluster config

Define a cluster config at `cluster_configs/<your-cluster>.yaml`. See
[the NeMo-Skills cluster-configs docs](https://nvidia-nemo.github.io/Skills/basics/cluster-configs/)
for the full schema; this recipe needs:

| Field | Why | Example |
|---|---|---|
| `executor: slurm` | recipe is SLURM-only | `slurm` |
| `ssh_tunnel:` | so `ns` submits jobs over SSH | `host:`, `user:`, `job_dir:`, `identity:` |
| `account` / `partition` / `cpu_partition` | SLURM accounting + queue | per-cluster values |
| `containers.vllm_dp_ray` | the policy server image | based on `vllm/vllm-openai:v0.18.1` (or any 0.18.x with `ray>=2.48`) |
| `containers.nemo-gym` | the Gym + xCOMET-XXL image | any image with `nemo-gym[dev]` + `unbabel-comet` installed |
| `containers.sandbox` | required by `ns nemo_gym_rollouts` even though wmt24pp doesn't sandbox | any NeMo-Skills sandbox image |
| `mounts:` | a directory for the HF cache (paired with the `HF_HOME` env var below — Skills requires both) and a writable workspace dir for `--output_dir` | `<host-hf-dir>:/models`, `<host-workspace>:/workspace` |
| `env_vars:` | `HF_HOME=<in-container-path>` is **required** by Skills and must point inside one of your mounts (so the COMET prefetch survives across jobs). Optionally bump `VLLM_ENGINE_READY_TIMEOUT_S` above vLLM's 600s default if your model's cold-load (weights + KV-cache init + warmup) exceeds it — typical triggers are very large models or cross-node TP. | `HF_HOME=/models/hf-cache` (matching the `:/models` mount above) |

The two container fields that aren't trivial:

- **`vllm_dp_ray`**: any vLLM 0.18.x image. The Skills repo ships
  `dockerfiles/Dockerfile.vllm` which builds on `vllm/vllm-openai:v0.18.1`
  with `ray[cgraph]` + audio/Qwen-VL extras. **The bundled `ray` version
  in this image MUST match the `ray` resolved by `nemo-gym`'s `uv.lock`**
  — cross-container Ray-cluster joins fail with `ConnectionError: Could
  not read 'temp_dir' from GCS` on protocol mismatch.
- **`nemo-gym`**: any image where `pip install -e <gym>[dev]` resolves
  cleanly AND has `unbabel-comet`, `torch>=2.5`, `sacrebleu` baked in.
  The lazy-install path in `resources_servers/wmt_translation/.venv`
  works as a fallback but adds 2–3 min to first-job startup.

#### Prepare benchmark data on the cluster

The local `ng_prepare_benchmark` from [above](#prepare-benchmark-data)
writes the JSONL to your dev workstation. For a SLURM run, the JSONL
plus the `Unbabel/XCOMET-XXL` cache need to live on the cluster's
filesystem. Dispatch the prepare via `ns run_cmd` with the `nemo-gym`
container (which has `unbabel-comet` so the prefetch step actually
runs):

```bash
ns run_cmd \
    --cluster <your-cluster> \
    --container nemo-gym \
    --expname wmt24pp_prepare \
    --command 'ng_prepare_benchmark "+config_paths=[benchmarks/wmt24pp/config.yaml]"'
```

This populates `benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl` and
prefetches `Unbabel/XCOMET-XXL` + its `xlm-roberta-xxl` tokenizer into
the cluster's `HF_HOME`. Subsequent rollout jobs read both from the
shared filesystem.

### 2-node smoke topology (1 model node + 1 extra_gpu COMET node)

Sized to fit on an `interactive` partition for fast iteration. Bump
`server_nodes` for DP>1 (`server_nodes = dp_size + num_extra_gpu_nodes`)
and switch to a batch partition for larger evaluations.

```bash
# Pick a translation-capable policy model accessible from your cluster.
# The PR's parity numbers come from nvidia/Nemotron-3-Nano-30B-A3B-BF16.
MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"

ns nemo_gym_rollouts \
    --cluster <your-cluster> \
    --partition interactive \
    --server_type vllm_dp_ray \
    --server_gpus 8 \
    --server_nodes 2 \
    --server_args "--tensor-parallel-size 8 --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-backend ray --distributed-executor-backend ray --api-server-count 1 --reasoning-parser deepseek_r1 --trust-remote-code --dtype auto --enforce-eager" \
    --model "$MODEL" \
    --config_paths "benchmarks/wmt24pp/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml" \
    --input_file benchmarks/wmt24pp/data/wmt24pp_benchmark.jsonl \
    --output_dir /workspace/wmt24pp_smoke \
    --expname wmt24pp_smoke \
    -- \
    +agent_name=wmt24pp_wmt_translation_simple_agent \
    +prompt_config=benchmarks/wmt24pp/prompts/default.yaml \
    +num_repeats=1 \
    +limit=20 \
    +num_samples_in_parallel=64 \
    ++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=true
```

`--reasoning-parser deepseek_r1` is required for the Nemotron-3-Nano
family above; drop it for non-reasoning models.

The job allocates 2 nodes (1 model node hosting `server_gpus` vLLM
workers + 1 extra_gpu node hosting the xCOMET-XXL actor pool), starts
vLLM in DP-on-Ray mode on the model node, schedules the actor pool onto
the extra node via the custom `extra_gpu` Ray resource, and writes
`rollouts.jsonl` (with per-row `comet_score`) plus
`rollouts_aggregate_metrics.json` to `--output_dir`.

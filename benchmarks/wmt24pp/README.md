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

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/wmt24pp/config.yaml"
ng_run "+config_paths=[$config_paths]"
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
# 1. Install NeMo-Skills, which provides the `ns` CLI.
pip install git+https://github.com/NVIDIA-NeMo/Skills.git

# 2. Define a cluster config at cluster_configs/<your-cluster>.yaml.
#    See https://nvidia-nemo.github.io/Skills/basics/cluster-configs/
#    for the schema. It must declare the `vllm` (or `vllm_dp_ray`),
#    `nemo-rl`, and `sandbox` containers and either mount or pre-cache
#    Unbabel/XCOMET-XXL + the policy model under HF_HOME.
```

### 2-node smoke topology (1 model node + 1 extra_gpu COMET node)

Sized to fit on an `interactive` partition for fast iteration. Bump
`server_nodes` for DP>1 (`server_nodes = dp_size + num_extra_gpu_nodes`)
and switch to a batch partition for larger evaluations.

```bash
ns nemo_gym_rollouts \
    --cluster <your-cluster> \
    --partition interactive \
    --server_type vllm_dp_ray \
    --server_gpus 8 \
    --server_nodes 2 \
    --server_args "--tensor-parallel-size 8 --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-backend ray --distributed-executor-backend ray --api-server-count 1 --trust-remote-code --dtype auto --enforce-eager" \
    --model <your-translation-model> \
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

The job allocates 2 nodes (`server_gpus * 1 model node + 1 extra_gpu
node`), starts vLLM in DP-on-Ray mode on the model node, schedules the
xCOMET-XXL actor pool onto the extra node via the custom `extra_gpu`
Ray resource, and writes `rollouts.jsonl` (with per-row `comet_score`)
plus `rollouts_aggregate_metrics.json` to `--output_dir`.

# FLORES-200 Translation Benchmark

Multilingual segment-level translation across the default six-language set
(`en`, `de`, `es`, `fr`, `it`, `ja`) from the
[`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus)
`devtest` split. The benchmark interleaves all 30 directed pairs (6 sources ×
6 targets, minus self-pairs) yielding 30 360 rows
(30 × 1012 sentences).

Verification reuses the shared `wmt_translation` resource server: deterministic
corpus-level BLEU (sacrebleu) per language pair plus the cross-pair
aggregations `xx->xx`, `<src>->xx`, `xx->{tgt}`, optionally augmented by
xCOMET-XXL neural QE when `compute_comet: true` is set on the server.

See `resources_servers/wmt_translation/README.md` for verifier details and
the Ray GPU-scheduled COMET path.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/flores200/config.yaml]"
```

Writes `data/flores200_devtest_benchmark.jsonl` and pre-fetches the
xCOMET-XXL checkpoint plus its xlm-roberta-xxl tokenizer into `HF_HOME`
(when `unbabel-comet` is installed in the active env). That keeps the
resource server's Ray actors fully offline at runtime — no HF Hub calls
during `verify()`, no rate-limit retries.

The default downloads six FLORES+ language configs (`eng_Latn`, `deu_Latn`,
`spa_Latn`, `fra_Latn`, `ita_Latn`, `jpn_Jpan`) and produces a JSONL whose
order, fields, and contents are byte-equivalent to the reference
[NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills) prepare script
(`nemo_skills/dataset/flores200/prepare.py`) under the same defaults.

## Running servers

The xCOMET-XXL actor pool requires the `extra_gpu` Ray resource which is
only advertised on multi-node SLURM deployments via NeMo-Skills'
`get_ray_server_cmd`. Local / single-node runs disable COMET via Hydra
override and rely on corpus-BLEU only:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/flores200/config.yaml"
ng_run "+config_paths=[$config_paths]" \
    "++flores200_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=false"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=flores200_wmt_translation_simple_agent \
    +prompt_config=benchmarks/flores200/prompts/default.yaml \
    +input_jsonl_fpath=benchmarks/flores200/data/flores200_devtest_benchmark.jsonl \
    +output_jsonl_fpath=results/flores200_rollouts.jsonl \
    +num_repeats=4
```

## End-to-end on a SLURM cluster (via NeMo-Skills)

The commands above assume the Gym head and a Ray cluster have been brought
up manually. For a fully reproducible run on SLURM, use NeMo-Skills'
`ns nemo_gym_rollouts` CLI — it handles vLLM bring-up with the right Ray
topology (the `vllm_dp_ray` server type reserves a hidden `extra_gpu` node
for the streaming xCOMET-XXL actor pool), placement-group setup, and Gym
launch in one shot. See
`benchmarks/wmt24pp/README.md` for the full SLURM recipe — flores200 reuses
the exact same `wmt_translation` server and topology requirements.

### Prepare benchmark data on the cluster

```bash
ns run_cmd \
    --cluster <your-cluster> \
    --container nemo-gym \
    --expname flores200_prepare \
    --command 'ng_prepare_benchmark "+config_paths=[benchmarks/flores200/config.yaml]"'
```

This populates `benchmarks/flores200/data/flores200_devtest_benchmark.jsonl`
and prefetches `Unbabel/XCOMET-XXL` plus its `xlm-roberta-xxl` tokenizer
into the cluster's `HF_HOME`. Subsequent rollout jobs read both from the
shared filesystem.

### 2-node smoke topology (1 model node + 1 extra_gpu COMET node)

Sized to fit on an `interactive` partition for fast iteration. Bump
`server_nodes` for DP>1 (`server_nodes = dp_size + num_extra_gpu_nodes`)
and switch to a batch partition for larger evaluations.

```bash
# Pick a translation-capable policy model accessible from your cluster.
MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"

ns nemo_gym_rollouts \
    --cluster <your-cluster> \
    --partition interactive \
    --server_type vllm_dp_ray \
    --server_gpus 8 \
    --server_nodes 2 \
    --server_args "--tensor-parallel-size 8 --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-backend ray --distributed-executor-backend ray --api-server-count 1 --reasoning-parser deepseek_r1 --trust-remote-code --dtype auto --enforce-eager" \
    --model "$MODEL" \
    --config_paths "benchmarks/flores200/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml" \
    --input_file benchmarks/flores200/data/flores200_devtest_benchmark.jsonl \
    --output_dir /workspace/flores200_smoke \
    --expname flores200_smoke \
    -- \
    +agent_name=flores200_wmt_translation_simple_agent \
    +prompt_config=benchmarks/flores200/prompts/default.yaml \
    +num_repeats=1 \
    +limit=20 \
    +num_samples_in_parallel=64 \
    ++flores200_wmt_translation_resources_server.resources_servers.wmt_translation.compute_comet=true
```

`--reasoning-parser deepseek_r1` is required for the Nemotron-3-Nano family
above; drop it for non-reasoning models.

The job allocates 2 nodes (1 model node hosting `server_gpus` vLLM workers +
1 extra_gpu node hosting the xCOMET-XXL actor pool), starts vLLM in
DP-on-Ray mode on the model node, schedules the actor pool onto the extra
node via the custom `extra_gpu` Ray resource, and writes `rollouts.jsonl`
(with per-row `comet_score`) plus `rollouts_aggregate_metrics.json` to
`--output_dir`.

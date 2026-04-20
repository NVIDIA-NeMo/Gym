# MRCR benchmark

Benchmark wrapper over the [`mrcr` resources server](../../resources_servers/mrcr/README.md)
for the [openai/mrcr](https://huggingface.co/datasets/openai/mrcr) dataset.

Each task is a multi-turn conversation with a final-turn "prepend `<prefix>`
to the Nth occurrence and reproduce it exactly" instruction. Scoring:
`SequenceMatcher.ratio()` between stripped response and stripped expected
answer, gated on the response starting with the random prefix.

### Launch local vllm server
```bash
pip install -U "vllm>=0.12.0"

wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 10240 \
  --trust-remote-code \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

The reasoning parser is required — see the `mrcr` resources server
[README](../../resources_servers/mrcr/README.md) for why.

### Set `env.yaml` in `Gym/`
```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### Install Gym
```bash
cd Gym/
uv venv
source .venv/bin/activate
uv sync
```

### Prepare benchmark data
Downloads the HF dataset, token-counts each sample with `tiktoken o200k_base`,
and writes `benchmarks/mrcr/data/mrcr_benchmark.jsonl`. Samples over 200000
input tokens are dropped to leave headroom for model-side tokenizers (which
can be 7–10% heavier than tiktoken) to stay under a 262144-token native
context.

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/mrcr/config.yaml]"
```

### Start MRCR environment
```bash
ng_run "+config_paths=[benchmarks/mrcr/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

### Collect rollouts
```bash
ng_collect_rollouts \
    +agent_name=mrcr_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/mrcr/data/mrcr_benchmark.jsonl \
    +output_jsonl_fpath=results/mrcr_rollouts.jsonl \
    +num_repeats=4
```

Notes on common footguns:

- **`num_repeats` for `type: benchmark` datasets is a CLI flag**
  (`+num_repeats=N`), not a honored YAML field. The `num_repeats: 4` entry
  under `datasets:` in `benchmarks/mrcr/config.yaml` is a documentation
  hint only — it only triggers row duplication for
  `type: train`/`type: validation`.
- **Do NOT pair with `+num_repeats_add_seed=true`.** That flag writes
  `seed=<rollout_idx>` into `responses_create_params`, but
  `NeMoGymResponseCreateParamsNonStreaming` has `extra="forbid"` and
  rejects every rollout with `extra_forbidden`. Temperature-1.0 sampling
  alone gives sufficient variance across repeats.

### Metrics

`compute_metrics()` in the resources server emits:
- `pass@k/accuracy`, `pass@1[avg-of-k]/accuracy` via
  `compute_pass_majority_metrics`
- Per-`n_needles` subset breakdown via
  `compute_subset_metrics(subset_key="n_needles")` — exposes stratified
  pass@k keys like `2/pass@4/accuracy`, `4/pass@4/accuracy`,
  `8/pass@4/accuracy`

For training, see the
[docs](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/index.html).

# compute_eval

Port of NeMo-Skills' [`compute-eval`](https://github.com/NVIDIA/compute-eval)
benchmark. CUDA / C / C++ / Python kernel-implementation problems compiled
with **nvcc** and graded against the upstream hidden test suite.

Dataset: [nvidia/compute-eval](https://huggingface.co/datasets/nvidia/compute-eval)
(gated — set `HF_TOKEN` before running prepare).

Verification is delegated to the
[`compute_eval`](../../resources_servers/compute_eval/) resources server
(requires `nvcc` on PATH). The prompt is character-for-character with
NeMo-Skills' `nemo_skills/prompt/config/compute-eval/baseline.yaml`.

## Example usage

```bash
# Prepare benchmark data (HF_TOKEN must be set; dataset is gated)
HF_TOKEN=$HF_TOKEN ng_prepare_benchmark "+config_paths=[benchmarks/compute_eval/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/compute_eval/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=compute_eval_compute_eval_simple_agent \
    +input_jsonl_fpath=benchmarks/compute_eval/data/compute_eval_benchmark.jsonl \
    +output_jsonl_fpath=results/compute_eval_rollouts.jsonl \
    +prompt_config=benchmarks/prompts/compute-eval/baseline.yaml \
    +num_repeats=4
```

`prepare.py` accepts a `--release` flag (e.g. `2025-1`, `2025-2`) matching
the upstream HuggingFace dataset release naming.

# AA-LCR

AA-LCR provides two versioned configs in one benchmark:

- `aalcr/config_v1_1`: the recommended AA-LCR v1.1 grading update introduced with the
  [Artificial Analysis Intelligence Index v4.2](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-2).
  It pins [dataset commit `9a77ef5`](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/commit/9a77ef56b717057ade24ceab4d273712a0b4f19e),
  which corrects 16 answer keys, adds the published judge system prompt, uses
  JSON verdicts, and specifies GPT-5.6 Luna (medium) as the equality checker.
- `aalcr`: the original v1.0 dataset and Gym's historical plain-text judge
  protocol, retained at the existing selector for backward compatibility.

## v1.1

### Prepare data
```bash
gym eval prepare --benchmark aalcr/config_v1_1
```

The v1.1 config always sends its grading requests to
`openai/openai/gpt-5.6-luna` on NVIDIA's inference gateway
(`https://inference-api.nvidia.com/v1`) with medium reasoning effort.

Before running the evaluation, set `JUDGE_API_KEY` to an NVIDIA inference
gateway API key that has access to GPT-5.6 Luna:

```bash
export JUDGE_API_KEY='<your NVIDIA inference gateway API key>'
```

`JUDGE_API_KEY` authenticates requests to the Luna judge; it is not an
AA-LCR-specific credential. Data preparation does not require it. Evaluation
fails during judge startup when it is absent and never falls back to another
judge.

### Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark aalcr/config_v1_1 \
    ++output_jsonl_fpath=results/benchmarks/aalcr_v1_1.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++ray_head_node_address=auto \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<>
```

## v1.0 compatibility

### Prepare data
```bash
gym eval prepare --benchmark aalcr
```

### Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark aalcr \
    ++output_jsonl_fpath=results/benchmarks/aalcr_v1_0.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++resume_from_cache=true \
    ++ray_head_node_address=auto \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=<> \
    ++policy_api_key=<> \
    ++policy_model_name=<> \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.base_url=<>' \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.model=<>' \
    '++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.api_key=<>'
```

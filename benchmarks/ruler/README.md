# Prerequisites
Please ensure that you have git-lfs installed (used to fetch the `cwe` task corpus)!

Linux: `apt update && apt install -y git-lfs`

# Prepare data
Prepares RULER at 1M context (13 tasks x 100 samples) with an answer-prefix assistant turn:
```bash
gym eval prepare --benchmark ruler
```

# Serve the model
RULER at 1M context needs a vLLM endpoint served with a large `--max-model-len` (>= the prepared
context length plus room to generate). Thinking is disabled for the eval, so serve **without** a
`--reasoning-parser`:
```bash
vllm serve <model> \
    --served-model-name <served-model-name> \
    --tensor-parallel-size 8 \
    --port <port> \
    --max-model-len 1300000 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
```
`--tensor-parallel-size` (and multi-node) depends on the model size and hardware. The benchmark
config enables `continue_final_assistant_message` and `chat_template_kwargs.enable_thinking=false`
on the model server, so the model continues the answer-prefix turn and emits the answer directly.

# Run
```bash
gym eval run \
    --model-type vllm_model \
    --benchmark ruler \
    --output results/benchmarks/ruler.jsonl \
    --split benchmark \
    --model-url <endpoint-served-above>/v1 \
    --model-api-key <> \
    --model <served-model-name> \
    --resume \
    ++overwrite_metrics_conflicts=true \
    ++reuse_existing_data_preparation=true
```

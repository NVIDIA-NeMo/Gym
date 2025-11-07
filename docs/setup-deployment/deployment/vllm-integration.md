(setup-deployment-vllm)=

# vLLM Integration

Connect NeMo Gym to vLLM-hosted models without native Responses API support.

---

## Overview

### Why vLLM Integration?

```{list-table}
:header-rows: 1
:widths: 40 30 30

* - Scenario
  - Without vLLM Middleware
  - With vLLM Middleware
* - **API Compatibility**
  - Only GPT-4o and similar
  - Any vLLM-supported model
* - **Token Tracking**
  - Built-in Responses API
  - Via `/tokenize` endpoint
* - **Tool Calls**
  - Native support
  - Converted from Chat Completions
* - **Reasoning Traces**
  - Native format
  - Parsed from `<think>` tags
```

:::{note}
As of November 2025, few models natively support OpenAI's Responses API. The `vllm_model` middleware translates between Responses API (used by NeMo Gym) and Chat Completions API (provided by vLLM).
:::

### How It Works

```{mermaid}
graph LR
    A[NeMo Gym] -->|Responses API Request| B[vllm_model Middleware]
    B -->|Chat Completions Request| C[vLLM Server]
    C -->|Chat Completions Response| B
    B -->|Responses API Response| A
    
    style B fill:#4CAF50,stroke:#2E7D32,color:#fff
```

The middleware:
1. **Converts** Responses API calls to Chat Completions format
2. **Uses** vLLM's `/tokenize` endpoint for accurate token tracking
3. **Parses** reasoning from `<think>` tags in model output
4. **Maintains** token-level accuracy for RL training

---

## Quick Start

### Step 1: Start vLLM Server

#### Option A: Standard Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install vLLM
pip install vllm

# Start server with tool support
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

#### Option B: With HuggingFace Transfer (Faster Downloads)

```bash
# Install with HF transfer
pip install hf_transfer vllm

# Download model first
HF_HUB_ENABLE_HF_TRANSFER=1 \
huggingface-cli download Qwen/Qwen3-30B-A3B

# Start server
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

:::{important}
**Do NOT use** vLLM's `--reasoning-parser` flag. The middleware handles reasoning parsing internally via `uses_reasoning_parser: true` config to maintain Responses API compatibility.
:::

### Step 2: Configure NeMo Gym

Use the `vllm_model` configuration:

```bash
# Replace openai_model with vllm_model
ng_run "+config_paths=[
    resources_servers/multineedle/configs/multineedle.yaml,
    responses_api_models/vllm_model/configs/vllm_model.yaml
]"
```

Or update `env.yaml`:

```yaml
# env.yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY  # vLLM doesn't require key for local
policy_model_name: Qwen/Qwen3-30B-A3B
```

### Step 3: Verify Connection

```bash
# Test vLLM server
curl http://localhost:10240/v1/models

# Run NeMo Gym task
ng_run "+config_paths=[
    resources_servers/simple_task/configs/simple.yaml,
    responses_api_models/vllm_model/configs/vllm_model.yaml
]"
```

---

## Configuration Reference

### vLLM Server Options

```{list-table}
:header-rows: 1
:widths: 25 50 25

* - Parameter
  - Description
  - Recommended Value
* - `--dtype`
  - Model precision
  - `auto` (FP16/BF16)
* - `--tensor-parallel-size`
  - Number of GPUs
  - Match GPU count
* - `--gpu-memory-utilization`
  - GPU memory fraction
  - `0.9` (leave headroom)
* - `--enable-auto-tool-choice`
  - Enable tool calling
  - Required for tasks
* - `--tool-call-parser`
  - Tool format parser
  - `hermes` for Qwen
* - `--host`
  - Server host
  - `0.0.0.0` for network access
* - `--port`
  - Server port
  - `10240` (or custom)
```

### NeMo Gym vLLM Model Config

Default configuration (`responses_api_models/vllm_model/configs/vllm_model.yaml`):

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}       # http://localhost:10240/v1
      api_key: ${policy_api_key}         # EMPTY for local
      model: ${policy_model_name}        # Model identifier
      return_token_id_information: false # Enable for training
      uses_reasoning_parser: true        # Parse <think> tags
```

### CLI Overrides

```bash
# Custom vLLM server location
ng_run "+config_paths=[$config_paths]" \
    +policy_model.vllm_model.base_url=http://gpu-server:10240/v1 \
    +policy_model.vllm_model.model_name=Qwen/Qwen3-30B-A3B

# Enable token tracking for training
ng_run "+config_paths=[$config_paths]" \
    +policy_model.vllm_model.return_token_id_information=true
```

---

## Supported Models

### Tested Configurations

::::{tab-set}

:::{tab-item} Qwen Models

```bash
# Qwen3-30B-A3B (Recommended)
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Features**:
- ✅ Native tool calling support
- ✅ Reasoning via `<think>` tags
- ✅ Strong instruction following
- ✅ Efficient inference
:::

:::{tab-item} Llama Models

```bash
# Llama 3.1 70B
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --dtype auto \
    --tensor-parallel-size 8 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

**Features**:
- ✅ Strong reasoning capabilities
- ⚠️ Tool calling via JSON mode
- ⚠️ Larger memory footprint
:::

:::{tab-item} Custom Models

For other models, verify:
- ✅ vLLM compatibility
- ✅ Tool calling support (check vLLM docs for parser)
- ✅ Instruction tuning (base models may be unreliable)

```bash
vllm serve your-org/your-model \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser [check-vllm-docs]
```
:::

::::

### Tool Call Parser Reference

```{list-table}
:header-rows: 1
:widths: 30 40 30

* - Model Family
  - Tool Call Parser
  - Flag
* - Qwen
  - Hermes format
  - `--tool-call-parser hermes`
* - Llama 3.1+
  - Llama JSON format
  - `--tool-call-parser llama3_json`
* - Mistral
  - Mistral format
  - `--tool-call-parser mistral`
* - Other
  - Check vLLM docs
  - See [vLLM tool calling](https://docs.vllm.ai/)
```

---

## Advanced Configuration

### Multi-GPU Setup

For large models across multiple GPUs:

```bash
# 8x A100 80GB
vllm serve Qwen/Qwen3-72B-A3B \
    --dtype auto \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

:::{tip}
Use `--tensor-parallel-size` equal to the number of GPUs for maximum efficiency. For very large models, consider pipeline parallelism with `--pipeline-parallel-size`.
:::

### Token Tracking for Training

Enable detailed token information for RL training:

```yaml
# vllm_model_for_training.yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true  # Enable tracking
      uses_reasoning_parser: true
```

This exposes:
- `prompt_token_ids`: Input token IDs
- `generation_token_ids`: Generated token IDs
- `generation_log_probs`: Log probabilities per token

### Reasoning Parser Customization

The middleware parses reasoning from `<think>` tags by default:

```python
# In model output:
"<think>I need to calculate 2+2...</think>The answer is 4."

# Parsed as:
# reasoning: "I need to calculate 2+2..."
# content: "The answer is 4."
```

For custom reasoning formats, modify `VLLMConverter` in `responses_api_models/vllm_model/app.py`.

---

## Troubleshooting

::::{dropdown} **Connection Refused**

```console
ERROR: Connection refused to http://localhost:10240
```

**Solutions**:
1. Verify vLLM server is running: `ps aux | grep vllm`
2. Check port: `lsof -i :10240`
3. Test directly: `curl http://localhost:10240/v1/models`
4. Check firewall rules if on remote server
::::

::::{dropdown} **Tool Calls Not Working**

```console
ERROR: Tool call parsing failed
```

**Solutions**:
1. Verify `--enable-auto-tool-choice` flag is set
2. Check correct parser for model: `--tool-call-parser hermes`
3. Ensure model supports tool calling (instruction-tuned models)
4. Test with simple tool call to vLLM directly
::::

::::{dropdown} **Out of Memory**

```console
ERROR: CUDA out of memory
```

**Solutions**:
1. Reduce `--gpu-memory-utilization` (try 0.85)
2. Lower `--max-model-len` (e.g., 16384 instead of 32768)
3. Use smaller batch size: `--max-num-seqs 128`
4. Enable CPU offload: `--cpu-offload-gb 8`
::::

::::{dropdown} **Reasoning Not Parsed**

```console
WARNING: Expected reasoning but got raw content
```

**Solutions**:
1. Verify `uses_reasoning_parser: true` in config
2. Check model outputs `<think>` tags (prompt engineering)
3. Test reasoning format directly with vLLM
4. Adjust prompt to encourage reasoning tags
::::

::::{dropdown} **Token Count Mismatch**

```console
WARNING: Token count differs from expected
```

**Solutions**:
1. Ensure `/tokenize` endpoint is accessible
2. Verify `return_token_id_information: true` if needed for training
3. Check vLLM version compatibility
4. Test `/tokenize` directly: `curl -X POST http://localhost:10240/tokenize`
::::

---

## Performance Optimization

### Batch Processing

For high throughput:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --max-num-seqs 256 \            # Increase batch size
    --max-num-batched-tokens 8192 \ # Token budget
    --dtype auto \
    --tensor-parallel-size 4
```

### Context Length

Adjust for your tasks:

```bash
# Long context tasks
vllm serve Qwen/Qwen3-30B-A3B \
    --max-model-len 32768 \
    --tensor-parallel-size 4

# Short context (more throughput)
vllm serve Qwen/Qwen3-30B-A3B \
    --max-model-len 8192 \
    --tensor-parallel-size 4
```

---

## Next Steps

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item}
```{button-ref} local-development
:color: primary
:outline:
:expand:

Set Up Locally First
```
:::

:::{grid-item}
```{button-ref} distributed-computing
:color: primary
:outline:
:expand:

Scale with Ray
```
:::

:::{grid-item}
```{button-ref} ../../models/index
:color: primary
:outline:
:expand:

Browse Tested Models
```
:::

::::

:::{seealso}
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai/)
- **Model Performance**: {doc}`../../models/index`
- **Configuration Reference**: {doc}`../configuration/reference`
- **Debugging**: {doc}`../configuration/debugging`
:::


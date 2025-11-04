(models-vllm-quick-start)=

# Quick Start

Get vLLM running with NeMo Gym in under 5 minutes.

---

## Before You Start

Ensure you have these prerequisites before deploying vLLM with NeMo Gym:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **NeMo Gym installed**
  - Complete [Setup and Installation](../../get-started/setup-installation.md) first
* - **Python 3.10+**
  - Required by vLLM (check with `python3 --version`)
* - **CUDA-capable GPU**
  - NVIDIA GPU with CUDA support for inference
* - **Hugging Face account**
  - Optional - only required for gated models like Llama
* - **GPU Memory**
  - **8B models**: ~16GB VRAM • **30B models**: ~60GB VRAM (2-4 GPUs) • **70B models**: ~140GB VRAM (4-8 GPUs)
* - **Disk Space**
  - 10-100GB+ depending on model size (weights stored locally)
* - **Network Bandwidth**
  - Required for initial model download from Hugging Face
```

:::{tip}
**New to NeMo Gym?** Start with the [Get Started tutorials](../../get-started/index.md) using OpenAI first. Once you understand the workflow, return here to deploy your own models with vLLM.
:::

---

## Set Up

:::{admonition} Already have a vLLM server running?
:class: tip
Skip to [Step 4: Configure NeMo Gym](vllm-quickstart-configure) if you already have a vLLM server deployed and just need to connect NeMo Gym to it.
:::

1. Install vLLM:

   ```bash
   # Create a virtual environment
   uv venv --python 3.12 --seed
   source .venv/bin/activate

   # Install vLLM with dependencies
   uv pip install hf_transfer datasets vllm --torch-backend=auto
   ```

2. Download a Model:

   ```bash
   # Example: Download Qwen3-30B-A3B (supports tool calling)
   HF_HOME=.cache/ \
   HF_HUB_ENABLE_HF_TRANSFER=1 \
       hf download Qwen/Qwen3-30B-A3B
   ```

   :::{dropdown} Popular models for NeMo Gym

   **Models with tool calling support** (recommended):
   - `Qwen/Qwen3-30B-A3B` - 30B parameters, excellent tool calling
   - `meta-llama/Llama-3.1-70B-Instruct` - 70B parameters, high quality
   - `meta-llama/Llama-3.1-8B-Instruct` - 8B parameters, fastest inference

   **For gated models** (Llama, Gemma):

   ```bash
   # Login to Hugging Face first
   huggingface-cli login

   # Then download
   HF_HOME=.cache/ HF_HUB_ENABLE_HF_TRANSFER=1 \
       hf download meta-llama/Llama-3.1-8B-Instruct
   ```

   See [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) for the complete list.
   :::

3. Start vLLM Server:

   ```bash
   HF_HOME=.cache/ \
   HOME=. \
   vllm serve \
       Qwen/Qwen3-30B-A3B \
       --dtype auto \
       --tensor-parallel-size 4 \
       --gpu-memory-utilization 0.9 \
       --enable-auto-tool-choice --tool-call-parser hermes \
       --host 0.0.0.0 \
       --port 10240
   ```

:::{tip}
**Reasoning tokens handled automatically**: NeMo Gym's vLLM adapter parses reasoning tokens (like `<think>` tags) transparently, so you don't need to configure vLLM's `--reasoning-parser` flag. This ensures compatibility with the Responses API format used throughout NeMo Gym.
:::

**✅ Success check**: Visit `http://localhost:10240/health` - you should see a health status response.

---

(vllm-quickstart-configure)=
## Configure

1. Create or update `env.yaml` in your NeMo Gym repository root:

   ```yaml
   policy_base_url: http://localhost:10240/v1  # Your vLLM server URL
   policy_api_key: EMPTY                       # Use EMPTY if no auth configured
   policy_model_name: Qwen/Qwen3-30B-A3B       # Must match model loaded in vLLM
   ```

2. Validate config. 

:::{dropdown} If you have an existing vLLM server elsewhere

Update the configuration to point to your server:

```yaml
policy_base_url: http://your-vllm-server:8000/v1
policy_api_key: EMPTY  # or your API key
policy_model_name: meta-llama/Llama-3.1-8B-Instruct  # match your model
```

Verify your server is accessible:
```bash
curl http://your-vllm-server:8000/v1/models
```
:::

---

## Run

1. Start NeMo Gym Servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"
```

**✅ Success check**: You should see multiple servers starting, including the head server on port 11000.

---

### Test the Integration

```bash
ng_test +entrypoint=responses_api_models/vllm_model
```

**✅ Success check**: All tests should pass.

---

## Next Steps

Now that vLLM is configured, you can:

- **[Collect rollouts](../../get-started/collecting-rollouts.md)** for training data generation
- **[Configure additional parameters](configuration.md)** like token IDs for training
- **[Set up load balancing](optimization.md)** for production throughput


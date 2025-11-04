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

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

---

## Set Up vLLM Server

:::{admonition} Already have a vLLM server?
:class: tip
If you already have a vLLM server running elsewhere, skip to [Configure NeMo Gym](#configure-nemo-gym) and use the "Existing vLLM Server" tab.
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
## Configure NeMo Gym

1. Choose your configuration based on where your vLLM server is running:

   ::::{tab-set}

   :::{tab-item} Local vLLM Server
   If you completed steps 1-3 above and started vLLM on localhost:

   **Create `env.yaml`** in your NeMo Gym repository root:

   ```yaml
   policy_base_url: http://localhost:10240/v1  # Your local vLLM server
   policy_api_key: EMPTY                       # No auth for local server
   policy_model_name: Qwen/Qwen3-30B-A3B       # Must match model you started
   ```
   :::

   :::{tab-item} Existing vLLM Server
   If you have a vLLM server already running elsewhere:

   **Create or update `env.yaml`** in your NeMo Gym repository root:

   ```yaml
   policy_base_url: http://your-vllm-server:8000/v1  # Your vLLM server URL
   policy_api_key: EMPTY                              # Or your API key if configured
   policy_model_name: meta-llama/Llama-3.1-8B-Instruct  # Must match loaded model
   ```

   **Verify server accessibility**:
   ```bash
   # Check server is reachable
   curl http://your-vllm-server:8000/v1/models
   ```

   You should see a JSON response listing the available models.
   :::

   ::::

2. Validate your configuration** (optional but recommended):

   :::{dropdown} Test vLLM connection before starting NeMo Gym

   Catch configuration issues early by testing your vLLM server:

   ```bash
   python -c "
   import openai
   import yaml
   import requests
   
   # Load your configuration
   with open('env.yaml') as f:
       config = yaml.safe_load(f)
   
   base_url = config['policy_base_url']
   
   # Test 1: Check vLLM health endpoint
   print('🔍 Testing vLLM server health...')
   health_url = base_url.replace('/v1', '/health')
   health_response = requests.get(health_url)
   print(f'✅ Health check: {health_response.status_code}')
   
   # Test 2: Verify model is loaded
   print(f'\n🔍 Checking if model is loaded...')
   client = openai.OpenAI(
       api_key=config['policy_api_key'],
       base_url=base_url
   )
   models = client.models.list()
   available_models = [m.id for m in models.data]
   print(f'✅ Available models: {available_models}')
   
   # Test 3: Simple completion
   print(f'\n🔍 Testing completion with {config[\"policy_model_name\"]}...')
   response = client.chat.completions.create(
       model=config['policy_model_name'],
       messages=[{'role': 'user', 'content': 'Say hello'}],
       max_tokens=10
   )
   print(f'✅ Model response: {response.choices[0].message.content}')
   print(f'\n✨ All checks passed! Your vLLM configuration is ready.')
   "
   ```

   **✅ Success check**: You should see three green checkmarks confirming health, model availability, and successful completion.

   **Common errors**:
   - `Connection refused`: vLLM server not running or wrong port
   - `Model not found`: Model name in `env.yaml` doesn't match vLLM
   - `404 Not Found`: Check that `base_url` includes `/v1` path

   :::

---

## Run

1. Start NeMo Gym Servers

   ```bash
   config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
   resources_servers/simple_weather/configs/simple_weather.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

    **✅ Success check**: You should see multiple servers starting, including the head server (default port `11000`).

2. Test with a simple agent interaction:

   ```bash
   # Run a single rollout to verify the full stack works
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/vllm_test.jsonl \
     +limit=1
   ```

   **✅ Success check**: This tests the complete flow: NeMo Gym servers → vLLM adapter → vLLM inference → agent response. You should see the agent complete one interaction and write to `results/vllm_test.jsonl`. This confirms vLLM is responding through NeMo Gym. If this works, you're ready for [rollout collection](gs-collecting-rollouts)!

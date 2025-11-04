(models-nvidia-nim-quick-start)=

# Quick Start

Get NVIDIA NIM connected to NeMo Gym in under 5 minutes.

---

## Before You Start

Ensure you have these prerequisites before configuring NVIDIA NIM with NeMo Gym:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **NeMo Gym installed**
  - Complete [Setup and Installation](../../get-started/setup-installation.md) first
* - **NVIDIA NIM deployed**
  - NIM instance accessible via network (cloud or on-premises)
* - **API endpoint URL**
  - Base URL for NIM inference service (e.g., `https://nim.example.com/v1`)
* - **API key**
  - Authentication credentials from NIM deployment
* - **Model identifier**
  - Name of model deployed in NIM (e.g., `meta/llama-3.1-8b-instruct`)
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

---

## Configure NeMo Gym

1. Set Environment Variables:

   Create or update `env.yaml` in your NeMo Gym repository root:

   ```yaml
   # env.yaml
   policy_base_url: https://your-nim-endpoint.nvidia.com/v1
   policy_api_key: your-nvidia-api-key
   policy_model_name: meta/llama-3.1-8b-instruct
   ```

2. Validate Configuration (optional but recommended):

   :::{dropdown} Test NIM connection before starting NeMo Gym

   Catch configuration issues early by testing your NIM endpoint:

   ```bash
   python -c "
   import openai
   import yaml
   import requests
   
   # Load configuration
   with open('env.yaml') as f:
       config = yaml.safe_load(f)
   
   base_url = config['policy_base_url']
   
   # Test 1: Check NIM health endpoint
   print('🔍 Testing NIM server health...')
   health_url = base_url.replace('/v1', '/health')
   try:
       health_response = requests.get(health_url, timeout=5)
       print(f'✅ Health check: {health_response.status_code}')
   except Exception as e:
       print(f'❌ Health check failed: {e}')
   
   # Test 2: Verify model is available
   print(f'\n🔍 Checking if model is available...')
   client = openai.OpenAI(
       api_key=config['policy_api_key'],
       base_url=base_url
   )
   try:
       models = client.models.list()
       available_models = [m.id for m in models.data]
       print(f'✅ Available models: {available_models}')
   except Exception as e:
       print(f'❌ Model list failed: {e}')
   
   # Test 3: Simple completion
   print(f'\n🔍 Testing completion with {config[\"policy_model_name\"]}...')
   try:
       response = client.chat.completions.create(
           model=config['policy_model_name'],
           messages=[{'role': 'user', 'content': 'Say hello'}],
           max_tokens=10
       )
       print(f'✅ Model response: {response.choices[0].message.content}')
       print(f'\n✨ All checks passed! Your NIM configuration is ready.')
   except Exception as e:
       print(f'❌ Completion failed: {e}')
   "
   ```

   **✅ Success check**: You should see three green checkmarks confirming health, model availability, and successful completion.

   **Common errors**:
   - `Connection refused`: NIM server not accessible or wrong URL
   - `Model not found`: Model name doesn't match NIM deployment
   - `401 Unauthorized`: API key incorrect or invalid

   :::

---

## Run

1. Start NeMo Gym Servers:

   ```bash
   config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
   resources_servers/simple_weather/configs/simple_weather.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

   **✅ Success check**: You should see multiple servers starting, including the head server (default port `11000`).

2. Test with Single Rollout:

   Verify the complete stack before large-scale collection:

   ```bash
   # Run a single rollout to test end-to-end
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/nim_test.jsonl \
     +limit=1
   ```

   **✅ Success check**: This tests NeMo Gym servers → NIM endpoint → agent response. You should see the agent complete one interaction and write to `results/nim_test.jsonl`. If this works, you're ready for rollout collection at scale!

3. Collect Rollouts at Scale:

   Generate large-scale training data using NVIDIA NIM:

   ```bash
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/nim_rollouts.jsonl \
     +limit=1000 \
     +concurrency=50
   ```

   :::{tip}
   **Production tip**: NIM's enterprise infrastructure handles high-concurrency workloads well. Start with `+concurrency=50` and increase based on your NIM deployment capacity and rate limits.
   :::


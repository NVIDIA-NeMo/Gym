(models-openai-quick-start)=

# Quick Start

Get OpenAI connected to NeMo Gym in under 3 minutes.

---

## Before You Start

Ensure you have these prerequisites before configuring OpenAI with NeMo Gym:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **NeMo Gym installed**
  - Complete [Setup and Installation](../../get-started/setup-installation.md) first
* - **OpenAI API account**
  - Sign up at [platform.openai.com](https://platform.openai.com)
* - **OpenAI API key**
  - Generate from OpenAI dashboard under API keys
* - **Model selection**
  - Choose model (e.g., `gpt-4-turbo`, `gpt-3.5-turbo`)
* - **Payment method**
  - Add payment method for API usage (pay-per-token)
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
   policy_base_url: https://api.openai.com/v1
   policy_api_key: sk-your-openai-api-key-here
   policy_model_name: gpt-4-turbo
   ```

   :::{tip}
   **Cost optimization**: Start with `gpt-3.5-turbo` for initial testing - it's 10-30x cheaper than GPT-4 models and still very capable for many tasks.
   :::

2. Validate Configuration (optional but recommended):

   :::{dropdown} Test OpenAI connection before starting NeMo Gym

   Catch configuration issues early by testing your OpenAI API key:

   ```bash
   python -c "
   import openai
   import yaml
   
   # Load configuration
   with open('env.yaml') as f:
       config = yaml.safe_load(f)
   
   # Test connection
   print('🔍 Testing OpenAI API connection...')
   client = openai.OpenAI(
       api_key=config['policy_api_key'],
       base_url=config['policy_base_url']
   )
   
   try:
       # Test 1: List available models
       print('✅ API key valid')
       
       # Test 2: Simple completion
       print(f'\n🔍 Testing completion with {config[\"policy_model_name\"]}...')
       response = client.chat.completions.create(
           model=config['policy_model_name'],
           messages=[{'role': 'user', 'content': 'Say hello'}],
           max_tokens=10
       )
       print(f'✅ Model response: {response.choices[0].message.content}')
       print(f'\n✨ All checks passed! Your OpenAI configuration is ready.')
   except Exception as e:
       print(f'❌ Error: {e}')
   "
   ```

   **✅ Success check**: You should see confirmation that your API key is valid and the model responds.

   **Common errors**:
   - `Authentication failed`: API key incorrect or expired
   - `Model not found`: Model name typo or no access to that model
   - `Quota exceeded`: No credits or payment method not configured

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
     +output_jsonl_fpath=results/openai_test.jsonl \
     +limit=1
   ```

   **✅ Success check**: This tests NeMo Gym servers → OpenAI API → agent response. You should see the agent complete one interaction and write to `results/openai_test.jsonl`. If this works, you're ready for rollout collection!

3. Collect Rollouts:

   Generate training data using OpenAI:

   ```bash
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/openai_rollouts.jsonl \
     +limit=100 \
     +concurrency=10
   ```

   :::{tip}
   **Cost management**: Start with lower concurrency (`+concurrency=10`) to control costs. OpenAI charges per token, so monitor usage in the OpenAI dashboard as you scale up.
   :::


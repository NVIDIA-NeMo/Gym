(models-openrouter-quick-start)=

# Quick Start

Get OpenRouter connected to NeMo Gym in under 3 minutes.

---

## Before You Start

Ensure you have these prerequisites before configuring OpenRouter with NeMo Gym:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **NeMo Gym installed**
  - Complete [Setup and Installation](../../get-started/setup-installation.md) first
* - **OpenRouter account**
  - Sign up at [openrouter.ai](https://openrouter.ai)
* - **OpenRouter API key**
  - Generate from OpenRouter dashboard
* - **Model selection**
  - Browse [OpenRouter model catalog](https://openrouter.ai/models) for available models
* - **Payment method**
  - Add credits or payment method for API usage
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Configure NeMo Gym

1. Set Environment Variables:

   Create or update `env.yaml` in your NeMo Gym repository root:

   ```yaml
   # env.yaml
   policy_base_url: https://openrouter.ai/api/v1
   policy_api_key: sk-or-your-openrouter-api-key
   policy_model_name: openai/gpt-4-turbo
   ```

   :::{tip}
   **Model format**: OpenRouter uses `provider/model-name` format (e.g., `openai/gpt-4-turbo`, `anthropic/claude-3-sonnet`).
   :::

2. Validate Configuration (optional but recommended):

   :::{dropdown} Test OpenRouter connection before starting NeMo Gym

   Catch configuration issues early by testing your OpenRouter API key:

   ```bash
   python -c "
   import openai
   import yaml
   
   # Load configuration
   with open('env.yaml') as f:
       config = yaml.safe_load(f)
   
   # Test connection
   print('🔍 Testing OpenRouter API connection...')
   client = openai.OpenAI(
       api_key=config['policy_api_key'],
       base_url=config['policy_base_url']
   )
   
   try:
       # Test completion
       print(f'🔍 Testing model: {config[\"policy_model_name\"]}...')
       response = client.chat.completions.create(
           model=config['policy_model_name'],
           messages=[{'role': 'user', 'content': 'Say hello'}],
           max_tokens=10
       )
       print(f'✅ Model response: {response.choices[0].message.content}')
       print(f'\n✨ All checks passed! Your OpenRouter configuration is ready.')
   except Exception as e:
       print(f'❌ Error: {e}')
   "
   ```

   **✅ Success check**: You should see confirmation that the model responds.

   **Common errors**:
   - `Authentication failed`: API key incorrect or doesn't start with `sk-or-`
   - `Model not found`: Model identifier format incorrect or model not available
   - `Insufficient credits`: No credits or payment method not configured

   :::

---

## Run

1. Start NeMo Gym Servers:

   ```bash
   config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
   resources_servers/simple_weather/configs/simple_weather.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

   :::{note}
   **Why `openai_model`?** OpenRouter implements the OpenAI API specification, so it uses the same `openai_model` adapter. Just change the `base_url` in your `env.yaml` to point to OpenRouter—no separate adapter needed.
   :::

   **✅ Success check**: You should see multiple servers starting, including the head server (default port `11000`).

2. Test with Single Rollout:

   Verify the complete stack before large-scale collection:

   ```bash
   # Run a single rollout to test end-to-end
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/openrouter_test.jsonl \
     +limit=1
   ```

   **✅ Success check**: This tests NeMo Gym servers → OpenRouter → provider → agent response. You should see the agent complete one interaction and write to `results/openrouter_test.jsonl`. If this works, you're ready for rollout collection!

3. Collect Rollouts:

   Generate training data using OpenRouter:

   ```bash
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/openrouter_rollouts.jsonl \
     +limit=null \
     +num_samples_in_parallel=10
   ```

   :::{note}
   The example dataset contains 5 samples. For production use, replace with your full dataset path and adjust `limit` as needed (or use `+limit=null` to process all).
   :::

   :::{tip}
   **Cost optimization**: OpenRouter shows per-request costs in the dashboard. Monitor usage and compare prices between providers to optimize costs.
   :::


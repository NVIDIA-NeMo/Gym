(models-azure-openai-quick-start)=

# Quick Start

Get Azure OpenAI connected to NeMo Gym in under 5 minutes.

---

## Before You Start

Ensure you have these prerequisites before configuring Azure OpenAI with NeMo Gym:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **NeMo Gym installed**
  - Complete [Setup and Installation](../../get-started/setup-installation.md) first
* - **Azure subscription**
  - Active Azure subscription with billing enabled
* - **Azure OpenAI access**
  - Request access at [azure.microsoft.com/products/ai-services/openai-service](https://azure.microsoft.com/products/ai-services/openai-service)
* - **Azure OpenAI resource**
  - Deployed Azure OpenAI resource in Azure portal
* - **Model deployment**
  - At least one model deployed (e.g., GPT-4, GPT-3.5)
* - **API endpoint and key**
  - From Azure portal under Keys and Endpoint
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Set Up Azure OpenAI Resource

If you don't have an Azure OpenAI resource yet:

1. Create Azure OpenAI Resource:

   - Navigate to [Azure portal](https://portal.azure.com)
   - Search for "Azure OpenAI"
   - Click "Create" and fill in resource details
   - Choose region (consider data residency requirements)
   - Wait for deployment to complete

2. Deploy a Model:

   - Open your Azure OpenAI resource
   - Navigate to "Model deployments"
   - Click "Create new deployment"
   - Choose model (e.g., `gpt-4`, `gpt-35-turbo`)
   - Give it a deployment name (e.g., `my-gpt-4-deployment`)
   - Note the deployment name - you'll use this in configuration

3. Get Endpoint and Key:

   - Navigate to "Keys and Endpoint"
   - Copy "Endpoint" URL
   - Copy one of the keys (Key 1 or Key 2)

:::{tip}
**Deployment names are custom**: Unlike standard OpenAI, Azure uses your custom deployment names, not base model names like `gpt-4`.
:::

---

## Configure NeMo Gym

1. Set Environment Variables:

   Create or update `env.yaml` in your NeMo Gym repository root:

   ```yaml
   # env.yaml
   policy_base_url: https://your-resource.openai.azure.com
   policy_api_key: your-azure-api-key
   policy_model_name: my-gpt-4-deployment
   ```

   :::{tip}
   **Endpoint format**: Use the base Azure OpenAI resource URL without any path suffix. Example: `https://myresource.openai.azure.com`
   :::

2. Validate Configuration (optional but recommended):

   :::{dropdown} Test Azure OpenAI connection before starting NeMo Gym

   Catch configuration issues early by testing your Azure endpoint:

   ```bash
   python -c "
   import openai
   import yaml
   
   # Load configuration
   with open('env.yaml') as f:
       config = yaml.safe_load(f)
   
   # Test connection
   print('🔍 Testing Azure OpenAI connection...')
   client = openai.AzureOpenAI(
       api_key=config['policy_api_key'],
       api_version='2024-10-21',  # Use latest from Azure docs
       azure_endpoint=config['policy_base_url']
   )
   
   try:
       # Test completion
       print(f'🔍 Testing deployment: {config[\"policy_model_name\"]}...')
       response = client.chat.completions.create(
           model=config['policy_model_name'],
           messages=[{'role': 'user', 'content': 'Say hello'}],
           max_tokens=10
       )
       print(f'✅ Model response: {response.choices[0].message.content}')
       print(f'\n✨ All checks passed! Your Azure OpenAI configuration is ready.')
   except Exception as e:
       print(f'❌ Error: {e}')
   "
   ```

   **✅ Success check**: You should see confirmation that your deployment responds.

   **Common errors**:
   - `DeploymentNotFound`: Deployment name doesn't match Azure portal
   - `Unauthorized`: API key incorrect or expired
   - `InvalidApiVersion`: API version not supported

   :::

---

## Run

1. Start NeMo Gym Servers:

   ```bash
   config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml,\
   resources_servers/simple_weather/configs/simple_weather.yaml"

   ng_run "+config_paths=[${config_paths}]" \
       +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
   ```

   :::{tip}
   **API version**: Check [Azure OpenAI API versions](https://learn.microsoft.com/azure/ai-services/openai/reference) for the latest stable version.
   :::

   **✅ Success check**: You should see multiple servers starting, including the head server (default port `11000`).

2. Test with Single Rollout:

   Verify the complete stack before large-scale collection:

   ```bash
   # Run a single rollout to test end-to-end
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/azure_test.jsonl \
     +limit=1
   ```

   **✅ Success check**: This tests NeMo Gym servers → Azure OpenAI → agent response. You should see the agent complete one interaction and write to `results/azure_test.jsonl`. If this works, you're ready for rollout collection!

3. Collect Rollouts:

   Generate training data using Azure OpenAI:

   ```bash
   ng_collect_rollouts \
     +agent_name=simple_weather_simple_agent \
     +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
     +output_jsonl_fpath=results/azure_rollouts.jsonl \
     +limit=100 \
     +num_samples_in_parallel=10
   ```

   :::{tip}
   **Azure billing**: Monitor usage in Azure portal under Cost Management. Azure OpenAI charges per token like standard OpenAI.
   :::


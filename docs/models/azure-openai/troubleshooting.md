(models-azure-openai-troubleshooting)=

# Troubleshooting

Common issues and solutions when working with Azure OpenAI in NeMo Gym, organized by issue type.

---

## Authentication Issues

Problems with API keys and Azure access.

:::{dropdown} {octicon}`lock;1em;sd-text-danger` Unauthorized or authentication failure

**Solution**:

1. Verify API key from Azure portal:
   - Navigate to your Azure OpenAI resource
   - Click "Keys and Endpoint"
   - Use Key 1 or Key 2
   - Ensure key is copied completely without extra spaces

2. Check key in `env.yaml`:
   ```yaml
   # Verify format and spacing
   policy_api_key: your-actual-key-here
   ```

3. Test key directly:
   ```bash
   # Test with curl
   curl -X POST "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/chat/completions?api-version=2024-10-21" \
     -H "api-key: YOUR-API-KEY" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hi"}]}'
   ```

4. Regenerate key if needed:
   - Azure portal → Keys and Endpoint → Regenerate keys
   - Update `env.yaml` with new key
   - Restart NeMo Gym servers

:::

:::{dropdown} {octicon}`shield;1em;sd-text-warning` Access denied or subscription issues

**Solution**:

1. Verify Azure OpenAI access approved:
   - Azure OpenAI requires application approval
   - Check [Azure OpenAI access](https://learn.microsoft.com/azure/ai-services/openai/overview#how-do-i-get-access-to-azure-openai)
   - Wait for approval email before proceeding

2. Check subscription status:
   - Ensure Azure subscription is active
   - Verify billing is enabled
   - Check for spending limits

3. Verify resource access:
   - Ensure you have Contributor or Owner role on resource
   - Check Azure RBAC permissions
   - Verify resource group access

:::

---

## Configuration Issues

Settings specific to Azure OpenAI deployment.

:::{dropdown} {octicon}`search;1em;sd-text-danger` Deployment not found

**Solution**:

1. Verify deployment name (not model name):
   ```yaml
   # Correct - your deployment name
   policy_model_name: my-gpt-4-deployment
   
   # Incorrect - base model name
   policy_model_name: gpt-4
   ```

2. Check deployment exists in Azure portal:
   - Navigate to Azure OpenAI resource
   - Click "Model deployments"
   - Find your deployment name
   - Copy exact name to `env.yaml`

3. Verify deployment is active:
   - Check deployment status in portal
   - Ensure deployment completed successfully
   - Wait if deployment is in progress

:::

:::{dropdown} {octicon}`alert;1em;sd-text-danger` API version not supported

**Solution**:

1. Update to latest stable version:
   ```bash
   ng_run "+config_paths=[${config_paths}]" \
       +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21
   ```

2. Check supported versions:
   - Visit [Azure OpenAI API versions](https://learn.microsoft.com/azure/ai-services/openai/reference)
   - Use latest stable version (not preview for production)
   - Format: `YYYY-MM-DD`

3. Common version issues:
   - Wrong format (use `2024-10-21` not `v2024-10-21`)
   - Deprecated version (check documentation)
   - Preview version in production (use stable)

:::

:::{dropdown} {octicon}`globe;1em;sd-text-warning` Endpoint URL format incorrect

**Check**:

1. Verify URL format:
   ```yaml
   # Correct format
   policy_base_url: https://your-resource.openai.azure.com
   
   # Common mistakes
   policy_base_url: https://api.openai.com/v1                # OpenAI, not Azure
   policy_base_url: https://your-resource.azure.com          # Missing .openai
   policy_base_url: https://your-resource.openai.azure.com/  # Trailing slash
   ```

2. Find correct endpoint:
   - Azure portal → Your resource → Keys and Endpoint
   - Copy "Endpoint" URL exactly as shown
   - Should end with `.openai.azure.com`

3. Check resource name:
   - Must match your Azure OpenAI resource name
   - Case-sensitive
   - No typos

:::

---

## Regional & Availability Issues

Problems specific to Azure regional deployment.

:::{dropdown} {octicon}`location;1em;sd-text-warning` Model not available in region

**Solution**:

1. Check model availability for your region:
   - Visit [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models)
   - Find your Azure region
   - Check which models are available

2. Options to resolve:
   - Deploy model in a different Azure region
   - Use different model available in your region
   - Request access for specific model in region

3. Common regional limitations:
   - GPT-4: Limited to specific regions
   - GPT-3.5: More widely available
   - New models: Gradual regional rollout

:::

:::{dropdown} {octicon}`clock;1em;sd-text-info` Higher latency than expected

**Diagnose**:

1. Check geographic proximity:
   - Azure resource region vs NeMo Gym location
   - Cross-region latency can be significant
   - Consider deploying closer to infrastructure

2. Test network latency:
   ```bash
   # Measure latency to Azure endpoint
   curl -w "Time: %{time_total}s\n" -o /dev/null -s \
     https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/chat/completions?api-version=2024-10-21
   ```

3. Consider alternatives:
   - Deploy Azure OpenAI in same region as NeMo Gym
   - Use Azure networking (VNet, Express Route)
   - Evaluate if latency acceptable for use case

:::

---

## Runtime Issues

Problems during rollout collection.

:::{dropdown} {octicon}`stop;1em;sd-text-danger` Rate limiting or quota exceeded

**Solution**:

1. Check quota limits:
   - Azure portal → Your resource → Quotas
   - View tokens per minute (TPM) limits
   - Check requests per minute (RPM) limits

2. Request quota increase:
   - Azure portal → Your resource → Quotas
   - Click "Request Quota Increase"
   - Fill out form with justification
   - Wait for approval (usually 24-48 hours)

3. Optimize usage:
   ```bash
   # Reduce concurrency
   ng_collect_rollouts +num_samples_in_parallel=5  # Lower value
   ```

4. Monitor usage:
   - Azure portal → Your resource → Metrics
   - Track token usage
   - Set up alerts for quota limits

:::

:::{dropdown} {octicon}`x;1em;sd-text-danger` Context length exceeded

**How Azure OpenAI handles this**: Azure OpenAI returns a 400 error when the context length is exceeded. This error propagates through NeMo Gym's exception handling and is returned as a 500 status code.

**To prevent**:
- Use deployments with appropriate context lengths
- Monitor conversation history
- Implement truncation strategies
- Context limits by model:
  - `gpt-4`: 8K tokens
  - `gpt-4-32k`: 32K tokens
  - `gpt-4-turbo`: 128K tokens
  - `gpt-35-turbo`: 16K tokens

:::

:::{dropdown} {octicon}`report;1em;sd-text-warning` Content filtering triggered

**Azure-specific**: Azure OpenAI has content filtering enabled by default.

**Solution**:

1. Review content filters:
   - Azure portal → Your resource → Content filters
   - See what triggered filter
   - Review severity levels

2. Modify prompts if needed:
   - Adjust agent instructions
   - Remove potentially problematic content
   - Test with different phrasings

3. Configure filters (if needed):
   - Content filters can be customized
   - Requires appropriate permissions
   - Contact Azure support for enterprise needs

:::

---

## Cost & Billing Issues

Azure-specific billing and cost management.

:::{dropdown} {octicon}`credit-card;1em;sd-text-warning` Unexpected costs or billing

**Diagnose**:

1. Check actual usage:
   - Azure portal → Cost Management
   - Filter by Azure OpenAI resource
   - Review token usage and costs

2. Common causes:
   - Higher traffic than expected
   - Expensive model (GPT-4 vs GPT-3.5)
   - Long prompts or responses
   - Multi-turn conversations

3. Set up cost controls:
   - Azure portal → Cost Management → Budgets
   - Create budget alerts
   - Set spending limits
   - Monitor daily usage

4. Optimize costs:
   - Use GPT-3.5 for development
   - Reduce max_tokens
   - Lower concurrency during testing
   - Monitor token usage patterns

:::

---

## Getting Help

If you're still experiencing issues:

1. Check Azure status: Visit [Azure status dashboard](https://status.azure.com) for service outages
2. Review Azure logs: Check Activity log and Diagnostics logs in Azure portal
3. Test Azure API directly: Use Azure OpenAI Studio to isolate issues
4. Verify configuration: Use `ng_dump_config` to see resolved configuration
5. Run tests: Execute `ng_test +entrypoint=responses_api_models/azure_openai_model`
6. Contact Microsoft support: Azure enterprise customers have support channels

:::{seealso}
- **[Configuration Reference](configuration.md)** - Verify your settings
- **[Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)** - Official Azure docs
- **[Quick Start](quick-start.md)** - Validate your setup
:::


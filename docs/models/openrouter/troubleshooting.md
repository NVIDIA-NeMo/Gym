(models-openrouter-troubleshooting)=

# Troubleshooting

Common issues and solutions when working with OpenRouter in NeMo Gym, organized by issue type.

---

## Authentication Issues

Problems with API keys and OpenRouter access.

:::{dropdown} {octicon}`lock;1em;sd-text-danger` Authentication failure or invalid API key

**Solution**:

1. Verify API key format:
   - OpenRouter API keys start with `sk-or-`
   - Check for extra spaces or newlines in `env.yaml`
   - Ensure key is copied completely

2. Check API key status:
   ```bash
   # Test key directly
   curl https://openrouter.ai/api/v1/auth/key \
     -H "Authorization: Bearer sk-or-your-key"
   ```

3. Verify key in OpenRouter dashboard:
   - Visit [openrouter.ai/keys](https://openrouter.ai/keys)
   - Ensure key hasn't been revoked
   - Check key permissions

4. Regenerate key if needed:
   - Delete old key in dashboard
   - Create new key
   - Update `env.yaml`

:::

:::{dropdown} {octicon}`credit-card;1em;sd-text-danger` Insufficient credits

**Problem**: API calls fail due to no credits or payment method.

**Solution**:

1. Add credits:
   - Visit OpenRouter dashboard
   - Navigate to Billing section
   - Add credits or configure payment method

2. Check current balance:
   - Dashboard shows remaining credits
   - Set up balance alerts
   - Monitor spending during large runs

3. Verify payment method:
   - Ensure credit card is valid
   - Check for payment failures
   - Update payment method if needed

:::

---

## Model & Provider Issues

Problems with model selection and provider availability.

:::{dropdown} {octicon}`search;1em;sd-text-danger` Model not found

**Solution**:

1. Verify model identifier format:
   ```yaml
   # Correct - provider/model-name format
   policy_model_name: openai/gpt-4-turbo
   policy_model_name: anthropic/claude-3-sonnet
   
   # Incorrect - missing provider prefix
   policy_model_name: gpt-4-turbo
   policy_model_name: claude-3-sonnet
   ```

2. Check model availability:
   - Visit [OpenRouter model catalog](https://openrouter.ai/models)
   - Verify model is currently available
   - Check for model name changes

3. Test model directly:
   ```bash
   curl https://openrouter.ai/api/v1/chat/completions \
     -H "Authorization: Bearer sk-or-your-key" \
     -H "Content-Type: application/json" \
     -d '{"model":"openai/gpt-4-turbo","messages":[{"role":"user","content":"Hi"}]}'
   ```

:::

:::{dropdown} {octicon}`alert;1em;sd-text-warning` Provider unavailable

**Problem**: Underlying provider (OpenAI, Anthropic, etc.) is down.

**Solution**:

1. Check provider status:
   - OpenRouter dashboard shows provider status
   - Visit provider status pages:
     - [OpenAI Status](https://status.openai.com)
     - [Anthropic Status](https://status.anthropic.com)

2. Use fallback model:
   ```bash
   # Switch to different provider temporarily
   ng_run "+config_paths=[${config_paths}]" \
       +policy_model_name=anthropic/claude-3-sonnet
   ```

3. Wait for provider recovery:
   - OpenRouter queues requests during brief outages
   - Extended outages require switching providers

:::

:::{dropdown} {octicon}`versions;1em;sd-text-info` Model version or deprecation

**Problem**: Model identifier changed or deprecated.

**Solution**:

1. Check model catalog for updates:
   - OpenRouter catalog shows current model names
   - Look for deprecation notices
   - Update to new model identifiers

2. Update configuration:
   ```yaml
   # Example: Update to new model version
   policy_model_name: openai/gpt-4-turbo  # Current
   # policy_model_name: openai/gpt-4  # Older version
   ```

:::

---

## Configuration Issues

Problems with endpoint URLs and settings.

:::{dropdown} {octicon}`globe;1em;sd-text-warning` Incorrect endpoint URL

**Check**:

1. Verify base URL format:
   ```yaml
   # Correct
   policy_base_url: https://openrouter.ai/api/v1
   
   # Incorrect
   policy_base_url: https://openrouter.ai          # Missing /api/v1
   policy_base_url: https://api.openrouter.ai/v1   # Wrong subdomain
   ```

2. Test endpoint accessibility:
   ```bash
   curl https://openrouter.ai/api/v1/models \
     -H "Authorization: Bearer sk-or-your-key"
   ```

:::

:::{dropdown} {octicon}`gear;1em;sd-text-info` Using wrong model adapter

**Problem**: Trying to use a custom OpenRouter adapter instead of OpenAI adapter.

**Solution**: OpenRouter uses OpenAI-compatible endpoints. Always use the `openai_model` configuration:

```bash
# Correct
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,..."

# Incorrect - there is no openrouter_model adapter
config_paths="responses_api_models/openrouter_model/configs/..."
```

:::

---

## Runtime Issues

Problems during rollout collection.

:::{dropdown} {octicon}`stop;1em;sd-text-danger` Rate limiting

**Solution**:

1. Check rate limits for your plan:
   - Free tier: Limited requests per minute
   - Paid tier: Higher limits
   - Visit OpenRouter dashboard for current limits

2. Reduce concurrency:
   ```bash
   ng_collect_rollouts +concurrency=5  # Lower value
   ```

3. Upgrade plan if needed:
   - Contact OpenRouter for higher limits
   - Consider distributing across multiple accounts (not recommended)

:::

:::{dropdown} {octicon}`clock;1em;sd-text-warning` Request timeout

**Solution**:

1. Check provider status: Some providers may have higher latency.

2. Verify network connectivity: Ensure stable connection to OpenRouter.

3. Try different provider:
   ```bash
   # If one provider is slow, try another
   ng_run "+config_paths=[${config_paths}]" \
       +policy_model_name=anthropic/claude-3-sonnet
   ```

4. Monitor OpenRouter status: Check dashboard for provider latency metrics.

:::

:::{dropdown} {octicon}`x;1em;sd-text-danger` Context length exceeded

**How NeMo Gym handles this**: The OpenAI adapter automatically catches context length errors and returns an empty response.

**To prevent**:
- Check context limits for your chosen model in OpenRouter catalog
- Different models have different context lengths
- Implement conversation history truncation
- Choose models with appropriate context windows

:::

---

## Cost Issues

Managing costs across multiple providers.

:::{dropdown} {octicon}`graph;1em;sd-text-warning` Higher than expected costs

**Diagnose and fix**:

1. Check usage in dashboard:
   - OpenRouter dashboard shows detailed usage
   - Filter by model and provider
   - Review per-request costs

2. Compare provider costs:
   - Some providers charge more than others
   - Check OpenRouter catalog for current pricing
   - Consider switching to cheaper alternatives

3. Optimize model selection:
   - Use cheaper models for development (GPT-3.5, Claude Haiku)
   - Reserve expensive models for production
   - Test with open-source models first

4. Monitor token usage:
   - Long prompts increase costs
   - Verbose responses cost more
   - Optimize prompt length and max_tokens

:::

---

## Getting Help

If you're still experiencing issues:

1. Check OpenRouter status: Visit [OpenRouter status](https://openrouter.ai/status) or dashboard
2. Review OpenRouter logs: Check request logs in OpenRouter dashboard
3. Test OpenRouter API directly: Use `curl` to isolate NeMo Gym vs OpenRouter issues
4. Verify configuration: Use `ng_dump_config` to see resolved configuration values
5. Run tests: Execute `ng_test +entrypoint=responses_api_models/openai_model`
6. Contact OpenRouter support: Visit [OpenRouter Discord](https://discord.gg/openrouter) or support channels

:::{seealso}
- **[Configuration Reference](configuration.md)** - Verify your settings
- **[OpenRouter Documentation](https://openrouter.ai/docs)** - Official OpenRouter docs
- **[Quick Start](quick-start.md)** - Validate your setup
:::


(models-openai-troubleshooting)=

# Troubleshooting

Common issues and solutions when working with OpenAI in NeMo Gym, organized by issue type.

---

## Authentication Issues

Problems with API keys and access.

:::{dropdown} Authentication failure or invalid API key
:icon: lock
:color: danger

**Solution**:

1. Verify API key format:
   - OpenAI API keys typically start with `sk-`
   - Check for extra spaces or newlines in `env.yaml`
   - Ensure key is copied completely
   - Note: NeMo Gym passes the key as-is to OpenAI without format validation

2. Check API key status:
   ```bash
   # Test key directly
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer sk-your-api-key"
   ```

3. Verify key in OpenAI dashboard:
   - Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Ensure key hasn't been revoked or expired
   - Check key permissions if using restricted keys

4. Common fixes:
   - Regenerate API key if compromised
   - Remove and re-add key to `env.yaml`
   - Restart NeMo Gym servers after updating key

:::

:::{dropdown} Insufficient quota or credits
:icon: credit-card
:color: danger

**Problem**: API calls fail due to billing issues.

**Solution**:

1. Check usage limits:
   - Visit OpenAI dashboard Usage section
   - Review current usage vs limits
   - Check if trial credits expired

2. Add payment method:
   - Navigate to Billing section
   - Add credit card for pay-as-you-go
   - Verify payment method is active

3. Check spending limits:
   - Review hard and soft spending limits
   - Increase limits if needed
   - Set up budget alerts

:::

---

## Rate Limiting Issues

Problems with request throttling and API limits.

:::{dropdown} Too many requests error (429)
:icon: stop
:color: danger

**Solution**:

1. Reduce concurrency:
   ```bash
   # Lower concurrent requests
   ng_collect_rollouts +num_samples_in_parallel=5  # Start low
   ```

2. Check rate limits for your tier:
   - Free tier: Lower limits
   - Pay-as-you-go: Higher limits
   - Enterprise: Custom limits

3. Implement delays:
   - NeMo Gym handles retries automatically
   - Very high concurrency may still hit limits
   - Gradually increase `+num_samples_in_parallel` to find your limit

4. Upgrade tier if needed:
   - Pay-as-you-go has higher limits than free tier
   - Contact OpenAI for enterprise limits

:::

:::{dropdown} Request timeout errors
:icon: clock
:color: warning

**Solution**:

1. Check network connectivity: Ensure stable connection to OpenAI API.

2. Review model selection: Larger models (GPT-4) may have longer response times.

3. Reduce response length:
   ```yaml
   # In your agent configuration
   max_tokens: 500  # Lower value for faster responses
   ```

4. Monitor OpenAI status: Check [status.openai.com](https://status.openai.com) for outages.

:::

---

## Configuration Issues

Settings that need to be configured correctly.

:::{dropdown} Model not found or not available
:icon: search
:color: warning

**Solution**:

1. Verify model name:
   ```yaml
   # Correct model names
   policy_model_name: gpt-4-turbo        # ✅
   policy_model_name: gpt-3.5-turbo      # ✅
   
   # Common typos
   policy_model_name: gpt4-turbo         # ❌ Missing hyphen
   policy_model_name: gpt-3.5            # ❌ Incomplete name
   ```

2. Check model access:
   - Some models require special access
   - Beta models may not be available to all accounts
   - Visit OpenAI dashboard to see available models

3. Use latest model names:
   - OpenAI updates model names periodically
   - Check [OpenAI Models documentation](https://platform.openai.com/docs/models)
   - Update `env.yaml` with current model identifiers

:::

:::{dropdown} Incorrect endpoint URL
:icon: alert
:color: warning

**Check**:

1. Verify base URL format:
   ```yaml
   # Correct
   policy_base_url: https://api.openai.com/v1
   
   # Incorrect
   policy_base_url: https://api.openai.com     # Missing /v1
   policy_base_url: https://openai.com/v1      # Wrong domain
   ```

2. Check for typos: Double-check URL spelling and format.

3. Test endpoint directly:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer sk-your-api-key"
   ```

:::

---

## Runtime Issues

Problems that occur during rollout collection.

:::{dropdown} Context length exceeded errors
:icon: x
:color: danger

**How NeMo Gym handles this**: When OpenAI returns a context length error, it will propagate through NeMo Gym's error handling and cause the individual rollout to fail. The rollout collection will continue processing remaining samples.

**To prevent**:
- Use models with appropriate context lengths
- Monitor conversation history length
- Implement truncation in your resource server logic
- Consider context window sizes:
  - `gpt-3.5-turbo`: 16K tokens
  - `gpt-4`: 8K tokens (base model)
  - `gpt-4-32k`: 32K tokens
  - `gpt-4-turbo`: 128K tokens
  - `gpt-4o`: 128K tokens
  - See [OpenAI Models documentation](https://platform.openai.com/docs/models) for current limits

**Note**: If you need automatic fallback behavior for context length errors (returning empty responses to continue rollouts), consider implementing custom error handling in your resource server or agent logic.

:::

:::{dropdown} Content policy violations
:icon: report
:color: warning

**Problem**: Requests rejected due to content policy.

**Solution**:

1. Review OpenAI content policy:
   - Check for prohibited content in prompts
   - Review agent outputs that might trigger filters
   - Visit [OpenAI Usage Policies](https://openai.com/policies/usage-policies)

2. Modify prompts if needed:
   - Remove potentially problematic content
   - Adjust agent instructions
   - Test with different phrasings

3. Appeal if incorrect:
   - Contact OpenAI support if legitimate content is blocked
   - Provide context for your use case

:::

:::{dropdown} Inconsistent response quality
:icon: iterations
:color: info

**Diagnose**:

1. Check temperature settings: Higher temperature = more variability.

2. Review prompt clarity: Ambiguous prompts lead to inconsistent responses.

3. Consider model selection:
   - GPT-3.5: Faster but less consistent
   - GPT-4: Slower but more consistent
   - Test both for your use case

4. Monitor response patterns: Look for systematic issues in outputs.

:::

---

## Cost Issues

Managing unexpected costs or budget concerns.

:::{dropdown} Higher than expected costs
:icon: graph
:color: warning

**Diagnose and fix**:

1. Check actual token usage:
   - Review usage in OpenAI dashboard
   - Compare to estimates
   - Identify high-token rollouts

2. Common causes:
   - Very long system prompts
   - Verbose model responses
   - Multi-turn conversations with full history
   - Using GPT-4 when GPT-3.5 would suffice

3. Optimization strategies:
   - Switch to `gpt-3.5-turbo` for development
   - Reduce `max_tokens` parameter
   - Shorten system prompts
   - Implement conversation history truncation

4. Set up alerts:
   - Configure budget alerts in OpenAI dashboard
   - Monitor daily usage during large runs
   - Use spending limits to prevent overruns

:::

---

## Getting Help

If you're still experiencing issues:

1. Check OpenAI status: Visit [status.openai.com](https://status.openai.com) for service outages
2. Review OpenAI logs: Check API usage logs in OpenAI dashboard
3. Test API directly: Use `curl` or OpenAI Python client to isolate NeMo Gym vs OpenAI issues
4. Verify configuration: Use `ng_dump_config` to see resolved configuration values
5. Run tests: Execute `ng_test +entrypoint=responses_api_models/openai_model`
6. Contact OpenAI support: For API-specific issues or billing questions

:::{seealso}
- **[Configuration Reference](configuration.md)** - Verify your settings
- **[OpenAI Documentation](https://platform.openai.com/docs)** - API reference and guides
- **[Quick Start](quick-start.md)** - Validate your setup
:::


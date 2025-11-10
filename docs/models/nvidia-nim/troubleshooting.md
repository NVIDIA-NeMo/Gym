(models-nvidia-nim-troubleshooting)=

# Troubleshooting

Common issues and solutions when working with NVIDIA NIM in NeMo Gym, organized by issue type.

---

## Connection & Authentication Issues

Problems that prevent basic connectivity or authentication.

:::{dropdown} Connection refused or cannot connect to NIM server
:icon: x-circle
:color: danger

**Check these**:

1. **NIM endpoint is accessible**:
   ```bash
   curl https://your-nim-endpoint.nvidia.com/health
   ```
   Should return health status.

2. **Correct URL in configuration**:
   ```yaml
   # Correct - includes /v1 path
   policy_base_url: https://nim.example.com/v1
   
   # Incorrect - missing /v1
   policy_base_url: https://nim.example.com
   ```

3. **Network connectivity**: Ensure firewall rules and security groups allow access to NIM endpoint.

4. **VPN/Private endpoints**: If using private NIM deployment, verify VPN connection or private endpoint configuration.

:::

:::{dropdown} API key rejected or unauthorized access
:icon: lock
:color: danger

**Solution**:

1. **Verify API key format and validity**:
   - Check API key is correct in `env.yaml`
   - Ensure API key hasn't expired
   - Verify API key has necessary permissions

2. **Test authentication directly**:
   ```bash
   curl https://your-nim-endpoint.nvidia.com/v1/models \
     -H "Authorization: Bearer your-api-key"
   ```

3. **Check access controls**: Verify your API key has access to the specific model you're trying to use.

4. **Rotate key if needed**: Generate new API key from NIM management console if current key is compromised.

:::

---

## Configuration Issues

Settings that need to be configured correctly for specific features.

:::{dropdown} Model not found or not available
:icon: search
:color: warning

**Solution**:

1. **Verify model is deployed in NIM**:
   ```bash
   curl https://your-nim-endpoint.nvidia.com/v1/models \
     -H "Authorization: Bearer your-api-key"
   ```
   Check that your model name appears in the response.

2. **Match model identifier exactly**:
   ```yaml
   # Model identifier must match exactly what NIM returns
   policy_model_name: meta/llama-3.1-8b-instruct
   ```

3. **Check model deployment status**: Verify model is fully deployed and healthy in NIM management console.

4. **Verify model access permissions**: Ensure your API key has permission to access the specific model.

:::

:::{dropdown} Incorrect API version or endpoint format
:icon: alert
:color: warning

**Check**:

1. **Endpoint includes `/v1`**:
   ```yaml
   # Correct
   policy_base_url: https://nim.example.com/v1
   ```

2. **Using OpenAI-compatible format**: NIM requires OpenAI adapter configuration, not a custom NIM adapter.

3. **Check NIM API version**: Some NIM deployments may use different API versions - consult your NIM documentation.

:::

:::{dropdown} Using wrong model adapter
:icon: gear
:color: info

**Problem**: Trying to use a NIM-specific adapter instead of OpenAI adapter.

**Solution**: NVIDIA NIM uses OpenAI-compatible endpoints. Always use the `openai_model` configuration:

```bash
# Correct
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,..."

# Incorrect - there is no nvidia_nim_model adapter
config_paths="responses_api_models/nvidia_nim_model/configs/..."
```

:::

---

## Runtime & Performance Issues

Problems that occur during execution or affect throughput and speed.

:::{dropdown} Slow inference or unexpected latency
:icon: clock
:color: warning

**Optimize**:

1. **Check NIM instance capacity**:
   - Monitor GPU utilization in NIM console
   - Verify instance isn't overloaded
   - Consider scaling up NIM deployment

2. **Adjust concurrency**:
   ```bash
   # Start conservative and increase
   ng_collect_rollouts +num_samples_in_parallel=25  # Lower concurrency
   ```

3. **Review network latency**: If NIM is in different region/cloud, network latency may be significant.

4. **Enable monitoring**: Use NIM's built-in metrics to identify bottlenecks.

5. **Check auto-scaling**: Ensure NIM auto-scaling is configured and responding to load.

:::

:::{dropdown} Rate limiting or throttling
:icon: stop
:color: danger

**Solution**:

1. **Check rate limits**: Review your NIM deployment's rate limit configuration.

2. **Reduce concurrency**:
   ```bash
   ng_collect_rollouts +num_samples_in_parallel=10  # Lower value
   ```

3. **Implement backoff**: NeMo Gym handles retries, but very aggressive concurrency may hit limits.

4. **Scale NIM deployment**: Contact NVIDIA support to increase capacity or adjust rate limits.

5. **Review API key limits**: Some API keys have per-key rate limits - check your NIM configuration.

:::

:::{dropdown} Context length exceeded errors
:icon: x
:color: danger

**How NeMo Gym handles this**: Context length errors from NIM are returned as HTTP errors. These errors will cause the individual rollout to fail, but rollout collection continues with remaining tasks.

**To fix**:
- Use models with larger context windows
- Reduce conversation history length in your resource server
- Check NIM model documentation for context limits
- Configure truncation strategies in your agent logic
- Monitor rollout logs for context length failures

:::

:::{dropdown} Timeout errors
:icon: alert
:color: warning

**Solution**:

1. **Check NIM health**: Verify NIM deployment is healthy and responsive.

2. **Increase timeout values**: If using custom timeout configuration, increase values for large models.

3. **Monitor model load time**: First request after idle period may be slower due to model loading.

4. **Check request size**: Very long prompts may exceed timeout - consider breaking into smaller requests.

:::

---

## Production Issues

Issues specific to production deployments.

:::{dropdown} Inconsistent performance
:icon: graph
:color: info

**Diagnose**:

1. **Check auto-scaling behavior**: Review scaling events and patterns.

2. **Monitor request patterns**: Look for traffic spikes or unusual patterns.

3. **Review caching**: Ensure response caching is configured appropriately.

4. **Check NIM version**: Ensure all NIM instances are running same version.

5. **Enable detailed monitoring**: Use NIM observability features to identify root cause.

:::

:::{dropdown} Security or compliance issues
:icon: shield
:color: warning

**Address**:

1. **Review audit logs**: Check for unauthorized access attempts.

2. **Verify encryption**: Ensure TLS/SSL is configured for all endpoints.

3. **Check access controls**: Review and update API key permissions.

4. **Enable security features**: Configure IP allowlisting, rate limiting.

5. **Contact NVIDIA support**: For compliance questions or security incidents.

:::

---

## Getting Help

If you're still experiencing issues:

1. Check NIM management console: Review deployment health, logs, and metrics
2. Test NIM directly: Use NIM API directly to isolate NeMo Gym vs NIM issues
3. Verify configuration: Use `ng_dump_config` to see resolved configuration values
4. Run tests: Execute `ng_test +entrypoint=responses_api_models/openai_model`
5. Contact NVIDIA support: Enterprise customers have access to dedicated support

:::{seealso}
- **[Configuration Reference](configuration.md)** - Verify your settings
- **[NVIDIA NIM Documentation](https://developer.nvidia.com/nim)** - NIM-specific details
- **[Quick Start](quick-start.md)** - Validate your setup
:::


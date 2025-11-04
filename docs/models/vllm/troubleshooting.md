(models-vllm-troubleshooting)=

# Troubleshooting

Common issues and solutions when working with vLLM in NeMo Gym, organized by issue type.

---

## Connection & Server Issues

Problems that prevent basic connectivity or server communication.

:::{dropdown} Connection refused or cannot connect to vLLM server
:icon: x-circle
:color: danger

**Check these**:

1. **vLLM server is running**:
   ```bash
   # Check vLLM server health (vLLM native endpoint)
   curl http://localhost:10240/health
   ```
   Should return health status from the vLLM server.

2. **Correct URL in configuration**:
   ```yaml
   # Correct - includes /v1 path
   policy_base_url: http://localhost:10240/v1
   
   # Incorrect - missing /v1
   policy_base_url: http://localhost:10240
   ```

3. **Network connectivity**:
   ```bash
   # Test vLLM server API (vLLM native endpoint)
   curl http://your-vllm-server:8000/v1/models
   ```

4. **Firewall rules**: Ensure the vLLM server port is accessible from the machine running NeMo Gym.

:::

:::{dropdown} Model not found or invalid model identifier
:icon: search
:color: warning

**Solution**:

1. **Verify model loaded in vLLM server**:
   ```bash
   # Query vLLM server models endpoint
   curl http://localhost:10240/v1/models
   ```
   Check that your model name appears in the response.

2. **Match model identifier exactly**:
   ```yaml
   # If vLLM shows: meta-llama/Llama-3.1-8B-Instruct
   # Use exactly that:
   policy_model_name: meta-llama/Llama-3.1-8B-Instruct
   ```

3. **Check vLLM logs**: Look for model loading errors or warnings.

:::

---

## Configuration Issues

Settings that need to be configured correctly for specific features.

:::{dropdown} Reasoning tokens not appearing in responses
:icon: comment-discussion
:color: info

**Check**:

1. **Reasoning parser enabled**:
   ```yaml
   uses_reasoning_parser: true  # Should be true (default)
   ```

2. **vLLM reasoning parser NOT used**:
   ```bash
   # Incorrect - do not use
   vllm serve model --reasoning-parser qwen3
   
   # Correct - no reasoning parser argument
   vllm serve model --tool-call-parser hermes
   ```

3. **Model generates reasoning**: Not all models produce reasoning tokens. Check model documentation.

:::

:::{dropdown} Tool calling not working
:icon: tools
:color: warning

**Check**:

1. **vLLM started with tool parser**:
   ```bash
   vllm serve model --enable-auto-tool-choice --tool-call-parser hermes
   ```

2. **Model supports tool calling**: Not all models have tool-calling capabilities. Common models that do:
   - Qwen/Qwen3-30B-A3B (use hermes parser)
   - meta-llama/Llama-3.1-70B-Instruct (use hermes parser)
   - mistralai models (use mistral parser)

3. **Tool call parser matches model**: Use `--tool-call-parser hermes` for most models, `--tool-call-parser mistral` for Mistral family.

:::

:::{dropdown} Token IDs or log probabilities missing
:icon: code
:color: info

**Enable training mode**:

```yaml
# Use vllm_model_for_training.yaml
policy_model:
  responses_api_models:
    vllm_model:
      return_token_id_information: true  # Enable this
```

**Verify configuration loaded correctly**:
```bash
ng_dump_config "+config_paths=[...]"  # View resolved configuration
```

See [Configuration Reference](configuration.md#configuration-parameters) for all training mode parameters.

:::

---

## Runtime & Performance Issues

Problems that occur during execution or affect throughput and speed.

:::{dropdown} Context length exceeded errors
:icon: stop
:color: danger

**How NeMo Gym handles this**: The vLLM adapter automatically catches context length errors (HTTP 400 with "context length" message) and returns an empty response, allowing rollout collection to continue without crashing.

**To fix**:
- Use models with larger context windows
- Reduce conversation history length in your resource server
- Configure vLLM server with `--max-model-len 16384` to match your needs (adjust based on your model's capabilities)

:::

:::{dropdown} Slow inference or low throughput
:icon: clock
:color: warning

**Optimize**:

1. **vLLM server parameters** (at server startup):
   ```bash
   vllm serve model \
       --tensor-parallel-size 4 \          # Use multiple GPUs
       --gpu-memory-utilization 0.95 \     # Maximize GPU usage
       --max-num-seqs 256 \                # Increase batch size
       --enable-prefix-caching              # Cache common prefixes
   ```

2. **Load balancing** (NeMo Gym configuration):
   ```yaml
   base_url:
     - http://vllm-1:8000/v1
     - http://vllm-2:8000/v1
     - http://vllm-3:8000/v1  # More servers = higher throughput
   ```

3. **NeMo Gym concurrency** (at rollout collection):
   ```bash
   ng_collect_rollouts +num_samples_in_parallel=100  # Increase concurrent requests
   ```

4. **Monitor vLLM server metrics**: Check GPU utilization, batch sizes, and queue lengths using vLLM's metrics endpoint.

See [Optimization Guide](optimization.md) for detailed performance tuning.

:::

---

## Getting Help

If you're still experiencing issues:

1. **Check vLLM server logs**: Look for errors or warnings from the vLLM server process
2. **Test vLLM directly**: Use vLLM's API endpoints directly (via `curl` or OpenAI client) to isolate NeMo Gym adapter vs vLLM server issues
3. **Verify configuration**: Use `ng_dump_config` to inspect all resolved configuration values
4. **Run adapter tests**: Execute `ng_test +entrypoint=responses_api_models/vllm_model` to verify the adapter is working correctly

:::{seealso}
- **[Configuration Reference](configuration.md)** - Verify your settings
- **[Optimization Guide](optimization.md)** - Performance troubleshooting
- **[vLLM Documentation](https://docs.vllm.ai/)** - vLLM-specific issues
:::


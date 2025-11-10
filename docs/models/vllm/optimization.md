(models-vllm-optimization)=

# Optimization

Performance tuning and production deployment strategies for vLLM with NeMo Gym.

---

## Load Balancing

Distribute inference requests across multiple vLLM servers for higher throughput.

### Configuration

Specify multiple vLLM endpoints using either comma-separated values or YAML list syntax:

::::{tab-set}

:::{tab-item} Layer 1: env.yaml

Comma-separated URLs in your environment file:

```yaml
# env.yaml
policy_base_url: http://vllm-1:8000/v1,http://vllm-2:8000/v1,http://vllm-3:8000/v1
policy_api_key: EMPTY
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

**When to use**: Quick setup, works with standard vLLM config files.

:::

:::{tab-item} Layer 2: Config YAML

Explicit list syntax in the config file:

```yaml
# responses_api_models/vllm_model/configs/vllm_model.yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url:
        - http://vllm-1:8000/v1
        - http://vllm-2:8000/v1
        - http://vllm-3:8000/v1
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

**When to use**: Custom config files, when you prefer explicit YAML list syntax.

:::

::::

### How It Works

**Session-based routing**: NeMo Gym assigns each client session to one vLLM server. All requests from that session route to the same server, while new sessions are distributed round-robin across available servers.

**Why session-based?**
- **KV cache efficiency**: vLLM can reuse cached tokens within a session
- **Consistent behavior**: Same server handles the entire agent interaction
- **Multi-turn support**: Conversation context stays on one server

**Load distribution**: New sessions are assigned round-robin, providing good distribution for high-concurrency workloads with many independent rollouts.

:::{tip}
Load balancing is most effective when:
- Running high concurrency rollout collection (`+num_samples_in_parallel=100+`)
- Each vLLM server has similar capacity
- All servers have the same model loaded
:::

---

## vLLM Server Optimization

Optimize vLLM server startup parameters for better performance:

```bash
vllm serve model \
    --tensor-parallel-size 4 \          # Use multiple GPUs
    --gpu-memory-utilization 0.95 \     # Maximize GPU usage
    --max-num-seqs 256 \                # Increase batch size
    --enable-prefix-caching \           # Cache common prefixes
    --enable-auto-tool-choice \         # Enable tool calling
    --tool-call-parser hermes           # Tool format parser
```

**Key parameters**:

```{list-table}
:header-rows: 1
:widths: 30 15 20 35

* - Parameter
  - Type
  - Typical Value
  - Description
* - `--tensor-parallel-size`
  - int
  - 4 or 8
  - Number of GPUs to use for tensor parallelism
* - `--gpu-memory-utilization`
  - float
  - 0.90-0.95
  - Fraction of GPU memory to allocate (higher = more capacity, less headroom)
* - `--max-num-seqs`
  - int
  - 256
  - Maximum batch size for concurrent sequence processing
* - `--enable-prefix-caching`
  - flag
  - -
  - Enable KV cache reuse for common prompt prefixes
* - `--enable-auto-tool-choice`
  - flag
  - -
  - Enable automatic tool selection for function calling
* - `--tool-call-parser`
  - string
  - `hermes` or `mistral`
  - Parser format for tool calling (model-dependent)
```

---

## Switch from OpenAI to vLLM

Migrating from OpenAI? Switch by changing one configuration file reference:

**Before** (OpenAI):
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
```

**After** (vLLM):
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"
```

Update `env.yaml` to point to your vLLM server and everything else remains the same.

---

## Monitoring and Metrics

Monitor these vLLM server metrics for optimal performance:

**GPU utilization**: Should be 90-95% during peak load
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

**vLLM metrics**: Check batch sizes and queue lengths
```bash
curl http://localhost:8000/metrics  # Prometheus metrics endpoint
```

Monitor these metrics to identify bottlenecks and optimize vLLM server configuration for your workload.

(models-vllm-optimization)=

# Optimization

Performance tuning and production deployment strategies for vLLM with NeMo Gym.

---

## Load Balancing

Distribute inference requests across multiple vLLM servers for higher throughput.

### Configuration

Specify multiple vLLM endpoints using either comma-separated values or YAML list syntax:

**Option 1: Comma-separated in env.yaml**
```yaml
policy_base_url: http://vllm-1:8000/v1,http://vllm-2:8000/v1,http://vllm-3:8000/v1
policy_api_key: EMPTY
policy_model_name: meta-llama/Llama-3.1-8B-Instruct
```

**Option 2: YAML list in config file**
```yaml
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

### How It Works

**Session-based routing**: NeMo Gym assigns each client session to one vLLM server. All requests from that session route to the same server, while new sessions are distributed round-robin across available servers.

**Why session-based?**
- **KV cache efficiency**: vLLM can reuse cached tokens within a session
- **Consistent behavior**: Same server handles the entire agent interaction
- **Multi-turn support**: Conversation context stays on one server

**Load distribution**: New sessions are assigned round-robin, providing good distribution for high-concurrency workloads with many independent rollouts.

:::{tip}
Load balancing is most effective when:
- Running high concurrency rollout collection (`+concurrency=100+`)
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
- `--tensor-parallel-size`: Number of GPUs to use (4 or 8 typical)
- `--gpu-memory-utilization`: How much GPU memory to use (0.90-0.95 recommended)
- `--max-num-seqs`: Maximum batch size for concurrent requests
- `--enable-prefix-caching`: Reuse KV cache for common prompt prefixes

---

## NeMo Gym Concurrency Tuning

Increase concurrent requests to maximize vLLM throughput:

```bash
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/vllm_rollouts.jsonl \
  +limit=1000 \
  +concurrency=100  # Increase for higher throughput
```

**Finding optimal concurrency**:
1. Start with `+concurrency=50`
2. Monitor vLLM server GPU utilization
3. Increase concurrency until GPU reaches 90-95% utilization
4. If latency increases significantly, reduce concurrency

:::{note}
Optimal concurrency depends on:
- vLLM server GPU count and memory
- Model size and batch configuration
- Network latency between NeMo Gym and vLLM
:::

---

## Separate Policy and Judge Servers

Use different vLLM models or servers for policy decisions vs verification:

```yaml
# env.yaml
# Policy model (larger, more capable)
policy_base_url: http://policy-server:8000/v1
policy_api_key: EMPTY
policy_model_name: meta-llama/Llama-3.1-70B-Instruct

# Judge model (smaller, faster)
judge_base_url: http://judge-server:8000/v1
judge_api_key: EMPTY
judge_model_name: Qwen/Qwen3-30B-A3B
```

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
responses_api_models/vllm_model/configs/vllm_judge_model.yaml,\
resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"

ng_run "+config_paths=[${config_paths}]"
```

**Benefits**:
- Use larger model for policy (quality)
- Use smaller/faster model for judge (throughput)
- Independent scaling of policy vs judge workloads

:::{seealso}
See [Separate Policy and Judge Models](../../tutorials/separate-policy-and-judge-models.md) for comprehensive patterns and examples.
:::

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

**NeMo Gym throughput**: Track rollouts generated per second
```bash
# Monitor output file growth
watch -n 1 "wc -l results/vllm_rollouts.jsonl"
```

---

## Next Steps

- **[Configuration reference](configuration.md)** - All vLLM adapter parameters
- **[Troubleshooting](troubleshooting.md)** - Performance issues and solutions
- **[Collect rollouts](../../get-started/collecting-rollouts.md)** - Start generating training data


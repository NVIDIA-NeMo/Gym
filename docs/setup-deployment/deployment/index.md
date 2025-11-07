(setup-deployment-scenarios)=

# Deployment

Deploy NeMo Gym in different environments—local development, remote servers, or containerized infrastructure.

---

## Local Development

Set up NeMo Gym for local development and testing:

```bash
# 1. Clone and install
git clone <repo>
cd Gym
pip install -e ".[dev]"

# 2. Create env.yaml with secrets
cat > env.yaml << EOF
policy_api_key: sk-your-openai-key
EOF

# 3. Test with simple config
ng_run "+config_paths=[responses_api_agents/simple_agent/config.yaml]"
```

## Remote Servers

Deploy components on remote machines:

1. Install NeMo Gym on target machine
2. Set up `env.yaml` with production credentials
3. Configure network access (open required ports)
4. Run with production config:

```bash
ng_run "+config_paths=[production_config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

## Containers

Package NeMo Gym using Docker (example):

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["ng_run", "+config_paths=[config.yaml]"]
```

---

## Using vLLM with Non-Responses Models

Most models lack native Responses API support. NeMo Gym provides a middleware layer (`vllm_model`) that maps Responses API to Chat Completions.

### Why Use VLLMModel?

As of November 2025, few models support OpenAI Responses API natively (GPT-4o and similar). NeMo Gym is first-party Responses API, making it difficult to use with standard models. The `vllm_model` middleware solves this by translating between formats.

### Quick Start with vLLM

**Step 1: Start vLLM server**

```bash
# Create environment
uv venv --python 3.12 --seed 
source .venv/bin/activate
uv pip install hf_transfer datasets vllm --torch-backend=auto

# Download model (example: Qwen3-30B-A3B)
HF_HOME=.cache/ \
HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download Qwen/Qwen3-30B-A3B

# Start vLLM server with tool support
HF_HOME=.cache/ \
HOME=. \
vllm serve \
    Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

**Step 2: Use vllm_model configuration**

```bash
# Replace openai_model with vllm_model
config_paths="resources_servers/multineedle/configs/multineedle.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

That's it! NeMo Gym now works with your vLLM-hosted model.

### VLLMModel Configuration

The `vllm_model` middleware:
- Converts Responses API calls to Chat Completions
- Uses vLLM-specific endpoints like `/tokenize` for accurate token tracking
- Handles tool calls and reasoning traces
- Maintains token-level accuracy for RL training

**Important configuration notes**:

```{important}
**Do NOT use** vLLM's `--reasoning-parser` flag. The Gym middleware handles reasoning parsing internally to maintain Responses API compatibility.
```

**Tool call parsers**:
- Qwen models: `--tool-call-parser hermes`
- Other models: Check vLLM documentation for appropriate parser

**Model examples**:
- ✅ Qwen/Qwen3-30B-A3B (NeMo RL compatible)
- ✅ Any vLLM-supported model with tool capabilities
- ✅ Models without native Responses API support

### Custom vLLM Configuration

Edit `responses_api_models/vllm_model/configs/vllm_model.yaml` or override via CLI:

```bash
ng_run "+config_paths=[$config_paths]" \
    +policy_model.vllm_model.base_url=http://your-vllm-server:10240 \
    +policy_model.vllm_model.model_name=your-model-name
```

:::{seealso}
See {doc}`../../models/index` for tested model configurations and performance characteristics.
:::

---

## Distributed Computing with Ray

NeMo Gym automatically initializes Ray for distributed CPU-intensive tasks like verification and data processing.

### Automatic Ray Setup

Ray initializes automatically when you run NeMo Gym:

```bash
ng_run "+config_paths=[$config_paths]"
```

**What happens**:
1. Main process starts a Ray cluster
2. Ray address stored in global configuration
3. Child server processes connect to the same cluster
4. All processes share the same Ray runtime

**Console output**:

```
Starting Ray cluster...
Started Ray cluster at ray://127.0.0.1:6379
```

### Using a Custom Ray Cluster

Connect to an existing Ray cluster for multi-node distributed computing:

```yaml
# config.yaml or env.yaml
ray_head_node_address: "ray://your-cluster-address:10001"
```

Then run normally:

```bash
ng_run "+config_paths=[config.yaml]"
```

**When to use custom clusters**:
- Multi-node training with NeMo-RL
- Large-scale distributed rollout collection
- High-throughput verification across compute cluster
- Training frameworks manage Ray cluster for you

Training frameworks like [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) configure the Ray cluster automatically for distributed training.

### Parallelizing Tasks in Resource Servers

Use Ray to parallelize CPU-intensive operations in your resource server:

```python
import ray
from nemo_gym.base_resources_server import SimpleResourcesServer

class MyResourcesServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Process many items in parallel
        results = await self._parallel_verify(body.items)
        return BaseVerifyResponse(reward=compute_reward(results))
    
    async def _parallel_verify(self, items: list) -> list:
        # Submit all tasks to Ray cluster
        futures = [verify_item.remote(item) for item in items]
        # Get results in parallel
        results = ray.get(futures)
        return results

# CPU-intensive function runs on Ray worker
@ray.remote(scheduling_strategy="SPREAD")
def verify_item(item):
    # Expensive computation here
    result = expensive_verification(item)
    return result
```

**Best practices**:
- Use `scheduling_strategy="SPREAD"` to distribute across nodes
- Keep Ray tasks stateless (no shared mutable state)
- Batch small tasks to reduce overhead
- Use Ray for CPU-bound work, async for I/O-bound work

**Real-world example**: The `library_judge_math` server uses Ray to verify 1000+ math problems in parallel during rollout collection.

### Ray Configuration Options

Advanced Ray configuration:

```yaml
# config.yaml
ray_head_node_address: "ray://cluster:10001"  # Custom cluster
# Ray will use default configuration if not specified
```

**Environment variables**:
- `RAY_ADDRESS`: Alternative way to set cluster address
- Ray respects standard Ray environment variables

:::{seealso}
For advanced Ray usage, refer to [Ray documentation](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) for decorators, actors, and distributed patterns.
:::

---

## Scaling

For high-throughput production workloads:

### Horizontal Scaling

- Run multiple resource server instances
- Use load balancers for distribution
- Scale horizontally by adding more servers
- Monitor resource usage and adjust accordingly

### Ray-Based Scaling

- Connect to multi-node Ray cluster for distributed processing
- Parallelize verification across compute nodes
- Scale CPU-intensive tasks automatically

### High-Throughput Scenarios

For extreme concurrency (16k+ concurrent requests):
- Use Ray for parallel processing
- Profile servers to identify bottlenecks (see {doc}`../operations/index`)
- Optimize verify() methods for efficiency
- Consider multiple vLLM worker instances




(infra-ray-distributed)=
# Distributed Computing with Ray

Scale CPU-intensive tasks in NeMo Gym using [Ray](https://www.ray.io/) for distributed parallel execution.

---

## Overview

NeMo Gym uses Ray for parallelizing CPU-intensive operations, particularly in resources servers where verification logic can be computationally expensive. Ray enables:

- Parallel execution of verification tasks across CPU cores
- Distribution of work across multiple nodes in a cluster
- Seamless integration with training frameworks like NeMo-RL

```{note}
Ray is **not** used for rollout collection parallelism. Rollout collection uses async HTTP with aiohttp for high-concurrency request handling. See the {ref}`FAQ <reference-faq>` for details on the async HTTP architecture.
```

## Ray Initialization

:::::{tab-set}

::::{tab-item} Automatic Setup

Ray initializes automatically when you start NeMo Gym servers:

```bash
ng_run "+config_paths=[$config_paths]"
```

The initialization flow:

1. **Main process** (`cli.py`): Ray cluster starts when `RunHelper.start()` is called
2. **Server processes** (`server_utils.py`): Each server calls `initialize_ray()` and connects to the same cluster
3. **Shared state**: The Ray cluster address is stored in the global config for all child processes

::::

::::{tab-item} Connecting to an Existing Cluster

For production training with NeMo-RL or other frameworks, connect to an existing Ray cluster:

```yaml
# In your config or env.yaml
ray_head_node_address: "ray://your-cluster-address:10001"
```

When `ray_head_node_address` is specified, NeMo Gym connects to that cluster instead of starting a new one. This enables resources servers to run distributed tasks across all nodes in the training cluster.

::::

:::::

## Parallelizing CPU-Intensive Tasks

### When to Use Ray

Use Ray's `@ray.remote` decorator for:

- **Verification logic** that involves expensive computation
- **Batch processing** where items can be processed independently
- **Any CPU-bound operation** that would benefit from parallelization

Do **not** use Ray for:
- HTTP requests to model servers (use async/await instead)
- I/O-bound operations (use asyncio)
- Simple operations where parallelization overhead exceeds benefit

### Using @ray.remote

Decorate CPU-intensive functions with `@ray.remote`:

```python
import ray

@ray.remote(scheduling_strategy="SPREAD")
def cpu_intensive_task(data):
    """Expensive computation distributed across nodes."""
    result = expensive_computation(data)
    return result

def process_data_parallel(data_list):
    # Submit all tasks to Ray
    futures = [cpu_intensive_task.remote(data) for data in data_list]
    
    # Collect results
    results = ray.get(futures)
    return results
```

The `scheduling_strategy="SPREAD"` distributes tasks across different nodes for better parallelization.

### Example: Parallel Verification

Here's a pattern for parallelizing verification in a resources server:

```python
import ray
from typing import List

@ray.remote(scheduling_strategy="SPREAD")
def verify_single(answer: str, expected: str) -> bool:
    """CPU-intensive verification for a single answer."""
    # Your expensive verification logic here
    return compute_match(answer, expected)

class MyResourcesServer(BaseResourcesServer):
    async def verify(self, request: VerifyRequest) -> VerifyResponse:
        # Submit verification tasks in parallel
        futures = [
            verify_single.remote(ans, exp) 
            for ans, exp in zip(request.answers, request.expected)
        ]
        
        # Gather results
        results = ray.get(futures)
        
        return VerifyResponse(correct=results)
```

## Configuration

### Version Requirements

```{important}
Ray versions must match exactly between the main process and all child processes. NeMo Gym automatically constrains the Ray version in child server environments to match the parent.
```

The version constraint is managed in `global_config.py`:

```python
# Child servers receive this dependency constraint
f"ray[default]=={ray_version}"
```

### Resource Allocation

Configure Ray resources based on your workload:

```python
# CPU-only task (default)
@ray.remote
def cpu_task(data):
    pass

# Task requiring specific CPU count
@ray.remote(num_cpus=2)
def multi_cpu_task(data):
    pass
```

```{note}
GPU workloads in NeMo Gym typically go through model servers (vLLM, OpenAI API) rather than Ray GPU allocation. Use model server configuration for GPU resource management.
```

## Monitoring

### Ray Dashboard

Access the Ray dashboard for cluster monitoring:

```
http://<head-node>:8265
```

The dashboard provides:

- **Cluster overview**: Node status, resource utilization
- **Task timeline**: Execution progress and timing
- **Logs**: Aggregated logs from all workers
- **Metrics**: CPU, memory, and throughput statistics

## Troubleshooting

::::{dropdown} Version Mismatch Errors
:icon: alert

If you see errors about Ray version incompatibility:

1. Ensure all nodes use the same Ray version
2. Check that `ray[default]` extra matches the top-level `pyproject.toml`
3. Verify Python versions match (Ray is sensitive to Python version)

::::

::::{dropdown} Sandboxed Environment Errors
:icon: alert

If you encounter:

```
PermissionError: [Errno 1] Operation not permitted (originated from sysctl())
```

This occurs in sandboxed environments where Ray's `psutil` calls are blocked. Solutions:

1. Run outside the sandbox in a regular terminal
2. Grant additional permissions if your environment supports it
3. See the {ref}`FAQ <reference-faq>` for environment-specific workarounds

::::

::::{dropdown} Connection Issues
:icon: alert

If servers fail to connect to the Ray cluster:

1. Verify `ray_head_node_address` is correct and reachable
2. Check firewall rules allow Ray's ports (default: 6379 for GCS, 8265 for dashboard)
3. Ensure the Ray cluster is running before starting NeMo Gym servers

::::

## Related Resources

- {doc}`deployment-topology` - Production deployment patterns
- {ref}`FAQ: Use Ray for parallelizing CPU-intensive tasks <reference-faq>` - Additional examples
- [Ray Documentation](https://docs.ray.io/) - Comprehensive Ray reference

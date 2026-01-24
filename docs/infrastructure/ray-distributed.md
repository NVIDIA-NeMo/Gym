(infra-ray-distributed)=

# Distributed Computing with Ray

Scale CPU-intensive verification tasks using [Ray](https://www.ray.io/) for distributed parallel execution.

---

## Overview

NeMo Gym uses Ray for parallelizing CPU-intensive operations in resources servers where verification logic can be computationally expensive. Ray enables:

- Parallel execution of verification tasks across CPU cores
- Distribution of work across multiple nodes in a cluster
- Integration with training frameworks like NeMo RL

```{note}
Ray is **not** used for rollout collection parallelism. Rollout collection uses async HTTP with aiohttp for high-concurrency request handling. See the {ref}`FAQ <reference-faq>` for details on the async HTTP architecture.
```

## Ray Initialization

:::::{tab-set}

::::{tab-item} Automatic Setup

Ray initializes automatically when you start NeMo Gym servers:

```bash
# Example: Start with math verification config
ng_run "+config_paths=[resources_servers/math/configs/gsm8k.yaml]"
```

The initialization flow:

1. **Main process** (`cli.py:177`): Ray cluster starts when `RunHelper.start()` calls `initialize_ray()`
2. **Server processes** (`server_utils.py:563`): Servers call `initialize_ray()` during setup, connecting to the same cluster
3. **Shared state**: The Ray cluster address is stored in `global_config_dict["ray_head_node_address"]` for all child processes

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
    """Expensive computation distributed across nodes.
    
    Use for operations >100ms that benefit from parallelization.
    """
    # Your CPU-bound logic here
    result = process(data)
    return result

def process_data_parallel(data_list):
    # Submit all tasks to Ray
    futures = [cpu_intensive_task.remote(data) for data in data_list]
    
    # Collect results
    results = ray.get(futures)
    return results
```

The `scheduling_strategy="SPREAD"` distributes tasks across different nodes for better parallelization.

### Real Example: Code Execution Verification

From `resources_servers/code_gen/lcb_integration/compute_code_generation_metrics.py`:

```python
import ray

# Using SPREAD scheduling so that Ray assigns tasks to as many distinct nodes as possible.
@ray.remote(scheduling_strategy="SPREAD")
def check_correctness_remote(sample, generation, timeout, debug=True):
    """Ray wrapper of check_correctness for remote execution."""
    return check_correctness(sample, generation, timeout, debug)
```

Usage pattern for batch verification:

```python
# Submit all tasks to Ray
futures = [
    check_correctness_remote.remote(sample, gen, timeout)
    for sample, gen in zip(samples, generations)
]

# Collect results (blocks until all complete)
results = ray.get(futures)
```

See also: `resources_servers/swerl_gen/eval/singularity_utils.py:202` for SWE-bench evaluation using Ray.

## Configuration

### Version Requirements

```{important}
Ray versions must match exactly between the main process and all child processes. NeMo Gym automatically constrains the Ray version in child server environments to match the parent.
```

The version constraint is managed in `global_config.py:288-291`:

```python
# Child servers receive this dependency constraint
global_config_dict[HEAD_SERVER_DEPS_KEY_NAME] = [
    f"ray[default]=={ray_version}",
    f"openai=={openai_version}",
]
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
GPU workloads in NeMo Gym go through model servers (vLLM, OpenAI API). The `local_vllm_model` server uses Ray for vLLM process scheduling, but GPU allocation is managed by vLLM itself, not Ray's resource system.
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

## Where Ray is Used

| Component | File | Purpose |
|---|---|---|
| Code generation | `code_gen/lcb_integration/compute_code_generation_metrics.py` | Parallel test execution |
| SWE-bench evaluation | `swerl_gen/eval/singularity_utils.py` | Parallel patch verification |
| Local vLLM | `local_vllm_model/app.py` | vLLM server scheduling |

## Related Resources

- {doc}`deployment-topology` — Production deployment patterns
- {ref}`FAQ: Use Ray for parallelizing CPU-intensive tasks <reference-faq>` — Additional examples
- [Ray Documentation](https://docs.ray.io/) — Comprehensive Ray reference

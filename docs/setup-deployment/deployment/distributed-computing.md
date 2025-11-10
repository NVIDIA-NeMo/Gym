(setup-deployment-ray)=

# Distributed Computing with Ray

Scale NeMo Gym with Ray for high-throughput workloads and multi-node training.

---

## Overview

### When to Use Ray Clusters

```{list-table}
:header-rows: 1
:widths: 40 30 30

* - Use Case
  - Single-Node Ray
  - Multi-Node Ray Cluster
* - **Local Development**
  - ✅ Automatic
  - ❌ Not needed
* - **Single-Server Production**
  - ✅ Automatic
  - ❌ Not needed
* - **Multi-Node Training**
  - ❌ Insufficient
  - ✅ Required
* - **Large-Scale Rollout Collection**
  - ❌ Limited throughput
  - ✅ High throughput
* - **Distributed Verification**
  - ⚠️ Limited by cores
  - ✅ Scales horizontally
```

:::{tip}
For most use cases, NeMo Gym's automatic Ray initialization is sufficient. Custom clusters are needed primarily for multi-node training or large-scale data collection.
:::

---

## Automatic Ray Initialization

### How It Works

NeMo Gym automatically starts a Ray cluster when you run `ng_run`:

```{mermaid}
sequenceDiagram
    participant User
    participant MainProcess
    participant Ray
    participant ChildServers
    
    User->>MainProcess: ng_run
    MainProcess->>Ray: Initialize Ray cluster
    Ray-->>MainProcess: ray://127.0.0.1:6379
    MainProcess->>MainProcess: Store address in config
    MainProcess->>ChildServers: Start server processes
    ChildServers->>Ray: Connect to existing cluster
    Ray-->>ChildServers: Connected
```

**What happens**:

1. **Main process** calls `initialize_ray()` (line 126 in `cli.py`)
2. **Ray cluster** starts locally or connects to existing
3. **Address stored** in global config: `ray_head_node_address`
4. **Child processes** connect to same cluster automatically
5. **All processes** share the same Ray runtime

### Console Output

```console
$ ng_run "+config_paths=[config.yaml]"

Starting Ray cluster...
Started Ray cluster at ray://127.0.0.1:6379
Starting Head Server on http://127.0.0.1:8000
Starting Simple Agent on http://127.0.0.1:8001
...
```

:::{note}
The Ray address `ray://127.0.0.1:6379` is stored in global configuration and shared with all server processes for unified cluster access.
:::

---

## Custom Ray Clusters

### Use External Ray Cluster

Connect to an existing Ray cluster for distributed computing by setting the `ray_head_node_address` configuration parameter:

::::{tab-set}

:::{tab-item} Via Configuration File

```yaml
# config.yaml
ray_head_node_address: "ray://your-cluster:10001"
```

```bash
ng_run "+config_paths=[config.yaml]"
```
:::

:::{tab-item} Via env.yaml

```yaml
# env.yaml
ray_head_node_address: "ray://gpu-cluster.example.com:10001"
```

```bash
ng_run "+config_paths=[config.yaml]"
```
:::

:::{tab-item} Via CLI Override

```bash
ng_run "+config_paths=[config.yaml]" \
    +ray_head_node_address="ray://your-cluster:10001"
```
:::

::::

:::{note}
The `ray_head_node_address` configuration parameter is NeMo Gym's standard way to connect to external Ray clusters. The address is stored in the global configuration and shared with all server processes.
:::

### Console Output with Custom Cluster

```console
$ ng_run "+config_paths=[config.yaml]"

Connecting to Ray cluster at specified address: ray://gpu-cluster:10001
Ray already initialized
Starting Head Server on http://127.0.0.1:8000
...
```

---

## Setting Up a Ray Cluster

### Single-Node Cluster (Automatic)

No setup required—NeMo Gym handles this automatically.

### Multi-Node Cluster (Manual)

#### Head Node

Start Ray head node on your primary server:

```bash
# On head node (e.g., gpu-server-01)
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265
```

**Expected output**:

```console
Local node IP: 10.0.1.100
Ray runtime started at: ray://10.0.1.100:10001

To connect to this cluster from another node:
    ray start --address='10.0.1.100:6379'
    
Dashboard: http://10.0.1.100:8265
```

#### Worker Nodes

Connect additional nodes to the cluster:

```bash
# On worker nodes (e.g., gpu-server-02, gpu-server-03, ...)
ray start --address='10.0.1.100:6379'
```

**Expected output**:

```console
Local node IP: 10.0.1.101
Connected to Ray cluster at 10.0.1.100:6379
```

#### Verify Cluster

```bash
# Check cluster status
ray status

# Expected output:
# ======== Cluster Resources ========
# Total CPUs: 192 (64 per node × 3 nodes)
# Total GPUs: 24 (8 per node × 3 nodes)
# Total Memory: 768.0 GB
```

---

## Using Ray in Resource Servers

### Parallelizing Verification

Leverage Ray for CPU-intensive verification tasks:

```python
import ray
from nemo_gym.base_resources_server import SimpleResourcesServer

class MathVerificationServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Verify many math problems in parallel."""
        
        # Submit all verification tasks to Ray cluster
        futures = [
            verify_single_problem.remote(problem)
            for problem in body.problems
        ]
        
        # Wait for all results
        results = ray.get(futures)
        
        # Compute reward
        reward = sum(1 for r in results if r.correct) / len(results)
        
        return BaseVerifyResponse(
            reward=reward,
            metadata={"correct": sum(1 for r in results if r.correct)}
        )


@ray.remote(scheduling_strategy="SPREAD")
def verify_single_problem(problem: dict) -> dict:
    """CPU-intensive verification (runs on Ray worker)."""
    
    # Expensive computation: parsing, execution, comparison
    user_answer = execute_math_solution(problem["solution"])
    correct_answer = problem["expected"]
    
    return {
        "correct": user_answer == correct_answer,
        "problem_id": problem["id"]
    }
```

### Best Practices

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} ✅ Do: Spread Tasks Across Nodes

```python
@ray.remote(scheduling_strategy="SPREAD")
def my_task(data):
    ...
```

Distributes tasks across all available nodes for maximum parallelism.
:::

:::{grid-item-card} ✅ Do: Keep Tasks Stateless

```python
@ray.remote
def process_item(item):
    # No shared mutable state
    return compute(item)
```

Stateless tasks avoid synchronization overhead and race conditions.
:::

:::{grid-item-card} ✅ Do: Batch Small Tasks

```python
# Instead of 10,000 tiny tasks
futures = [process.remote([batch]) 
           for batch in chunked(items, 100)]
```

Reduces scheduling overhead for small operations.
:::

:::{grid-item-card} ⚠️ Avoid: Passing Large Objects

```python
# Bad: serialize large data
@ray.remote
def process(large_data):
    ...

# Good: use Ray object store
ref = ray.put(large_data)
process.remote(ref)
```

Use object store for large data to avoid serialization costs.
:::

::::

---

## Real-World Examples

### Example 1: Competitive Coding Verification

The `comp_coding` server verifies code solutions in parallel using Ray:

```python
# From resources_servers/comp_coding/lcb_integration/compute_code_generation_metrics.py

# Using SPREAD scheduling so Ray assigns tasks to distinct nodes
@ray.remote(scheduling_strategy="SPREAD")
def check_correctness_remote(sample, generation, timeout, debug=True):
    """Ray wrapper for remote code execution and verification."""
    return check_correctness(sample, generation, timeout, debug)
```

```python
# From resources_servers/comp_coding/app.py

class CompCodingResourcesServer(SimpleResourcesServer):
    async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
        # Extract code from model output
        code = extract_code(model_out, LMStyle.OpenAIChat)
        
        # Submit verification task to Ray cluster
        async with self._semaphore:
            task_args = (
                {"input_output": tests.model_dump_json()},
                code,
                self.config.unit_test_timeout_secs,
                self.config.debug,
            )
            
            future = check_correctness_remote.remote(*task_args)
            result, metadata = await loop.run_in_executor(None, ray.get, future)
        
        # Return verification results
        reward = 1.0 if all(r == True for r in result) else 0.0
        return CompCodingVerifyResponse(reward=reward, result=result, metadata=metadata)
```

**Performance** (approximate):
- **Single-threaded**: ~50-100 verifications/minute
- **Ray (8 cores)**: ~400-600 verifications/minute
- **Ray cluster (3 nodes, 24 cores)**: ~1,200-1,800 verifications/minute

:::{note}
Performance varies based on test complexity, timeout values, and hardware specifications.
:::

### Example 2: Distributed Rollout Collection

:::{tip}
NeMo Gym's rollout collection automatically uses Ray for parallelization. When you run `ng_collect_rollouts`, requests are distributed across available Ray workers.
:::

For custom distributed patterns, you can use Ray actors:

```python
# Conceptual example for advanced use cases
@ray.remote
class RolloutWorker:
    """Dedicated worker for rollout collection."""
    
    def __init__(self, model_config):
        self.model = load_model(model_config)
    
    def collect_rollout(self, prompt):
        """Generate response and collect trajectory."""
        response = self.model.generate(prompt)
        reward = verify_response(response)
        
        return {
            "prompt": prompt,
            "response": response,
            "reward": reward
        }


# Create workers across cluster
num_workers = 4  # Adjust based on cluster size
workers = [RolloutWorker.remote(model_config) for _ in range(num_workers)]

# Distribute prompts to workers
from itertools import cycle
futures = [
    worker.collect_rollout.remote(prompt)
    for worker, prompt in zip(cycle(workers), prompts)
]

rollouts = ray.get(futures)
```

:::{note}
This is a conceptual example showing Ray patterns. For production rollout collection, use NeMo Gym's built-in `ng_collect_rollouts` command which handles Ray distribution automatically.
:::

---

## Monitoring Ray Clusters

### Ray Dashboard

Access the web dashboard:

```bash
# Dashboard runs on head node
http://your-head-node:8265
```

**Features**:
- Cluster resource utilization
- Task execution timeline
- Node health and status
- Memory usage tracking
- Task profiling

### CLI Monitoring

```bash
# Cluster status
ray status

# Resource usage
ray memory

# Running tasks
ray list tasks --limit 20

# Stop cluster
ray stop
```

---

## Troubleshooting

::::{dropdown} **Ray Fails to Start**

```console
ERROR: Failed to initialize Ray
```

**Solutions**:
1. Check port 6379 is available: `lsof -i :6379`
2. Ensure sufficient disk space for `/tmp/ray`
3. Verify network connectivity between nodes
4. Check firewall rules (Ray uses ports 6379, 8265, 10001)
::::

::::{dropdown} **Worker Nodes Not Connecting**

```console
ERROR: Cannot connect to Ray cluster at 10.0.1.100:6379
```

**Solutions**:
1. Verify head node address: `ping 10.0.1.100`
2. Check firewall allows port 6379
3. Ensure Ray version matches across nodes
4. Check head node Ray status: `ray status`
::::

::::{dropdown} **Tasks Not Distributing**

```console
WARNING: All tasks running on head node
```

**Solutions**:
1. Use `scheduling_strategy="SPREAD"`:
   ```python
   @ray.remote(scheduling_strategy="SPREAD")
   ```
2. Verify worker nodes are healthy: `ray status`
3. Check resource requirements don't exceed worker capacity
4. Ensure tasks are remote functions, not local
::::

::::{dropdown} **Out of Memory**

```console
ERROR: Ray object store out of memory
```

**Solutions**:
1. Reduce object store usage:
   ```python
   ray.init(object_store_memory=10 * 1024 * 1024 * 1024)  # 10 GB
   ```
2. Delete large objects explicitly: `ray.internal.free(ref)`
3. Process data in smaller batches
4. Use Ray's distributed memory efficiently
::::

---

## Integration with NeMo-RL

NeMo-RL training framework automatically manages Ray clusters:

```bash
# NeMo-RL handles Ray cluster setup
python -m nemo_rl.train \
    --config training_config.yaml
```

NeMo-RL:
1. **Creates** Ray cluster across training nodes
2. **Configures** `ray_head_node_address` for NeMo Gym
3. **Distributes** rollout collection across workers
4. **Manages** cluster lifecycle (start/stop)

:::{seealso}
Complete training guide: {doc}`../../training/index`
:::

---

## Performance Tuning

### Optimal Task Granularity

```python
# Too fine-grained (high overhead)
futures = [process_item.remote(x) for x in range(100000)]

# Better: batch into chunks
CHUNK_SIZE = 1000
chunks = [items[i:i+CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
futures = [process_batch.remote(chunk) for chunk in chunks]
```

**Rule of thumb**: Tasks should run for at least 10-100ms to amortize scheduling overhead.

### Resource Allocation

```python
# Specify resource requirements
@ray.remote(num_cpus=4, num_gpus=1)
def gpu_intensive_task(data):
    ...

# For CPU-bound tasks
@ray.remote(num_cpus=1)
def cpu_task(data):
    ...
```

### Object Store Optimization

```python
# Put large objects in object store once
large_data_ref = ray.put(large_data)

# Pass reference to all tasks
futures = [
    process_with_data.remote(large_data_ref, item)
    for item in items
]
```

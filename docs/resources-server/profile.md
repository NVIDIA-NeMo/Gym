(resources-server-profile)=

# Profile Resources Server

Measure and optimize the performance of your resources server for high-throughput rollout collection.

**Goal**: Identify and fix performance bottlenecks in your resources server.

**Prerequisites**: A working resources server ({doc}`/tutorials/creating-resource-server`)

---

For large-scale RL training, resources servers can receive thousands of concurrent requests. This guide covers NeMo Gym's built-in profiling tools and optimization patterns.

## Additional Profiling Tools

NeMo Gym includes [yappi](https://github.com/sumerc/yappi) (Yet Another Python Profiler), a CPU profiler that supports multithreaded and async code. For additional profiling tools:

```bash
pip install memory_profiler
pip install py-spy  # May require sudo for some operations
```

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Tool latency** | Time per tool call | < 100ms for simple tools |
| **Verify latency** | Time per verification | < 50ms |
| **Throughput** | Rollouts per second | Depends on tools |
| **Memory usage** | RAM per session | Minimize |
| **Event loop lag** | Async responsiveness | < 10ms |

---

## Built-in Profiling with Yappi

NeMo Gym includes a yappi-based CPU profiler that runs during server lifecycle.

### Enable Profiling

Start servers with profiling enabled:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/math_with_judge
```

**Configuration options**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `profiling_enabled` | bool | Enable profiling (default: `false`) |
| `profiling_results_dirpath` | str | Directory for profiling logs |

:::{note}
Profiling adds slight overhead. Disable for production workloads.
:::

### Run Load Test

In a separate terminal, generate traffic:

```bash
ng_collect_rollouts +agent_name=math_with_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/math_with_judge/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/math_with_judge_rollouts.jsonl \
    +limit=1024 \
    +num_repeats=1
```

Adjust `+limit` and `+num_repeats` to control sample size.

### View Live Stats

While servers are running, query the `/stats` endpoint. The port is defined in your resources server config (typically under `host` and `port` fields):

```bash
# Check your config for the port, e.g., resources_servers/my_server/configs/my_server.yaml
curl http://localhost:8000/stats
```

### Collect Results

After `ng_collect_rollouts` completes, stop the servers (Ctrl+C). Profiling logs are written to `profiling_results_dirpath`:

```text
results/profiling/math_with_judge/
└── SimpleResourcesServer___math_with_judge.log
```

---

## Interpreting Yappi Output

Log files contain per-function timing data:

```text
name                                                                              ncall       tsub      ttot      tavg      
.../resources_servers/math_with_judge/app.py:118 LibraryJudgeMathResourcesServer.verify   1024   0.009755  17.98387  0.017562
.../resources_servers/math_with_judge/app.py:145 LibraryJudgeMathResourcesServer._verify_answer   1024   0.002933  17.87998  0.017461
```

**Column definitions**:

| Column | Description |
|--------|-------------|
| `ncall` | Number of times the function was called |
| `tsub` | Time spent in the function itself (excluding called functions) |
| `ttot` | Total time including all nested calls |
| `tavg` | Average time per call (`ttot / ncall`) |

**Interpretation**:

- **High `tsub`**: The function itself is expensive—optimize its code
- **High `ttot` with low `tsub`**: Bottleneck is in called functions—drill down
- **High `ncall`**: Function called frequently—consider caching or batching

---

## Quick Profiling

### Inline Timing

Add timing to individual tools:

```python
import time

async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
    start = time.perf_counter()
    result = await self._do_work(body)
    elapsed = time.perf_counter() - start
    print(f"Tool latency: {elapsed*1000:.2f}ms")
    return result
```

### Timing with Response Metadata

Track timing in verification responses:

```python
from time import time

async def verify(self, body: CompCodingVerifyRequest) -> CompCodingVerifyResponse:
    start_time = time()
    
    # ... verification logic ...
    
    return CompCodingVerifyResponse(
        **body.model_dump(),
        reward=reward,
        unit_tests_time_taken=time() - start_time,
    )
```

**Verified in**: `resources_servers/code_gen/app.py:128-149`

---

## Identifying Bottlenecks

### I/O Bound Operations

For async I/O (API calls, file reads), ensure you're not blocking the event loop:

:::::{tab-set}

::::{tab-item} ❌ Blocking

```python
import requests

async def my_tool(self, body: ToolRequest) -> ToolResponse:
    # BAD: Blocks the event loop
    response = requests.get(url)
    return ToolResponse(data=response.json())
```

::::

::::{tab-item} ✅ Non-blocking

```python
import aiohttp

async def my_tool(self, body: ToolRequest) -> ToolResponse:
    # GOOD: Async HTTP
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
    return ToolResponse(data=data)
```

::::

:::::

:::{tip}
For simple use cases, synchronous `requests` is acceptable. For high concurrency (1000+ parallel requests), use `aiohttp`. See {doc}`integrate-apis` for details.
:::

### CPU Bound Operations

Offload CPU-intensive work to avoid blocking the event loop.

:::::{tab-set}

::::{tab-item} Semaphore + Ray

Use a semaphore to limit concurrency and Ray for distributed execution:

```python
from asyncio import Semaphore, get_running_loop
import ray

class MyResourcesServer(SimpleResourcesServer):
    def model_post_init(self, context):
        # model_post_init runs after Pydantic model initialization
        # Set semaphore value based on available CPU cores or config
        self._semaphore = Semaphore(value=self.config.num_processes)

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        async with self._semaphore:
            loop = get_running_loop()
            future = compute_expensive_task.remote(body.data)
            result = await loop.run_in_executor(None, ray.get, future)
        return VerifyResponse(reward=result)
```

**Verified in**: `resources_servers/code_gen/app.py:74-149`

:::{tip}
Size your semaphore based on resource constraints. For CPU-bound tasks, use `os.cpu_count()`. For I/O-bound tasks with external rate limits, match the rate limit.
:::

::::

::::{tab-item} Thread Pool

For non-Ray workloads, use a thread pool:

```python
from asyncio import get_running_loop
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def verify(self, body: VerifyRequest) -> VerifyResponse:
    loop = get_running_loop()
    result = await loop.run_in_executor(executor, cpu_heavy_function, body.data)
    return VerifyResponse(reward=result)
```

:::{warning}
Thread pools hold threads in memory. For long-running servers, ensure `max_workers` is bounded to prevent memory growth.
:::

::::

:::::

---

## Optimization Strategies

### Connection Pooling

Reuse HTTP sessions across requests:

```python
def model_post_init(self, context):
    self._session = aiohttp.ClientSession()
```

NeMo Gym provides a global aiohttp client with configurable limits:

| Config | Default | Description |
|--------|---------|-------------|
| `global_aiohttp_connector_limit` | 102400 | Total connection pool size |
| `global_aiohttp_connector_limit_per_host` | 1024 | Per-host connection limit |

**Verified in**: `nemo_gym/server_utils.py:76-121`

### Concurrency Control

Use semaphores to prevent resource exhaustion:

```python
from asyncio import Semaphore

class MyResourcesServer(SimpleResourcesServer):
    def model_post_init(self, context):
        self._semaphore = Semaphore(value=self.config.max_concurrent)

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        async with self._semaphore:
            # Limited concurrent execution
            result = await self._expensive_operation(body)
        return VerifyResponse(reward=result)
```

### Caching

Cache expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def parse_expression(expr: str) -> ParsedResult:
    # Expensive parsing cached by input
    return expensive_parse(expr)
```

For async caching, consider `aiocache` or similar libraries.

### Batching

Batch multiple requests to external services:

```python
async def _batch_verify(self, items: list[VerifyItem]) -> list[float]:
    # Single API call for multiple items
    response = await self._api.batch_score(items)
    return response.scores
```

---

## Benchmarking

### Establish Baselines

Before optimizing, measure baseline performance:

```bash
# Baseline measurement
ng_collect_rollouts +agent_name=my_agent \
    +input_jsonl_fpath=test_data.jsonl \
    +output_jsonl_fpath=baseline_results.jsonl \
    +num_samples_in_parallel=50 \
    +limit=1000
```

Record:

- Total time
- Rollouts per second
- `tavg` for key functions

### Before/After Comparisons

After optimization, run the same test:

```bash
# After optimization
ng_collect_rollouts +agent_name=my_agent \
    +input_jsonl_fpath=test_data.jsonl \
    +output_jsonl_fpath=optimized_results.jsonl \
    +num_samples_in_parallel=50 \
    +limit=1000
```

Compare `tavg` values in profiling logs to quantify improvement.

---

## Related

- {doc}`/about/performance` — Global performance controls
- {doc}`integrate-apis` — HTTP client patterns
- {doc}`/reference/faq` — Profiling FAQ

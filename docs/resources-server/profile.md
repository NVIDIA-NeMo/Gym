(resources-server-profile)=

# Profile Resources Server

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #547](https://github.com/NVIDIA-NeMo/Gym/issues/547) for details.
```

Measure and optimize the performance of your resources server for high-throughput rollout collection.

---

## Prerequisites

Install profiling tools:

```bash
pip install memory_profiler
pip install py-spy  # May require sudo for some operations
```

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Tool latency** | Time per tool call | < 100ms for simple tools |
| **Verify latency** | Time per verification | < 50ms |
| **Throughput** | Rollouts per second | Depends on tools |
| **Memory usage** | RAM per session | Minimize |
| **Event loop lag** | Async responsiveness | < 10ms |

## Quick Profiling

### Inline Timing

```python
import time

async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
    start = time.perf_counter()
    result = await self._do_work(body)
    elapsed = time.perf_counter() - start
    print(f"Tool latency: {elapsed*1000:.2f}ms")
    return result
```

### Load Testing with ng_collect_rollouts

```bash
# Test with concurrent requests
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=test_data.jsonl \
    +output_jsonl_fpath=results.jsonl \
    +num_samples_in_parallel=50 \
    +limit=1000
```

## Deep Profiling Tools

### cProfile

<!-- TODO: Document Python profiling with cProfile -->

### py-spy

<!-- TODO: Document py-spy for production profiling -->

### Memory Profiling

<!-- TODO: Document memory profiling with memory_profiler -->

### Async Profiling

Since resources servers are async FastAPI applications, standard profilers may miss coroutine-level timing:

<!-- TODO: Document async-specific profiling techniques -->
<!-- - Event loop blocking detection -->
<!-- - Coroutine timing -->
<!-- - aiomonitor or similar tools -->

## Interpreting Results

### Reading cProfile Output

<!-- TODO: Document how to interpret cProfile output -->
<!-- - cumtime vs tottime -->
<!-- - Call counts -->
<!-- - Sorting by different metrics -->

### Understanding Flame Graphs

<!-- TODO: Document flame graph interpretation -->
<!-- - Width = time spent -->
<!-- - Stack depth = call hierarchy -->
<!-- - Hot paths identification -->

### Common Patterns

<!-- TODO: Document common profiling patterns -->
<!-- - Synchronous code blocking event loop -->
<!-- - Excessive memory allocation -->
<!-- - N+1 API call patterns -->

## Identifying Bottlenecks

### I/O Bound Operations

<!-- TODO: Document async I/O optimization -->

### CPU Bound Operations

<!-- TODO: Document CPU optimization -->

### External API Calls

<!-- TODO: Document API call optimization -->

## Optimization Strategies

### Connection Pooling

```python
# Reuse HTTP sessions across requests
def model_post_init(self, context):
    self._session = aiohttp.ClientSession()
```

See {doc}`integrate-apis` for more HTTP client patterns.

### Caching

<!-- TODO: Document caching strategies -->

### Batching

<!-- TODO: Document batching strategies -->

## Benchmarking & Validation

### Establishing Baselines

<!-- TODO: Document baseline measurement methodology -->

### Before/After Comparisons

<!-- TODO: Document A/B performance comparison -->

### Automated Regression Testing

<!-- TODO: Document CI integration for performance tests -->

## Examples

For profiling examples, see:

- `resources_servers/` — Review existing implementations for performance patterns

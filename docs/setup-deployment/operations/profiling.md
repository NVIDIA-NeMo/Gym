(setup-operations-profiling)=

# Performance Profiling

Profile resource servers to identify bottlenecks and optimize for high-throughput training (16k+ concurrent requests).

---

## Enable Profiling

Configure NeMo Gym to collect profiling data during server execution:

**Step 1: Start servers with profiling enabled**

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"

ng_run "+config_paths=[$config_paths]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/library_judge_math
```

**Configuration options**:

```{list-table}
:header-rows: 1
:widths: 40 15 45

* - Parameter
  - Type
  - Description
* - `profiling_enabled`
  - bool
  - Enable profiling (disabled by default due to slight overhead)
* - `profiling_results_dirpath`
  - str
  - Directory for profiling results (previous results overwritten)
```

:::{tip}
Profiling introduces minimal overhead (~1-2%) but should be disabled in production unless actively investigating performance issues.
:::

**Step 2: Run workload** (in separate terminal)

```bash
ng_collect_rollouts +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=resources_servers/library_judge_math/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/library_judge_math_rollouts.jsonl \
    +limit=1024 \
    +num_repeats=1
```

Use `limit` and `num_repeats` to control workload size for representative profiling.

**Step 3: Stop servers** (Ctrl+C)

Profiling results save automatically on shutdown.

---

## Reading Profiling Results

Profiling results are saved to `<profiling_results_dirpath>/<server_name>.log`.

### Example Output

```
name                                                                     ncall   tsub      ttot      tavg      
.../library_judge_math/app.py:118 LibraryJudgeMath...Server.verify      1024    0.009755  17.98387  0.017562
.../library_judge_math/app.py:145 ...Server._verify_answer              1024    0.002933  17.87998  0.017461
.../library_judge_math/app.py:173 ...Server._verify_answer_with_library 1024    0.007851  17.87704  0.017458
.../library_judge_math/app.py:191 <genexpr>                             2339    0.001695  0.029082  0.000012
.../library_judge_math/app.py:163 _mute_output                          2048    0.007473  0.016538  0.000008
```

### Understanding Profiling Metrics

**Column definitions**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Metric
  - Description
* - `ncall`
  - Number of times function was called
* - `tsub`
  - Time in function itself (excluding nested calls) — "self time"
* - `ttot`
  - Total time including all nested function calls
* - `tavg`
  - Average time per call (ttot / ncall)
```

**Example interpretation**:

```
verify:                      ttot=17.98s, tsub=0.009s  → Bottleneck in nested calls
_verify_answer:              ttot=17.87s, tsub=0.002s  → Still in nested calls
_verify_answer_with_library: ttot=17.87s, tsub=0.007s  → Actual bottleneck here
```

The `verify` function took 17.98s total, but only 0.009s in the function itself. The remaining ~17.87s was spent in nested calls, specifically in `_verify_answer_with_library`.

---

## Identifying Bottlenecks

Use these patterns to find optimization opportunities:

### Analysis Checklist

**Look for**:

1. **High `ttot`** - Functions consuming most total time
   - These are your primary optimization targets
   - Focus on functions with `ttot` > 10% of total runtime

2. **High `tsub`** - Functions with expensive direct operations
   - These functions have heavy computation in the function itself
   - Optimize algorithms or move to compiled code

3. **High `tavg` with many calls** - Good targets for O(n) improvements
   - Small per-call improvements have big impact
   - Consider caching or vectorization

4. **High `ttot - tsub`** - Bottleneck is in nested calls
   - Drill down into called functions
   - Consider parallelization of nested calls

### Optimization Priority Matrix

```{list-table}
:header-rows: 1
:widths: 35 35 30

* - Pattern
  - Optimization Strategy
  - Example
* - High `tavg` × high `ncall`
  - Biggest impact potential
  - Cache repeated calculations
* - High `tsub`
  - Direct code optimization
  - Use faster algorithms
* - Synchronous external calls
  - Parallelization
  - Use `async` or Ray
* - Repeated heavy operations
  - Memoization
  - LRU cache decorators
```

---

## Real-Time Profiling Stats

Access profiling data while servers are running without stopping them:

```bash
# Get current profiling statistics
curl http://localhost:<server-port>/stats
```

**Example usage**:

```bash
# Monitor resource server stats
curl http://localhost:8003/stats

# Compare before and after optimization
curl http://localhost:8003/stats > before.json
# ... make code changes ...
curl http://localhost:8003/stats > after.json
diff before.json after.json
```

:::{tip}
Use the `/stats` endpoint to monitor profiling metrics in production without restarting servers or interrupting workloads.
:::

---

## Profiling Workflow

### Recommended Profiling Workflow

**Step 1: Establish Baseline**

```bash
# Profile current implementation
ng_run "+config_paths=[config.yaml]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/baseline

# Run representative workload
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test.jsonl \
    +output_jsonl_fpath=results/baseline_rollouts.jsonl \
    +limit=1000

# Stop servers (Ctrl+C)
# Review results/profiling/baseline/resource_server.log
```

**Step 2: Identify Bottlenecks**

Review profiling results and identify:
- Functions with highest `ttot`
- Functions called frequently with high `tavg`
- Synchronous operations that could be parallelized

**Step 3: Optimize**

Apply optimization strategies:

```python
# Before: Synchronous external calls
def verify(self, question, answer):
    result1 = external_api.call(answer)
    result2 = another_api.call(answer)
    return result1 and result2

# After: Parallel external calls
async def verify(self, question, answer):
    result1, result2 = await asyncio.gather(
        external_api.call_async(answer),
        another_api.call_async(answer)
    )
    return result1 and result2
```

**Step 4: Compare Results**

```bash
# Profile optimized implementation
ng_run "+config_paths=[config.yaml]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/optimized

# Run same workload
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test.jsonl \
    +output_jsonl_fpath=results/optimized_rollouts.jsonl \
    +limit=1000

# Compare results
diff results/profiling/baseline/resource_server.log \
     results/profiling/optimized/resource_server.log
```

---

## Common Bottlenecks and Solutions

### Synchronous External API Calls

**Problem**: External API calls block execution

```python
# Slow: Synchronous calls
def verify(self, answer):
    for test_case in self.test_cases:
        result = external_api.verify(test_case, answer)  # Blocks
        if not result:
            return False
    return True
```

**Solution**: Use async or Ray for parallelization

```python
# Fast: Parallel async calls
async def verify(self, answer):
    tasks = [
        external_api.verify_async(test_case, answer)
        for test_case in self.test_cases
    ]
    results = await asyncio.gather(*tasks)
    return all(results)
```

### Heavy Computation in Verify Function

**Problem**: CPU-intensive verification logic

```python
# Slow: Heavy computation
def verify(self, question, answer):
    for i in range(len(large_dataset)):
        if complex_computation(answer, large_dataset[i]):
            return True
    return False
```

**Solution**: Optimize algorithm or use Ray for parallelization

```python
# Fast: Parallel computation with Ray
import ray

@ray.remote
def verify_chunk(answer, chunk):
    return any(complex_computation(answer, item) for item in chunk)

def verify(self, question, answer):
    chunks = split_into_chunks(large_dataset, num_chunks=8)
    futures = [verify_chunk.remote(answer, chunk) for chunk in chunks]
    results = ray.get(futures)
    return any(results)
```

### Repeated Expensive Operations

**Problem**: Same expensive computation repeated multiple times

```python
# Slow: Repeated computation
def verify(self, question, answer):
    parsed = expensive_parse(answer)  # Called on every verification
    return validate(parsed)
```

**Solution**: Use caching

```python
# Fast: Memoization
from functools import lru_cache

@lru_cache(maxsize=1024)
def expensive_parse(answer):
    # ... expensive parsing logic
    return parsed_result

def verify(self, question, answer):
    parsed = expensive_parse(answer)  # Cached
    return validate(parsed)
```

### Inefficient Data Structures

**Problem**: O(n) lookups in lists

```python
# Slow: Linear search
def verify(self, question, answer):
    valid_answers = [...]  # Large list
    return answer in valid_answers  # O(n) lookup
```

**Solution**: Use appropriate data structures

```python
# Fast: Constant-time lookup
def __init__(self):
    self.valid_answers = set([...])  # Convert to set

def verify(self, question, answer):
    return answer in self.valid_answers  # O(1) lookup
```

---

## Profiling Best Practices

### When to Profile

**Profile during**:
- Before production deployment
- After adding new verification logic
- When experiencing performance issues
- To validate optimization improvements

**Don't profile**:
- During production workloads (adds overhead)
- With unrealistic workload sizes
- Without representative data

### Profiling Configuration

**Workload size selection**:

```bash
# Small workload (quick feedback)
+limit=100

# Medium workload (realistic profiling)
+limit=1000

# Large workload (production-scale testing)
+limit=10000
```

**Choose workload based on**:
- Available time (small for iteration, large for validation)
- Representative diversity (ensure coverage of edge cases)
- Production scale (profile at expected throughput)

### Interpreting Results

**Red flags**:
- Any single function with `ttot` > 50% of total runtime
- Average call time (`tavg`) increasing with dataset size
- Synchronous external calls in hot paths

**Green flags**:
- Bottlenecks in external services (optimize API usage)
- Even distribution of time across functions
- Low `tavg` for frequently called functions

---

## Scale Testing for Production

For high-throughput training scenarios (16k+ concurrent requests):

### Stress Testing

```bash
# Simulate high concurrency
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/large_dataset.jsonl \
    +output_jsonl_fpath=results/stress_test.jsonl \
    +limit=16384 \
    +num_repeats=1
```

### Distributed Profiling

**Profile Ray-based distributed deployment**:

```bash
# Start Ray cluster
ray start --head

# Profile with Ray parallelization
ng_run "+config_paths=[config.yaml]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/distributed \
    +use_ray=true \
    +num_workers=8
```

:::{seealso}
For Ray deployment patterns, refer to {doc}`../deployment/distributed-computing`.
:::

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Run Tests
:link: testing
:link-type: doc

Validate optimizations with comprehensive tests
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Monitor Production
:link: monitoring
:link-type: doc

Set up ongoing performance monitoring
:::

::::


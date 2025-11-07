(setup-operations-profiling)=

# Performance Profiling

Profile resource servers to identify performance bottlenecks and optimize verification logic.

---

## Quick Start

Enable profiling during rollout collection to capture function-level performance metrics.

:::::{tab-set}

::::{tab-item} Basic Profiling

**Terminal 1: Start with profiling**

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"

ng_run "+config_paths=[$config_paths]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/library_judge_math
```

**Terminal 2: Run workload**

```bash
ng_collect_rollouts +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=resources_servers/library_judge_math/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/library_judge_math_rollouts.jsonl \
    +limit=1024
```

**Terminal 1: Stop and save** (Ctrl+C)

Results save automatically to `results/profiling/library_judge_math/<server_name>.log`

::::

::::{tab-item} Live Monitoring

Monitor profiling stats without stopping servers:

```bash
# First, find your resource server's port from ng_run startup output
# Look for: "🔍 Enabled profiling for <server_name>"
# The server info will show its port (auto-assigned, varies per run)

# Check real-time profiling data (replace <PORT> with actual port)
curl http://localhost:<PORT>/stats

# Example: If your resource server is on port 54321
curl http://localhost:54321/stats

# Track changes during optimization
curl http://localhost:<PORT>/stats > baseline.txt
# ... make code changes and deploy ...
curl http://localhost:<PORT>/stats > optimized.txt
diff baseline.txt optimized.txt
```

:::{tip}
To find the port: Check `ng_run` output for server startup messages, or use `lsof -i -P | grep LISTEN` to see all listening ports.
:::

::::

:::::

### Configuration

These parameters control profiling behavior and output location.

```{list-table}
:header-rows: 1
:widths: 30 15 55

* - Parameter
  - Type
  - Description
* - `profiling_enabled`
  - `bool`
  - Enable profiling (disabled by default; adds minimal overhead)
* - `profiling_results_dirpath`
  - `str`
  - Directory for results (relative to project root; overwrites existing)
```

---

## Understanding Results

Profiling output shows function call counts, execution times, and performance bottlenecks.

### Output Format

Profiling results show function-level performance metrics:

```
name                                                                     ncall   tsub      ttot      tavg      
.../library_judge_math/app.py:118 LibraryJudgeMath...Server.verify      1024    0.009755  17.98387  0.017562
.../library_judge_math/app.py:145 ...Server._verify_answer              1024    0.002933  17.87998  0.017461
.../library_judge_math/app.py:173 ...Server._verify_answer_with_library 1024    0.007851  17.87704  0.017458
```

```{list-table}
:header-rows: 1
:widths: 15 85

* - Metric
  - Meaning
* - `ncall`
  - Number of times function was called
* - `tsub`
  - Time spent in function itself (excluding nested calls)
* - `ttot`
  - Total time including all nested function calls
* - `tavg`
  - Average time per call (`ttot / ncall`)
```

### Reading the Data

**Finding bottlenecks**:

1. **High `ttot`** → Primary time consumers (optimize these first)
2. **High `tsub`** → Expensive direct computation (optimize algorithm)
3. **High `tavg` × many `ncall`** → Small improvements have big impact (add caching)
4. **High `ttot`, low `tsub`** → Bottleneck in nested calls (drill down)

---

## Optimization Strategies

Apply these techniques to reduce verification latency and improve throughput.

### Common Bottlenecks

These patterns cause most performance issues in resource server implementations.



:::::{tab-set}

::::{tab-item} Synchronous API Calls

**Problem**: Blocking external calls

```python
# Before: Sequential blocking
def verify(self, answer):
    for test in self.test_cases:
        result = api.verify(test, answer)  # Blocks
        if not result:
            return False
    return True
```

**Solution**: Parallelize with async

```python
# After: Concurrent requests
async def verify(self, answer):
    tasks = [api.verify_async(test, answer) for test in self.test_cases]
    results = await asyncio.gather(*tasks)
    return all(results)
```

::::

::::{tab-item} Repeated Computation

**Problem**: Redundant expensive operations

```python
# Before: Recomputes every time
def verify(self, question, answer):
    parsed = expensive_parse(answer)
    return validate(parsed)
```

**Solution**: Add caching

```python
# After: Cache results
from functools import lru_cache

@lru_cache(maxsize=1024)
def expensive_parse(answer):
    return parsed_result

def verify(self, question, answer):
    parsed = expensive_parse(answer)  # Cached
    return validate(parsed)
```

::::

::::{tab-item} Inefficient Data Structures

**Problem**: Linear search in large collections

```python
# Before: O(n) lookup
def verify(self, question, answer):
    valid_answers = [...]  # Large list
    return answer in valid_answers  # O(n)
```

**Solution**: Use appropriate data structure

```python
# After: O(1) lookup
def __init__(self):
    self.valid_answers = set([...])

def verify(self, question, answer):
    return answer in self.valid_answers  # O(1)
```

::::

:::::

::::{dropdown} Advanced: Ray parallelization for CPU-intensive tasks

For computationally heavy verification logic:

```python
import ray

@ray.remote
def verify_chunk(answer, chunk):
    return any(complex_computation(answer, item) for item in chunk)

def verify(self, question, answer):
    chunks = split_into_chunks(self.large_dataset, num_chunks=8)
    futures = [verify_chunk.remote(answer, chunk) for chunk in chunks]
    return any(ray.get(futures))
```

:::{seealso}
For Ray setup, refer to {doc}`../deployment/distributed-computing`
:::

::::


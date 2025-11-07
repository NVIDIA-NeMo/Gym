(setup-operations)=

# Operations

Monitor, test, and debug your NeMo Gym deployment to ensure reliable operation.

---

## Monitoring

Set up health checks and monitoring:

### Health Checks

Each server exposes a health endpoint:

```bash
curl http://localhost:8000/health
```

### Log Monitoring

Monitor logs for errors:

```bash
# View logs
ng_run "+config_paths=[config.yaml]" --log-level DEBUG

# Check for errors
grep ERROR ng_gym.log
```

### Resource Usage

Monitor CPU, memory, and network usage:

```bash
# System monitoring
top
htop

# Network monitoring
netstat -an | grep 8000
```

---

## Testing

Test your NeMo Gym setup to ensure reliability and correctness.

### Run Server Tests

Test individual servers (agents, models, or resource servers):

```bash
# Test simple agent
ng_test +entrypoint=responses_api_agents/simple_agent

# Test resource server
ng_test +entrypoint=resources_servers/simple_weather

# Test with data validation
ng_test +entrypoint=resources_servers/multineedle +should_validate_data=true
```

**What `ng_test` does**:
1. Sets up isolated virtual environment
2. Installs dependencies from `requirements.txt`
3. Runs pytest in server directory
4. Optionally validates example data (5 examples required)

### Detailed Test Execution

After running `ng_test` once, run tests directly in the server environment:

```bash
cd resources_servers/simple_weather
source .venv/bin/activate

# Run all tests
pytest

# Verbose output
pytest -v

# Specific test
pytest tests/test_app.py::test_verify

# With coverage
pytest --cov=.
```

### Test Requirements for Resource Servers

Every resource server must include:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Artifact
  - Purpose
* - `tests/test_app.py`
  - At least one test (you own correctness)
* - `data/example.jsonl`
  - 5 example inputs for the agent
* - `data/example_metrics.json`
  - Metrics from `ng_prepare_data`
* - `data/example_rollouts.jsonl`
  - Rollouts from `ng_collect_rollouts`
```

```{important}
Test coverage is NOT enforced by CI. **You are responsible for your server's correctness and functionality.**
```

### Configuration Validation

Validate configuration loads correctly before running:

```bash
# Validate configuration
ng_dump_config "+config_paths=[config.yaml]"

# Pipe to grep for specific checks
ng_dump_config "+config_paths=[config.yaml]" | grep api_key
```

### Integration Testing

Test the complete pipeline:

```bash
# Small-scale integration test
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/test_rollouts.jsonl \
    +limit=10
```

### Functional Testing

Run the full test suite:

```bash
# Run unit tests
pytest tests/unit_tests/

# Run functional tests
bash tests/functional_tests/run_tests.sh
```

---

## Performance Profiling

Profile resource servers to optimize for high-throughput training (16k+ concurrent requests).

### Enable Profiling

**Step 1: Start servers with profiling enabled**

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[$config_paths]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/library_judge_math
```

**Configuration options**:
- `profiling_enabled` (bool): Enable profiling (disabled by default due to slight overhead)
- `profiling_results_dirpath` (str): Directory for profiling results (previous results overwritten)

**Step 2: Run workload** (in separate terminal)

```bash
ng_collect_rollouts +agent_name=library_judge_math_simple_agent \
    +input_jsonl_fpath=resources_servers/library_judge_math/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/library_judge_math_rollouts.jsonl \
    +limit=1024 \
    +num_repeats=1
```

Use `limit` and `num_repeats` to control workload size.

**Step 3: Stop servers** (Ctrl+C)

Profiling results save automatically on shutdown.

### Reading Profiling Results

Results saved to `<profiling_results_dirpath>/<server_name>.log`:

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

### Identifying Bottlenecks

**Look for**:
1. **High `ttot`** - Functions consuming most total time
2. **High `tsub`** - Functions with expensive direct operations
3. **High `tavg` with many calls** - Good targets for O(n) improvements
4. **High `ttot - tsub`** - Bottleneck is in nested calls

**Optimization priorities**:
- Functions with high `tavg` × high `ncall` (biggest impact)
- Functions with high `tsub` (direct code optimization)
- Functions calling expensive operations (parallelization opportunities)

### Real-Time Profiling Stats

While servers are running, access profiling data via HTTP:

```bash
curl http://localhost:<server-port>/stats
```

This endpoint returns current profiling statistics without stopping the server.

### Profiling Best Practices

**When to profile**:
- Before production deployment
- After adding new verification logic
- When experiencing performance issues
- To validate optimization improvements

**Common bottlenecks**:
- Synchronous external API calls (use async)
- Heavy computation in verify() (use Ray for parallelization)
- Inefficient data structures (optimize or cache)
- Repeated expensive operations (memoize)

```{tip}
For large-scale training, your resource server may handle 16k+ concurrent requests. Profile with realistic workloads to identify issues before production.
```

:::{seealso}
Use Ray for parallelizing CPU-intensive bottlenecks. See {doc}`../deployment/index` for Ray usage patterns.
:::

---

## Debugging

Common debugging techniques:

### Enable Debug Logging

```bash
ng_run "+config_paths=[config.yaml]" --log-level DEBUG
```

### Check Server Connectivity

```bash
# Test if servers are reachable
curl http://localhost:8000/health

# Check which ports are listening
lsof -i :8000
netstat -an | grep 8000
```

### Common Issues

**Issue**: `Connection refused`
**Solution**: Check if server is running and port is correct

**Issue**: `API key invalid`
**Solution**: Verify env.yaml contains correct API keys

**Issue**: `Module not found`
**Solution**: Ensure you installed with `pip install -e .`

### Configuration Debugging

```bash
# Dump full resolved configuration
ng_dump_config "+config_paths=[config.yaml]"

# Check specific value
ng_dump_config "+config_paths=[config.yaml]" | grep policy_api_key
```




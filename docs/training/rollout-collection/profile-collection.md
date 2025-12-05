(profile-rollout-collection)=

# Profile Rollout Collection

Identify performance bottlenecks in your resource servers and rollout pipeline.

---

## How It Works

For large-scale training, your resource server needs to handle thousands of concurrent requests efficiently. Gym provides profiling tools to understand where time is spent.

---

## Enable Profiling

Start servers with profiling enabled:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/math_with_judge
```

### Profiling Parameters

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Parameter
  - Default
  - Description
* - `+profiling_enabled`
  - false
  - Enable profiling (incurs slight overhead)
* - `+profiling_results_dirpath`
  - (required)
  - Directory to save profiling results
```

---

## Run a Load Test

Generate a significant number of rollouts to collect meaningful profiling data:

```bash
ng_collect_rollouts +agent_name=math_with_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/math_with_judge/data/dapo17k_bytedtsinghua_train.jsonl \
    +output_jsonl_fpath=temp/math_with_judge_rollouts.jsonl \
    +limit=1024 \
    +num_repeats=1
```

After collection completes, stop your servers (Ctrl+C) to finalize the profiling logs.

---

## Interpret Results

The profiling log shows function-level timing:

```text
name                                             ncall    tsub      ttot      tavg      
LibraryJudgeMathResourcesServer.verify           1024     0.009755  17.98387  0.017562
LibraryJudgeMathResourcesServer._verify_answer   1024     0.002933  17.87998  0.017461
LibraryJudgeMathResourcesServer._verify_with_lib 1024     0.007851  17.87704  0.017458
<genexpr>                                        2339     0.001695  0.029082  0.000012
_mute_output                                     2048     0.007473  0.016538  0.000008
```

### Column Definitions

```{list-table}
:header-rows: 1
:widths: 15 85

* - Column
  - Description
* - `ncall`
  - Number of times the function was called
* - `tsub`
  - Time spent **inside** the function itself (excluding calls to other functions)
* - `ttot`
  - **Total** time including all nested function calls
* - `tavg`
  - Average time per call (`ttot / ncall`)
```

### Reading the Example

From the output above:
- `verify()` was called 1024 times (once per rollout)
- Total time in `verify()` and its children: 17.98s
- Average per call: 17.5ms
- The function itself only took 0.01s — most time is in `_verify_answer`

---

## Common Bottlenecks

### Verification Logic

If `verify()` is slow:
- Check for synchronous I/O (database calls, file reads)
- Consider caching repeated computations
- Use async operations where possible

### Tool Execution

If tool calls are slow:
- Profile individual tool functions
- Check external API latency
- Consider batching or caching

### Model Calls

If generation is the bottleneck:
- Increase `num_samples_in_parallel` (if backend can handle it)
- Use a faster generation backend
- Reduce `max_output_tokens` if responses are unnecessarily long

---

## Optimization Checklist

- [ ] Profile with realistic load (1000+ rollouts)
- [ ] Identify functions with high `ttot`
- [ ] Check if `tsub` is high (function itself) or low (time in children)
- [ ] Optimize the actual bottleneck, not assumptions
- [ ] Re-profile after changes to verify improvement


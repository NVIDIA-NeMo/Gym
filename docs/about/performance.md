(about-performance)=
# Performance

```{note}
This page is a stub. Content is being developed.
```

NeMo Gym is designed for high-throughput rollout collection to support large-scale RL training.

## Throughput Benchmarks

<!-- TODO: Add throughput benchmarks for different configurations -->

## Concurrent Rollout Collection

The `ng_collect_rollouts` command supports concurrent requests:

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=data.jsonl \
    +output_jsonl_fpath=rollouts.jsonl \
    +num_samples_in_parallel=10  # Concurrent requests
```

## Latency Optimization

### Model Server Latency

<!-- TODO: Document model server latency considerations and vLLM optimizations -->

### Tool Execution Latency

<!-- TODO: Document tool execution latency and async best practices -->

## Resource Utilization

### CPU Resources

<!-- TODO: Document CPU utilization patterns -->

### GPU Resources

<!-- TODO: Document GPU utilization for model inference -->

### Memory Management

<!-- TODO: Document memory management and session lifecycle -->

## Profiling

For detailed performance analysis, see {doc}`/resources-server/profile`.

(about-performance)=
# Performance

This page lists the performance controls and profiling hooks in Gym.

## TL;DR

- **Concurrency**: Use `+num_samples_in_parallel` to cap concurrent rollout requests.
- **Profiling**: Enable with `+profiling_enabled=true` to identify bottlenecks.
- **Connection limits**: Configure `global_aiohttp_connector_limit` for high-throughput workloads.
- **Workers**: Set `num_workers` for Uvicorn server worker count.

## Rollout collection concurrency

Use `+num_samples_in_parallel` to cap concurrent rollout requests (see {doc}`/about/concepts/key-terminology`).
When set, the collector uses a semaphore to limit concurrency during rollout collection.

```bash
ng_collect_rollouts \
    +agent_name=example_single_tool_call_simple_agent \
    +input_jsonl_fpath=weather_query.jsonl \
    +output_jsonl_fpath=weather_rollouts.jsonl \
    +limit=100 \
    +num_repeats=4 \
    +num_samples_in_parallel=10
```

After collection, the command averages numeric fields from rollout responses and prints the aggregated metrics as JSON.

## Server profiling

Enable CPU profiling by setting these global config fields:

- `profiling_enabled`
- `profiling_results_dirpath`

When profiling is enabled, the server:

- Starts a Yappi CPU profiler during the FastAPI lifespan
- Writes a `.log` file per server instance to `profiling_results_dirpath`
- Exposes a `/stats` endpoint with current profiler stats

Example (start servers with profiling enabled):

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[${config_paths}]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/math_with_judge
```

Profiling logs for the same server name overwrite the existing `.log` file in that directory.

For profiling walkthroughs, see {doc}`/resources-server/profile`.

## Connection limits and workers

Gym sets a global aiohttp client for server requests with:

- `global_aiohttp_connector_limit`
- `global_aiohttp_connector_limit_per_host`

For servers, `num_workers` sets the Uvicorn worker count.

## File descriptor limit

On startup, Gym tries to raise the file descriptor limit to `65535` before starting the Uvicorn server.

## Where global config fields come from

Gym resolves the global config once per run, merging:

1. Config files from `config_paths`
2. Local `env.yaml`
3. Command line overrides

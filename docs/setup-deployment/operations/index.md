(setup-operations)=

# Operations

Monitor, test, profile, and debug NeMo Gym deployments to ensure reliable, high-performance operation.

---

## Operations Topics

Choose the operational task that matches your current need:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Monitoring
:link: monitoring
:link-type: doc

Monitor health, logs, and resource usage across your deployment.
+++
{bdg-secondary}`health-checks` {bdg-secondary}`logs` {bdg-secondary}`metrics`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Testing
:link: testing
:link-type: doc

Test servers, validate configurations, and run integration tests.
+++
{bdg-secondary}`unit-tests` {bdg-secondary}`integration` {bdg-secondary}`validation`
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Performance Profiling
:link: profiling
:link-type: doc

Profile resource servers to identify and optimize performance bottlenecks.
+++
{bdg-secondary}`profiling` {bdg-secondary}`optimization` {bdg-secondary}`scale`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: debugging
:link-type: doc

Debug configuration, connectivity, and runtime issues.
+++
{bdg-secondary}`troubleshooting` {bdg-secondary}`logs` {bdg-secondary}`errors`
:::

::::

---

## Operations Workflow

Recommended operational workflow for production deployments:

::::{tab-set}

:::{tab-item} Development

**Local development and testing**:

1. **Test**: Validate servers with `ng_test`
2. **Debug**: Use debug logging to catch issues early
3. **Profile**: Identify bottlenecks before deployment
4. **Monitor**: Check health endpoints during development

**Quick commands**:

```bash
# Test server
ng_test +entrypoint=resources_servers/your_server

# Run with debug logging
ng_run "+config_paths=[config.yaml]" --log-level DEBUG

# Profile development workload
ng_run "+config_paths=[config.yaml]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling
```

:::

:::{tab-item} Staging

**Pre-production validation**:

1. **Test**: Run integration tests with production-like data
2. **Profile**: Run at scale to identify performance issues
3. **Monitor**: Set up health checks and log aggregation
4. **Debug**: Resolve issues before production deployment

**Quick commands**:

```bash
# Integration test
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/staging_test.jsonl \
    +output_jsonl_fpath=results/staging_rollouts.jsonl \
    +limit=1000

# Profile at scale
ng_run "+config_paths=[config.yaml]" \
    +profiling_enabled=true \
    +profiling_results_dirpath=results/profiling/staging

# Monitor health
for port in 8000 8001 8002 8003; do
  curl -s http://localhost:$port/health || echo "Port $port unhealthy"
done
```

:::

:::{tab-item} Production

**Ongoing operations**:

1. **Monitor**: Continuous health and performance monitoring
2. **Test**: Automated testing on each deployment
3. **Debug**: Quick incident response with logs and metrics
4. **Profile**: Periodic profiling to catch regressions

**Quick commands**:

```bash
# Health monitoring script
watch -n 10 'curl -s http://production-host:8000/health'

# Check logs for errors
grep ERROR /var/log/nemo_gym/*.log

# Get profiling stats without restart
curl http://production-host:8003/stats

# Run smoke test
ng_collect_rollouts +agent_name=prod_agent \
    +input_jsonl_fpath=data/smoke_test.jsonl \
    +output_jsonl_fpath=results/smoke_test.jsonl \
    +limit=10
```

:::

::::

---

## Common Operations Tasks

### Before Deployment

**Pre-deployment checklist**:

- [ ] Run unit tests: `pytest tests/unit_tests/`
- [ ] Test servers: `ng_test +entrypoint=...`
- [ ] Validate configuration: `ng_dump_config "+config_paths=[config.yaml]"`
- [ ] Profile with realistic workload
- [ ] Set up health check monitoring

### During Operation

**Ongoing monitoring**:

- [ ] Monitor health endpoints
- [ ] Check logs for errors
- [ ] Track resource usage (CPU, memory, GPU)
- [ ] Review profiling stats periodically

### Troubleshooting

**When issues occur**:

1. Check server health: `curl http://localhost:8000/health`
2. Review logs: `grep ERROR ng_gym.log`
3. Enable debug logging: `--log-level DEBUG`
4. Validate configuration: `ng_dump_config "+config_paths=[config.yaml]"`
5. Profile to identify bottlenecks

---

## Operations Best Practices

### Testing

**Comprehensive testing strategy**:

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Test Type
  - When to Run
  - Command
* - Unit Tests
  - Every code change
  - `pytest tests/unit_tests/`
* - Server Tests
  - After server changes
  - `ng_test +entrypoint=...`
* - Integration Tests
  - Before deployment
  - `ng_collect_rollouts ... +limit=10`
* - Functional Tests
  - Weekly, pre-release
  - `bash tests/functional_tests/run_tests.sh`
```

### Monitoring

**What to monitor**:

- **Health**: All server `/health` endpoints
- **Logs**: Error and warning patterns
- **Resources**: CPU, memory, GPU utilization
- **Performance**: Request latency, throughput

:::{tip}
Set up automated alerts for health check failures and error rate spikes to catch issues before they impact training.
:::

### Profiling

**When to profile**:

- Before production deployment
- After adding new verification logic
- When experiencing performance issues
- Periodically to catch regressions

**Profiling targets**:

- Resource servers (verify functions are critical path)
- High-concurrency scenarios (16k+ requests)
- End-to-end pipeline performance

### Debugging

**Systematic debugging approach**:

1. **Reproduce**: Create minimal reproduction case
2. **Isolate**: Test components individually
3. **Log**: Enable debug logging for detailed output
4. **Validate**: Check configuration with `ng_dump_config`
5. **Fix**: Apply fix and verify with tests

---

## Next Steps

Start with the operational task most relevant to your current need:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Set Up Monitoring
:link: monitoring
:link-type: doc

Configure health checks and log monitoring
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Run Tests
:link: testing
:link-type: doc

Validate your deployment with comprehensive tests
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Profile Performance
:link: profiling
:link-type: doc

Identify and optimize bottlenecks for production
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debug Issues
:link: debugging
:link-type: doc

Troubleshoot configuration and runtime problems
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

monitoring
testing
profiling
debugging

```

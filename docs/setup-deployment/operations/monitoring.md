(setup-operations-monitoring)=

# Monitoring

Monitor NeMo Gym training runs and server deployments through built-in progress tracking, log analysis, and system resource monitoring.

---

## Built-In Monitoring Features

NeMo Gym provides automatic monitoring during training data collection.

### Real-Time Progress Tracking

`ng_collect_rollouts` displays live progress with throughput metrics:

```bash
ng_collect_rollouts \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl

# Displays:
# Collecting rollouts: 45%|████▌     | 450/1000 [02:15<02:45, 3.33it/s]
```

**Key metric**: `it/s` (items per second) shows current throughput. Use this to identify bottlenecks during collection.

### Automatic Metric Aggregation

After collection completes, NeMo Gym displays aggregated metrics for all numeric fields from your resource server:

```json
{
  "reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1
}
```

:::{seealso}
For details on metric aggregation, refer to {ref}`concepts-rc-fundamentals`.
:::

---

## Log Monitoring

Monitor server activity and diagnose issues through log analysis.

### Enable Debug Logging

:::::{tab-set}

::::{tab-item} Standard Logging
```bash
ng_run "+config_paths=[config.yaml]"
```
::::

::::{tab-item} Debug Logging
```bash
ng_run "+config_paths=[config.yaml]" --log-level DEBUG
```
::::

::::{tab-item} Save to File
```bash
ng_run "+config_paths=[config.yaml]" > logs/ng_gym.log 2>&1
```
::::

:::::

### Analyze Logs

:::::{tab-set}

::::{tab-item} Search for Errors
```bash
# Find errors
grep ERROR ng_gym.log

# Find warnings
grep WARN ng_gym.log

# Follow in real-time
tail -f ng_gym.log | grep ERROR
```
::::

::::{tab-item} Common Patterns

```{list-table}
:header-rows: 1
:widths: 30 50 20

* - Log Pattern
  - Meaning
  - Action
* - `Server started on port 8000`
  - Server initialized
  - Normal operation
* - `Connection refused`
  - Cannot reach server
  - Verify server running
* - `API key invalid`
  - Authentication failure
  - Check `env.yaml`
* - `Timeout after 30s`
  - Request too slow
  - Check {doc}`profiling`
```

::::

::::{tab-item} Multi-Server Logs
```bash
# Timestamped log files
ng_run "+config_paths=[config.yaml]" \
    > logs/ng_gym_$(date +%Y%m%d_%H%M%S).log 2>&1

# Aggregate errors
cat logs/*.log | grep ERROR | sort
```
::::

:::::

:::{seealso}
For troubleshooting configuration issues, refer to {doc}`debugging`.
:::

---

## System Resource Monitoring

Track CPU, memory, network, and GPU resources to optimize deployments.

::::{dropdown} CPU and Memory Monitoring
:icon: cpu

```bash
# Real-time process monitoring
top

# Enhanced monitoring (install: brew install htop / apt install htop)
htop

# Monitor NeMo Gym processes
ps aux | grep ng_run
```

::::

::::{dropdown} Network Monitoring
:icon: broadcast

```bash
# Check ports in use
lsof -i :8000
lsof -i :8001-8010

# View active connections
netstat -an | grep 8000

# Monitor network traffic
tcpdump -i lo0 port 8000
```

::::

::::{dropdown} GPU Monitoring
:icon: device-desktop

For deployments using GPU inference (vLLM, NVIDIA NIM):

```bash
# Monitor GPU usage
nvidia-smi

# Continuous monitoring (refresh every 1s)
watch -n 1 nvidia-smi

# Monitor specific GPU
nvidia-smi -i 0
```

**Key metrics**:
- **GPU utilization**: Should be 90-95% during inference
- **Memory usage**: Track to avoid OOM errors
- **Temperature**: Monitor for thermal throttling

::::

---

## Server Availability Checks

Verify servers are running by testing their API endpoints.

### Check Individual Servers

NeMo Gym servers expose these endpoints:

:::::{tab-set}

::::{tab-item} Resource Server
```bash
# Test verification endpoint
curl -X POST http://localhost:8003/verify \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Endpoints**: `/seed_session`, `/verify`
::::

::::{tab-item} Agent Server
```bash
# Test agent endpoint
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Endpoints**: `/v1/responses`, `/run`
::::

::::{tab-item} Model Server
```bash
# Test model endpoint (OpenAI-compatible)
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Endpoints**: `/v1/chat/completions`, `/v1/responses`
::::

:::::

### Multi-Server Availability Script

Check all configured servers:

```bash
#!/bin/bash
# check_servers.sh

# Resource server
curl -sf http://localhost:8003/seed_session -X POST \
  -H "Content-Type: application/json" -d '{}' > /dev/null \
  && echo "✓ Resource server (8003): running" \
  || echo "✗ Resource server (8003): not responding"

# Agent server  
curl -sf http://localhost:8001/v1/responses -X POST \
  -H "Content-Type: application/json" -d '{}' > /dev/null \
  && echo "✓ Agent server (8001): running" \
  || echo "✗ Agent server (8001): not responding"

# Model server
curl -sf http://localhost:8002/v1/chat/completions -X POST \
  -H "Content-Type: application/json" -d '{}' > /dev/null \
  && echo "✓ Model server (8002): running" \
  || echo "✗ Model server (8002): not responding"
```

:::{tip}
For Kubernetes deployments, use these endpoint checks in liveness and readiness probes.
:::

---

## External Monitoring Integration

### vLLM Prometheus Metrics

If using vLLM as a model server, vLLM exposes Prometheus-compatible metrics:

```bash
# Check vLLM metrics endpoint
curl http://localhost:8000/metrics
```

**Available metrics**:
- Request throughput
- Batch sizes
- Queue lengths
- Token generation rates

:::{seealso}
Refer to {doc}`../../models/vllm/optimization` for vLLM-specific monitoring guidance.
:::

### Custom Monitoring Integration

For production deployments, integrate with your existing monitoring stack:

::::{dropdown} Example: Prometheus + Grafana
:icon: graph

**1. Expose custom metrics** from your resource server:

```python
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
response_time = Histogram('response_seconds', 'Response time')

async def verify(self, body: BaseVerifyRequest):
    request_count.inc()
    with response_time.time():
        # Your verification logic
        pass
```

**2. Configure Prometheus** to scrape vLLM and custom metrics

**3. Create Grafana dashboards** to visualize:
- Request throughput per server
- Response latencies (p50, p95, p99)
- Error rates
- GPU utilization

::::

::::{dropdown} Example: Alert Configuration
:icon: alert

```yaml
# Example alert rules (adapt to your monitoring system)
alerts:
  - name: HighErrorRate
    expr: rate(requests_failed[5m]) > 0.05
    action: notify_on_call
  
  - name: SlowVerification
    expr: histogram_quantile(0.95, response_seconds) > 5.0
    action: investigate
  
  - name: GPUMemoryHigh
    expr: gpu_memory_usage_percent > 90
    action: scale_up
```

::::

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debug Issues
:link: debugging
:link-type: doc

Troubleshoot configuration and connectivity problems
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Profile Performance
:link: profiling
:link-type: doc

Identify and optimize performance bottlenecks
:::

::::
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

### Logging Configuration

:::::{tab-set}

::::{tab-item} Standard Logging
```bash
ng_run "+config_paths=[config.yaml]"
```
::::

::::{tab-item} Save to File
```bash
ng_run "+config_paths=[config.yaml]" > logs/ng_gym.log 2>&1
```
::::

::::{tab-item} Filter 200 OK Messages
```bash
# Hide successful health check requests (default behavior)
ng_run "+config_paths=[config.yaml]"

# Show all requests including 200 OK (verbose)
ng_run "+config_paths=[config.yaml]" +uvicorn_logging_show_200_ok=true
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

:::{note}
NeMo Gym servers do not have dedicated health check endpoints. The examples below test functional endpoints to verify server availability. For simple port checking, use `nc -zv localhost <port>` or `lsof -i :<port>`.

**Port Assignment**: Servers use auto-assigned ports (varies per run). Find actual ports in `ng_run` startup output or use `lsof -i -P | grep LISTEN`.
:::

### Check Individual Servers

NeMo Gym servers expose these endpoints (replace `<PORT>` with actual port from startup output):

:::::{tab-set}

::::{tab-item} Resource Server
```bash
# Test verification endpoint (replace <PORT> with actual port)
curl -X POST http://localhost:<PORT>/verify \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Endpoints**: `/seed_session`, `/verify`
::::

::::{tab-item} Agent Server
```bash
# Test agent endpoint (replace <PORT> with actual port)
curl -X POST http://localhost:<PORT>/run \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Endpoints**: `/v1/responses`, `/run`
::::

::::{tab-item} Model Server
```bash
# Test model endpoint - OpenAI-compatible (replace <PORT> with actual port)
curl -X POST http://localhost:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Endpoints**: `/v1/chat/completions`, `/v1/responses`
::::

:::::

### Multi-Server Availability Script

Check all configured servers with simple port checks. First, extract actual ports from your startup output:

```bash
#!/bin/bash
# check_servers.sh
# Note: Update port numbers based on your ng_run startup output

# Example ports (your actual ports will differ)
RESOURCE_PORT=54321
AGENT_PORT=54322
MODEL_PORT=54323

# Resource server
nc -zv localhost $RESOURCE_PORT 2>&1 | grep -q succeeded \
  && echo "✓ Resource server ($RESOURCE_PORT): running" \
  || echo "✗ Resource server ($RESOURCE_PORT): not responding"

# Agent server  
nc -zv localhost $AGENT_PORT 2>&1 | grep -q succeeded \
  && echo "✓ Agent server ($AGENT_PORT): running" \
  || echo "✗ Agent server ($AGENT_PORT): not responding"

# Model server
nc -zv localhost $MODEL_PORT 2>&1 | grep -q succeeded \
  && echo "✓ Model server ($MODEL_PORT): running" \
  || echo "✗ Model server ($MODEL_PORT): not responding"
```

:::{tip}
**Finding ports**: Check `ng_run` output for startup messages like "Server running on port XXXXX", or use `lsof -i -P | grep LISTEN` to see all active ports.

For Kubernetes deployments, configure fixed ports in your YAML configs for predictable service discovery, then use these port checks in liveness probes.
:::

---

## External Monitoring Integration

### vLLM Server Prometheus Metrics

If using vLLM as your model server, the vLLM server itself exposes Prometheus-compatible metrics:

```bash
# Check vLLM server metrics endpoint
# Port 8000 is vLLM's default port (not NeMo Gym's)
curl http://localhost:8000/metrics
```

:::{note}
This `/metrics` endpoint is provided by the vLLM server itself, not by NeMo Gym. The port number will match your vLLM server configuration. Refer to vLLM documentation for the complete list of available metrics.
:::

**Available metrics from vLLM**:
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
# Install prometheus_client separately: pip install prometheus-client
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
response_time = Histogram('response_seconds', 'Response time')

async def verify(self, body: BaseVerifyRequest):
    request_count.inc()
    with response_time.time():
        # Your verification logic
        pass
```

:::{note}
The `prometheus-client` library is not included with NeMo Gym. Install it separately if you want to add custom metrics.
:::

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

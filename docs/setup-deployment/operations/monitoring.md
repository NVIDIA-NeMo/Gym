(setup-operations-monitoring)=

# Monitoring

Monitor NeMo Gym deployments to ensure reliable operation through health checks, log analysis, and resource tracking.

---

## Health Checks

Each server exposes a health endpoint for availability monitoring:

```bash
# Check server health
curl http://localhost:8000/health

# Check specific server
curl http://localhost:8001/health  # Agent
curl http://localhost:8002/health  # Policy model
curl http://localhost:8003/health  # Resource server
```

**Use health checks to**:
- Verify server is running and accepting requests
- Monitor uptime in production deployments
- Integrate with load balancers and orchestration tools
- Validate server startup in automation scripts

:::{tip}
Health checks return HTTP 200 status when the server is healthy. Use these endpoints in Kubernetes liveness probes, Docker healthchecks, or monitoring dashboards.
:::

---

## Log Monitoring

Monitor logs to track server activity and diagnose issues:

### Enable Logging

```bash
# Standard logging
ng_run "+config_paths=[config.yaml]"

# Debug-level logging
ng_run "+config_paths=[config.yaml]" --log-level DEBUG

# Save logs to file
ng_run "+config_paths=[config.yaml]" > ng_gym.log 2>&1
```

### Analyze Logs

**Search for errors**:

```bash
# Find errors in logs
grep ERROR ng_gym.log

# Find warnings
grep WARN ng_gym.log

# Follow logs in real-time
tail -f ng_gym.log | grep ERROR
```

**Common log patterns**:

```{list-table}
:header-rows: 1
:widths: 30 50 20

* - Log Pattern
  - Meaning
  - Action
* - `Server started on port 8000`
  - Server initialized successfully
  - Normal operation
* - `Connection refused`
  - Cannot reach dependent server
  - Verify server is running
* - `API key invalid`
  - Authentication failure
  - Check env.yaml credentials
* - `Timeout after 30s`
  - Request taking too long
  - Investigate performance
```

:::{seealso}
For debugging specific configuration issues, refer to {doc}`debugging`.
:::

---

## Resource Usage Monitoring

Track system resources to optimize deployment and prevent bottlenecks:

### CPU and Memory Monitoring

```bash
# Real-time process monitoring
top

# Enhanced interactive monitoring
htop

# Monitor specific process
ps aux | grep ng_run
```

### Network Monitoring

```bash
# Check which ports are in use
lsof -i :8000
lsof -i :8001-8010

# View active connections
netstat -an | grep 8000

# Monitor network traffic
tcpdump -i lo0 port 8000
```

### GPU Monitoring

For deployments using GPU inference (vLLM, NVIDIA NIM):

```bash
# Monitor GPU usage
nvidia-smi

# Continuous monitoring (refresh every 1s)
watch -n 1 nvidia-smi

# Monitor specific GPU
nvidia-smi -i 0
```

---

## Production Monitoring Patterns

### Multi-Server Health Monitoring

**Check all servers at once**:

```bash
# Create health check script
cat > check_health.sh << 'EOF'
#!/bin/bash
servers=(8000 8001 8002 8003)
for port in "${servers[@]}"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)
  if [ $status -eq 200 ]; then
    echo "✓ Port $port: healthy"
  else
    echo "✗ Port $port: unhealthy (status: $status)"
  fi
done
EOF

chmod +x check_health.sh
./check_health.sh
```

### Log Aggregation

**For multi-server deployments**:

```bash
# Run with separate log files per server
ng_run "+config_paths=[config.yaml]" \
    > logs/nemo_gym_$(date +%Y%m%d_%H%M%S).log 2>&1

# Aggregate and analyze
cat logs/*.log | grep ERROR | sort
```

### Metrics Collection

**Track request counts and response times**:

Most resource servers log metrics to help track performance:

```bash
# View metrics from logs
grep "Request processed" ng_gym.log | \
  awk '{print $NF}' | \
  datamash mean 1 median 1 max 1
```

:::{seealso}
For detailed performance analysis, refer to {doc}`profiling`.
:::

---

## Integration with External Tools

### Prometheus Metrics

Some NeMo Gym servers expose metrics endpoints compatible with Prometheus:

```bash
# Check if metrics endpoint exists
curl http://localhost:8000/metrics
```

### Grafana Dashboards

Create custom dashboards to visualize:
- Request throughput per server
- Response latencies
- Error rates
- Resource utilization

### Alert Configuration

**Example alert rules** (adapt to your monitoring system):

```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 0.05
    action: notify_on_call
  
  - name: SlowResponseTime
    condition: p95_latency > 5000ms
    action: investigate
  
  - name: ServerDown
    condition: health_check_failed
    action: restart_server
```

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


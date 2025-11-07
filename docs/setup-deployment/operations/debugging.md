(setup-operations-debugging)=

# Debugging

Systematically diagnose and resolve configuration, server, and runtime issues in NeMo Gym deployments.

:::{admonition} Target Audience
:class: tip
ML engineers and researchers debugging local development environments or distributed deployments.
:::

---

## Quick Diagnosis

Start here to identify the issue category:

::::{tab-set}

:::{tab-item} Configuration Issues
**Symptoms**: Missing variables, unresolved `${VAR}`, KeyError

**Quick check**:
```bash
ng_dump_config "+config_paths=[config.yaml]"
```

→ Jump to {ref}`debug-configuration`
:::

:::{tab-item} Server Connectivity
**Symptoms**: Connection refused, timeout, port conflicts

**Quick check**:
```bash
# Check if servers are running
ps aux | grep ng_run

# Check port availability
lsof -i :8000
```

→ Jump to {ref}`debug-connectivity`
:::

:::{tab-item} Ray/Distributed
**Symptoms**: Ray connection errors, worker failures

**Quick check**:
```bash
ray status
```

→ Jump to {ref}`debug-ray`
:::

:::{tab-item} Runtime Errors
**Symptoms**: Python exceptions, verification failures

**Quick check**:
```bash
# Check server output for errors
grep -i error logs/*.log
```

→ Jump to {ref}`debug-runtime`
:::

::::

---

(debug-configuration)=
## Configuration Debugging

### Inspect Resolved Configuration

Use `ng_dump_config` to view the final configuration after variable substitution:

::::{tab-set}

:::{tab-item} Full Config
```bash
ng_dump_config "+config_paths=[config.yaml]"
```
:::

:::{tab-item} Specific Section
```bash
ng_dump_config "+config_paths=[config.yaml]" | grep -A 10 policy_model
```
:::

:::{tab-item} Find Unresolved Variables
```bash
# Look for ${VAR} patterns that weren't substituted
ng_dump_config "+config_paths=[config.yaml]" | grep '\${'
```

If this returns results, those variables are missing from `env.yaml`.
:::

::::

### Common Configuration Problems

::::{tab-set}

:::{tab-item} Missing API Key
**Error**: `KeyError: 'policy_api_key'`

**Solution**:

1. Check current configuration:
   ```bash
   ng_dump_config "+config_paths=[config.yaml]" | grep api_key
   ```

2. Add to `env.yaml`:
   ```yaml
   policy_api_key: sk-your-actual-key
   ```

3. Verify substitution:
   ```bash
   ng_dump_config "+config_paths=[config.yaml]" | grep policy_api_key
   ```
:::

:::{tab-item} Unsubstituted Variable
**Symptom**: `${VAR_NAME}` appears in dumped config

**Solution**:

Ensure the variable exists in your environment file:

```yaml
# env.yaml
policy_api_key: sk-...
policy_model_name: gpt-4
```

Verify it's being loaded:
```bash
ng_dump_config "+config_paths=[config.yaml]" "+dotenv_path=env.yaml"
```
:::

:::{tab-item} Config File Not Found
**Error**: `FileNotFoundError: config.yaml`

**Solution**:

1. Check working directory:
   ```bash
   pwd
   ls -la config.yaml
   ```

2. Use absolute path:
   ```bash
   ng_run "+config_paths=[$(pwd)/config.yaml]"
   ```

3. Or verify relative path:
   ```bash
   # Config should be relative to where you run ng_run
   ng_run "+config_paths=[./configs/my_config.yaml]"
   ```
:::

::::

---

(debug-connectivity)=
## Server Connectivity Debugging

### Verify Servers Are Running

::::{tab-set}

:::{tab-item} Check Processes
```bash
# List all NeMo Gym server processes
ps aux | grep ng_run

# Check specific server by port
lsof -i :8000
```
:::

:::{tab-item} Test Server Endpoints
NeMo Gym servers do not have built-in health endpoints. To verify a server is running:

```bash
# Check if port is listening
nc -zv localhost 8000

# Or use telnet
telnet localhost 8000
```

For servers with profiling enabled, you can test the `/stats` endpoint:
```bash
# Only works if profiling_enabled=true
curl http://localhost:8000/stats
```
:::

:::{tab-item} View Server Output
```bash
# Run servers in foreground to see output
ng_run "+config_paths=[config.yaml]"

# Or capture to file
ng_run "+config_paths=[config.yaml]" 2>&1 | tee server.log
```
:::

::::

### Resolve Port Conflicts

::::{tab-set}

:::{tab-item} Identify Conflict
```bash
# Find process using the port
lsof -i :8000

# On Linux, also try
netstat -tulpn | grep 8000
```
:::

:::{tab-item} Change Port
```bash
# Use different default port
ng_run "+config_paths=[config.yaml]" +default_port=9000

# Or configure specific server ports
ng_run "+config_paths=[config.yaml]" \
    +head_server.port=8000 \
    +policy_model.port=8001 \
    +resource_server.port=8002
```
:::

:::{tab-item} Kill Conflicting Process
```bash
# Find PID from lsof output
lsof -i :8000

# Kill the process
kill <PID>

# Force kill if needed
kill -9 <PID>
```
:::

::::

---

(debug-ray)=
## Ray Distributed Debugging

### Check Ray Cluster Status

::::{tab-set}

:::{tab-item} Cluster Status
```bash
# Check if Ray is running and view cluster info
ray status

# Expected output shows nodes, CPUs, GPUs
```
:::

:::{tab-item} Ray Logs
```bash
# View Ray head node logs
cat /tmp/ray/session_latest/logs/raylet.out

# View most recent errors
grep -i error /tmp/ray/session_latest/logs/*.log
```
:::

:::{tab-item} Ray Dashboard
```bash
# Ray dashboard runs on port 8265 by default
# Open in browser: http://localhost:8265
```
:::

::::

### Common Ray Issues

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Problem
  - Diagnosis
  - Solution
* - Ray not started
  - `ray status` returns error
  - Run `ray start --head`
* - Wrong Ray address
  - Connection timeout
  - Verify `ray_address` in config or set `RAY_ADDRESS` env var
* - Worker out of memory
  - Check Ray dashboard
  - Reduce batch size or increase worker memory
* - Worker crashed
  - Check `ray status` for failed workers
  - Check Ray logs for exceptions
```

:::{seealso}
For distributed deployment patterns, refer to {doc}`../deployment/distributed-computing`.
:::

---

(debug-runtime)=
## Runtime Debugging

### Interactive Debugging with pdb

Add breakpoints directly in your resource server code:

```python
# In your resource server's verify method
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    # Add breakpoint
    import pdb; pdb.set_trace()
    
    # Your verification logic
    result = self._check_answer(body.question, body.answer)
    return BaseVerifyResponse(is_correct=result)
```

:::{dropdown} pdb Commands Reference
```
n (next)       - Execute current line, move to next
s (step)       - Step into function call
c (continue)   - Continue execution until next breakpoint
p <var>        - Print variable value
pp <var>       - Pretty-print variable
l (list)       - Show code context
w (where)      - Print stack trace
q (quit)       - Quit debugger
```
:::

### Test-Driven Debugging

Isolate issues in unit tests for faster iteration:

```python
# tests/test_resource_server.py
import pytest
from your_server import YourResourceServer

def test_specific_failing_case():
    """Reproduce the exact failure."""
    server = YourResourceServer()
    
    # Use the exact input that fails
    result = await server.verify(
        question="What is 2+2?",
        answer="5"  # Intentionally wrong to test error handling
    )
    
    assert result.is_correct == False
    assert "incorrect" in result.feedback.lower()
```

Run with pytest debugger:

::::{tab-set}

:::{tab-item} Drop to pdb on Failure
```bash
pytest --pdb tests/test_resource_server.py::test_specific_failing_case
```
:::

:::{tab-item} Verbose Output
```bash
pytest -vv -s tests/test_resource_server.py
```
:::

:::{tab-item} Run Single Test
```bash
pytest tests/test_resource_server.py::test_specific_failing_case
```
:::

::::

### Common Runtime Errors

::::{dropdown} ModuleNotFoundError: No module named 'nemo_gym'
**Cause**: NeMo Gym not installed or wrong Python environment

**Solution**:
```bash
# Check installation
pip list | grep nemo-gym

# Install in editable mode
pip install -e .

# Verify correct environment
which python
source .venv/bin/activate
```
::::

::::{dropdown} API Authentication Errors (401 Unauthorized)
**Cause**: Missing or invalid API key

**Solution**:
1. Check if key is in `env.yaml`:
   ```bash
   grep api_key env.yaml
   ```

2. Verify it's being loaded:
   ```bash
   ng_dump_config "+config_paths=[config.yaml]" | grep api_key
   ```

3. Ensure variable name matches:
   ```yaml
   # config.yaml
   api_key: ${OPENAI_API_KEY}
   
   # env.yaml
   OPENAI_API_KEY: sk-...
   ```
::::

::::{dropdown} Request Timeouts
**Cause**: Slow verification logic or external API delays

**Diagnosis**:
```bash
# Enable profiling to identify bottlenecks
ng_run "+config_paths=[config.yaml]" +profiling_enabled=true

# Query stats endpoint while servers are running
curl http://localhost:8000/stats
```

**Solutions**:
- Optimize verification logic (see {doc}`profiling`)
- Increase timeout in configuration
- Use async calls for external APIs
- Check resource usage: `htop` or `top`
::::

---

## Systematic Debugging Workflow

Follow this checklist for any issue:

::::{tab-set}

:::{tab-item} 1. Capture Information
```bash
# Capture full error output
ng_run "+config_paths=[config.yaml]" 2>&1 | tee debug.log

# Note the exact error message
grep -i error debug.log

# Check system resources
df -h  # Disk space
free -h  # Memory (Linux)
top  # CPU usage
```
:::

:::{tab-item} 2. Verify Configuration
```bash
# Dump and review configuration
ng_dump_config "+config_paths=[config.yaml]" > config_dump.yaml

# Check for unresolved variables
grep '\${' config_dump.yaml

# Verify file paths exist
# (check paths in config_dump.yaml)
```
:::

:::{tab-item} 3. Isolate Component
```bash
# Test resource server in isolation
ng_test +entrypoint=resources_servers/your_server

# Run with minimal configuration
ng_run "+config_paths=[minimal_config.yaml]"

# Test external dependencies separately
# (e.g., API endpoints, databases)
```
:::

:::{tab-item} 4. Document & Report
When reporting issues, include:

1. **Error message**: Complete stack trace
2. **Configuration**: Sanitized (remove API keys)
3. **Environment**:
   ```bash
   echo "OS: $(uname -a)"
   echo "Python: $(python --version)"
   pip show nemo-gym
   ```
4. **Steps to reproduce**: Exact commands
5. **Expected vs actual behavior**
:::

::::

---

## Logging Best Practices

### Capture Logs Effectively

::::{tab-set}

:::{tab-item} Standard Output
```bash
# Capture stdout and stderr
ng_run "+config_paths=[config.yaml]" 2>&1 | tee run.log
```
:::

:::{tab-item} Filter Noise
```bash
# NeMo Gym automatically filters 200 OK messages
# This is enabled by default via uvicorn_logging_show_200_ok=false

# To show all requests (noisy):
ng_run "+config_paths=[config.yaml]" +uvicorn_logging_show_200_ok=true
```
:::

:::{tab-item} Monitor in Real-Time
```bash
# Run in background
ng_run "+config_paths=[config.yaml]" > server.log 2>&1 &

# Tail and filter errors
tail -f server.log | grep -i error

# Or warnings
tail -f server.log | grep -i warn
```
:::

::::

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`meter;1.5em` Performance Profiling
:link: profiling
:link-type: doc

Profile slow verification logic
:::

:::{grid-item-card} {octicon}`pulse;1.5em` System Monitoring
:link: monitoring
:link-type: doc

Set up continuous monitoring
:::

:::{grid-item-card} {octicon}`beaker;1.5em` Testing Guide
:link: testing
:link-type: doc

Write tests to prevent regressions
:::

:::{grid-item-card} {octicon}`question;1.5em` Configuration Reference
:link: ../configuration/reference
:link-type: doc

Full configuration options
:::

::::


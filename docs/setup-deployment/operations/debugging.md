(setup-operations-debugging)=

# Debugging

Debug and troubleshoot NeMo Gym deployments to quickly resolve configuration, connectivity, and runtime issues.

---

## Enable Debug Logging

Increase logging verbosity to diagnose issues:

```bash
# Standard logging
ng_run "+config_paths=[config.yaml]"

# Debug-level logging
ng_run "+config_paths=[config.yaml]" --log-level DEBUG

# Save debug logs to file
ng_run "+config_paths=[config.yaml]" --log-level DEBUG > debug.log 2>&1
```

**Log levels**:

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Level
  - When to Use
  - Information Shown
* - `INFO`
  - Normal operation
  - Server startup, request counts, basic status
* - `DEBUG`
  - Troubleshooting
  - Configuration values, request details, function calls
* - `WARNING`
  - Production monitoring
  - Deprecations, potential issues, recoverable errors
* - `ERROR`
  - Critical issues
  - Failures, exceptions, unrecoverable errors
```

:::{tip}
Use `DEBUG` logging when troubleshooting, but return to `INFO` or `WARNING` for production to reduce log volume.
:::

---

## Configuration Debugging

Debug configuration issues using `ng_dump_config`:

### Validate Configuration

```bash
# Dump full resolved configuration
ng_dump_config "+config_paths=[config.yaml]"

# Check specific section
ng_dump_config "+config_paths=[config.yaml]" | grep policy_model

# Validate environment variable substitution
ng_dump_config "+config_paths=[config.yaml]" | grep '\${'
# If this shows results, those variables are missing from env.yaml
```

### Common Configuration Issues

**Issue**: `KeyError: 'policy_api_key'`

```bash
# Diagnose: Check if variable exists in configuration
ng_dump_config "+config_paths=[config.yaml]" | grep policy_api_key

# Solution: Add to env.yaml
echo "policy_api_key: sk-your-key" >> env.yaml
```

**Issue**: Variable not being substituted (`${VAR}` appears in config)

```bash
# Diagnose: Find unsubstituted variables
ng_dump_config "+config_paths=[config.yaml]" | grep '\${'

# Solution: Ensure variable is in env.yaml or dotenv file
ng_dump_config "+config_paths=[config.yaml]" "+dotenv_path=env.yaml"
```

**Issue**: Configuration file not found

```bash
# Diagnose: Check path and working directory
pwd
ls config.yaml

# Solution: Use absolute path or correct relative path
ng_run "+config_paths=[/absolute/path/to/config.yaml]"
```

:::{seealso}
For detailed configuration debugging techniques, refer to {doc}`../configuration/debugging`.
:::

---

## Server Connectivity Debugging

Debug server communication and network issues:

### Check Server Health

```bash
# Test if server is running
curl http://localhost:8000/health

# Test all servers
for port in 8000 8001 8002 8003; do
  echo "Testing port $port..."
  curl -s http://localhost:$port/health || echo "Port $port not responding"
done
```

### Identify Port Conflicts

```bash
# Check which process is using a port
lsof -i :8000

# Check all NeMo Gym server ports
lsof -i :8000-8010

# Network statistics
netstat -an | grep 8000
```

**Resolve port conflicts**:

```bash
# Option 1: Kill conflicting process
kill <PID>

# Option 2: Use different port
ng_run "+config_paths=[config.yaml]" +default_port=9000

# Option 3: Specify port per server
ng_run "+config_paths=[config.yaml]" \
    +head_server.port=8000 \
    +policy_model.port=8001
```

### Test Network Connectivity

```bash
# Test basic connectivity
ping localhost

# Test TCP connection
telnet localhost 8000

# Test HTTP endpoint
curl -v http://localhost:8000/health
```

**Firewall issues**:

```bash
# Check if firewall is blocking ports (macOS)
sudo pfctl -sr | grep 8000

# Check firewall status (Linux)
sudo ufw status
sudo iptables -L
```

---

## Common Issues and Solutions

### Connection Refused

**Symptoms**: `Connection refused` error when making requests

**Diagnose**:

```bash
# Check if server is running
ps aux | grep ng_run

# Check if port is listening
lsof -i :8000

# Check server logs for startup errors
grep ERROR ng_gym.log
```

**Solutions**:

1. **Server not started**: Run `ng_run "+config_paths=[config.yaml]"`
2. **Wrong port**: Verify port in configuration matches request
3. **Server crashed**: Check logs for errors, fix issue, restart

---

### API Key Invalid

**Symptoms**: `401 Unauthorized` or `API key invalid`

**Diagnose**:

```bash
# Check if API key is set
ng_dump_config "+config_paths=[config.yaml]" | grep api_key

# Verify env.yaml contains key
cat env.yaml | grep api_key
```

**Solutions**:

1. **Missing key**: Add to `env.yaml`:
   ```yaml
   policy_api_key: sk-your-actual-key
   ```

2. **Wrong variable name**: Ensure variable name in config matches env.yaml:
   ```yaml
   # config.yaml
   api_key: ${policy_api_key}
   
   # env.yaml
   policy_api_key: sk-...
   ```

3. **Invalid key**: Verify key is correct and has proper permissions

---

### Module Not Found

**Symptoms**: `ModuleNotFoundError: No module named 'nemo_gym'`

**Diagnose**:

```bash
# Check if NeMo Gym is installed
pip list | grep nemo-gym

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**Solutions**:

1. **Not installed**: Install in editable mode:
   ```bash
   pip install -e .
   ```

2. **Wrong environment**: Activate correct virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. **Wrong Python**: Verify using correct Python:
   ```bash
   which python
   python --version
   ```

---

### Server Timeout

**Symptoms**: Requests timeout after 30-60 seconds

**Diagnose**:

```bash
# Check if server is processing requests
curl http://localhost:8003/stats

# Monitor server logs in real-time
tail -f ng_gym.log | grep ERROR
```

**Solutions**:

1. **Slow verification**: Profile to identify bottlenecks (see {doc}`profiling`)
2. **External API timeout**: Increase timeout or use async calls
3. **Resource exhaustion**: Check CPU/memory usage with `top` or `htop`

---

### Ray Connection Issues

**Symptoms**: `ray.exceptions.RaySystemError: Failed to connect to Ray`

**Diagnose**:

```bash
# Check Ray status
ray status

# Check Ray logs
cat /tmp/ray/session_latest/logs/raylet.out
```

**Solutions**:

1. **Ray not started**: Start Ray cluster:
   ```bash
   ray start --head
   ```

2. **Wrong Ray address**: Verify Ray address in configuration:
   ```bash
   ng_dump_config "+config_paths=[config.yaml]" | grep ray_address
   ```

3. **Ray cluster shutdown**: Restart Ray and NeMo Gym servers

:::{seealso}
For Ray deployment patterns, refer to {doc}`../deployment/distributed-computing`.
:::

---

## Interactive Debugging

### Python Debugger (pdb)

**Add breakpoint in code**:

```python
# In resource server code
def verify(self, question, answer):
    import pdb; pdb.set_trace()  # Debugger will stop here
    # ... verification logic
```

**Run and debug**:

```bash
# Server will pause at breakpoint
ng_run "+config_paths=[config.yaml]"

# Use debugger commands:
# (Pdb) n        - next line
# (Pdb) s        - step into function
# (Pdb) c        - continue execution
# (Pdb) p var    - print variable
# (Pdb) l        - show code context
# (Pdb) q        - quit debugger
```

### Test-Driven Debugging

**Isolate issue in test**:

```python
# tests/test_app.py
def test_specific_failure():
    """Reproduce specific issue."""
    server = ResourceServer()
    
    # Exact input that causes issue
    result = server.verify(
        question="problematic input",
        answer="expected output"
    )
    
    assert result["is_correct"] == True  # Will fail, showing actual result
```

**Run test with debugger**:

```bash
# Drop into debugger on failure
pytest --pdb tests/test_app.py::test_specific_failure

# Run with verbose output
pytest -vv -s tests/test_app.py::test_specific_failure
```

---

## Debugging Checklist

When encountering an issue, follow this systematic approach:

### Step 1: Gather Information

- [ ] Read complete error message
- [ ] Check server logs for errors: `grep ERROR ng_gym.log`
- [ ] Verify server is running: `curl http://localhost:8000/health`
- [ ] Note exact command that triggered issue

### Step 2: Validate Configuration

- [ ] Dump configuration: `ng_dump_config "+config_paths=[config.yaml]"`
- [ ] Check for unresolved variables: `ng_dump_config ... | grep '\${'`
- [ ] Verify API keys are set: `ng_dump_config ... | grep api_key`

### Step 3: Test Components Individually

- [ ] Test server in isolation: `ng_test +entrypoint=...`
- [ ] Test health endpoints: `curl http://localhost:<port>/health`
- [ ] Run with debug logging: `--log-level DEBUG`

### Step 4: Search and Document

- [ ] Search error message in documentation
- [ ] Check GitHub issues for similar problems
- [ ] Document solution for future reference

---

## Getting Help

### Provide Debugging Information

When reporting issues, include:

1. **Error message** (complete stack trace)
2. **Configuration** (sanitized, no API keys)
3. **Environment**: OS, Python version, NeMo Gym version
4. **Steps to reproduce**
5. **Expected vs actual behavior**

**Example bug report**:

```
Issue: Connection refused when starting servers

Environment:
- OS: macOS 14.0
- Python: 3.10.12
- NeMo Gym: 0.1.0

Configuration:
<paste sanitized config>

Steps to reproduce:
1. ng_run "+config_paths=[config.yaml]"
2. curl http://localhost:8000/health

Error:
<paste error message>
```

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Monitor Health
:link: monitoring
:link-type: doc

Set up monitoring to catch issues early
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Validate with Tests
:link: testing
:link-type: doc

Run comprehensive tests to verify fixes
:::

::::


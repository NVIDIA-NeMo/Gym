(setup-operations-testing)=

# Testing

Test your NeMo Gym setup to ensure reliability and correctness across development, integration, and production environments.

---

## Server Testing with `ng_test`

Test individual servers (agents, models, or resource servers) to verify correctness:

### Basic Usage

```bash
# Test simple agent
ng_test +entrypoint=responses_api_agents/simple_agent

# Test resource server
ng_test +entrypoint=resources_servers/simple_weather

# Test with data validation
ng_test +entrypoint=resources_servers/multineedle +should_validate_data=true
```

### What `ng_test` Does

When you run `ng_test`, it performs these steps automatically:

1. **Environment Setup**: Creates isolated virtual environment for the server
2. **Dependency Installation**: Installs dependencies from `requirements.txt`
3. **Test Execution**: Runs pytest in the server directory
4. **Data Validation** (optional): Validates example data format and completeness

**Example output**:

```
Setting up test environment for simple_weather...
Installing dependencies...
Running tests...
=============================== test session starts ===============================
collected 3 items

tests/test_app.py::test_verify PASSED                                       [ 33%]
tests/test_app.py::test_health PASSED                                       [ 66%]
tests/test_app.py::test_invalid_input PASSED                                [100%]

================================ 3 passed in 0.54s ================================
✓ All tests passed
```

:::{important}
`ng_test` validates that your server can be deployed and tested in an isolated environment. It catches dependency issues, test failures, and data format problems before deployment.
:::

---

## Detailed Test Execution

After running `ng_test` once to set up the environment, you can run tests directly for faster iteration:

```bash
cd resources_servers/simple_weather
source .venv/bin/activate

# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test
pytest tests/test_app.py::test_verify

# Run with coverage report
pytest --cov=.
```

### Pytest Options

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Purpose
* - `-v` or `--verbose`
  - Show detailed test names and results
* - `-s`
  - Show print statements and logging output
* - `-k EXPRESSION`
  - Run tests matching expression (e.g., `-k test_verify`)
* - `--cov=.`
  - Generate coverage report
* - `--pdb`
  - Drop into debugger on failure
* - `-x`
  - Stop after first failure
* - `--lf`
  - Run only tests that failed last time
```

**Example**: Debug failing test

```bash
# Run specific test with output and debugger
pytest -s --pdb tests/test_app.py::test_verify
```

---

## Test Requirements for Resource Servers

Every resource server must include these artifacts to ensure quality and reproducibility:

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

### Minimum Test Implementation

**Basic test structure** (`tests/test_app.py`):

```python
import pytest
from app import ResourceServer

@pytest.fixture
def server():
    return ResourceServer()

def test_verify(server):
    """Test verification logic."""
    response = server.verify(
        question="What is 2+2?",
        answer="4"
    )
    assert response["is_correct"] == True

def test_health(server):
    """Test health endpoint."""
    response = server.health()
    assert response["status"] == "healthy"
```

:::{warning}
Test coverage is **NOT enforced by CI**. You are responsible for your server's correctness and functionality. Write comprehensive tests to catch issues before deployment.
:::

---

## Configuration Validation

Validate that configuration files load correctly before running workloads:

### Dump Configuration

```bash
# Validate and display full configuration
ng_dump_config "+config_paths=[config.yaml]"

# Check specific values
ng_dump_config "+config_paths=[config.yaml]" | grep api_key

# Validate with environment-specific settings
ng_dump_config "+config_paths=[config.yaml]" \
    "+dotenv_path=env.staging.yaml" | grep policy_model_name
```

### Common Validation Checks

**Verify required variables are set**:

```bash
# Check policy model configuration
ng_dump_config "+config_paths=[config.yaml]" | grep -E "base_url|api_key|model_name"

# Check resource server settings
ng_dump_config "+config_paths=[config.yaml]" | grep resources_servers
```

**Test variable substitution**:

```bash
# Ensure env.yaml variables are resolved
ng_dump_config "+config_paths=[config.yaml]" | grep -v '\${'
# If this shows any ${VAR} patterns, those variables are missing from env.yaml
```

:::{seealso}
For detailed configuration debugging, refer to {doc}`../configuration/debugging`.
:::

---

## Integration Testing

Test the complete pipeline with small-scale workloads before production deployment:

### End-to-End Test

```bash
# Test complete rollout collection pipeline
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/test_rollouts.jsonl \
    +limit=10
```

**What to verify**:
- All servers start correctly
- Requests flow through the system
- Rollouts are generated with correct format
- Output file contains expected results

### Multi-Server Integration Test

```bash
# Test with multiple resource servers
ng_run "+config_paths=[config1.yaml,config2.yaml]"

# In separate terminal, test both servers
curl http://localhost:8003/health
curl http://localhost:8004/health
```

### Test with Different Models

```bash
# Test with cheaper model for development
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/test_rollouts.jsonl \
    +policy_model_name=gpt-4o-mini \
    +limit=10
```

---

## Functional Testing

Run the complete NeMo Gym test suite to validate framework functionality:

### Unit Tests

Test individual components:

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific test module
pytest tests/unit_tests/test_config_types.py

# Run with coverage
pytest tests/unit_tests/ --cov=nemo_gym --cov-report=html
```

### Functional Tests

Test complete workflows:

```bash
# Run functional test suite
bash tests/functional_tests/run_tests.sh

# Run specific functional test
pytest tests/functional_tests/test_rollout_collection.py
```

---

## Testing Best Practices

### Development Testing Workflow

**Recommended sequence**:

1. **Unit Test**: Verify individual functions with `pytest`
2. **Server Test**: Validate server with `ng_test`
3. **Config Test**: Check configuration with `ng_dump_config`
4. **Integration Test**: Run small rollout collection (limit=10)
5. **Production Test**: Run full workload on staging environment

### Test Data Management

**Create minimal test datasets**:

```bash
# Extract 10 examples for testing
head -n 10 data/full_dataset.jsonl > data/test_dataset.jsonl

# Test with minimal data
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_dataset.jsonl \
    +output_jsonl_fpath=results/test_output.jsonl
```

### Continuous Integration

**Example CI pipeline**:

```bash
#!/bin/bash
# ci_test.sh

set -e  # Exit on error

# 1. Run unit tests
pytest tests/unit_tests/

# 2. Test resource servers
ng_test +entrypoint=resources_servers/simple_weather

# 3. Validate configuration
ng_dump_config "+config_paths=[config.yaml]"

# 4. Run integration test
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/ci_test.jsonl \
    +limit=5

echo "✓ All tests passed"
```

---

## Troubleshooting Test Failures

### Common Test Issues

**Issue**: `ImportError: No module named 'app'`

```bash
# Solution: Ensure NeMo Gym is installed in editable mode
pip install -e .
```

**Issue**: Test passes locally but fails in `ng_test`

```bash
# Solution: Missing dependency in requirements.txt
# Add missing packages to resources_servers/<server>/requirements.txt
```

**Issue**: `Connection refused` during integration test

```bash
# Solution: Servers not running or wrong port
# 1. Check server logs for startup errors
# 2. Verify ports in configuration
# 3. Check health endpoints: curl http://localhost:8000/health
```

### Debug Test Failures

```bash
# Run with verbose output and stop on first failure
pytest -v -x -s tests/test_app.py

# Drop into debugger on failure
pytest --pdb tests/test_app.py

# Show full diff for assertion failures
pytest -vv tests/test_app.py
```

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debug Issues
:link: debugging
:link-type: doc

Troubleshoot configuration and deployment problems
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Profile Performance
:link: profiling
:link-type: doc

Optimize performance for production workloads
:::

::::


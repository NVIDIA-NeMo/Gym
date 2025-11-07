(setup-operations-testing)=

# Testing

Validate your servers, configurations, and pipelines with NeMo Gym's testing tools.

---

## Quick Start

::::{tab-set}

:::{tab-item} Test a Server
```bash
# Test resource server
ng_test +entrypoint=resources_servers/simple_weather

# Test with data validation
ng_test +entrypoint=resources_servers/multineedle \
    +should_validate_data=true
```
:::

:::{tab-item} Validate Configuration
```bash
# Check configuration loads correctly
ng_dump_config "+config_paths=[config.yaml]"

# Verify specific values
ng_dump_config "+config_paths=[config.yaml]" | grep api_key
```
:::

:::{tab-item} Integration Test
```bash
# Test end-to-end pipeline with 10 samples
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/test_rollouts.jsonl \
    +limit=10
```
:::

::::

---

## Server Testing

### Understanding `ng_test`

`ng_test` validates servers in isolated environments:

1. Creates virtual environment (`.venv/`)
2. Installs dependencies from `requirements.txt`
3. Runs pytest tests
4. (Optional) Validates example data format

```{important}
`ng_test` catches dependency conflicts and test failures **before deployment**. Always run it after modifying server code.
```

### Required Test Artifacts

Every resource server needs these files:

```{list-table}
:header-rows: 1
:widths: 35 65

* - File
  - Purpose
* - `tests/test_app.py`
  - At least one test validating server logic
* - `data/example.jsonl`
  - 5 example inputs (first 5 from training data)
* - `data/example_metrics.json`
  - Output from `ng_prepare_data` validation
* - `data/example_rollouts.jsonl`
  - Output from `ng_collect_rollouts` on examples
```

:::{dropdown} Minimum test implementation
:icon: code

```python
import pytest
from app import YourResourceServer  # Replace with your actual server class name

@pytest.fixture
def server():
    # Replace YourResourceServer with your actual class
    # e.g., SimpleWeatherResourcesServer, MultiNeedleResourcesServer
    return YourResourceServer()

def test_verify(server):
    """Test verification logic."""
    response = server.verify(
        question="What is 2+2?",
        answer="4"
    )
    assert response["is_correct"] == True
```
:::

:::{warning}
Test **coverage thresholds** are **not enforced by CI**. You own correctness and test quality for your servers. CI runs tests but does not enforce minimum coverage percentages.
:::

### Direct pytest Usage

After `ng_test` sets up the environment, iterate faster with direct pytest:

```bash
cd resources_servers/simple_weather
source .venv/bin/activate
pytest
```

:::{dropdown} Common pytest options
:icon: terminal

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Run specific test
pytest tests/test_app.py::test_verify

# Stop on first failure
pytest -x

# Rerun last failures
pytest --lf

# Debug on failure
pytest --pdb
```
:::

---

## Configuration Validation

Validate configuration files before deployment:

```bash
# Display full resolved configuration
ng_dump_config "+config_paths=[config.yaml]"

# Check specific settings
ng_dump_config "+config_paths=[config.yaml]" | grep -E "base_url|api_key|model_name"

# Verify variable substitution (should show no ${VAR} patterns)
ng_dump_config "+config_paths=[config.yaml]" | grep -v '\${'
```

:::{seealso}
For configuration troubleshooting, refer to {doc}`debugging`.
:::

---

## Integration Testing

Validate end-to-end pipelines with small-scale workloads:

::::{tab-set}

:::{tab-item} Single Resource Server
```bash
# Test complete rollout collection
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/test_rollouts.jsonl \
    +limit=10
```

**Verify**: servers start, requests flow correctly, output format valid
:::

:::{tab-item} Multiple Resource Servers
```bash
# Start servers
ng_run "+config_paths=[config1.yaml,config2.yaml]"

# Check server availability (separate terminal)
lsof -i :8003 && echo "✓ Server 1 running"
lsof -i :8004 && echo "✓ Server 2 running"
```
:::

:::{tab-item} Different Models
```bash
# Test with cheaper/faster model
ng_collect_rollouts +agent_name=test_agent \
    +input_jsonl_fpath=data/test_input.jsonl \
    +output_jsonl_fpath=results/test_rollouts.jsonl \
    +policy_model_name=gpt-4o-mini \
    +limit=10
```
:::

::::

---

## Framework Testing

Test NeMo Gym framework components:

::::{tab-set}

:::{tab-item} Unit Tests
```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific module
pytest tests/unit_tests/test_global_config.py

# With coverage report
pytest tests/unit_tests/ --cov=nemo_gym --cov-report=html
```
:::

:::{tab-item} Functional Tests
```bash
# Run GPU-enabled functional test suite
bash tests/functional_tests/L2_Functional_Tests_GPU.sh
```
:::

::::

---

## Troubleshooting

:::{dropdown} ImportError: No module named 'app'
:icon: alert

**Cause**: NeMo Gym not installed in editable mode

**Solution**:
```bash
pip install -e .
```
:::

:::{dropdown} Test passes locally but fails in ng_test
:icon: alert

**Cause**: Missing dependency in `requirements.txt`

**Solution**: Add missing packages to `resources_servers/<server>/requirements.txt`
:::

:::{dropdown} Connection refused during integration test
:icon: alert

**Cause**: Servers not running or incorrect port configuration

**Solution**:
1. Check server logs for startup errors
2. Verify ports in configuration files
3. Check server availability: `lsof -i :8000` or `nc -zv localhost 8000`
:::

:::{dropdown} Debug failing tests
:icon: tools

```bash
# Verbose output, stop on first failure
pytest -v -x -s tests/test_app.py

# Drop into debugger on failure
pytest --pdb tests/test_app.py

# Show full diff for assertions
pytest -vv tests/test_app.py
```
:::

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


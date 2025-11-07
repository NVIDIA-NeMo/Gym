(setup-deployment-local)=

# Local Development Setup

Set up NeMo Gym for local experimentation and testing.

---

## Prerequisites

```{list-table}
:header-rows: 1
:widths: 20 50 30

* - Requirement
  - Description
  - Verification
* - Python 3.10+
  - Required for NeMo Gym
  - `python --version`
* - pip or uv
  - Package manager
  - `pip --version`
* - Git
  - For cloning repository
  - `git --version`
* - OpenAI API Key
  - For policy model (or use vLLM)
  - Account at platform.openai.com
```

---

## Quick Start

### Step 1: Install NeMo Gym

```bash
# Clone repository
git clone https://github.com/NVIDIA/NeMo-Gym.git
cd Gym

# Install with development dependencies
pip install -e ".[dev]"
```

:::{tip}
Use `pip install -e ".[dev]"` for editable install with dev tools (pytest, pre-commit, etc.)
:::

### Step 2: Configure Credentials

Create `env.yaml` with your API credentials:

```yaml
# env.yaml (automatically gitignored)
policy_api_key: sk-your-openai-api-key
policy_base_url: https://api.openai.com/v1
policy_model_name: gpt-4o-2024-08-06
```

:::{important}
The `env.yaml` file contains secrets and is automatically ignored by git. Never commit credentials.
:::

### Step 3: Run Test Configuration

```bash
# Use the simple agent example
ng_run "+config_paths=[responses_api_agents/simple_agent/config.yaml]"
```

**Expected output**:

```console
Starting Ray cluster...
Started Ray cluster at ray://127.0.0.1:6379
Starting Head Server on http://127.0.0.1:8000
Starting Simple Agent on http://127.0.0.1:8001
Starting Policy Model on http://127.0.0.1:8002
Starting Simple Resources Server on http://127.0.0.1:8003

All servers started successfully.
```

---

## Verify Installation

### Check Running Servers

```bash
# List running processes
ps aux | grep ng_run

# Test API endpoint
curl http://localhost:8001/health
```

Expected response:

```json
{"status": "healthy", "version": "0.1.0"}
```

### Run Example Task

```bash
# Execute simple task
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_turns": 1
  }'
```

---

## Common Issues

::::{dropdown} **Port Already in Use**

```console
ERROR: Port 8000 already in use
```

**Solution**: Stop existing processes or use different ports:

```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use custom ports
ng_run "+config_paths=[config.yaml]" +head_server.port=9000
```
::::

::::{dropdown} **Missing API Key**

```console
ERROR: policy_api_key not found in configuration
```

**Solution**: Ensure `env.yaml` exists with valid credentials:

```bash
# Check if env.yaml exists
ls -la env.yaml

# Validate format
cat env.yaml
```
::::

::::{dropdown} **Import Errors**

```console
ModuleNotFoundError: No module named 'nemo_gym'
```

**Solution**: Reinstall in editable mode:

```bash
pip install -e ".[dev]"
```
::::

---

## Development Workflow

### Project Structure

```
Gym/
├── nemo_gym/              # Core library
├── resources_servers/     # Task implementations
├── responses_api_agents/  # Agent implementations
├── responses_api_models/  # Model adapters
├── env.yaml               # Your credentials (gitignored)
└── outputs/               # Training data outputs
```

### Typical Development Cycle

1. **Modify Code**: Edit files in `nemo_gym/`, `resources_servers/`, etc.
2. **Run Tests**: `pytest tests/`
3. **Test Locally**: `ng_run "+config_paths=[your_config.yaml]"`
4. **Collect Data**: Check `outputs/` for training data
5. **Iterate**: Refine and repeat

### Creating Custom Resource Server

```bash
# Initialize new resource server
ng_init +entrypoint=resources_servers/my_task

# This creates:
# resources_servers/my_task/
# ├── app.py                      # Server implementation
# ├── configs/my_task.yaml        # Configuration
# └── data/                       # Datasets
#     ├── train.jsonl
#     ├── validation.jsonl
#     └── example.jsonl
```

:::{seealso}
Complete guide: {doc}`../../tutorials/custom-resource-server`
:::

---

## Configuration Options

### Minimal Configuration

```yaml
# Bare minimum for local testing
simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      resources_server:
        type: resources_servers
        name: my_resources_server
```

### Common Overrides

```bash
# Change ports
ng_run "+config_paths=[config.yaml]" +head_server.port=9000

# Use different model
ng_run "+config_paths=[config.yaml]" +policy_model_name=gpt-4o-mini

# Debug logging
ng_run "+config_paths=[config.yaml]" +log_level=DEBUG
```

:::{seealso}
Full configuration reference: {doc}`../configuration/reference`
:::

---

## IDE Setup

### VS Code

Recommended `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true
}
```

### PyCharm

1. **File → Settings → Project → Python Interpreter**
2. Select local virtualenv or create new
3. Mark `nemo_gym` as sources root
4. Configure pytest as test runner

---

## Next Steps

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item}
```{button-ref} ../configuration/index
:color: primary
:outline:
:expand:

Configure Your Setup
```
:::

:::{grid-item}
```{button-ref} vllm-integration
:color: primary
:outline:
:expand:

Use Local Models with vLLM
```
:::

:::{grid-item}
```{button-ref} ../../tutorials/index
:color: primary
:outline:
:expand:

Follow Tutorials
```
:::

::::

:::{seealso}
- **Debugging**: {doc}`../configuration/debugging`
- **Multi-Server Setup**: {doc}`../configuration/multi-server`
- **Examples**: Browse `resources_servers/` directory for working examples
:::


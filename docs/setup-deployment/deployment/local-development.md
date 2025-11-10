(setup-deployment-local)=

# Local Development

Configure your development environment for iterating on agents, resource servers, and models.

:::{note}
**Already completed the {doc}`../../get-started/index`?** This guide builds on that foundation with development workflows, IDE configuration, and advanced testing patterns.
:::

---

## Development Environment Setup

::::{tab-set}

:::{tab-item} VS Code
**Recommended `.vscode/settings.json`**:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true
}
```

**Workspace extensions**:
- Python (Microsoft)
- Pylance
- YAML (Red Hat)
:::

:::{tab-item} PyCharm
**Configure interpreter**:
1. **File → Settings → Project → Python Interpreter**
2. Select local virtualenv at `.venv/`
3. Mark `nemo_gym/` as sources root (right-click → Mark Directory As → Sources Root)

**Enable pytest**:
1. **File → Settings → Tools → Python Integrated Tools**
2. Set test runner to `pytest`
3. Configure test working directory to project root
:::

:::{tab-item} Neovim/Vim
**Install LSP support**:

```lua
-- Using nvim-lspconfig with pyright
require('lspconfig').pyright.setup{
  settings = {
    python = {
      pythonPath = ".venv/bin/python",
      analysis = {
        typeCheckingMode = "basic"
      }
    }
  }
}
```

**Test runner**: Use `vim-test` or `neotest-python` with pytest
:::

::::

---

## Development Workflows

### Iterative Testing Cycle

```{list-table}
:header-rows: 1
:widths: 10 40 30 20

* - Step
  - Action
  - Command
  - Expected Duration
* - 1
  - Modify code in `nemo_gym/`, `resources_servers/`, etc.
  - (edit files)
  - —
* - 2
  - Run unit tests for changed modules
  - `pytest tests/unit_tests/test_<module>.py`
  - < 30s
* - 3
  - Test locally with servers running
  - `ng_run "+config_paths=[config.yaml]"`
  - 5-10s startup
* - 4
  - Collect rollout data for validation
  - Check `outputs/<date>/`
  - —
* - 5
  - Run full test suite before committing
  - `pytest tests/`
  - 2-5 min
```

### Testing Individual Servers

::::{tab-set}

:::{tab-item} Resource Server
**Test a single resource server**:

```bash
# Test specific resource server
ng_test +entrypoint=resources_servers/simple_weather
```

Validates:

- ✅ Server starts without errors
- ✅ Endpoints respond correctly
- ✅ Verification logic produces expected results
:::

:::{tab-item} Agent
**Test agent workflows**:

```bash
# Test agent with minimal config
ng_run "+config_paths=[responses_api_agents/simple_agent/config.yaml]"

# Then in another terminal:
python responses_api_agents/simple_agent/client.py
```

Check for:

- ✅ Tool calls executed correctly
- ✅ Model responses formatted properly
- ✅ Rollout data saved to `outputs/`
:::

:::{tab-item} Model Server
**Test model integration**:

```bash
# Standalone model server test
ng_run "+config_paths=[responses_api_models/openai_model/configs/openai_model.yaml]"

# Verify with direct API call
curl -X POST http://localhost:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "test"}]}'
```

Validates:

- ✅ Authentication working
- ✅ Model accessible
- ✅ Response format correct
:::

::::

---

## Configuration Management

**Focus:** Quick overrides for iterative development. For complete configuration reference and production patterns, see {doc}`../configuration/index`.

### Environment-Specific Configs

```{list-table}
:header-rows: 1
:widths: 30 40 30

* - File
  - Purpose
  - Priority
* - `env.yaml`
  - Secrets and credentials (gitignored)
  - **High** (overrides everything)
* - `config.yaml`
  - Project-specific settings
  - Medium
* - CLI arguments
  - Runtime overrides
  - **Highest** (overrides all files)
```

**Example workflow**:

```bash
# Base configuration
cat > my_experiment.yaml << EOF
simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      max_turns: 10
      temperature: 0.7
EOF

# Override at runtime
ng_run "+config_paths=[my_experiment.yaml]" \
  +simple_agent.responses_api_agents.simple_agent.temperature=0.9 \
  +simple_agent.responses_api_agents.simple_agent.max_turns=5
```

:::{seealso}
{doc}`../configuration/reference` for complete configuration schema and hierarchy rules.
:::

### Common Development Overrides

::::{dropdown} Change Server Ports

```bash
# Avoid conflicts with other services
ng_run "+config_paths=[config.yaml]" \
  +head_server.port=9000
```

**When needed**: Port conflicts, running simultaneous instances.
::::

::::{dropdown} Use Different Models

```bash
# Quick model comparison
ng_run "+config_paths=[config.yaml]" \
  +policy_model_name=gpt-4o-mini  # Cheaper for iteration

ng_run "+config_paths=[config.yaml]" \
  +policy_model_name=gpt-4o  # Higher quality
```

**Tip**: Set defaults in `env.yaml`, override temporarily via CLI.
::::

::::{dropdown} Limit Data Collection

```bash
# Faster testing with fewer samples
ng_run "+config_paths=[config.yaml]" \
  +simple_agent.responses_api_agents.simple_agent.datasets.0.num_repeats=2
```

**Development mode**: Use `num_repeats=1` for speed. Production: 5-10+ for coverage.
::::

---

## Next Steps

### Build Custom Resource Servers

:::{seealso}
**Need domain-specific tools or verification?** See {doc}`../../tutorials/custom-resource-server` for a complete tutorial on building custom resource servers from scratch, including:
- Tool implementation patterns (APIs, databases, code execution)
- Verification strategies (binary, continuous, multi-metric)
- Dataset preparation and testing
- Deployment and troubleshooting
:::

---

## Debugging and Troubleshooting

### Inspecting Collected Data

```bash
# View collected training data
ng_viewer +jsonl_fpath=outputs/2025-11-07/train.jsonl
```

:::{note}
The `ng_viewer` command uses `+jsonl_fpath` as its parameter. This differs from `ng_collect_rollouts`, which uses `+input_jsonl_fpath` and `+output_jsonl_fpath`.
:::

**Opens interactive viewer** showing:

- Agent messages and tool calls
- Verification results and scores
- Aggregated metrics

### Configuration and Common Issues

:::{seealso}
**Troubleshooting configuration issues?** See {doc}`../configuration/debugging` for detailed debugging workflows including:
- Port conflicts and resolution
- Module import errors
- Configuration not loading
- API rate limits and quota errors
- Variable resolution and server references
- Pre-deployment validation checklists
:::

:::{seealso}
**Production monitoring and testing?** See {doc}`../operations/index` for operational concerns including:
- Health checks and monitoring
- Resource server validation with `ng_test`
- Performance profiling for production scale
- Integration and functional testing
:::


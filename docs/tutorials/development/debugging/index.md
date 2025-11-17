# Debugging

Learn how to debug NeMo Gym servers during development.

## Overview

NeMo Gym includes built-in debugging support that works with any IDE supporting the Python `debugpy` protocol. When debug mode is enabled, each server subprocess starts with a debugger listening on a specific port, allowing you to attach your IDE and set breakpoints.

## Configuration

Debug mode is controlled by two configuration options:

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `debug_mode` | bool | `false` | Enable debug mode for server subprocesses |
| `debug_port` | int | `5678` | Base port number. Each server gets sequential ports |

Enable debug mode via command-line:
```bash
ng_run debug_mode=true debug_port=5678
```

Or in your YAML config:
```yaml
debug_mode: true
debug_port: 5678
```

## How It Works

When `debug_mode=true`, the CLI automatically starts each server subprocess with:

```bash
python -m debugpy --listen 0.0.0.0:PORT app.py
```

Servers are assigned sequential debug ports:
- First server: 5678 (base port)
- Second server: 5679 (base + 1)
- Third server: 5680 (base + 2)
- And so on...

Your IDE can then attach to these ports to debug the running servers.

## IDE-Specific Guides

Choose your IDE for detailed setup instructions:

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Visual Studio Code
:link: vscode
:link-type: doc
Complete guide for debugging with VS Code, including launch configurations and troubleshooting.
+++
{bdg-secondary}`vscode` {bdg-secondary}`debugpy`
:::

::::

:::{note}
Other IDEs that support the `debugpy` protocol (PyCharm Professional, etc.) can also attach to the debug ports. The configuration shown here is IDE-agnostic; only the attachment UI differs between IDEs.
:::

## Prerequisites

Make sure `debugpy` is installed:

```bash
uv sync --extra dev
```

This is included in the dev dependencies and is the only requirement for debugging.

```{toctree}
:hidden:
:maxdepth: 1

vscode
```

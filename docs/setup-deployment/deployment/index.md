# Deployment Scenarios

(setup-deployment-scenarios)=

Deploy NeMo Gym in different environments based on your use case.

---

## Choose Your Deployment Scenario

Select the deployment approach that matches your needs:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Development
:link: local-development
:link-type: doc

Quick setup for local experimentation and testing.
+++
{bdg-primary}`Prototyping` {bdg-secondary}`Debugging` {bdg-secondary}`Data Collection`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` vLLM Integration
:link: vllm-integration
:link-type: doc

Connect NeMo Gym to vLLM-hosted models.
+++
{bdg-primary}`Open Source Models` {bdg-secondary}`Local Inference`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Distributed Computing
:link: distributed-computing
:link-type: doc

Scale with Ray clusters for high-throughput workloads.
+++
{bdg-primary}`Multi-Node Training` {bdg-secondary}`High Throughput`
:::

::::

---

## Component Architecture

Understanding what gets deployed:

```{list-table}
:header-rows: 1
:widths: 20 25 35 20

* - Component
  - Purpose
  - Network Role
  - Default Port
* - Head Server
  - Configuration distribution
  - Internal coordinator
  - 8000
* - Responses API Agent
  - Orchestrates interactions
  - External API endpoint
  - 8001
* - Policy Model
  - Generates responses
  - Internal service
  - 8002
* - Resources Server
  - Domain verification
  - Internal service
  - 8003
```

:::{tip}
All components run as **separate processes** that communicate over HTTP. You can run them on the same machine or distribute them across several servers.
:::

---

## Network Configuration

::::{tab-set}

:::{tab-item} Default (Local)

**Localhost-only access** (development and testing):

```yaml
# Default configuration
default_host: "127.0.0.1"  # Localhost only
default_port: 8000         # Starting port, auto-increments
```

Each server gets the next available port: 8000, 8001, 8002, 8003, and so on.

**Use when**:

- Local development
- Single-machine testing
- No external network access needed

:::

:::{tab-item} External Access (Production)

**Network-accessible servers** (production deployment):

Via CLI:

```bash
ng_run "+config_paths=[config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

Via configuration file:

```yaml
# production_config.yaml
default_host: 0.0.0.0
head_server:
  port: 8000
```

```{warning}
Binding to `0.0.0.0` exposes servers to network. Ensure:

- Configure firewall rules
- Secure API keys properly
- Use TLS/SSL for production traffic
```

**Use when**:

- Remote server deployment
- Container orchestration
- Multi-machine setups
- External API access required

:::

::::

```{toctree}
:hidden:
:maxdepth: 2

local-development
vllm-integration
distributed-computing

```

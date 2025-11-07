# Deployment Scenarios

(setup-deployment-scenarios)=

Deploy NeMo Gym in different environments based on your use case.

---

## Choose Your Deployment Scenario

Select the deployment approach that matches your needs:

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`device-desktop` Local Development
:link: local-development
:link-type: doc

Quick setup for local experimentation and testing.

**Best for**: Initial exploration, debugging, prototyping, data collection
:::

:::{grid-item-card} {octicon}`server` vLLM Integration
:link: vllm-integration
:link-type: doc

Connect NeMo Gym to vLLM-hosted models.

**Best for**: Using models without native Responses API support
:::

:::{grid-item-card} {octicon}`workflow` Distributed Computing
:link: distributed-computing
:link-type: doc

Scale with Ray clusters for high-throughput workloads.

**Best for**: Multi-node training, large-scale rollout collection
:::

::::

---

## Quick Start: Common Deployment Patterns

### Single-Node Development

Most common starting point:

```bash
# Install and run locally
pip install -e ".[dev]"
ng_run "+config_paths=[responses_api_agents/simple_agent/config.yaml]"
```

:::{seealso}
Complete setup instructions: {doc}`local-development`
:::

### Production Deployment

For remote servers or containers, you control:

- **Network configuration**: Set host/port for external access
- **Credentials**: Use `env.yaml` for API keys and secrets
- **Resource allocation**: Configure server resources per component

::::{tab-set}

:::{tab-item} Remote Server

```bash
# Production configuration
ng_run "+config_paths=[production_config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

**Key considerations**:

- Open required ports (default: 8000-8003)
- Manage API keys securely
- Ensure network accessibility between components
:::

:::{tab-item} Docker Container

```dockerfile
FROM python:3.10-slim
WORKDIR /app

# Install NeMo Gym
COPY . .
RUN pip install --no-cache-dir -e .

# Runtime configuration
ENV PYTHONUNBUFFERED=1
EXPOSE 8000-8003

CMD ["ng_run", "+config_paths=[config.yaml]"]
```

**Key considerations**:

- Map configuration files as volumes
- Expose port range for all servers
- Handle secrets via environment variables or mounted files
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

### Default Behavior

```yaml
# Defaults in config
default_host: "127.0.0.1"  # Localhost only
default_port: 8000         # Starting port, auto-increments
```

Each server gets the next available port: 8000, 8001, 8002, 8003, and so on.

### External Access

Override for production:

```bash
ng_run "+config_paths=[config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

Or in config:

```yaml
# production_config.yaml
default_host: 0.0.0.0
head_server:
  port: 8000
```

:::{warning}
Binding to `0.0.0.0` exposes servers to network. Ensure:

- Configure firewall rules
- Secure API keys properly
- Use TLS/SSL for production traffic
:::

---

## Credentials and Secrets

### Development: `env.yaml`

```yaml
# env.yaml (automatically gitignored)
policy_api_key: sk-your-openai-key
policy_base_url: https://api.openai.com/v1
judge_api_key: sk-your-judge-key
```

### Production Options

::::{tab-set}

:::{tab-item} Environment Variables

```bash
export POLICY_API_KEY=sk-prod-key
ng_run "+config_paths=[config.yaml]"
```

:::

:::{tab-item} Mounted Configuration

```bash
# Docker with mounted secrets
docker run -v /secure/env.yaml:/app/env.yaml nemo-gym
```

:::

:::{tab-item} Secret Management

```python
# Integration with secret managers
import boto3

secrets = boto3.client('secretsmanager')
api_key = secrets.get_secret_value(SecretId='nemo-gym-api-key')
```

:::

::::

---

## Next Steps

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item}

```{button-ref} local-development
:color: primary
:outline:
:expand:

Get Started Locally
```

:::

:::{grid-item}

```{button-ref} vllm-integration
:color: primary
:outline:
:expand:

Connect to vLLM
```

:::

:::{grid-item}

```{button-ref} distributed-computing
:color: primary
:outline:
:expand:

Scale with Ray
```

:::

::::

:::{seealso}
**Configuration Reference**: {doc}`../configuration/index`  
**Troubleshooting**: {doc}`../configuration/debugging`  
**Multi-Server Setup**: {doc}`../configuration/multi-server`
:::

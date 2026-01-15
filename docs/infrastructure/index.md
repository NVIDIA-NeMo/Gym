(infrastructure-index)=
# Infrastructure

Learn how to deploy and scale NeMo Gym for production workloads.

## Overview

NeMo Gym scales from single-machine development to multi-node production:

| Scale | Use Case | Components |
|-------|----------|------------|
| **Development** | Local testing | Single process, all servers |
| **Single-node** | Small-scale training | Multiple processes, one machine |
| **Multi-node** | Production training | Distributed across cluster |

## Deployment Patterns

### Local Development

```bash
# All servers on one machine
ng_run "+config_paths=[config.yaml]"
```

### Production Deployment

```bash
# Head server on coordinator node
ng_run --head-only

# Workers on GPU nodes
ng_run --worker --head-url=http://coordinator:11000
```

## Guides

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Topology
:link: deployment-topology
:link-type: doc
Production deployment patterns.
+++
{bdg-secondary}`deployment` {bdg-secondary}`topology`
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` Distributed Computing with Ray
:link: ray-distributed
:link-type: doc
Scale with Ray clusters.
+++
{bdg-secondary}`ray` {bdg-secondary}`distributed`
:::

::::

## Key Concepts

### Head Server

The head server (port 11000) provides:
- Service discovery for all components
- Configuration distribution
- Request routing

### Server Registration

Servers register with the head server on startup, enabling name-based routing:

```python
# Servers can call each other by name
await self.server_client.post(
    server_name="my_resources",
    url_path="/verify",
    json=payload
)
```

## Resource Requirements

### CPU-Only (NeMo Gym)

- 8GB+ RAM
- Standard x86_64 or ARM64 CPU
- Network connectivity

### GPU (Model Inference)

- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+
- GPU driver 450+

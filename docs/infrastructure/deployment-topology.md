(infra-deployment-topology)=
# Deployment Topology

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #293](https://github.com/NVIDIA-NeMo/Gym/issues/293) for details.
```

Understand how to deploy NeMo Gym components for different scales and use cases.

---

## Topology Patterns

NeMo Gym supports three deployment patterns, each suited to different stages of development and scale.

### Pattern 1: All-in-One (Development)

All servers run in a single process—ideal for local development and debugging:

```
┌─────────────────────────────────┐
│         Single Machine          │
│                                 │
│  ┌─────────────────────────┐   │
│  │      Head Server        │   │
│  │   ├── Model Server      │   │
│  │   ├── Resources Server  │   │
│  │   └── Agent Server      │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
```

**Best for:** Local development, debugging, quick iteration

### Pattern 2: Separated Services (Testing)

Services run as separate processes on a single machine:

```
┌─────────────────────────────────┐
│         Single Machine          │
│                                 │
│  ┌───────────┐ ┌────────────┐  │
│  │   Head    │ │   Model    │  │
│  │  Server   │ │   Server   │  │
│  └───────────┘ └────────────┘  │
│  ┌───────────┐ ┌────────────┐  │
│  │ Resources │ │   Agent    │  │
│  │  Server   │ │   Server   │  │
│  └───────────┘ └────────────┘  │
└─────────────────────────────────┘
```

**Best for:** Integration testing, validating service communication, Docker Compose setups

### Pattern 3: Distributed (Production)

Services distributed across multiple nodes for horizontal scaling:

```
┌─────────────┐     ┌─────────────┐
│ Coordinator │     │  GPU Node   │
│             │     │             │
│ ┌─────────┐ │     │ ┌─────────┐ │
│ │  Head   │ │────▶│ │ Model   │ │
│ │ Server  │ │     │ │ Server  │ │
│ └─────────┘ │     │ └─────────┘ │
└─────────────┘     └─────────────┘
        │
        │           ┌─────────────┐
        │           │ CPU Node    │
        └──────────▶│             │
                    │ ┌─────────┐ │
                    │ │Resources│ │
                    │ │ Server  │ │
                    │ └─────────┘ │
                    └─────────────┘
```

**Best for:** Production workloads, large-scale rollout collection, multi-GPU training

---

## Choosing a Pattern

Use this decision matrix to select the right deployment pattern:

| Factor | All-in-One | Separated | Distributed |
|--------|------------|-----------|-------------|
| **Team size** | Individual | Small team | Production team |
| **Data volume** | Small datasets | Medium datasets | Large datasets |
| **GPU requirements** | Single GPU or CPU | Single machine, multiple GPUs | Multi-node GPU cluster |
| **Fault tolerance** | None | Process isolation | Full isolation |
| **Setup complexity** | Minimal | Moderate | Higher |

### When to Scale

**Scale from All-in-One to Separated when:**
- You need to restart individual services without restarting everything
- Multiple team members are developing different components
- You want to simulate production networking locally

**Scale from Separated to Distributed when:**
- Single machine cannot handle rollout throughput
- GPU memory on one machine is insufficient
- You need fault isolation between coordinator and workers

---

## Configuration

### Single-Node

```bash
# All-in-one: everything in one process
ng_run "+config_paths=[config.yaml]"
```

### Multi-Node

```bash
# Start head server on coordinator node
ng_run --head-only

# Start workers on GPU nodes (run on each worker machine)
ng_run --worker --head-url=http://coordinator:11000
```

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HEAD_SERVER_URL` | — | URL of the head server (required for workers) |
| `NUM_EXPECTED_WORKERS` | `1` | Workers the head waits for before starting |
| `WORKER_HEARTBEAT_INTERVAL` | `30` | Seconds between worker health checks |

---

## Networking

### Service Discovery

The head server (port 11000) acts as a service registry. Servers register on startup, enabling name-based routing:

```python
# Servers call each other by name, not IP
await self.server_client.post(
    server_name="my_resources",
    url_path="/verify",
    json=payload
)
```

### Port Allocation

| Server | Default Port | Notes |
|--------|--------------|-------|
| Head server | 11000 | Must be accessible to all workers |
| Model server | 8000 | Per-worker |
| Resources server | 8080 | Per-worker |

### Network Requirements

| Requirement | Recommendation |
|-------------|----------------|
| **Latency** | < 10ms between head and workers for optimal throughput |
| **Bandwidth** | 1 Gbps minimum; 10 Gbps recommended for large model weights |
| **Firewall** | Allow TCP traffic on ports 11000, 8000, 8080 between nodes |

---

## Resource Sizing

### Per-Pattern Recommendations

| Component | All-in-One | Separated | Distributed |
|-----------|------------|-----------|-------------|
| **Coordinator CPU** | 4+ cores | 4+ cores | 8+ cores |
| **Coordinator RAM** | 16 GB | 16 GB | 32 GB |
| **Worker CPU** | — | — | 4+ cores per worker |
| **Worker GPU** | 1× 16GB+ | 1-8× 16GB+ | 1+ per worker |

### Capacity Planning

Estimate throughput based on your rollout complexity:

| Rollout Type | Single Worker | 4 Workers | 8 Workers |
|--------------|---------------|-----------|-----------|
| Simple (1-step) | ~100/min | ~400/min | ~800/min |
| Multi-step (5 steps) | ~20/min | ~80/min | ~160/min |
| Multi-turn (10 turns) | ~10/min | ~40/min | ~80/min |

```{note}
Actual throughput depends on model size, GPU type, and environment complexity.
```

---

## Migration Paths

### Development → Testing

1. Extract configuration into separate YAML files per server
2. Start servers in separate terminals/processes
3. Verify inter-service communication via logs

### Testing → Production

1. Containerize each server type (see {doc}`/resources-server/containerize`)
2. Deploy head server on coordinator node
3. Deploy workers with GPU resources
4. Configure networking and service discovery

### Scaling Existing Deployments

```bash
# Add workers to running cluster
ng_run --worker --head-url=http://coordinator:11000

# Workers auto-register; head distributes work automatically
```

---

## Monitoring

### Health Endpoints

All servers expose health endpoints:

```bash
# Check head server health
curl http://coordinator:11000/health

# Check worker health
curl http://worker-1:8080/health
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Worker registration | Number of active workers | < expected count |
| Rollout throughput | Rollouts completed per minute | < baseline |
| GPU utilization | Per-worker GPU usage | < 50% (underutilized) |
| Request latency | Head → worker round-trip | > 100ms |

---

## Security Considerations

### Network Security

- **Internal networks only**: NeMo Gym servers are designed for trusted networks
- **No built-in auth**: Use network-level security (VPC, firewall rules)
- **TLS**: Configure reverse proxy (nginx, Traefik) for TLS termination

### Production Checklist

- [ ] Servers not exposed to public internet
- [ ] Firewall rules restrict access to known IPs
- [ ] Monitoring alerts configured for anomalies
- [ ] Logs shipped to centralized logging system

---

## Related Guides

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Multi-Node Docker
:link: /environment-tutorials/multi-node-docker
:link-type: doc
Deploy with Docker Compose across multiple containers.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` Distributed Computing with Ray
:link: ray-distributed
:link-type: doc
Scale rollout collection with Ray clusters.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Containerize Resources Servers
:link: /resources-server/containerize
:link-type: doc
Package custom servers for deployment.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Architecture Overview
:link: /about/architecture
:link-type: doc
Understand the server-based architecture.
:::

::::

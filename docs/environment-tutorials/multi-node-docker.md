(env-multi-node-docker)=
# Multi-Node Docker Instances

```{warning}
This article was generated and has not been reviewed. Content may change.
```

Deploy NeMo Gym environments across multiple Docker instances for scalable rollout collection.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
30-60 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`/get-started/detailed-setup`
- Docker and Docker Compose installed
- Multiple machines or VMs (for true multi-node)

:::

::::

---

## Overview

Multi-node deployment distributes environment execution across multiple Docker containers, enabling:

- **Horizontal scaling** — Add workers to increase throughput
- **GPU distribution** — Spread GPU workloads across machines
- **Fault isolation** — Worker failures don't crash the head node

This guide covers Docker Compose-based deployment. Kubernetes orchestration documentation is coming soon.

---

## When to Use Multi-Node

Use multi-node deployment when:

- Single machine cannot handle rollout throughput
- Environment requires GPU resources across nodes
- Need to scale horizontally for large datasets
- Want isolation between head coordination and worker execution

---

## Architecture

```text
                    ┌─────────────────┐
                    │   Head Server   │
                    │   (port 11000)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
│   Worker 1    │   │   Worker 2    │   │   Worker 3    │
│ - Model Svc   │   │ - Model Svc   │   │ - Model Svc   │
│ - Resources   │   │ - Resources   │   │ - Resources   │
└───────────────┘   └───────────────┘   └───────────────┘
```

| Component | Role |
|-----------|------|
| **Head Server** | Coordinates work distribution, aggregates results |
| **Workers** | Execute rollouts, run model inference and resources servers |

---

## Quick Start

### 1. Create Docker Compose Configuration

```yaml
# docker-compose.yaml
version: '3.8'

services:
  head:
    image: nemo-gym:latest
    ports:
      - "11000:11000"
    command: ng_run --head-only
    environment:
      - NUM_EXPECTED_WORKERS=2

  worker-1:
    image: nemo-gym:latest
    environment:
      - HEAD_SERVER_URL=http://head:11000
    depends_on:
      - head
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker-2:
    image: nemo-gym:latest
    environment:
      - HEAD_SERVER_URL=http://head:11000
    depends_on:
      - head
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Start the Cluster

```bash
docker compose up -d
```

### 3. Verify Workers Registered

```bash
# Check head server logs
docker compose logs head | grep "worker registered"

# Expected output:
# worker-1 registered successfully
# worker-2 registered successfully
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HEAD_SERVER_URL` | — | URL of the head server (required for workers) |
| `NUM_EXPECTED_WORKERS` | `1` | Number of workers the head waits for before starting |
| `WORKER_HEARTBEAT_INTERVAL` | `30` | Seconds between worker health checks |
| `WORKER_TIMEOUT` | `300` | Seconds before marking a worker as unhealthy |

### Network Configuration

Workers discover the head server via the `HEAD_SERVER_URL` environment variable. Docker Compose's default bridge network handles DNS resolution:

```yaml
services:
  head:
    networks:
      - gym-network

  worker-1:
    environment:
      - HEAD_SERVER_URL=http://head:11000  # 'head' resolves via Docker DNS
    networks:
      - gym-network

networks:
  gym-network:
    driver: bridge
```

**Port requirements:**

| Port | Purpose |
|------|---------|
| `11000` | Head server coordination API |
| `8080` | Resources server (per worker) |
| `8000` | Model server (per worker) |

---

## Scaling & Resources

### Adding Workers

Scale workers dynamically:

```bash
# Scale to 5 workers
docker compose up -d --scale worker=5
```

Or define additional workers in your compose file for heterogeneous configurations.

### GPU Allocation

Assign specific GPUs to workers:

```yaml
worker-1:
  environment:
    - CUDA_VISIBLE_DEVICES=0
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']
            capabilities: [gpu]

worker-2:
  environment:
    - CUDA_VISIBLE_DEVICES=1
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['1']
            capabilities: [gpu]
```

### Load Balancing

The head server distributes work using round-robin by default. Workers pull tasks when ready, ensuring even distribution regardless of individual task duration.

---

## Monitoring

### Health Checks

Add health checks to detect failures:

```yaml
services:
  worker-1:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific worker
docker compose logs -f worker-1

# Head server only
docker compose logs -f head
```

---

## Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Workers not registering | Network isolation | Verify all services on same Docker network |
| `HEAD_SERVER_URL` connection refused | Head not ready | Add `depends_on` with health check condition |
| GPU not visible in worker | Missing device reservation | Add `deploy.resources.reservations.devices` |
| Uneven work distribution | Workers at different speeds | Check GPU memory, reduce batch size on slow workers |
| Worker marked unhealthy | Task timeout | Increase `WORKER_TIMEOUT` or optimize task |

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Connect your distributed environment to training.
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containerize Resources Servers
:link: /resources-server/containerize
:link-type: doc
Package custom servers for deployment.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Creating Training Environments
:link: creating-training-environment
:link-type: doc
Design effective environments for RL.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Multi-Step Environments
:link: multi-step
:link-type: doc
Build sequential tool-calling workflows.
:::

::::

(resources-server-containerize)=
# Containerize Resources Servers

Package your resources server as a Docker container for portable, reproducible deployments.

**Goal**: Deploy a resources server as a standalone Docker container that connects to a NeMo Gym head server.

**Prerequisites**:
- Docker installed ([Docker installation guide](https://docs.docker.com/get-docker/))
- A working resources server (see {doc}`/tutorials/creating-resource-server`)
- Network access to a NeMo Gym head server (for multi-node deployments)

---

## When to Containerize

Containerize your resources server when you need:

- **Consistent deployment** across development, staging, and production
- **Isolated dependencies** that won't conflict with host system
- **Orchestrator deployment** via Kubernetes, Docker Swarm, or similar
- **GPU access** in production environments

For local development, running directly with `ng_run` (NeMo Gym's CLI command) is simpler. See {doc}`/environment-tutorials/multi-node-docker` for multi-node deployment patterns.

---

## Dockerfile Template

Create a `Dockerfile` in your resources server directory. Choose the CPU or GPU variant:

:::::{tab-set}

::::{tab-item} CPU (Standard)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install NeMo Gym
RUN pip install --no-cache-dir nemo-gym

# Copy server code
COPY . .

EXPOSE 8080

# Server reads configuration from NEMO_GYM_CONFIG_DICT environment variable
CMD ["python", "app.py"]
```

::::

::::{tab-item} GPU (CUDA)

For GPU-enabled resources servers (e.g., LLM-as-judge verifiers):

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python from deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

# Install NeMo Gym
RUN python3 -m pip install --no-cache-dir nemo-gym

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]
```

```{note}
The NVIDIA Docker runtime (`--gpus all`) requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host.
```

::::

:::::

**Key points**:
- Install `nemo-gym` from PyPI, not from local editable install
- The server reads configuration from the `NEMO_GYM_CONFIG_DICT` environment variable

---

## Building the Container

```bash
cd resources_servers/my_server
docker build -t my-resources-server:latest .
```

---

## Running the Container

Pass configuration via the `NEMO_GYM_CONFIG_DICT` environment variable:

:::::{tab-set}

::::{tab-item} CPU

```bash
docker run -p 8080:8080 \
    -e NEMO_GYM_CONFIG_DICT='
head_server:
  host: "192.168.1.100"
  port: 11000
my_resources:
  resources_servers:
    my_server:
      entrypoint: app.py
      domain: agent
      host: "0.0.0.0"
      port: 8080
' \
    my-resources-server:latest
```

::::

::::{tab-item} GPU

```bash
docker run --gpus all -p 8080:8080 \
    -e NEMO_GYM_CONFIG_DICT='
head_server:
  host: "192.168.1.100"
  port: 11000
my_resources:
  resources_servers:
    my_server:
      entrypoint: app.py
      domain: agent
      host: "0.0.0.0"
      port: 8080
' \
    my-resources-server:latest
```

::::

:::::

**Configuration fields**:

| Field | Required | Description |
|-------|----------|-------------|
| `head_server.host` | Yes | IP or hostname of the head server |
| `head_server.port` | Yes | Head server port (default: 11000) |
| `entrypoint` | Yes | Python file to run (typically `app.py`) |
| `domain` | Yes | Task category for the server (used for metrics/grouping): `math`, `coding`, `agent`, `knowledge`, `instruction_following`, `safety`, `games`, etc. |
| `host` | No | Bind address (use `0.0.0.0` for containers) |
| `port` | No | Port to expose (auto-assigned if omitted) |

---

## Docker Compose

For local multi-container development:

```yaml
version: '3.8'
services:
  resources-server:
    build: ./resources_servers/my_server
    ports:
      - "8080:8080"
    environment:
      NEMO_GYM_CONFIG_DICT: |
        head_server:
          host: "host.docker.internal"
          port: 11000
        my_resources:
          resources_servers:
            my_server:
              entrypoint: app.py
              domain: agent
              host: "0.0.0.0"
              port: 8080
```

Start with:

```bash
docker compose up -d
```

---

:::{dropdown} Multi-Stage Builds
:icon: package
:open:

Reduce image size when your server has additional dependencies beyond NeMo Gym:

```dockerfile
# Build stage - compile dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools for packages with C extensions
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --wheel-dir=/wheels nemo-gym

# Runtime stage - minimal image
FROM python:3.12-slim

WORKDIR /app

# Install pre-built wheels (no compilation needed)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
```

**Benefits**:
- Smaller final image (no build tools like gcc)
- Faster builds with layer caching
- Better security (fewer packages in runtime image)

:::

---

## .dockerignore

Create a `.dockerignore` file to exclude unnecessary files from the build context:

```text
__pycache__/
*.pyc
*.pyo
.git/
.gitignore
*.md
.env
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
```

This reduces build context size and prevents sensitive files (like `.env`) from being included in the image.

---

## Logging Configuration

NeMo Gym servers use uvicorn for HTTP logging. Configure logging verbosity via the global config:

```yaml
# In NEMO_GYM_CONFIG_DICT
uvicorn_logging_show_200_ok: false  # Suppress 200 OK logs (cleaner output)
```

For container logging, use Docker's logging drivers:

```bash
docker run --log-driver=json-file --log-opt max-size=10m \
    -e NEMO_GYM_CONFIG_DICT='...' \
    my-resources-server:latest
```

Server logs include the server name prefix for multi-server deployments.

---

## Integration with NeMo Gym

### Connecting to a Head Server

The containerized resources server connects to the head server specified in its configuration. The head server tracks all running server instances and provides service discovery.

**Verify registration**:

```bash
# Check if server registered with head server
curl http://<head-server>:11000/server_instances
```

**Expected output**:

```json
[
  {
    "process_name": "my_resources",
    "server_type": "resources_servers",
    "name": "my_server",
    "host": "172.17.0.2",
    "port": 8080,
    "url": "http://172.17.0.2:8080"
  }
]
```

### Network Considerations

| Scenario | `host` value | Notes |
|----------|--------------|-------|
| Container on same host as head server | `host.docker.internal` | Use Docker's special DNS |
| Container on different host | Container's external IP | Must be routable from head server |
| Kubernetes | Service DNS name | Use Kubernetes service discovery |

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| "Could not connect to head server" | Head server unreachable | Verify `head_server.host` is correct and reachable from container |
| Server not in `/server_instances` | Configuration mismatch | Check `NEMO_GYM_CONFIG_DICT` format and required fields |
| Container exits immediately | Python error in `app.py` | Check logs with `docker logs <container>` |
| Port conflict | Port already in use | Change `port` in config or use `-p <new>:8080` |

---

## Health Checks

Add a health check to your Dockerfile for orchestrator integration:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1
```

This checks if the server responds to HTTP requests. Orchestrators like Kubernetes and Docker Swarm use this to determine container health and restart unhealthy containers.

For Kubernetes, use a liveness probe instead:

```yaml
livenessProbe:
  httpGet:
    path: /
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
```

---

## Production Checklist

Before deploying to production:

- [ ] **Network isolation**: Server not exposed to public internet
- [ ] **Resource limits**: Set container memory/CPU limits
- [ ] **Log aggregation**: Ship logs to centralized system
- [ ] **Restart policy**: Use `--restart=unless-stopped` or orchestrator equivalent
- [ ] **Health checks**: Add `HEALTHCHECK` instruction or orchestrator probes
- [ ] **TLS termination**: Use a reverse proxy (nginx, traefik) for HTTPS

```{warning}
NeMo Gym servers have no built-in authentication or TLS. Use network-level security (VPC, firewalls) and a reverse proxy for HTTPS.
```

---

## Next Steps

- {doc}`/environment-tutorials/multi-node-docker` — Full multi-node deployment guide
- {doc}`/infrastructure/deployment-topology` — Deployment architecture patterns
- {doc}`/tutorials/creating-resource-server` — Build a resources server from scratch

(resources-server-containerize)=
# Containerize Resource Servers

```{note}
This page is a stub. Content is being developed. See [GitHub Issue #546](https://github.com/NVIDIA-NeMo/Gym/issues/546) for details.
```

Package your resources server as a Docker container for portable, reproducible deployments.

---

## When to Containerize

Containerize your resources server when you need:
- Consistent deployment across environments
- Isolated dependencies
- Kubernetes/orchestrator deployment
- GPU access in production

## Dockerfile Template

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "app.py"]
```

## Building the Container

```bash
cd resources_servers/my_server
docker build -t my-resources-server:latest .
```

## Running the Container

```bash
docker run -p 8080:8080 \
    -e MY_API_KEY=${MY_API_KEY} \
    my-resources-server:latest
```

## GPU Support

For GPU-enabled containers:

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.12 python3-pip
```

## Docker Compose

```yaml
version: '3.8'
services:
  resources-server:
    build: ./resources_servers/my_server
    ports:
      - "8080:8080"
    environment:
      - API_KEY=${API_KEY}
```

## Multi-Stage Builds

<!-- TODO: Document multi-stage build optimization -->

## Health Checks

<!-- TODO: Document health check configuration -->

## Logging Configuration

<!-- TODO: Document logging in containers -->

## Integration with NeMo Gym

<!-- TODO: Document how to connect containerized servers to NeMo Gym -->

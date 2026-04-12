(kubernetes)=
# Kubernetes Deployment

NeMo Gym can run on a Kubernetes cluster, separate from the training framework. The typical split: NeMo RL runs on a Slurm/GPU cluster for training and inference, NeMo Gym runs on a K8s cluster for environment orchestration.

```
Slurm cluster (GPU)              K8s cluster (CPU)
+-----------------------+        +---------------------------+
| NeMo RL training      |        | Agent server pods         |
| vLLM inference        | <----> | Resources server pods     |
|                       |  HTTP  | Model proxy pod           |
+-----------------------+        +---------------------------+
```

The model server on K8s proxies inference requests to the vLLM endpoint on Slurm. Everything else runs on K8s without GPUs.

## Quick start

```bash
# Apply base resources
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/rbac.yaml

# Build a server image (example: math_with_code)
docker build --build-arg SERVER_PATH=resources_servers/math_with_code \
  -t nemogym/math-with-code:latest -f k8s/base/Dockerfile .
docker save nemogym/math-with-code:latest | ctr -n k8s.io images import -

# Deploy
kubectl apply -f k8s/examples/math-with-code.yaml
```

## Repository layout

```
k8s/
  base/
    Dockerfile           # Generic, parameterized by SERVER_PATH build arg
    namespace.yaml       # nemogym namespace
    rbac.yaml            # ServiceAccount + Role for K8s Job management
    server-template.yaml # Copy-and-fill template for new servers
  examples/
    math-with-code.yaml  # In-process resources server
    k8s-sandbox.yaml     # Container-per-request resources server
    model-server-proxy.yaml  # Proxies to Slurm vLLM
    simple-agent.yaml    # Agent wired to resources + model via service DNS
```

## Building images

The generic Dockerfile at `k8s/base/Dockerfile` builds any NeMo Gym server:

```bash
docker build \
  --build-arg SERVER_PATH=resources_servers/math_with_code \
  -t nemogym/math-with-code:latest \
  -f k8s/base/Dockerfile .
```

Server-specific pip dependencies are installed from the server's `requirements.txt` automatically.

For local clusters (kubeadm), load images into containerd:

```bash
docker save nemogym/math-with-code:latest | ctr -n k8s.io images import -
```

## Configuration

Each server pod gets its config via `NEMO_GYM_CONFIG_DICT` and `NEMO_GYM_CONFIG_PATH` environment variables, injected from a ConfigMap. See the example manifests for the JSON format.

Inter-server references use K8s service DNS:

```json
"resources_server": {
  "host": "math-with-code.nemogym.svc.cluster.local",
  "port": 8001
}
```

## Model server proxy

The model server runs on K8s as an HTTP proxy. It translates between the Responses API and Chat Completions API, forwarding to the vLLM endpoint on the Slurm cluster.

Set `policy_base_url` in the ConfigMap to the Slurm vLLM address:

```json
"policy_base_url": "http://slurm-head:8000/v1"
```

The K8s cluster and Slurm cluster must be network-reachable.

## Container-per-request servers

Servers that need isolated execution (k8s_sandbox, swe_agents_k8s) spawn K8s Jobs for each request using `K8sJobRunner`. These servers need the `nemogym` ServiceAccount (which has Job/Pod RBAC).

### k8s_sandbox

Each `/execute_code` call spawns a `python:3.12-slim` Job. Code is passed via env var. No shared filesystem needed.

### swe_agents_k8s

Each SWE-bench instance runs as a K8s Job using the instance's Docker image. Requires two PVCs:

| PVC | Access mode | Contents |
|-----|-------------|----------|
| `swe-workspace` | ReadWriteMany | Instance data, patches, trajectories (created at runtime) |
| `swe-setup` | ReadOnlyMany | Pre-built OpenHands, miniforge3, SWE-bench harness |

The setup PVC must be pre-populated before running SWE-bench tasks. The workspace PVC is used automatically.

## RBAC

The `nemogym` ServiceAccount in `k8s/base/rbac.yaml` grants:

| API group | Resource | Verbs |
|-----------|----------|-------|
| `batch` | `jobs` | create, get, list, delete |
| core | `pods` | list, get |
| core | `pods/log` | get |

Only servers that spawn K8s Jobs need this ServiceAccount. Pure HTTP servers can run without it.

## K8sJobRunner

`nemo_gym/k8s_runner.py` provides the `K8sJobRunner` class used by container-per-request servers. It wraps the synchronous Kubernetes Python client with async executors to avoid blocking the FastAPI event loop.

```python
runner = K8sJobRunner(namespace="nemogym")
exit_code, stdout, stderr = await runner.run_job(
    job_name="sandbox-abc123",
    image="python:3.12-slim",
    command=["python", "-c", "print('hello')"],
    timeout=30,
)
```

Supports `env`, `volume_mounts`, `volumes` (PVC, emptyDir, configMap, hostPath), and `resource_limits`.

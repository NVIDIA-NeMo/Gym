(kubernetes)=
# Kubernetes Deployment

NeMo Gym can run on a K8s cluster separate from the training framework. NeMo RL stays on Slurm with GPUs, NeMo Gym runs on K8s with CPUs, model server proxies to vLLM on Slurm.

## Quick start

```bash
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/rbac.yaml

docker build --build-arg SERVER_PATH=resources_servers/math_with_code \
  -t nemogym/math-with-code:latest -f k8s/base/Dockerfile .
docker save nemogym/math-with-code:latest | ctr -n k8s.io images import -

kubectl apply -f k8s/examples/math-with-code.yaml
```

## Layout

```
k8s/
  base/
    Dockerfile           # Generic, parameterized by SERVER_PATH build arg
    namespace.yaml
    rbac.yaml
    server-template.yaml # Copy-and-fill for new servers
  examples/
    math-with-code.yaml
    k8s-sandbox.yaml
    model-server-proxy.yaml
    simple-agent.yaml
```

## Building images

```bash
docker build --build-arg SERVER_PATH=resources_servers/math_with_code \
  -t nemogym/math-with-code:latest -f k8s/base/Dockerfile .
```

Server-specific deps installed from the server's `requirements.txt`.

For kubeadm clusters, load into containerd:

```bash
docker save nemogym/math-with-code:latest | ctr -n k8s.io images import -
```

## Configuration

Config injected via `NEMO_GYM_CONFIG_DICT` and `NEMO_GYM_CONFIG_PATH` env vars from a ConfigMap. See example manifests.

Inter-server refs use K8s service DNS:

```json
"resources_server": {
  "host": "math-with-code.nemogym.svc.cluster.local",
  "port": 8001
}
```

## Model server proxy

Proxies to vLLM on Slurm. Set `policy_base_url` in the ConfigMap:

```json
"policy_base_url": "http://slurm-head:8000/v1"
```

## Container-per-request servers

k8s_sandbox and swe_agents_k8s spawn K8s Jobs per request via `K8sJobRunner`. Need the `nemogym` ServiceAccount for RBAC.

### k8s_sandbox

Each `/execute_code` spawns a `python:3.12-slim` Job. Code passed via env var.

### swe_agents_k8s

Each SWE-bench instance runs as a K8s Job. Requires two PVCs:

| PVC | Access mode | Contents |
|-----|-------------|----------|
| `swe-workspace` | ReadWriteMany | Runtime data, patches, trajectories |
| `swe-setup` | ReadOnlyMany | Pre-built OpenHands, miniforge3, SWE-bench |

## RBAC

`k8s/base/rbac.yaml` grants the `nemogym` ServiceAccount:

| API group | Resource | Verbs |
|-----------|----------|-------|
| `batch` | `jobs` | create, get, list, delete |
| core | `pods` | list, get |
| core | `pods/log` | get |

Only needed for servers that spawn Jobs.

## K8sJobRunner

`nemo_gym/k8s_runner.py` -- async wrapper around the sync K8s Python client.

```python
runner = K8sJobRunner(namespace="nemogym")
exit_code, stdout, stderr = await runner.run_job(
    job_name="sandbox-abc123",
    image="python:3.12-slim",
    command=["python", "-c", "print('hello')"],
    timeout=30,
)
```

Supports `env`, `volume_mounts`, `volumes`, and `resource_limits`.

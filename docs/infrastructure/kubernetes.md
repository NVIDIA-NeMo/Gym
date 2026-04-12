(kubernetes)=
# Kubernetes Deployment

NeMo Gym can run on Kubernetes. This page covers the deployment manifests, RBAC, and the `k8s_sandbox` environment that demonstrates container-per-request execution using Kubernetes Jobs.

## Manifests

All manifests live under `resources_servers/k8s_sandbox/manifests/`.

```
resources_servers/k8s_sandbox/manifests/
├── Dockerfile          # Resources server image
├── namespace.yaml      # nemogym namespace
├── rbac.yaml           # ServiceAccount, Role, RoleBinding
├── deployment.yaml     # Deployment + ConfigMap
└── service.yaml        # ClusterIP on port 8001
```

### Build and deploy

```bash
docker build -t nemogym/k8s-sandbox:latest -f resources_servers/k8s_sandbox/manifests/Dockerfile .
docker save nemogym/k8s-sandbox:latest | ctr -n k8s.io images import -

kubectl apply -f resources_servers/k8s_sandbox/manifests/namespace.yaml
kubectl apply -f resources_servers/k8s_sandbox/manifests/rbac.yaml
kubectl apply -f resources_servers/k8s_sandbox/manifests/deployment.yaml
kubectl apply -f resources_servers/k8s_sandbox/manifests/service.yaml
```

### RBAC

The server pod creates Jobs and reads their logs. The Role grants:

| API group | Resource | Verbs |
|-----------|----------|-------|
| `batch` | `jobs` | create, get, list, delete |
| core | `pods` | list, get |
| core | `pods/log` | get |

The Role is namespaced to `nemogym`.

### Configuration

Server config is injected via `NEMO_GYM_CONFIG_DICT` and `NEMO_GYM_CONFIG_PATH` in the ConfigMap in `deployment.yaml`. Adjust `job_namespace`, `job_image`, and `execution_timeout` there.

## k8s_sandbox environment

`resources_servers/k8s_sandbox` is a resources server where each `/execute_code` call spawns a Kubernetes Job. The job runs the submitted Python code in a fresh `python:3.12-slim` container and returns stdout, stderr, and exit code. Jobs are deleted after completion.

Code is passed to the container via the `__CODE` environment variable:

```python
command=["python", "-c", "import os; exec(os.environ['__CODE'])"]
env={"__CODE": body.code}
```

### Verify

`/verify` checks whether `expected_output` appears in the stdout of the last tool call output. Returns `reward=1.0` on match, `0.0` otherwise.

### Run locally

```bash
export NEMO_GYM_CONFIG_PATH=k8s_sandbox
export NEMO_GYM_CONFIG_DICT='{
  "k8s_sandbox": {
    "resources_servers": {
      "k8s_sandbox": {
        "entrypoint": "app.py",
        "name": "k8s_sandbox",
        "host": "127.0.0.1",
        "port": 8765,
        "job_namespace": "nemogym",
        "job_image": "python:3.12-slim",
        "execution_timeout": 30
      }
    }
  },
  "head_server": {"host": "127.0.0.1", "port": 8000},
  "dry_run": false
}'

cd resources_servers/k8s_sandbox && python app.py
```

```bash
curl -X POST http://127.0.0.1:8765/execute_code \
  -H "Content-Type: application/json" \
  -d '{"code": "print(sum(range(1, 101)))"}'
```

```json
{"exit_code": 0, "stdout": "5050", "stderr": ""}
```

## K8sJobRunner

`nemo_gym/k8s_runner.py` provides `K8sJobRunner`, a reusable async wrapper around the synchronous Kubernetes Python client. Blocking K8s API calls run in a thread-pool executor so the FastAPI event loop is not blocked. The client is initialized lazily on first use, so importing the module is safe in environments without a kubeconfig.

```python
runner = K8sJobRunner(namespace="nemogym")
exit_code, stdout, stderr = await runner.run_job(
    job_name=f"sandbox-{uuid4().hex[:12]}",
    image="python:3.12-slim",
    command=["python", "-c", "print('hello')"],
    timeout=30,
)
```

`run_job` accepts optional `env`, `volume_mounts`, and `volumes` for more complex workloads. Jobs are deleted in a `finally` block after the call returns (`cleanup=True` by default); `ttl_seconds_after_finished=300` acts as a fallback if deletion fails.

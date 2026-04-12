(kubernetes)=
# kubernetes deployment

run nemo gym on k8s, separate from the training cluster. nemo rl stays on slurm with gpus. nemo gym runs on k8s with cpus. model server on k8s proxies inference requests to vllm on slurm.

## quick start

```bash
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/rbac.yaml

# build any server image
docker build --build-arg SERVER_PATH=resources_servers/math_with_code \
  -t nemogym/math-with-code:latest -f k8s/base/Dockerfile .
docker save nemogym/math-with-code:latest | ctr -n k8s.io images import -

# deploy
kubectl apply -f k8s/examples/math-with-code.yaml
```

## layout

```
k8s/
  base/
    Dockerfile             parameterized by SERVER_PATH build arg
    Dockerfile.head        head server image
    head_server_main.py    standalone head server entrypoint
    namespace.yaml
    rbac.yaml
    server-template.yaml   copy and fill in REPLACE_* values
  examples/
    head-server.yaml       service discovery, nodeport for nemo rl
    math-with-code.yaml
    k8s-sandbox.yaml
    model-server-proxy.yaml
    simple-agent.yaml
```

## nemo rl integration

no changes to nemo rl. the existing `NemoGym` environment class connects to a head server, fetches the config dict with all server addresses, and routes requests through `ServerClient`. on k8s the head server is its own pod with a nodeport. nemo rl just points `head_server_config` at that nodeport.

the head server configmap (`k8s/examples/head-server.yaml`) is the single source of truth for server addresses. when you add or remove a server on k8s, update that configmap.

## building images

```bash
docker build --build-arg SERVER_PATH=resources_servers/math_with_code \
  -t nemogym/math-with-code:latest -f k8s/base/Dockerfile .
```

server-specific deps from `requirements.txt` get installed automatically. for kubeadm clusters load into containerd:

```bash
docker save nemogym/math-with-code:latest | ctr -n k8s.io images import -
```

## config

each server pod gets config from a configmap with `NEMO_GYM_CONFIG_DICT` and `NEMO_GYM_CONFIG_PATH` env vars. see the example manifests for the json format.

servers find each other via k8s service dns:

```json
"resources_server": {
  "host": "math-with-code.nemogym.svc.cluster.local",
  "port": 8001
}
```

## model server proxy

proxies to vllm on slurm. set `policy_base_url` to the slurm vllm address:

```json
"policy_base_url": "http://slurm-head:8000/v1"
```

## servers that spawn containers

k8s_sandbox and swe_agents_k8s use `K8sJobRunner` to spawn k8s jobs per request. they need the `nemogym` serviceaccount for rbac (job create/delete/get, pod list, pod log).

most servers (math_with_code, code_gen, judges, etc) don't spawn containers and don't need this.

### k8s_sandbox

demo server. `/execute_code` spawns a `python:3.12-slim` job, passes code via env var, returns stdout/stderr/exit_code. just a proof of concept for the k8s job pattern -- real environments define their own tool endpoints.

### swe_agents_k8s

each swe-bench instance runs as a k8s job. needs two pvcs:

- `swe-workspace` (rwx) -- runtime data, patches, trajectories
- `swe-setup` (rox) -- pre-built openhands, miniforge3, swe-bench harness

## K8sJobRunner

`nemo_gym/k8s_runner.py` -- wraps the sync kubernetes python client with async executors so the fastapi event loop doesn't block.

```python
runner = K8sJobRunner(namespace="nemogym")
exit_code, stdout, stderr = await runner.run_job(
    job_name="sandbox-abc123",
    image="python:3.12-slim",
    command=["python", "-c", "print('hello')"],
    timeout=30,
)
```

supports `env`, `volume_mounts`, `volumes`, and `resource_limits`.

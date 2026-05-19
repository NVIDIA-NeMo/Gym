# NeMo-Gym on Kubernetes

Deploy NeMo-Gym benchmarks on any Kubernetes cluster using KubeRay for distributed Ray task execution. The base manifests are platform-agnostic — each benchmark gets its own Kustomize overlay, and optional platform overlays (e.g. OpenShift) compose on top.

## Architecture

```
┌─ Namespace: gym ─────────────────────────────────────────────────┐
│                                                                   │
│  RayCluster: gym-ray                                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Head Pod (coordination, dashboard on :8265)                │  │
│  │  Worker Pods (execute @ray.remote tasks)                    │  │
│  │  Resources Worker Pod ← runs benchmark server on :9080      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Deployment: gym-agent (simple_agent on :8080)                    │
│  Deployment: gym-model (vllm_model proxy on :8080)                │
│                                                                   │
│  Services: gym-agent-svc, gym-model-svc, gym-resources-svc        │
│  Ingress: gym-ray-dashboard (Ray dashboard, optional)             │
└───────────────────────────────────────────────────────────────────┘
        │
        │ HTTPS
        ▼
  External LLM endpoint (OpenAI-compatible /v1/chat/completions)
```

**Data flow:** Client → agent `/run` → model `/v1/responses` → external LLM → agent extracts code → resources `/verify` → Ray workers execute unit tests → reward (0.0 or 1.0)

## Directory Structure

```
k8s/
├── Dockerfile                              # Multi-stage: resources, agent, model targets
├── entrypoint.sh                           # Arbitrary UID handler (harmless on standard K8s)
├── README.md
├── base/                                   # Works on ANY Kubernetes cluster with KubeRay
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── serviceaccount.yaml
│   ├── configmap.yaml                      # Base Hydra config (agent + model)
│   ├── raycluster.yaml                     # Ray head, workers, resources worker group
│   ├── deployment-agent.yaml               # simple_agent (skips Ray)
│   ├── deployment-model.yaml               # vllm_model proxy (skips Ray)
│   ├── service-*.yaml                      # ClusterIP services
│   ├── networkpolicy.yaml                  # Internal pod-to-pod traffic
│   └── ingress-dashboard.yaml              # K8s Ingress for Ray dashboard (optional)
└── overlays/
    ├── openshift/                          # Platform: OpenShift (Kustomize Component)
    ├── code-gen/                           # Benchmark: code generation
    ├── example-single-tool-call/           # Benchmark: minimal example
    ├── code-gen-openshift/                 # Composition: code-gen + OpenShift
    └── example-single-tool-call-openshift/ # Composition: example + OpenShift
```

## Prerequisites

| Component | Requirement | How to verify |
|-----------|-------------|---------------|
| Kubernetes | 1.27+ | `kubectl version` |
| KubeRay operator | Installed via Helm | `kubectl get crd rayclusters.ray.io` |
| kustomize | 4.x+ | `kustomize version` |

**Install KubeRay:**

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.3.0 --namespace kuberay-system --create-namespace
```

## Quick Start

### 1. Build and push images (or use GHCR)

Pre-built images are available from GitHub Container Registry:

```
ghcr.io/nvidia-nemo/gym/nemo-gym-resources:latest
ghcr.io/nvidia-nemo/gym/nemo-gym-agent:latest
ghcr.io/nvidia-nemo/gym/nemo-gym-model:latest
```

Or build from source:

```bash
docker build -f k8s/Dockerfile --target resources -t ghcr.io/nvidia-nemo/gym/nemo-gym-resources:latest .
docker build -f k8s/Dockerfile --target agent -t ghcr.io/nvidia-nemo/gym/nemo-gym-agent:latest .
docker build -f k8s/Dockerfile --target model -t ghcr.io/nvidia-nemo/gym/nemo-gym-model:latest .
```

### 2. Configure LLM credentials

```bash
cp k8s/overlays/code-gen/secret.yaml.example k8s/overlays/code-gen/secret.yaml
# Edit secret.yaml with your LLM endpoint details
```

| Field | What to set | Example |
|-------|-------------|---------|
| `POLICY_BASE_URL` | OpenAI-compatible API base URL (must end with `/v1`) | `https://my-vllm.example.com/v1` |
| `POLICY_API_KEY` | API key or bearer token | `sk-abc123` |
| `POLICY_MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct` |

### 3. Deploy

```bash
kustomize build k8s/overlays/code-gen | kubectl apply -f -
```

### 4. Verify

```bash
kubectl get pods -n gym -w
```

Expected state:

| Pod | Ready | Description |
|-----|-------|-------------|
| `gym-ray-head-*` | 1/1 | Ray head node |
| `gym-ray-gym-workers-worker-*` | 1/1 | Ray code execution worker |
| `gym-ray-gym-resources-worker-*` | 1/1 | Resources server + Ray node |
| `gym-agent-*` | 1/1 | Agent server |
| `gym-model-*` | 1/1 | Model proxy |

### 5. Smoke test

```bash
kubectl port-forward -n gym svc/gym-agent-svc 8080:8080 &
curl -s http://localhost:8080/docs | head -5
kill %1
```

### 6. Ray dashboard

```bash
kubectl port-forward -n gym svc/gym-ray-head-svc 8265:8265
```

The base includes a Kubernetes Ingress for the Ray dashboard. If your cluster has an Ingress controller, it will be picked up automatically. Otherwise, use port-forwarding above.

## Available Overlays

Benchmark overlays are platform-agnostic. Platform overlays add cluster-specific resources. Composition overlays combine both.

| Overlay | Type | Description |
|---------|------|-------------|
| `code-gen` | Benchmark | Code generation benchmark (`resources_servers/code_gen`) |
| `example-single-tool-call` | Benchmark | Minimal single tool call example |
| `openshift` | Platform | [Kustomize Component](https://kubectl.docs.kubernetes.io/guides/config_management/components/) — adds OpenShift Route, OCP NetworkPolicies, removes Ingress |
| `code-gen-openshift` | Composition | `code-gen` + `openshift` |
| `example-single-tool-call-openshift` | Composition | `example-single-tool-call` + `openshift` |

Deploy a composition overlay the same way:

```bash
kustomize build k8s/overlays/code-gen-openshift | kubectl apply -f -
```

## Using a Different Image Registry

Override the default GHCR images in any overlay's `kustomization.yaml`:

```yaml
images:
  - name: ghcr.io/nvidia-nemo/gym/nemo-gym-resources
    newName: quay.io/myorg/nemo-gym-resources
    newTag: v1.0.0
  - name: ghcr.io/nvidia-nemo/gym/nemo-gym-agent
    newName: quay.io/myorg/nemo-gym-agent
    newTag: v1.0.0
  - name: ghcr.io/nvidia-nemo/gym/nemo-gym-model
    newName: quay.io/myorg/nemo-gym-model
    newTag: v1.0.0
```

## Adding a New Benchmark

Copy an existing overlay as a starting point:

```bash
cp -r k8s/overlays/code-gen k8s/overlays/my-benchmark
```

Three files to customize:

### `secret.yaml` — LLM credentials

Copy from `secret.yaml.example` and fill in your endpoint details.

### `configmap-patch.yaml` — Benchmark server config

Update `resources_instance` with your server's config:

```yaml
    resources_instance:
      resources_servers:
        my_server:
          entrypoint: app.py
          host: "${oc.env:RESOURCES_HOST}"
          port: 9080
          domain: coding
          # Server-specific fields:
          num_processes: 4
          timeout_secs: 30
```

### `kustomization.yaml` — RayCluster env patches

Update the JSON patches to point to your server:

```yaml
patches:
  - patch: |-
      - op: replace
        path: /spec/workerGroupSpecs/1/template/spec/containers/0/env/2/value
        value: "python resources_servers/my_server/app.py"
      - op: replace
        path: /spec/workerGroupSpecs/1/template/spec/containers/0/env/4/value
        value: "resources_instance"
    target:
      kind: RayCluster
      name: gym-ray
```

**Env index warning:** The RayCluster env vars are patched by array index. `env[2]` is `NEMO_GYM_SERVER_ENTRYPOINT` and `env[4]` is `NEMO_GYM_CONFIG_PATH`. Do not reorder the env vars in `base/raycluster.yaml`.

### Platform support for new benchmarks

Create a composition overlay that layers the platform component onto your benchmark:

```bash
mkdir k8s/overlays/my-benchmark-openshift
cat > k8s/overlays/my-benchmark-openshift/kustomization.yaml <<'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../my-benchmark

components:
  - ../openshift
EOF
```

## Scaling Ray Workers

Edit `k8s/base/raycluster.yaml`:

```yaml
workerGroupSpecs:
  - groupName: gym-workers
    replicas: 4
    minReplicas: 2
    maxReplicas: 8
```

Then reapply: `kustomize build k8s/overlays/code-gen | kubectl apply -f -`

## Port Layout

| Server | Port | Notes |
|--------|------|-------|
| Agent (simple_agent) | 8080 | Standard HTTP |
| Model (vllm_model) | 8080 | Standard HTTP |
| Resources (benchmark) | 9080 | Non-default to avoid conflict with Ray metrics on 8080 |
| Ray GCS | 6379 | Internal |
| Ray Dashboard | 8265 | Access via `kubectl port-forward` or Ingress |

## Teardown

```bash
kustomize build k8s/overlays/code-gen | kubectl delete -f -
```

## Troubleshooting

**Check recent events:**
```bash
kubectl get events -n gym --sort-by='.lastTimestamp' | tail -20
```

**Resources pod not reaching Ready:**
```bash
kubectl exec $(kubectl get pod -n gym -l ray.io/group=gym-resources -o name) -n gym -- cat /tmp/nemo-gym-server.log
```

**Agent returns connection errors:**
Check NetworkPolicies allow traffic between pods:
```bash
kubectl get networkpolicy -n gym
```

**Model returns connection errors:**
Verify the LLM endpoint is reachable from inside the cluster:
```bash
kubectl exec $(kubectl get pod -n gym -l app.kubernetes.io/name=gym-model -o name) -n gym -- \
  curl -s "$POLICY_BASE_URL/models" -H "Authorization: Bearer $POLICY_API_KEY"
```

**Ray dashboard:**
```bash
kubectl port-forward -n gym svc/gym-ray-head-svc 8265:8265
```

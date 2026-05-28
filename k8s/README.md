# NeMo-Gym on Kubernetes

Deploy NeMo-Gym benchmarks on any Kubernetes cluster using KubeRay for distributed Ray task execution. The base manifests are platform-agnostic — each benchmark gets its own Kustomize overlay with a unique `namePrefix`, and optional platform overlays (e.g. OpenShift) compose on top. Multiple benchmarks can run simultaneously in the same namespace without colliding.

## Architecture

```
┌─ Namespace: gym ──────────────────────────────────────────────────┐
│                                                                   │
│  Instance: cg- (code-gen)         Instance: estc- (example)       │
│  ┌─────────────────────────┐      ┌─────────────────────────┐     │
│  │ cg-gym-ray (RayCluster) │      │ estc-gym-ray (RayClust) │     │
│  │ cg-gym-agent (Deploy)   │      │ estc-gym-agent (Deploy) │     │
│  │ cg-gym-model (Deploy)   │      │ estc-gym-model (Deploy) │     │
│  │ cg-gym-head  (Deploy)   │      │ estc-gym-head  (Deploy) │     │
│  └─────────────────────────┘      └─────────────────────────┘     │
│                                                                   │
│  Isolation: namePrefix + app.kubernetes.io/instance label         │
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
├── head/
│   ├── Containerfile                       # HeadServer image (UBI 10, separate from main Dockerfile)
│   └── entrypoint.py                       # Standalone HeadServer startup script
├── entrypoint.sh                           # Arbitrary UID handler (harmless on standard K8s)
├── README.md
├── base/                                   # Works on ANY Kubernetes cluster with KubeRay
│   ├── kustomization.yaml
│   ├── configurations/
│   │   └── name-reference.yaml             # CRD field mappings for namePrefix propagation
│   ├── serviceaccount.yaml
│   ├── configmap.yaml                      # Base Hydra config (agent + model)
│   ├── raycluster.yaml                     # Ray head, workers, resources worker group
│   ├── deployment-agent.yaml               # simple_agent (skips Ray)
│   ├── deployment-model.yaml               # vllm_model proxy (skips Ray)
│   ├── deployment-head.yaml                # HeadServer (config server for CLI tools)
│   └── service-*.yaml                      # ClusterIP services
└── overlays/
    ├── openshift/                          # Platform: OpenShift (Kustomize Component)
    ├── code-gen/                           # Benchmark: code generation (prefix: cg-)
    ├── example-single-tool-call/           # Benchmark: minimal example (prefix: estc-)
    ├── code-gen-openshift/                 # Composition: code-gen + OpenShift
    └── example-single-tool-call-openshift/ # Composition: example + OpenShift
```

## Prerequisites

| Component | Requirement | How to verify |
|-----------|-------------|---------------|
| Kubernetes | 1.27+ | `kubectl version` |
| KubeRay operator | Installed via Helm | `kubectl get crd rayclusters.ray.io` |
| kustomize | 5.x+ | `kustomize version` |

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
ghcr.io/nvidia-nemo/gym/nemo-gym-head:latest
```

Or build from source:

```bash
docker build -f k8s/Dockerfile --target resources -t ghcr.io/nvidia-nemo/gym/nemo-gym-resources:latest .
docker build -f k8s/Dockerfile --target agent -t ghcr.io/nvidia-nemo/gym/nemo-gym-agent:latest .
docker build -f k8s/Dockerfile --target model -t ghcr.io/nvidia-nemo/gym/nemo-gym-model:latest .
docker build -f k8s/head/Containerfile --target head -t ghcr.io/nvidia-nemo/gym/nemo-gym-head:latest .
```

### 2. Create namespace and configure LLM credentials

```bash
kubectl create namespace gym

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

All resources are prefixed with the overlay's `namePrefix` (e.g. `cg-` for code-gen).

### 4. Verify

```bash
kubectl get pods -n gym -l app.kubernetes.io/instance=cg -w
```

Expected state:

| Pod | Ready | Description |
|-----|-------|-------------|
| `cg-gym-ray-head-*` | 3/3 | Ray head node |
| `cg-gym-ray-gym-workers-worker-*` | 1/1 | Ray code execution worker |
| `cg-gym-ray-gym-resources-worker-*` | 1/1 | Resources server + Ray node |
| `cg-gym-agent-*` | 1/1 | Agent server |
| `cg-gym-model-*` | 1/1 | Model proxy |
| `cg-gym-head-*` | 1/1 | HeadServer (config for CLI tools) |

### 5. Smoke test

```bash
kubectl port-forward -n gym svc/cg-gym-agent-svc 8080:8080 &
curl -s http://localhost:8080/docs | head -5
kill %1
```

### 6. Use CLI tools via port-forward

The HeadServer enables standard NeMo Gym CLI tools (`ng_collect_rollouts`, `ng_status`, `ng_reward_profile`) to work against the cluster. Set the env vars locally to point at the port-forwarded services:

```bash
kubectl port-forward -n gym svc/cg-gym-head-svc 11000:11000 &
kubectl port-forward -n gym svc/cg-gym-agent-svc 8080:8080 &

AGENT_HOST=localhost MODEL_HOST=localhost RESOURCES_HOST=localhost \
POLICY_BASE_URL=http://unused POLICY_API_KEY=unused POLICY_MODEL_NAME=unused \
ng_collect_rollouts +agent_name=simple_agent_instance \
  +input_jsonl_fpath=resources_servers/code_gen/data/example.jsonl \
  +output_jsonl_fpath=/tmp/rollouts.jsonl \
  +num_repeats=1 \
  "+config_paths=[resources_servers/code_gen/configs/code_gen.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

### 7. Ray dashboard

```bash
kubectl port-forward -n gym svc/cg-gym-ray-head-svc 8265:8265
```

No services are exposed externally — all access is via `kubectl port-forward`.

## Multi-Instance Deployment

Multiple benchmarks can run simultaneously in the same namespace. Each overlay has a unique `namePrefix` and `app.kubernetes.io/instance` label that ensures complete isolation — no resource name collisions, no service cross-selection, no data bleed.

```bash
# Deploy both benchmarks at once
kustomize build k8s/overlays/code-gen | kubectl apply -f -
kustomize build k8s/overlays/example-single-tool-call | kubectl apply -f -

# View resources by instance
kubectl get pods -n gym -l app.kubernetes.io/instance=cg
kubectl get pods -n gym -l app.kubernetes.io/instance=estc
```

### How isolation works

| Mechanism | What it does |
|-----------|-------------|
| `namePrefix` | Every resource gets a unique prefix (`cg-`, `estc-`, etc.) |
| `labels` with `includeSelectors` | Instance label injected into all Service selectors and Deployment matchLabels |
| `replacements` | Service DNS names in env vars (`AGENT_HOST`, `MODEL_HOST`, `RESOURCES_HOST`) updated to prefixed names |
| `configurations/name-reference.yaml` | RayCluster CRD's ServiceAccount, ConfigMap, and Secret refs updated by namePrefix |

Each instance is fully self-contained — separate ConfigMaps, Secrets, ServiceAccounts, RayClusters, Deployments, and Services.

## Available Overlays

Benchmark overlays are platform-agnostic. Platform overlays add cluster-specific resources. Composition overlays combine both.

| Overlay | Type | Prefix | Description |
|---------|------|--------|-------------|
| `code-gen` | Benchmark | `cg-` | Code generation benchmark (`resources_servers/code_gen`) |
| `example-single-tool-call` | Benchmark | `estc-` | Minimal single tool call example |
| `openshift` | Platform | — | [Kustomize Component](https://kubectl.docs.kubernetes.io/guides/config_management/components/) — sets `imagePullPolicy: Always` |
| `code-gen-openshift` | Composition | `cg-` | `code-gen` + `openshift` |
| `example-single-tool-call-openshift` | Composition | `estc-` | `example-single-tool-call` + `openshift` |

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
  - name: ghcr.io/nvidia-nemo/gym/nemo-gym-head
    newName: quay.io/myorg/nemo-gym-head
    newTag: v1.0.0
```

## Adding a New Benchmark

Copy an existing overlay as a starting point:

```bash
cp -r k8s/overlays/code-gen k8s/overlays/my-benchmark
```

Four things to customize:

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

### `kustomization.yaml` — Instance prefix and RayCluster env patches

Set a unique `namePrefix` and `app.kubernetes.io/instance` value for your benchmark. The `labels`, `replacements`, and RayCluster patches can be copied as-is from the template — only update the prefix/instance values and the RayCluster env patches:

```yaml
namePrefix: mb-

labels:
  - includeSelectors: true
    includeTemplates: true
    pairs:
      app.kubernetes.io/instance: mb
    fields:
      - path: spec/headGroupSpec/template/metadata/labels
        kind: RayCluster
        group: ray.io
        create: true
      - path: spec/workerGroupSpecs/template/metadata/labels
        kind: RayCluster
        group: ray.io
        create: true

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

replacements:
  # Copy the replacements block from code-gen/kustomization.yaml verbatim
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
| HeadServer | 11000 | Config server for CLI tools (`ng_collect_rollouts`, `ng_status`) |
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
# Replace INSTANCE with your instance label (cg, estc, etc.)
kubectl exec $(kubectl get pod -n gym -l app.kubernetes.io/instance=cg,ray.io/group=gym-resources -o name) -n gym -- cat /tmp/nemo-gym-server.log
```

**Model returns connection errors:**
Verify the LLM endpoint is reachable from inside the cluster:
```bash
kubectl exec deploy/cg-gym-model -n gym -- \
  curl -s "$POLICY_BASE_URL/models" -H "Authorization: Bearer $POLICY_API_KEY"
```

**Ray dashboard:**
```bash
kubectl port-forward -n gym svc/cg-gym-ray-head-svc 8265:8265
```

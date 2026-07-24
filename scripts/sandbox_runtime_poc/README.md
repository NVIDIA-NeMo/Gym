# OSWorld on OpenSandbox (runc + KVM) — deployment assets

Manifests and helpers for running the OSWorld environment
(`resources_servers/osworld` + `responses_api_agents/nemotron_osworld`) against an
OpenSandbox cell that serves KVM desktop VMs. Placeholders (`<...>`) are
environment-specific.

## Sandbox cluster (one-time, cluster admin)

| file | what it does |
|---|---|
| `osworld-qcow2-s3.yaml` | static PV + ROX PVC exposing the OSWorld guest image from S3 via the Mountpoint-S3 CSI driver (one image, mounted per-pod, read-only; COW overlay per VM) |
| `generic-device-plugin.yaml` | DaemonSet advertising `devic.es/kvm` + `devic.es/tun` on VM nodes so sandbox pods run non-privileged |
| `osworld-kvm-pool.yaml` | OpenSandbox `Pool` of pre-warmed QEMU/KVM Ubuntu desktops |
| `kata-metal-nodeoverlay.yaml` | Karpenter NodeOverlay declaring the device-plugin resources so metal nodes autoscale on VM demand (requires the `NodeOverlay` feature gate) |

Upload the guest image once: `aws s3 cp System.qcow2 s3://<YOUR_S3_BUCKET>/osworld/System.qcow2`.

## The eval Job (per run)

Inference is assumed to be an existing OpenAI-compatible endpoint — point
`POLICY_BASE_URL` at it (any vLLM deployment or hosted service serving
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` with its model-card recipe).

| file | what it does |
|---|---|
| `osworld-omni-gate.yaml` + `osworld-omni-gate.entrypoint.sh` | durable eval Job: runs `gym eval run` with per-rollout result flush and automatic `--resume` on pod retry |

## Dataset

`build_osworld_dataset.py` converts OSWorld task definitions (from the pinned fork
checkout) into the JSONL rows in `resources_servers/osworld/data/`.

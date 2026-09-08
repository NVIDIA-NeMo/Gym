# EnterpriseOps-Gym Resource Server

Run the [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
benchmark with Gym-managed MCP services and upstream-compatible SQL verification. See the
[benchmark README](../../../benchmarks/enterpriseops/README.md) for split names and evaluation commands.

The commands below assume an OpenAI-compatible model endpoint is already listening at the
configured URL.

## Quick start (x86_64)

```bash
# Prepare task rows and tool schemas.
gym eval prepare --benchmark enterpriseops

# Start seven managed EnterpriseOps services with the default Apptainer profile.
gym env start --benchmark enterpriseops --model-type openai_model \
  --model-url http://127.0.0.1:8000/v1 \
  --model-api-key EMPTY \
  --model <served-model-name>

# In another terminal, run the Oracle split.
gym eval run --no-serve \
  --agent enterpriseops_benchmark_simple_agent \
  --input benchmarks/enterpriseops/data/enterpriseops_oracle_benchmark.jsonl \
  --output results/enterpriseops_oracle.jsonl
```

The first `gym env start` downloads the pinned EnterpriseOps source checkout, its
`gym_dbs.zip` archive, and the seven digest-pinned service images into the local provider cache.
The resources server logs the detected architecture, sandbox provider, selected images, and
managed database storage.

## ARM64 (Apptainer)

The upstream service images are AMD64. Build native SIFs once, then use the same prepare,
start, and run commands shown above:

```bash
python -m resources_servers.enterpriseops_gym.arm64_images --all
```

By default, the builder and runtime both use
`~/.cache/nemo_gym/enterpriseops_gym/images`. To use another location, either pass
`--output-dir` while building and set `ENTERPRISEOPS_NATIVE_SIF_DIR` while running, or override
`native_sif_dir` in config. Explicit entries in `native_service_images` have highest precedence.

With the default Apptainer profile, Gym automatically gives each service a host-backed
`/app/mcp_databases` directory and removes it when the environment stops. No database-path
environment variable or bind override is required.

## Select another Sandbox API provider

Compose the selected provider configuration with the EOps remote overlay. The provider publishes
the service endpoints; EOps consumes those endpoints for MCP, database seeding, and verification.

```bash
gym env start --benchmark enterpriseops \
  --config <provider-config>.yaml \
  --config resources_servers/enterpriseops_gym/configs/enterpriseops_gym_remote.yaml \
  --model-type openai_model \
  --model-url <model-url> \
  --model-api-key <model-api-key> \
  --model <model-name>
```

The provider must support `start`, `exec`, `stop`, and resolving declared service ports.
Automatic database binds are added only when the resolved provider metadata identifies
`sandbox-api: apptainer-cli`; Docker, Enroot, remote, and custom providers are left unchanged.

## Runtime model

- `gym env start` starts one persistent service for each EOps domain: CSM, Teams, Calendar,
  Email, ITSM, HR, and Drive.
- `/seed_session` creates fresh per-domain databases from the task snapshot and associates their
  IDs with the Gym session.
- Tool calls are proxied to the matching MCP service with that database ID.
- `/verify` runs upstream-compatible verifiers, records reward/metrics, and deletes the seeded
  databases.

## Development checks

```bash
uv run pytest resources_servers/enterpriseops_gym/tests tests/unit_tests/test_apptainer_provider.py -q
uv run pre-commit run --all-files
```

## Data and licenses

- Upstream benchmark: [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym), Apache 2.0.
- Task data: [ServiceNow-AI/EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym).
- Tool schemas: [nvidia/NeMo-Gym-EnterpriseOps-Assets](https://huggingface.co/datasets/nvidia/NeMo-Gym-EnterpriseOps-Assets).

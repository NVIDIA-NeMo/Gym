# EnterpriseOps-Gym Reproducible Runtime

The NeMo Gym port is tied to these immutable inputs:

- Original port provenance: NVIDIA-NeMo/Gym PR 2142 at
  `245a9766a2e875df605ecb56fa8c43f386726e50`. This standalone integration was
  extracted from `0f4aa2c738b1694bf1c020db5a7ef0bfe204e440` and rebased onto main;
  the historical parity and performance reports describe their original runs.
- ServiceNow EnterpriseOps-Gym checkout:
  `de22905d21a080b83bf4a54258afe4250ee2dd55`.
- Hugging Face public oracle dataset:
  `ServiceNow-AI/EnterpriseOps-Gym` revision
  `c8e538eae8a6205294f0a86675fefdc1fac408f6` (649 rows).

## Checkout and seed databases

```bash
git clone https://github.com/ServiceNow/EnterpriseOps-Gym.git <EOG_ROOT>
git -C <EOG_ROOT> checkout --detach de22905d21a080b83bf4a54258afe4250ee2dd55
cd <EOG_ROOT>
unzip gym_dbs.zip
```

Set `seed_sql_root` to `<EOG_ROOT>`, not to its
`Domain Wise DBs and Task-DB Mappings` child.

## Digest-pinned services

```bash
docker run -d --platform linux/amd64 --name eog-csm --restart unless-stopped -p 8001:8005 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-csm@sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189
docker run -d --platform linux/amd64 --name eog-teams --restart unless-stopped -p 8002:8005 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-teams@sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78
docker run -d --platform linux/amd64 --name eog-calendar --restart unless-stopped -p 8003:8003 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-calendar@sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02
docker run -d --platform linux/amd64 --name eog-email --restart unless-stopped -p 8004:8005 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-email@sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd
docker run -d --platform linux/amd64 --name eog-itsm --restart unless-stopped -p 8006:8005 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-itsm@sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f
docker run -d --platform linux/amd64 --name eog-hr --restart unless-stopped -p 8008:8005 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-hr@sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d
docker run -d --platform linux/amd64 --name eog-drive --restart unless-stopped -p 8009:8005 \
  shivakrishnareddyma225/enterpriseops-gym-mcp-drive@sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32
```

Calendar is the port-mapping exception: its container listens on 8003; the other six
listen on 8005. All seven published images are `linux/amd64`. On Apple Silicon, use an
amd64-capable Docker VM (for example Colima with VZ/Rosetta) and keep the explicit
`--platform linux/amd64`; this is emulated and is not a native performance baseline.

```bash
for port in 8001 8002 8003 8004 8006 8008 8009; do
  curl -fsS "http://localhost:${port}/health"
done
```

## Prepare, run, and profile

Keep endpoint credentials in ignored `env.yaml`, for example:

```yaml
policy_base_url: https://example.openai-compatible.endpoint/v1
policy_api_key: ${oc.env:MODEL_API_KEY}
policy_model_name: provider/model-name
```

For NVIDIA Inference Hub, use `https://inference-api.nvidia.com/v1` as the base URL
and reference the credential as `${oc.env:INFERENCE_HUB_KEY}`; never put the key value
in this repository.

Reasoning-item handling is model-specific. Set `POLICY_DROP_REASONING_ITEMS=true` for
`nvidia/openai/gpt-oss-20b`, whose hosted endpoint rejects its content-less reasoning
items when they are echoed across tool turns. Keep it `false` for GPT-5-family models,
which require their encrypted reasoning item to be returned with subsequent tool outputs.

Then run:

```bash
gym eval prepare --benchmark enterpriseops

POLICY_DROP_REASONING_ITEMS=false gym env start \
  --config env.yaml \
  --config resources_servers/enterpriseops_gym/configs/enterpriseops_gym_turnlog.yaml \
  --model-type openai_model \
  "++enterpriseops_gym.resources_servers.enterpriseops_gym.seed_sql_root=<EOG_ROOT>"

# In a second terminal, use two repeats for an actual distribution while keeping the
# smoke bounded to one task.
gym eval run --no-serve \
  --config env.yaml \
  --config resources_servers/enterpriseops_gym/configs/enterpriseops_gym_turnlog.yaml \
  --model-type openai_model \
  --agent enterpriseops_gym_turn_logging_agent \
  --input resources_servers/enterpriseops_gym/data/example.jsonl \
  --output results/enterpriseops_model_smoke.jsonl \
  --limit 1 --num-repeats 2 --concurrency 1 \
  --temperature 0.0 --max-output-tokens 16384

gym eval profile \
  --inputs results/enterpriseops_model_smoke_materialized_inputs.jsonl \
  --rollouts results/enterpriseops_model_smoke.jsonl

python resources_servers/enterpriseops_gym/export_eval_telemetry.py \
  --input results/enterpriseops_model_smoke.jsonl \
  --output results/enterpriseops_model_smoke_turns.jsonl
```

For a valid benchmark/profile artifact, require every completed row to have
`num_verifiers_scored > 0`, reconcile materialized input and rollout task IDs, and retain
the generated aggregate metrics and reward-profiling files. Large campaigns should
restart the container fleet between passes because the upstream images have demonstrated
file-descriptor leakage over hundreds of task lifecycles.

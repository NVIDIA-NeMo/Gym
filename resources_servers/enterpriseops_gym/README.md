# EnterpriseOps-Gym Resources Server

Adapts the [ServiceNow EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
benchmark (Apache 2.0) to NeMo Gym: 1,150 stateful enterprise tool-use tasks across 8 domains
(Calendar, CSM, Drive, Email, HR, ITSM, Teams, Hybrid), executed against the upstream
MCP gym services and graded by SQL verifiers on final database state.

## Architecture

- `gym env start` owns seven Apptainer sandboxes, one per EOG domain. Each runs the upstream
  FastAPI/MCP application on its assigned endpoint. Per-rollout `/seed_session` seeds a fresh
  database from the task's SQL snapshot and pins `{gym -> database_id}` to the session cookie;
  a catch-all `POST /{tool_name}` proxies tool calls with the session's `x-database-id` and task
  context headers; `/verify` runs the task's verifiers (ported verbatim from EOG) and deletes
  the databases.
- `mcp_client.py` — pooled-aiohttp port of EOG's MCP/JSON-RPC client (one MCP session per gym
  server; per-call database ids; in-memory seed SQL cache).
- `verifier_engine.py` — line-for-line port of EOG's verifier engine (`database_state`,
  `response_check` LLM judge, `tool_execution`), including its **name-collapse scoring quirk**:
  duplicate-named verifiers overwrite each other and only the last one per name is scored.
  The headline `reward` preserves that for leaderboard parity; strict every-verifier metrics
  (`strict_success`, `strict_pass_rate`) are emitted alongside for RL reward shaping
  (`strict_verifiers: true` switches the reward to strict).
- `convert_tasks.py` / `snapshot_tools.py` — convert EOG task JSONs (local or the
  `ServiceNow-AI/EnterpriseOps-Gym` HF dataset) into NeMo Gym JSONL rows, baking in tool
  schemas from per-domain `tools/list` snapshots.

## Prerequisites

1. Apptainer available on the host. The runtime downloads the pinned EnterpriseOps source checkout
   and its verified `gym_dbs.zip` database archive into `cache_dir` automatically.
2. On x86_64, the runtime pulls and caches the seven digest-pinned upstream images automatically.

### ARM64 service images

The published upstream service images are AMD64. On ARM64, build the seven native SIFs once before
starting Gym. By default, `gym env start` looks for them in
`~/.cache/nemo_gym/enterpriseops_gym/images`; if any are missing, it fails before downloading
assets or starting a sandbox and prints the missing paths.

```bash
python -m resources_servers.enterpriseops_gym.arm64_images --all \
  --output-dir ~/.cache/nemo_gym/enterpriseops_gym/images
```

The command pulls the pinned AMD64 source images into a sibling `source-amd64/` cache, rebuilds
only missing native `<domain>-arm64.sif` files, and writes `<domain>-arm64.sif.provenance.json`
next to each result. It uses rootless `apptainer build --fakeroot`; add `--sudo` on systems where
fakeroot is unavailable. With a custom `cache_dir`, use `<cache_dir>/images` as `--output-dir`.
Set `native_sif_dir` to use a different shared read-only SIF directory.

## Prepare assets

The seven per-domain tool-schema snapshots are hosted on Hugging Face rather than
committed (~30k lines of generated JSON). Fetch them once:

```bash
python -m resources_servers.enterpriseops_gym.prepare
```

This downloads the pinned revision of
[`nvidia/NeMo-Gym-EnterpriseOps-Assets`](https://huggingface.co/datasets/nvidia/NeMo-Gym-EnterpriseOps-Assets),
verifies it against a checksum baked into `prepare.py`, and writes
`data/tools/*.json` (gitignored). Re-running is a no-op once the files match the pin.
`gym eval prepare --benchmark enterpriseops` calls this for you.

On a machine without Hub egress, fetch the snapshots elsewhere and point at them:

```bash
hf download nvidia/NeMo-Gym-EnterpriseOps-Assets --repo-type dataset \
    --revision <pinned-sha> --include 'enterpriseops_gym/tools/*' --local-dir /tmp/eog
export NEMO_GYM_EOG_TOOLS_DIR=/tmp/eog/enterpriseops_gym/tools
```

The directory is validated against the same checksum before use.

## Usage

```bash
# Re-capture a tool schema from a running gym server (see "Refreshing the snapshots";
# write to a scratch path -- data/tools/ is owned by prepare.py)
python resources_servers/enterpriseops_gym/snapshot_tools.py \
    --gym-url http://localhost:8001 --gym-name sn-csm-server \
    --output /tmp/eog-tools/csm.json

# Convert EOG task JSONs to a NeMo Gym dataset
python resources_servers/enterpriseops_gym/convert_tasks.py \
    --tasks-dir ../enterpriseops-gym/data/revised/csm \
    --tools-snapshot resources_servers/enterpriseops_gym/data/tools/csm.json \
    --domain csm --mode oracle \
    --output results/enterpriseops_csm_tasks.jsonl

# Run servers + collect rollouts
gym env start \
    --resources-server enterpriseops_gym \
    --model-type openai_model
gym eval run --no-serve \
    --agent enterpriseops_gym_simple_agent \
    --input results/enterpriseops_csm_tasks.jsonl \
    --output results/enterpriseops_csm.jsonl
```

### Refreshing the snapshots

A schema refresh is a revision bump, not a commit.

1. Bring up the seven upstream MCP gym containers.
2. Re-capture each domain with `snapshot_tools.py --output /tmp/eog-tools/<domain>.json`.
3. Upload the seven files to `enterpriseops_gym/tools/` in the assets dataset as a new
   commit.
4. Recompute the pin: `python -m resources_servers.enterpriseops_gym.prepare
   --print-hash /tmp/eog-tools`.
5. Update `DEFAULT_REVISION`, `TOOLS_FILE_COUNT`, and `TOOLS_TREE_SHA256` in
   `prepare.py`, and regenerate any dataset built from the old schemas.

Do not re-commit the JSON files.

## Parity notes

Ported bug-for-bug from EOG (do not "fix" here; see `verifier_engine.py` docstring):
SQL result extraction/comparison semantics, the verifier name-collapse, skipping verifiers
with unknown `gym_name`, judge prompts, and the model-observation encoding of tool results.
EOG judges `response_check` with the policy model itself; `judge_model_server` defaults to
`policy_model` to match.

## Licensing and data provenance

- **Upstream benchmark**: [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
  (ServiceNow) is licensed **Apache 2.0**; `verifier_engine.py` ports its scoring semantics
  and the parity fixtures were generated by running the original engine.
- **Task data**: the full benchmark dataset is the public HuggingFace dataset
  [`ServiceNow-AI/EnterpriseOps-Gym`](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym),
  downloaded at `benchmarks/enterpriseops/prepare.py` time and never committed.
- **Tool schemas**: `data/tools/*.json` are `tools/list` snapshots captured from the
  public EOG Docker containers by `snapshot_tools.py`. They are hosted at
  [`nvidia/NeMo-Gym-EnterpriseOps-Assets`](https://huggingface.co/datasets/nvidia/NeMo-Gym-EnterpriseOps-Assets)
  (Apache 2.0, attributing ServiceNow) and fetched by `prepare.py` against a pinned
  revision and checksum, so the bytes are reproducible without committing them.
- **Committed data** is limited to `data/example.jsonl` (5 CSM sample tasks),
  `data/example_metrics.json` and `data/example_rollouts.jsonl` (required by the
  resources-server data validation), and one hand-authored synthetic hybrid task
  (`data/hybrid_synthetic.jsonl`, written with LLM assistance against live container
  schemas). Tool schemas are baked into those rows, so they stay byte-reproducible
  independently of the hosted snapshots.
- **This integration**: Apache 2.0, same as NeMo Gym.

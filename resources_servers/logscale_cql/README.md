## LogScale CQL Benchmark — Gym Resources Server

Evaluates LLM ability to generate CrowdStrike LogScale (CQL) queries from natural language. Queries run against a live LogScale instance with synthetic Falcon telemetry data.

**162 verified Q&A pairs** | **3-dimensional scoring** (validity, semantic, execution) | **Multi-repo support** (sandbox, base_sensor, detections)

### Baseline Results

| Model | Prompt | Reward | Validity | Semantic | Execution |
|---|---|---|---|---|---|
| Nemotron Super v3 | baseline | 0.5181 | 0.4074 | 4.06 | 1.68 |
| Claude Opus 4.6 | baseline | 0.5128 | 0.4012 | 4.12 | 1.57 |
| GPT-OSS 120B | baseline | 0.5033 | 0.3889 | 4.06 | 1.55 |
| Nemotron Nano 30B A3B | baseline | 0.4848 | 0.3395 | 4.09 | 1.48 |
| Nemotron Super v3 | combo | ~0.75* | ~0.75* | — | — |

\* Combo prompt (with few-shot examples) estimated from 20-question sample. Full run pending.

---

### Quick Start (copy-paste)

Everything below runs from the repo root. Total setup time: ~5 min (+ ~5 min first-time image pull).

#### Step 0: Prerequisites

```bash
# Clone the repo and create a venv
git clone https://gitlab-master.nvidia.com/asteiner/nemo-gym.git
cd nemo-gym
git checkout asteiner/logscale-cql-v2
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Set your API keys
export INFERENCE_API_KEY=<your-nvidia-inference-api-key>
export LLM_MODEL=nvidia/nvidia/Nemotron-3-Nano-30B-A3B   # or any model
export POLICY_BASE_URL=https://inference-api.nvidia.com/v1
export POLICY_API_KEY="${INFERENCE_API_KEY}"
export POLICY_MODEL_NAME="${LLM_MODEL}"
```

#### Step 1: Download the datasets (one-time)

The synthetic telemetry CSVs are on the GitLab registry. Download and unzip:

```bash
# Option A: via the Gym CLI (needs env.yaml with mlflow creds)
ng_download_dataset_from_gitlab +dataset_name=logscale_cql +version=0.0.2 \
  +artifact_fpath=datasets/datasets_merged.zip \
  +output_fpath=resources_servers/logscale_cql/data/datasets_merged.zip

# Option B: download manually from GitLab MR #205 or the MLflow artifacts page

# Then unzip
cd resources_servers/logscale_cql/data
unzip datasets_merged.zip -d datasets_v2
cd ../../..
```

You should now have 8 CSV files in `resources_servers/logscale_cql/data/datasets_v2/`.

#### Step 2: Start LogScale + ingest data (automated)

`setup.sh` handles everything: starts the container, waits for healthy, installs the license, discovers repos, and ingests the datasets.

**Apptainer (default — HPC/Slurm/DGX, no Docker needed):**

```bash
cd resources_servers/logscale_cql
bash setup.sh
```

**Docker (local dev / macOS):**

```bash
cd resources_servers/logscale_cql
CONTAINER_RUNTIME=docker bash setup.sh
```

Wait for `=== Setup complete ===`. LogScale is now running on `http://localhost:8080` with data loaded.

> **No Apptainer?** Install without root: `conda install -c conda-forge apptainer`

#### Step 3: Run the benchmark

```bash
# Back to repo root
cd ../..
source .venv/bin/activate

# Start the Gym servers (runs in background)
ng_run \
  "+config_paths=[resources_servers/logscale_cql/configs/logscale_cql.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
  "+default_host=0.0.0.0" \
  "+policy_base_url=$POLICY_BASE_URL" \
  "+policy_api_key=$POLICY_API_KEY" \
  "+policy_model_name=$POLICY_MODEL_NAME" &

# Wait for "All servers ready" then run rollouts
MODEL_NAME=$(echo "$LLM_MODEL" | sed 's|.*/||')

ng_collect_rollouts +agent_name=logscale_cql_simple_agent \
  +input_jsonl_fpath=resources_servers/logscale_cql/data/all_questions.jsonl \
  +output_jsonl_fpath=resources_servers/logscale_cql/data/benchmark_rollouts_${MODEL_NAME}.jsonl \
  +num_samples_in_parallel=5
```

Results are saved to `resources_servers/logscale_cql/data/benchmark_rollouts_${MODEL_NAME}.jsonl`.

---

### How It Works

```
Question → LLM generates CQL → LogScale executes → Judge scores
                                                    ├── Validity (0/1): is CQL syntax valid?
                                                    ├── Semantic (1-5): does query match intent?
                                                    └── Execution (1-5): are results correct?

Reward = (validity + semantic/5 + execution/5) / 3
```

1. **162 Q&A pairs** — hand-written CQL questions with verified reference queries
2. **Synthetic data** — 8 merged CSVs (~1100 events) with shared entity pools
3. **Multi-repo** — data routed to `sandbox`, `base_sensor`, `detections` repos based on `#repo` tag
4. **Live execution** — model generates CQL, LogScale executes it, LLM judge scores the results

### Prompt Variants

| Variant | File | What it includes |
|---|---|---|
| **Baseline** | `all_questions.jsonl` | Field list + event types + CQL syntax reference |
| **Combo** | `all_questions_combo.jsonl` | Baseline + community CQL examples + few-shot Q&A pairs |

To run the combo variant, replace `all_questions.jsonl` with `all_questions_combo.jsonl` in the `ng_collect_rollouts` command.

### Container Requirements

LogScale runs as a single container (Docker or Apptainer):
- **Ports**: 8080 (HTTP API), 9200 (Elastic) — must be free on the host
- **Memory**: ~4GB recommended
- **Storage**: writable `/data` directory (Kafka + Humio data)
- **License**: embedded in `setup.sh` — no manual step needed
- **GeoLite DBs** (optional): mount `IpLocationDb.mmdb` and `IpAsnDb.mmdb` for `ipLocation()`/`asn()` queries

**Apptainer caveats** (tested on Ubuntu 24.04 + Apptainer 1.4.5):
- Use `apptainer run` (not `instance start`) — `instance start` does not execute the Docker CMD
- Use `APPTAINERENV_` prefix for env vars — `--env` flags don't propagate to child processes
- First run pulls and converts the Docker image to SIF (~5-10 min, then cached)

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_API_KEY` | (required) | API key for LLM judge + policy model |
| `POLICY_BASE_URL` | (required) | Policy model API URL (must end with `/v1`) |
| `POLICY_MODEL_NAME` | (required) | Model to benchmark |
| `JUDGE_MODEL` | `aws/anthropic/bedrock-claude-opus-4-6` | LLM judge model |
| `CONTAINER_RUNTIME` | `apptainer` | `apptainer` or `docker` (also settable in config YAML) |

### Files

| File | Description |
|---|---|
| `setup.sh` | One-command setup: container + license + data ingestion |
| `app.py` | Gym resources server (FastAPI + Ray) |
| `container_engine.py` | LogScale client — executes CQL, validates syntax |
| `data/ingest_datasets.py` | Ingests CSV datasets into LogScale |
| `configs/logscale_cql.yaml` | Server + agent + dataset config |
| `prompt_templates/` | Semantic and execution judge prompts |
| `test_standalone.py` | Quick test (no Gym needed): `python test_standalone.py --no-judge` |

### Datasets

8 merged CSVs covering 55 event types (1139 rows total). Download `datasets_merged.zip` from the GitLab registry (`logscale_cql` v0.0.2).

| File | Rows | Event Types |
|---|---|---|
| `process_events.csv` | 212 | ProcessRollup2 |
| `auth_events.csv` | 205 | UserLogon, UserLogonFailed, UserIdentity, SSO |
| `misc_events.csv` | 302 | ASEP, Firewall, Browser, Channel, RFM, etc. |
| `sensor_events.csv` | 131 | Heartbeat, AgentOnline, OsVersionInfo |
| `network_events.csv` | 106 | NetworkConnectIP4, NetworkListenIP4 |
| `dns_http_events.csv` | 87 | DnsRequest, HttpRequest/Response |
| `file_events.csv` | 66 | FileOpen, FileWritten, FileDeleted |
| `detection_events.csv` | 30 | Detection alerts (#repo=detections) |

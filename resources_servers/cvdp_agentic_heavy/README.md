# CVDP Agentic-Heavy Rollout Environment

This resource server adds NeMo-Gym runtime support for the public CVDP
agentic-heavy code-generation benchmark. It is scoped to benchmark rollout
execution only: no Trace2Skill, no skill evolution, and no commercial Xcelium
tool path.

Official public source: `nvidia/cvdp-benchmark-dataset` on Hugging Face. The
v1.1.0 agentic-heavy release includes:

- `cvdp_v1.1.0_agentic_heavy_code_generation.jsonl`
- public sanitized Git bundles under `cvdp_v1.1.0_agentic_heavy_code_generation_public/`

## Runtime Flow

1. `seed_session()` writes task `context_files` into a per-rollout sandbox.
2. The rollout agent calls constrained file and simulator tools against that sandbox.
3. Supported tools are `ls`, `cat`, `echo`, `edit`, `iverilog`, `vvp`, and `pwd`.
4. `verify()` writes hidden harness files, runs container verifier services, and
   returns reward `1.0` only when every service exits successfully.

## Leakage Boundary

The rollout model only receives `responses_create_params` and tool outputs.
During `seed_session()`, only `verifier_metadata.context_files` are written into
the sandbox. Hidden `harness_files` are written only inside `verify()`, after the
tool loop has ended. The official CVDP `patch` field is omitted by
`convert_to_gym.py` by default; pass `--include-solution-metadata` only for
offline analysis datasets that will not be used as model-visible rollout input.

## Convert The Official Dataset

Convert directly from the public Hugging Face JSONL and download public bundles
on demand:

```bash
python resources_servers/cvdp_agentic_heavy/scripts/convert_to_gym.py \
  --input hf://cvdp_v1.1.0_agentic_heavy_code_generation.jsonl \
  --output resources_servers/cvdp_agentic_heavy/data/gym_cvdp_v1.1.0_agentic_heavy_code_generation.jsonl \
  --repos-cache .cache/cvdp_repos_cache \
  --download-bundles
```

By default the converter removes commercial EDA/Xcelium-only harness services
and skips EDA-only rows. It also omits reference patch metadata.

## Customer Scripts

| Script | Purpose | Requires model key? |
|---|---|---:|
| `scripts/convert_to_gym.py` | Convert official Hugging Face CVDP rows to NeMo-Gym JSONL | No |
| `scripts/smoke_test_cvdp.py` | Validate converted JSONL leakage boundary and run verifier-container smoke tests | No |
| `scripts/build_apptainer_images.py` | Prebuild visible tool and hidden verifier SIF images for Apptainer rollouts | No |
| `scripts/run_cvdp_rollout.py` | Convert if needed, start servers, collect rollouts, and stop servers | Yes |

## Smoke-Test Without A Model Key

Validate the verifier path before spending model tokens:

```bash
python resources_servers/cvdp_agentic_heavy/scripts/smoke_test_cvdp.py --backend docker
```

To validate a converted CVDP JSONL and run one real hidden verifier harness as a
no-op attempt:

```bash
python resources_servers/cvdp_agentic_heavy/scripts/smoke_test_cvdp.py \
  --backend docker \
  --converted-jsonl resources_servers/cvdp_agentic_heavy/data/gym_cvdp_v1.1.0_agentic_heavy_code_generation.jsonl \
  --run-official-noop
```

The official no-op attempt may return reward `0.0`; that is normal because no
agent fix was applied.

## Apptainer Image Prebuild

For Apptainer runs, prebuild the visible tool and hidden verifier SIFs into the
same cache path used by the resource server:

```bash
python resources_servers/cvdp_agentic_heavy/scripts/build_apptainer_images.py \
  --converted-jsonl resources_servers/cvdp_agentic_heavy/data/gym_cvdp_v1.1.0_agentic_heavy_code_generation.jsonl \
  --sif-cache-dir .cache/sif
```

This prepares the visible tool container (`oss_sim_image`) and all verifier
images discovered from hidden harness compose files. Dockerfile-derived verifier
images are built with the same base-image-plus-post-commands logic used by the
resource server.

## Run A Full Rollout

Set owner-provided model endpoint values in the shell that launches the rollout:

```bash
export LLM_BASE_URL="https://api.example.com/v1"
export LLM_API_KEY="replace-with-your-key"
export ROLLOUT_MODEL="your-model-name"
```

The guarded launcher converts the dataset if needed, starts NeMo-Gym servers,
collects rollouts, and shuts servers down:

```bash
python resources_servers/cvdp_agentic_heavy/scripts/run_cvdp_rollout.py \
  --backend docker \
  --limit 2 \
  --output-jsonl results/cvdp_agentic_heavy_rollouts_smoke.jsonl
```

For production Apptainer runs, use the prebuilt SIF cache and omit `--limit`:

```bash
python resources_servers/cvdp_agentic_heavy/scripts/run_cvdp_rollout.py \
  --backend apptainer \
  --sif-cache-dir .cache/sif \
  --output-jsonl results/cvdp_agentic_heavy_rollouts.jsonl
```

Manual two-terminal flow:

```bash
ng_run \
  '+config_paths=[resources_servers/cvdp_agentic_heavy/configs/cvdp_agentic_heavy.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  '+policy_base_url=${oc.env:LLM_BASE_URL}' \
  '+policy_api_key=${oc.env:LLM_API_KEY}' \
  '+policy_model_name=${oc.env:ROLLOUT_MODEL}' \
  +cvdp_agentic_heavy.resources_servers.cvdp_agentic_heavy.execution_backend=docker
```

Then collect rollouts from another terminal:

```bash
ng_collect_rollouts \
  +agent_name=cvdp_agentic_heavy_agent \
  '+config_paths=[resources_servers/cvdp_agentic_heavy/configs/cvdp_agentic_heavy.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  '+policy_base_url=${oc.env:LLM_BASE_URL}' \
  '+policy_api_key=${oc.env:LLM_API_KEY}' \
  '+policy_model_name=${oc.env:ROLLOUT_MODEL}' \
  +input_jsonl_fpath=resources_servers/cvdp_agentic_heavy/data/gym_cvdp_v1.1.0_agentic_heavy_code_generation.jsonl \
  +output_jsonl_fpath=results/cvdp_agentic_heavy_rollouts.jsonl \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

## Gym JSONL Shape

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "<tool and rollout instructions>"},
      {"role": "user", "content": "<task prompt>"}
    ],
    "tools": ["ls", "cat", "echo", "edit", "iverilog", "vvp", "pwd"]
  },
  "verifier_metadata": {
    "task_id": "cvdp_agentic_heavy_<repo>_<n>",
    "categories": ["cid016", "hard"],
    "difficulty": "hard",
    "context_files": {"path/to/file.sv": "..."},
    "harness_files": {"docker-compose.yml": "...", "src/.env": "..."},
    "origin": {"repo": "...", "commit": "..."}
  }
}
```

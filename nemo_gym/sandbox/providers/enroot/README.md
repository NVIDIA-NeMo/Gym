# Enroot Sandbox Provider

A [NeMo Gym](../../../../README.md) sandbox provider backed by the local
[enroot](https://github.com/NVIDIA/enroot) CLI.

## Quick start

**Step 1 — start the NeMo Gym container:**

```bash
TAG=latest_tag_here
docker run --gpus all -it --rm \
    --privileged \
    --shm-size=15g \
    -p 1024:1024 \
    --entrypoint bash \
    nvcr.io/nvidia/nemo-gym:$TAG
```

**Step 2 — pre-convert container images to squashfs (run once per image):**

```python
import hashlib, subprocess
from pathlib import Path

images = [
    "docker.io/swebench/sweb.eval.x86_64.django_1776_django-10973:latest",
    "docker.io/swebench/sweb.eval.x86_64.pylint-dev_1776_pylint-4551:latest",
    "docker.io/swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-8595:latest",
    "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-20916:latest",
    "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-14141:latest",
]

sqsh_dir = Path("/tmp/enroot_sqshs")
sqsh_dir.mkdir(parents=True, exist_ok=True)

for image in images:
    key = hashlib.sha256(image.encode()).hexdigest()[:16]
    out = sqsh_dir / f"{key}.sqsh"
    if not out.exists():
        # enroot treats docker.io as a literal registry path; strip it so the
        # real Hub API (registry-1.docker.io) is used instead
        enroot_ref = image.removeprefix("docker.io/")
        subprocess.run(["enroot", "import", "-o", str(out), f"docker://{enroot_ref}"], check=True)
        print(f"cached {image} → {out}")
    else:
        print(f"already cached: {image}")
```

**Step 3 — start the env stack:**

The example below uses Qwen3-27B-FP8 — swap `--model` and the
`vllm_serve_kwargs` overrides for any other model.

```bash
gym env start \
    --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
    --config nemo_gym/sandbox/providers/enroot/configs/enroot.yaml \
    +skip_venv_if_present=true \
    --model-type local_vllm_model \
    --model Qwen/Qwen3.6-27B-FP8 \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=2' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY=strict' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.enable_auto_tool_choice=true' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tool_call_parser=qwen3_coder' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.reasoning_parser=qwen3' \
    '++policy_model.responses_api_models.local_vllm_model.uses_reasoning_parser=true' \
    '++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.quantization=fp8' \
    '++sandbox.enroot.create.sqsh_cache_dir=/tmp/enroot_sqshs' \
    '++sandbox.enroot.create.bypass_entrypoint=false'
```

**Step 4 — run evaluation:**

```bash
gym eval run --no-serve \
    --agent mini_swe_agent_2 \
    --input responses_api_agents/mini_swe_agent_2/data/example.jsonl \
    --output results/mini_swe_agent_2_v2.jsonl \
    --limit 5 \
    --num-repeats 1 \
    --temperature 0.5 \
    --max-output-tokens 2048
```

## Selecting and configuring the provider

The provider config is a single-key mapping: `{"enroot": {<kwargs>}}`. The kwargs
are grouped into three optional sections, each of which accepts a plain mapping
(e.g. from Hydra YAML) or the corresponding dataclass. A ready-to-use config is
shipped at [`configs/enroot.yaml`](./configs/enroot.yaml).

```yaml
enroot:
  create:
    base_dir: null            # provider-scoped enroot home (auto = per-user /tmp dir)
    data_path: null           # ENROOT_DATA_PATH override
    cache_path: null          # ENROOT_CACHE_PATH override
    runtime_path: null        # ENROOT_RUNTIME_PATH override
    sqsh_cache_dir: null      # where imported .sqsh images are cached
    rw: true
    remap_root: false
    start_timeout_s: 600
  exec:
    default_timeout_s: 180
    concurrency: 32
  probe:
    deadline_s: 180
    stable_count: 2
```



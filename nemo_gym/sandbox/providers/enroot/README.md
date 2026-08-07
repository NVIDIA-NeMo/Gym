# Enroot Sandbox Provider

A [NeMo Gym](../../../../README.md) sandbox provider backed by the local
[enroot](https://github.com/NVIDIA/enroot) CLI.

## Requirements

- The **`enroot` binary** must be installed and on `PATH`. The provider does
  **not** auto-install it; constructing the provider raises `RuntimeError` if it
  is missing. See the
  [enroot install guide](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).
- Unprivileged user namespaces must be enabled (`unprivileged_userns_clone=1`),
  which enroot needs for rootless operation.
- A container **image**: a local `.sqsh` squashfs, an enroot import URI, or a
  Docker image reference such as `ubuntu:22.04` or `nvcr.io/nvidia/pytorch:24.01`.
- **All enroot calls must run as the same OS user.** `enroot exec` re-enters
  namespaces owned by the launching user, so the provider (a single process)
  must own the whole container lifecycle. Switching to another *container* user
  is done with `su` inside the container.

## Quick start

**Step 1 — set up the main virtual environment:**

```bash
uv venv && uv sync --extra dev
source .venv/bin/activate
```

**Step 2 — set up the agent virtual environment (run once):**

The `mini_swe_agent_2` server manages its own venv. On Ubuntu 24.04 or newer
(glibc ≥ 2.39):

```bash
cd responses_api_agents/mini_swe_agent_2
uv venv --seed --python 3.12 .venv
uv pip install -r requirements.txt
cd ../..
```

On Ubuntu 22.04 (glibc < 2.39), `openshell` (pulled in by the `[sandbox]` extra)
has no compatible wheel and no source distribution. Since `openshell` is only
needed for the OpenShell provider — not for enroot — install the deps directly
without the `[sandbox]` extra. Pass `--python .venv/bin/python` so uv targets
the server venv rather than any currently activated venv:

```bash
cd responses_api_agents/mini_swe_agent_2
uv venv --seed --python 3.12 .venv
uv pip install --python .venv/bin/python -e "../../[dev]" mini-swe-agent==2.1.0 swebench==4.1.0
cd ../..
```

**Step 3 — pre-convert container images to squashfs (run once per image):**

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

**Step 4 — start the env stack:**

The example below uses Qwen3-27B-FP8 — swap `--model` and the
`vllm_serve_kwargs` overrides for any other model. The `+skip_venv_if_present=true`
flag reuses the agent venv built in step 2:

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

**Step 5 — run evaluation:**

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

## Python API

```python
from nemo_gym.sandbox.api import Sandbox
from nemo_gym.sandbox.providers import SandboxSpec

spec = SandboxSpec(
    image="ubuntu:22.04",              # or "nvcr.io/nvidia/pytorch:24.01", or "/path/to/image.sqsh"
    workdir="/sandbox",
    env={"GREETING": "hello"},
    files={"/sandbox/input.txt": "some seed content"},
)

with Sandbox({"enroot": {}}, spec) as sandbox:
    sandbox.start()

    result = sandbox.exec("echo $GREETING && cat /sandbox/input.txt")
    print(result.return_code, result.stdout)

    sandbox.upload("./local_script.sh", "/sandbox/script.sh")
    sandbox.download("/sandbox/result.txt", "./result.txt")
# leaving the `with` block kills the container and cleans up
```

> **Lifecycle contract:** download anything you want to keep *before* the sandbox
> is stopped. Stopping is teardown — it kills the container init, removes the
> rootfs (`enroot remove -f`), and deletes the host staging directory.

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

### `create` — `EnrootCreateConfig`

| Field | Default | Meaning |
|---|---|---|
| `mount_point` | `/sandbox` | Absolute path inside the container where the host staging dir is mounted. Powers the file-transfer fast path. |
| `base_dir` | auto | Base dir for the provider-scoped enroot paths when the specific paths below are unset. Defaults to `${TMPDIR}/nemo-gym-enroot-<uid>`. |
| `data_path` / `cache_path` / `runtime_path` | env → base | Pinned `ENROOT_DATA_PATH` / `ENROOT_CACHE_PATH` / `ENROOT_RUNTIME_PATH`, passed to **every** enroot subprocess (see below). |
| `sqsh_cache_dir` | `<base>/sqsh` | Where imported squashfs images are cached (keyed by image name). |
| `rw` | `true` | Start the container with a writable root filesystem (`--rw`). |
| `remap_root` | `false` | Remap the launching user to root inside the container (`--root`). |
| `init_command` | `while true; do sleep 86400; done` | The long-lived init keeping the container alive between `exec` calls (portable across busybox/coreutils). |
| `import_timeout_s` | `1800` | Max seconds for `enroot import` (image pull/convert). |
| `create_timeout_s` | `600` | Max seconds for `enroot create` (rootfs unpack). |
| `start_timeout_s` | `600` | Max seconds to wait for the container init PID to appear. |
| `start_poll_s` | `0.5` | Polling interval while waiting for the init PID. |
| `extra_import_args` / `extra_create_args` / `extra_start_args` | `[]` | Extra raw flags appended to the respective enroot command. |

### `exec` — `EnrootExecConfig`

| Field | Default | Meaning |
|---|---|---|
| `default_timeout_s` | `180` | Default per-command timeout when the caller doesn't pass one. |
| `default_mounts` | `[]` | Extra `-m src:dst` mounts added at container start. |
| `extra_exec_args` | `[]` | Extra raw flags appended to every `enroot exec`. |
| `concurrency` | `32` | Upper bound on concurrent `enroot` subprocesses (shared semaphore). |

### `probe` — `EnrootProbeConfig`

Same shape as the apptainer provider's probe. The default probe writes to and
reads back from `/sandbox`, so a returned sandbox is guaranteed to have a live,
writable staging mount. Set `command: null` to skip the probe.

### Relevant `SandboxSpec` fields

| Field | Used for |
|---|---|
| `image` | `.sqsh` path, enroot URI, or Docker reference. Required. |
| `env` | Passed as `-e KEY=VALUE` at start and re-applied on every `exec`. |
| `workdir` | Default working directory for `exec` (applied as a `cd` prefix). |
| `files` | Seed files written into the sandbox at `start()` (via the API's `upload`). |
| `resources.gpu` | Mapped to `NVIDIA_VISIBLE_DEVICES` (see below). `cpu`/`memory_mib`/`disk_gib` are **ignored with a warning**. |
| `provider_options` | `mounts`: a `"src:dst[:type:opts]"` string or list of enroot fstab entries — extra per-sandbox mounts added at start. |
| `ttl_s` | **Not supported** — ignored with a warning. Tear down via `stop()`/`close()`. |


## Limitations

- **No `ttl_s`.** enroot has no native auto-expiry; the field is ignored. Manage
  lifetime with `stop()` / `close()`.
- **No CPU/memory enforcement standalone.** Use pyxis/Slurm for cgroup limits.
- **Root inside the container** requires `create.remap_root: true`; without it,
  `exec(..., user="root")` runs as the launching user.
- **Same-user requirement.** The provider process must own the container for
  `enroot exec` to work; cross-user exec is not supported.
- **Runtime-failure detection is heuristic** — it keys off stderr markers, so a
  user command whose own output contains `[ERROR]` could be misclassified.



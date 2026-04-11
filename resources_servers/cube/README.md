# CUBE resources server

OpenEnv-style adapter: **YAML** chooses the benchmark via **`environment`** (plus OSWorld-specific fields). The cube **resources** server keeps **`entrypoint: app.py`**; on startup, **`app.py`** calls **`bootstrap.maybe_install_environment_extras()`** so optional wheels from `environments/<environment>/requirements.txt` install before importing **`osworld_cube`**. Without NeMo-Gym env vars (e.g. ad-hoc `python app.py`), that hook skips extras. **`domain`** is the NeMo Gym resources-server field (`config_types.Domain`, e.g. `other`) for dataset naming—do not overload it for CUBE.

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Cube resources server `entrypoint`: under `__main__`, runs `bootstrap.maybe_install_environment_extras()` then `CubeResourcesServer.run_webserver()` |
| `bootstrap.py` | `maybe_install_environment_extras()` — optional `uv pip install -r environments/<environment>/…` (invoked from `app.py`, not a separate YAML entrypoint) |
| `schemas.py` | Shared HTTP contract (`Cube*`) |
| `adapters.py` | CUBE `Observation` / tools → NeMo-Gym |
| `registry.py` | Maps YAML `config.environment` → `CubeEnvironmentBase` (`register_domain` / `instantiate_domain`) |
| `environments/base.py` | `CubeEnvironmentBase` ABC (`ensure_loaded`, `warm_on_startup`, `empty_reset_obs_detail`) |
| `environments/osworld/` | `environment: osworld` — `env.py` (`OSWorldEnvironment`, `ensure_osworld_tasks` → `osworld_cube`), `requirements.txt` (git `osworld-cube`, Pillow) |
| `server.py` | `CubeResourcesServer` — HTTP + CUBE task loop; delegates load/warmup to `CubeEnvironmentBase` |
| `host_tools.py` | Host binary checks (e.g. ``qemu-img`` for QEMU VM backend) before benchmark setup |
| `requirements.txt` | Base venv for `ng_run`: editable `nemo-gym[dev]` only; per-`environment` extras install from `app.py` via `bootstrap` |

## Configuration

Top-level fields on `CubeResourcesServerConfig`:

- **`environment`** — `osworld` today; more values can be added by subclassing `CubeEnvironmentBase`, calling `register_domain` in `registry.py`, and setting YAML `environment` to that name.
- **`domain`** — Required NeMo Gym field (`math`, `coding`, `other`, …); see `nemo_gym.config_types.Domain`. Use e.g. **`other`** for CUBE OSWorld unless you have a closer category.
- **`eager_benchmark_init`**, **`eager_osworld_vm_warmup`**, **`eager_osworld_warmup_task_idx`** — see [Eager benchmark init and VM warmup](#eager-benchmark-init-and-vm-warmup).

Additional keys are **domain-specific** (merged via `extra="allow"`), e.g. OSWorld’s `vm_backend_class`, `test_set_name`, `tasks_file`, `use_som`, `action_space`, etc.

Example: `configs/cube_osworld.yaml`.

## `ng_run` and `ng_collect_rollouts` (OSWorld / CUBE)

Run everything from the **NeMo Gym repository root** (the directory that contains the top-level `pyproject.toml`). Use the **root** virtualenv (`uv sync --extra dev` and `source .venv/bin/activate`), not only the per-server venv under `resources_servers/cube/.venv`—`ng_run` spawns servers with their own envs.

Put API keys and policy defaults in repo-root **`env.yaml`** (see the main [README](../../README.md) quick start). You need a working **policy** block (`policy_base_url`, `policy_api_key`, `policy_model_name`, or equivalent) because `configs/cube_osworld.yaml` wires the agent to `policy_model` from the OpenAI model config.

### Start servers (`ng_run`)

This stack is **three** child servers (CUBE resources, `cube_agent`, OpenAI policy) plus the head process—wait until the log reports all servers ready. The first run is often slow: **`eager_benchmark_init`** plus **`eager_osworld_vm_warmup`** (defaults) load the benchmark, then **boot and tear down one disposable QEMU task** before the cube server accepts HTTP—so cold VM start happens during startup, not on the first client request. `uv` also builds `resources_servers/cube/.venv`.

**`2 / 3 servers ready` with `cube_osworld_resources_server` missing:** Until eager startup finishes, the cube process **does not bind its HTTP port** yet. The other two children (agent + policy) may already respond, so the head poller shows **2 / 3** for a long time—that is **normal** during image download / QEMU warmup (often many minutes on first run), not necessarily a hang. NeMo-Gym sets Python logging to **`INFO`** before building each child app (and to **`DEBUG`** when that server has **`debug: true`**), so **`INFO`** lines from the cube server and OSWorld bootstrap show during eager init. **`debug: true`** in `cube_osworld.yaml` also sets uvicorn’s **`log_level`** to debug after listen (access lines, etc.).

**`KeyboardInterrupt` (and sometimes `CancelledError`) on the cube log:** With **`eager_benchmark_init: true`**, heavy startup runs **before** uvicorn starts, so **Ctrl+C** during image download or VM warmup usually stops with a single **`KeyboardInterrupt`**—no Starlette lifespan involved. After the server is listening, or with **`eager_benchmark_init: false`**, uvicorn/Starlette shutdown can still log **`CancelledError`** in some versions; that is interruption noise, not a separate failure mode.

To **listen for HTTP sooner** (at the cost of moving full QEMU boot to the first **`/seed_session`**), set **`eager_osworld_vm_warmup: false`** in `configs/cube_osworld.yaml`.

```bash
source .venv/bin/activate   # repo root

ng_run "+config_paths=[resources_servers/cube/configs/cube_osworld.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

**Ray:** Child processes call `ray.init()` and join the address in the merged config. If `env.yaml` sets **`ray_head_node_address`** to a **remote** cluster, it must be reachable (correct host, VPN, etc.); otherwise you get **GCS connection timeouts** and servers exit early. For **single-machine** local runs, **omit** `ray_head_node_address` so NeMo-Gym starts a local Ray cluster and injects the address for children.

**Stop:** **Ctrl+C** in the `ng_run` terminal (child servers and Ray are shut down by the runner).

### Collect rollouts (`ng_collect_rollouts`)

Use a **second** terminal with the same repo root and venv. **`ng_run` must still be running.`** Pass the **same** `config_paths` as above so the client resolves the head server and the same agent/model wiring.

The **agent name** passed to `+agent_name` must match the **top-level server key** for the agent in `cube_osworld.yaml`: **`cube_osworld_agent`** (not `cube_agent`).

```bash
source .venv/bin/activate

ng_collect_rollouts \
  "+config_paths=[resources_servers/cube/configs/cube_osworld.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
  +agent_name=cube_osworld_agent \
  +input_jsonl_fpath=resources_servers/cube/data/osworld/example.jsonl \
  +output_jsonl_fpath=cube_osworld_rollouts.jsonl \
  +limit=1 \
  +num_samples_in_parallel=1 \
  +debug=true
```

Use **real paths** in `+config_paths`, `+input_jsonl_fpath`, and `+output_jsonl_fpath` (as above). Copy-pasted placeholders such as `...` or `[...]` are treated as literal filenames and produce **`FileNotFoundError`**.

The **`+debug=true`** flag prints a timestamped line when each **`POST /run`** starts and when it finishes (with duration), refreshes the progress bar more often, enables **`global_aiohttp_client_request_debug`**, and turns on DEBUG logging—useful when a rollout looks hung (often the agent is still in `seed_session`, policy calls, or QEMU).

**`+debug_each_step=true`** adds **`debug_each_step`** to each rollout **`POST /run`** body. **`cube_agent`** attaches **`debug_step_events`** at the **top level** of the verify payload (not inside the OpenAI-shaped **`response`** object) so it survives HTTP JSON round-trip; **`ng_collect_rollouts`** prints one line per step on **your terminal**, then strips that field before writing JSONL. The agent server still logs to its own stdout if you look there. Combine with **`+debug=true`** for per-`/run` timing and HTTP debug. Other agents ignore the extra field.

OSWorld rollouts are heavy (QEMU per task). Keep **`+num_samples_in_parallel=1`** (or low) unless you know the host can run multiple VMs.

NeMo-Gym also writes **`cube_osworld_rollouts_materialized_inputs.jsonl`** next to the output (materialized rows after merges/repeats). Use **`+resume_from_cache=true`** to skip inputs already processed when re-running the same output path.

Useful overrides (Hydra):

```bash
ng_collect_rollouts +help=true
```

Examples: `+debug=true`, `+debug_each_step=true`, `+num_repeats=4`, `+num_repeats_add_seed=true`, `"+responses_create_params={max_output_tokens: 8192, temperature: 0.7}"`.

Input JSONL lines must include **`task_idx`** (index into the benchmark’s task list for your `test_set_name`) and **`responses_create_params.input`** (developer + user messages prepended before the first environment observation). The bundled **`data/osworld/example.jsonl`** has two rows aligned with **`TEST_SMALL`** and the OSWorld commit pinned in `osworld_cube` (**`e695a10`**): **`task_idx` 0** is the first Chrome task (Bing as default search engine), **`task_idx` 1** is the second (Amazon cookies / privacy). Add more lines for other indices or switch `test_set_name` / `tasks_file` in YAML as needed.

### Policy `POST /v1/responses` returns 500 / cube_agent 502

The first model call after `seed_session` sends a **large** payload (~screenshots + tools). If **`policy_model`** returns HTTP 500, check the **`(policy_model)`** server log in the `ng_run` terminal and the JSON error from OpenAI (after recent NeMo-Gym versions, **`cube_agent`** logs the policy response body on failure).

Typical causes:

- **Context length** — request exceeds the model’s window; try a **vision-capable** model with a larger context, or reduce observation size (e.g. YAML / CUBE settings if available).
- **Unsupported input** — use a model that supports **multimodal** Responses / image inputs for OSWorld.
- **Transient OpenAI 500s** — NeMo-Gym’s OpenAI client retries a few times; repeated failures surface a **`RuntimeError`** with the **last error body** from the API.

## Eager benchmark init and VM warmup

YAML field **`eager_benchmark_init`** (default `true`): **before** uvicorn binds the port, run `ensure_task_configs()` so OSWorld clones metadata, resolves tasks, and downloads base VM images as needed—not deferred to the first `POST /seed_session`.

YAML fields **`eager_osworld_vm_warmup`** (default `true`) and **`eager_osworld_warmup_task_idx`** (default `0`): after the benchmark is loaded, create **one** disposable `OSWorldTask`, call **`reset()`** (QEMU launch, snapshot restore, setup scripts, stabilization wait), then **`close()`**. That forces a full **VM boot** while startup is still in progress, so by the time NeMo-Gym marks the cube server “ready,” QEMU and the guest have already completed a cold start once. Failures surface at boot instead of mid-rollout.

**Note:** `osworld-cube` still creates a **new** task and **new** VM for every `/seed_session`, so the first **real** episode pays boot cost again; warmup mainly validates the stack, warms caches, and shifts “is QEMU working?” errors to server startup. Set **`eager_osworld_vm_warmup: false`** to skip the disposable boot (faster dev, first request pays full cold start).

Set **`eager_benchmark_init: false`** to defer **all** of the above until the first `seed_session` (HTTP comes up immediately).

## Dependencies

Use **`requirements.txt`** in this directory: `ng_run` runs **`uv pip install -r requirements.txt`** here to populate `resources_servers/cube/.venv` (editable **`nemo-gym[dev]`** from the repo root, `-e … @ ../../`). **Heavy extras** for each **`environment`** live under **`environments/<environment>/requirements.txt`**. On process start, **`app.py`** calls **`bootstrap.maybe_install_environment_extras()`**, which reads **`NEMO_GYM_CONFIG_DICT`** and **`NEMO_GYM_CONFIG_PATH`** (set by NeMo-Gym before `python`), resolves **`environment`**, and runs **`uv pip install --python <venv> -r environments/<environment>/requirements.txt`** if that file exists and its content changed since last run (tracked by **`.venv/.cube_environment_extras.stamp`**). OSWorld extras: **`osworld-cube`** (git, **pinned SHA** in `environments/osworld/requirements.txt`) and **Pillow**. The pin avoids **`main`** layouts that declare **`[tool.uv.sources]`** monorepo paths (`../../cube-standard`, …)—those paths are invalid for a standalone `uv pip install` of `cubes/osworld-cube` and make resolution fail with a bogus `cube-standard` git subdirectory. Bump the SHA only after you verify a newer revision installs cleanly outside the full cube-harness checkout. If there is no checkout-level `pyproject.toml` two levels above this folder, the setup command substitutes **`nemo-gym`** from PyPI and drops lines containing **`../..`** from **`requirements.txt`** only (env-specific files are unchanged). Optional: **`NEMO_GYM_CUBE_BOOTSTRAP_VERBOSE=1`** for `uv -v`; **`NEMO_GYM_CUBE_BOOTSTRAP_LOG_LEVEL=DEBUG`** for bootstrap logging.

With **`skip_venv_if_present: true`** and an **existing** cube `.venv`, NeMo-Gym does not re-run the base **`requirements.txt`** install; the **`app.py`** bootstrap hook still runs when the server starts and will install or refresh extras unless the stamp matches. Delete **`.venv/.cube_environment_extras.stamp`** to force a reinstall of env-specific wheels. For **`python app.py`** without NeMo-Gym env vars (e.g. local debugging), bootstrap skips extras—install **`environments/<environment>/requirements.txt`** manually if imports fail.

Adapter-only tests from the repo root usually use the root environment and do not need this venv.

## Host prerequisites (`environment: osworld`, QEMU backend)

The default **`OSWorldQEMUVMBackend`** shells out to **`qemu-img`** and a QEMU system binary. Those come from your **OS package manager**, not `pip`. If you see:

`FileNotFoundError: ... 'qemu-img'`

install QEMU tools on the machine running `ng_run`, for example:

- **Debian / Ubuntu:** `sudo apt install qemu-system-x86 qemu-utils`
- **Fedora / RHEL:** `sudo dnf install qemu-img qemu-kvm` (names may vary slightly by release)

Verify the tools after install:

```bash
which qemu-img
qemu-img --version
```

### KVM (recommended for speed)

For reasonable VM startup and step latency you usually want **hardware acceleration via KVM**:

- **`/dev/kvm`** must exist and be readable by your user (often `crw-rw----+ root kvm`).
- Your user should be in the **`kvm`** group, then **log out and back in** (or reboot) so the new group applies. Until then, existing shells do not see the new group; run **`newgrp kvm`** in the terminal where you start **`ng_run`**, or open a new login session.
- If QEMU prints **`Could not access KVM kernel module: Permission denied`** / **`failed to initialize kvm: Permission denied`**, you are almost always missing **`kvm`** group membership for **this** process (or `/dev/kvm` permissions are wrong—check **`ls -la /dev/kvm`**). After **`sudo usermod -aG kvm "$USER"`**, confirm with **`groups`** that **`kvm`** appears before retrying.
- If NeMo-Gym / QEMU already runs **inside another VM**, enable **nested virtualization** in the outer hypervisor and ensure the guest kernel exposes KVM (Intel VT-x / AMD-V).

**Install KVM (host is bare metal or a VM with nested virt)**

- **Debian / Ubuntu**
  - `sudo apt install qemu-kvm qemu-system-x86 qemu-utils`
  - `sudo usermod -aG kvm "$USER"` then sign out and back in.
  - Optional: `sudo apt install virt-manager` if you want a GUI.
- **Fedora / RHEL / CentOS Stream**
  - `sudo dnf install @virtualization` or at minimum `qemu-kvm libvirt`.
  - `sudo usermod -aG kvm "$USER"` (group name is usually `kvm`).

Check that the module loaded and the device is there:

```bash
ls -la /dev/kvm
```

If `/dev/kvm` is missing, the CPU may not expose virtualization in BIOS/UEFI, the kernel module failed (`kvm`, `kvm_intel`, or `kvm_amd`), or you are in a cloud VM without nested/KVM support—in those cases QEMU may fall back to **TCG** (software emulation), which is much slower but can still work for light tests.

When the OSWorld adapter loads (eager server startup or first `seed_session`), NeMo-Gym runs **`require_qemu_img_if_qemu_backend`** if your YAML `vm_backend_class` name contains `QEMU`, so you get a clear error **before** image download / `task.reset()` instead of an opaque `FileNotFoundError` from `subprocess`. This does **not** run during adapter-only tests that never call `ensure_osworld_tasks`.

To skip the check (e.g. custom backend naming): set environment variable **`NEMO_GYM_CUBE_SKIP_QEMU_HOST_CHECK=1`**.

## Tests

```bash
uv run pytest resources_servers/cube/tests/test_adapters.py
```

## Agents

Rollout agents use `responses_api_agents/cube_agent` (same HTTP contract as `resources_servers.cube.schemas`).

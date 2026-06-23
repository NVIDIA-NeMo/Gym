<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `swe_env` verifier

The required, provider-neutral, **fresh-sandbox** verification entry point for the
decoupled SWE environment (issue #1249). `verify()` takes an agent's patch, grades
it in its **own fresh sandbox**, and returns a non-nullable `reward` (`1.0`/`0.0`,
masked infra failures = `reward=0.0` + `mask_sample`). It imports the reusable
[`responses_api_agents/swe_env`](../../responses_api_agents/swe_env) library
(harness recipes, parsing, sandbox providers, lifecycle) — so any agent can reuse
the same env over HTTP, or in-process via that library.

Sandbox providers (selected by config `sandbox_provider`):
- **`docker`** — runs the SWE-bench eval Docker images directly (no `.sif` needed).
- **`apptainer`** — runs `.sif` images (ports the legacy on-prem path).
- **`opensandbox`** — the #1377 k8s provider (flat families).

## Running the full SWE-bench Verified eval (gold-patch validation)

`scripts/run_swebench_verified.py` runs the **decoupled sandbox infra over SWE-bench
Verified**: for each instance it provisions the official SWE-bench Docker image
through the `swe_env` provider + lifecycle, applies the **gold** patch, runs the real
per-repo SWE-bench `eval_script`, and grades with the official `swebench` parser. A
gold run should resolve ~all instances and validates the provider/lifecycle at full scale.

### Setup
```bash
# extra deps for the driver (not needed by the server itself)
uv pip install swebench datasets
# docker (default provider) must be installed; for --provider apptainer, apptainer + uidmap too.
```
SWE-bench images are pulled automatically from Docker Hub (`swebench` namespace,
`sweb.eval.x86_64.<instance_id>` with `__`→`_1776_`). The full Verified set needs
**~120 GB+** of disk; the driver `docker rmi`s each image after grading to bound usage.
There are **no pre-built `.sif` files**; `--provider apptainer` converts each image on
the fly (`apptainer build docker-daemon://…`). Set `HF_HOME` to a writable dir if your
`~/.cache/huggingface` is not writable.

### Examples
```bash
# smoke: first 5 instances on docker
python resources_servers/swe_env/scripts/run_swebench_verified.py --limit 5

# FULL 500, 4 in parallel, incremental results (resumable log)
python resources_servers/swe_env/scripts/run_swebench_verified.py \
    --concurrency 4 --output results/swebench_verified_gold.jsonl

# apptainer provider (builds a .sif per instance, then removes it)
python resources_servers/swe_env/scripts/run_swebench_verified.py --provider apptainer --limit 5

# specific instances
python resources_servers/swe_env/scripts/run_swebench_verified.py \
    --instances astropy__astropy-13453,django__django-11099
```
Flags: `--limit N`, `--instances id1,id2`, `--provider docker|apptainer`,
`--concurrency K`, `--eval-timeout S`, `--keep-images`, `--output PATH`.

Output: a per-instance line (`PASS`/`fail`/`ERR`) and a final
`resolved N/total (P%)`. Each result row (`{instance_id, resolved, status, error}`)
is appended to `--output` as it completes.

### Validated
- A real instance (`astropy__astropy-13453`) resolves end-to-end on **both** the
  `docker` and `apptainer` providers (`reward=1.0`).
- Unit tests (FakeSandbox) + env-gated real-container tests live in `tests/`.

## Tests
```bash
RAY_TMPDIR=/tmp ng_test +entrypoint=resources_servers/swe_env
# env-gated real-container tests (need docker / apptainer):
SWE_ENV_DOCKER_ITEST=1 pytest resources_servers/swe_env/tests/test_verify.py -k docker_real
SWE_ENV_REAL_SWEBENCH=1 pytest resources_servers/swe_env/tests/test_swebench_real_instance.py
```

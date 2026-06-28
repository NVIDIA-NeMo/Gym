# swe_env

Shared library for provisioning and grading SWE (software-engineering) task
environments. It is imported by `anyswe_agent` (and usable by any other Gym
agent that self-drives inside a SWE task sandbox); it is **not** a runnable
server and has no config or entrypoint.

Everything is provider-neutral, running over the `nemo_gym.sandbox` providers
(docker / apptainer / opensandbox):

- **`harnesses/`** — per-dataset-family recipes (SWE-bench, SWE-bench
  Multilingual, SWE-bench-ext, R2E-Gym, SWE-rebench, NV-internal). Each builds
  the task sandbox spec, materializes the model patch, runs the evaluation, and
  grades the result host-side via the official per-repo parser (falling back to
  a generic parser only where the official one is unavailable).
- **`sandbox.py`** — async sandbox lifecycle (`AsyncSweEnvironment`,
  `acquire_sandbox`) with always-teardown semantics.
- **`self_drive.py`** — provision a writable sandbox, inject a sandbox-reachable
  model endpoint / egress env, run an opaque agent launch command, and extract
  the resulting `git diff` patch.
- **`verify_task.py`** — grade a patch inline in a fresh sandbox (no separate
  `/verify` server), returning a mask-aware reward.
- **`parsing/`** — test-log parsers (relocated verbatim from `swe_agents`).

## Reward-profiling baseline — SWE-bench Verified gold patches

A gold-patch census validates the grader end-to-end: feed each instance's ground-truth
patch through the flat grader and it should resolve. On the **docker** provider
(pull-on-demand images, host-side flat grading) the full 500-instance census resolves
**493 / 500** (`patch_exists` 500/500, **0** infra errors), matching the apptainer / `.sif`
nested reference of **492 / 500** to within environment noise. An empty patch resolves
**0 / 500**, as expected.

The two are at parity (docker and apptainer both use the host-side flat grader — same parser,
same result; the `.sif` figure is swebench's *nested* `run_evaluation`). Their misses are a small
symmetric difference: 4 are shared genuine upstream env-flaky gold-failures (astropy-7606/8707/8872,
django-10097); docker additionally resolves 4 that `.sif` misses (pylint-6528/7277, sphinx-8595/9711)
and misses 3 that `.sif` resolves (sphinx-8120/8265/8269, instance-specific parser/eval quirks).

Reaching parity required closing two flat↔nested **reconstruction** gaps the census surfaced
(445 → 486 → 493):
- **`PYTEST_ADDOPTS=-rA`** — swebench 4.1.0's eval command for some families (sphinx via tox,
  several sklearn) runs pytest without `-rA`, so passing tests print only as dots and the host-side
  parser (`parse_log_pytest_v2`) saw zero passes (445 → 486).
- **drop `GIT_CONFIG_GLOBAL=/dev/null`** — older instance images' git can't parse `/dev/null`, so
  the eval script's `git checkout` + test-patch `git apply` failed and required tests came back
  "absent" (486 → 493). See `harnesses/swebench.py`.

### Reproduce

Grade gold patches on docker (no model, no agent — pure grader validation). Bound concurrency
with a semaphore and `docker rmi` each image after grading to cap disk:

```python
import asyncio, dataclasses, json
from datasets import load_dataset
from responses_api_agents.anyswe_agent.app import _build_swetask
from responses_api_agents.swe_env.verify_task import verify_task

async def gold_resolves(inst) -> bool:
    pinfo = {
        "instance_id": inst["instance_id"],
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "container_formatter": "docker://swebench/sweb.eval.x86_64.{instance_id}",
        "instance_dict": json.dumps(dict(inst)),
    }
    task = _build_swetask(pinfo, flat_eval=True)
    report = await verify_task({"docker": {}}, dataclasses.replace(task, model_patch=inst["patch"]))
    return bool(report.resolved)  # error_kind is None on a clean (non-infra) grade

rows = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
```

Swap `{"docker": {}}` for `{"apptainer": {}}` (with a `.sif` `container_formatter`) to grade on
apptainer instead.

## Tests

The unit tests run against a scripted fake `SandboxProvider`, so they need no
Docker/apptainer and execute in CI:

```bash
ng_test +entrypoint=responses_api_agents/swe_env
```

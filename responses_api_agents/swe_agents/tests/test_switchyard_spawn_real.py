"""Real-process regression test for the per-rollout Switchyard lifecycle.

The core of this test runs INSIDE a real Ray task with the production
decorator options — deliberately unmocked. The rest of the suite mocks
``asyncio.create_subprocess_exec``, which verifies the argv we build but can
never fail on what actually broke in production three separate times: state
the Gym server process has that a Ray worker does not (PATH, environment,
``sys.path``). Two 32-node training allocations were lost to bugs this file's
checks catch in under a minute on one CPU.

Covered in one task round-trip:
- the ``switchyard`` CLI resolves without PATH (interpreter-adjacent fallback);
- the spawned proxy reaches READY (binds and accepts TCP) — a proxy that
  spawns but exits pre-bind fails every rollout;
- params cross the boundary exactly as production sends them (``model_dump``
  -> ``model_validate``);
- task-side retrieval reconstructs a real trace from seeded records, and the
  payload (pydantic models included) survives the Ray object store;
- teardown is clean.

The venv wrapper's env derivation (NEMO_GYM_CONFIG_DICT ->
SWITCHYARD_POLICY_MODEL) is deployment-specific and exercised by the
deployment's own gate; the routing profile here uses literal values so the
test runs on any install with the real CLI.

Skips when ray or the switchyard CLI is unavailable. If the Ray worker
unexpectedly has the CLI on PATH, the environment does not reproduce
production and the test reports itself inconclusive via skip rather than
asserting a vacuous pass.
"""

import asyncio
import json
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

import pytest


ray = pytest.importorskip("ray")


_CLI_AVAILABLE = shutil.which("switchyard") is not None or (Path(sys.executable).parent / "switchyard").exists()

pytestmark = pytest.mark.skipif(not _CLI_AVAILABLE, reason="switchyard CLI not installed")

_PROFILE_YAML = """\
defaults:
  api_key: dummy
  base_url: http://127.0.0.1:9999/v1
  format: openai
routes:
  default:
    type: model
    target: test-model
    format: openai
"""


def _seed_records(rl_log_dir: Path, session_id: str = "ses_root", turns: int = 3) -> None:
    """Records in Switchyard's on-disk shape: cumulative messages, extending tokens."""
    history = [1, 2, 3]
    d = rl_log_dir / "sessions" / f"dir_{session_id}"
    d.mkdir(parents=True, exist_ok=True)
    for turn in range(turns):
        generation = [100 + turn, 101 + turn]
        record = {
            "schema_version": 1,
            "session_id": session_id,
            "parent_session_id": None,
            "uuid": f"{session_id}-{turn:03d}",
            "captured_at": f"2026-01-01T00:00:{turn:02d}",
            "model": "test-model",
            "is_valid": True,
            "finish_reason": "stop",
            "messages": [
                m
                for t in range(turn + 1)
                for m in (
                    {"role": "user", "content": f"turn {t}"},
                    {"role": "assistant", "content": f"reply {t}"},
                )
            ],
            "tools": [],
            "tool_choice": None,
            "request_id": f"req-{turn}",
            "token_count": len(history) + len(generation),
            "prompt_token_ids": list(history),
            "generation_token_ids": generation,
            "generation_log_probs": [-0.1] * len(generation),
        }
        (d / f"{record['uuid']}.json").write_text(json.dumps(record))
        history = history + generation


def _task_body(gym_dir: str, params_dict: dict) -> dict:
    """Runs inside the Ray worker.

    Under pytest this module has a real dotted name, so Ray exports this
    function BY REFERENCE: the worker imports the module before running the
    body, which is why the decorator passes PYTHONPATH in runtime_env — the
    driver's sys.path does not cross the boundary. The inserts below are what
    the body itself needs for its bare `import app`."""
    sys.path.insert(0, gym_dir)
    sys.path.insert(0, str(Path(gym_dir) / "responses_api_agents" / "swe_agents"))

    import app as swe_app

    swe_app.SWEBenchWrapperInstanceConfig.model_rebuild(force=True)
    params = swe_app.SWEBenchWrapperInstanceConfig.model_validate(params_dict)
    out: dict = {}

    async def _run() -> None:
        # Negative control: the pre-fix form must fail here, else this worker
        # has the venv on PATH and the resolution fallback goes unexercised.
        try:
            proc = await asyncio.create_subprocess_exec(
                "switchyard", "--help", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()
            out["control_raised"] = False
        except FileNotFoundError:
            out["control_raised"] = True

        process = None
        try:
            base_url, process = await swe_app._spawn_switchyard_local(params)
            out["spawn_ok"] = True
            host, port = base_url.removeprefix("http://").split(":")
            deadline = time.monotonic() + 5
            ok = False
            while time.monotonic() < deadline and not ok:
                with socket.socket() as s:
                    s.settimeout(1)
                    ok = s.connect_ex((host, int(port))) == 0
            out["tcp_accepting"] = ok
            out["process_alive"] = process.returncode is None
        finally:
            if process is not None:
                await swe_app._teardown_switchyard_process(process)
                out["teardown_ok"] = True

        out["payload"] = swe_app._collect_switchyard_payload(params)

    asyncio.run(_run())
    return out


def test_spawn_ready_and_retrieval_in_real_ray_worker(tmp_path: Path) -> None:
    from swe_agents.tests.test_app import _make_instance_config

    gym_dir = str(Path(__file__).resolve().parents[3])
    profile = tmp_path / "profile.yaml"
    profile.write_text(_PROFILE_YAML)

    with tempfile.TemporaryDirectory() as tmpdir:
        params = _make_instance_config(
            tmpdir,
            switchyard_spawn_routing_profile=str(profile),
            agent_framework="opencode",
            opencode_source="opencode",
        )
        _seed_records(params.persistent_dir / "switchyard_traces")

        ray.init(ignore_reinit_error=True, include_dashboard=False)
        try:
            task = ray.remote(
                runtime_env={
                    "py_executable": sys.executable,
                    # by-reference export: the worker must be able to import this
                    # module's package (responses_api_agents/<swe_agents>) itself
                    "env_vars": {"PYTHONPATH": str(Path(gym_dir) / "responses_api_agents")},
                },
                num_cpus=0.1,
            )(_task_body)
            out = ray.get(task.remote(gym_dir, params.model_dump()), timeout=540)
        finally:
            ray.shutdown()

    if out.get("control_raised") is False:
        pytest.skip("Ray worker has switchyard on PATH — environment does not reproduce production")

    assert out["spawn_ok"], "spawn failed in the Ray worker"
    assert out["tcp_accepting"], "proxy never reached READY (bound port)"
    assert out["process_alive"]
    assert out["teardown_ok"]

    payload = out["payload"]  # crossed the object store via ray.get
    assert payload is not None and payload.get("error") is None
    assert payload["root_id"] == "ses_root"
    assert payload["trace"].output_items, "reconstructed trace lost its tokens crossing the object store"

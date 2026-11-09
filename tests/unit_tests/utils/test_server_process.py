# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import nemo_gym
from nemo_gym.utils.server_process import (
    IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME,
    IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME,
)


# A server entrypoint with the module-level adapter, started the way Gym starts one (`python <entrypoint>.py`).
# Its main block starts a worker the way uvicorn does: a spawned process that imports the entrypoint by module name.
_ENTRYPOINT = textwrap.dedent(
    """
    import importlib
    import multiprocessing
    import os
    from functools import partial

    from nemo_gym.utils.server_process import is_nemo_gym_fastapi_entrypoint


    def build_server():
        with open(os.environ["BUILD_LOG"], "a") as log:
            log.write(__name__ + "\\n")


    if __name__ == "__main__":
        os.environ["{worker}"] = "1"
        os.environ["{entrypoint}"] = "fake_server/app.py"
        worker = multiprocessing.get_context("spawn").Process(
            target=partial(importlib.import_module, "fake_server.app")
        )
        worker.start()
        worker.join(timeout=60)
        if worker.is_alive():
            worker.kill()
            raise SystemExit("the spawned worker did not finish within 60 seconds")
        raise SystemExit(worker.exitcode)
    elif is_nemo_gym_fastapi_entrypoint(__file__):
        build_server()
    """
).format(worker=IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, entrypoint=IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME)


def test_a_spawned_worker_builds_the_server_once(tmp_path: Path) -> None:
    package = tmp_path / "fake_server"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "app.py").write_text(_ENTRYPOINT)
    build_log = tmp_path / "builds"
    env = {key: value for key, value in os.environ.items() if not key.startswith("IS_NEMO_GYM_FASTAPI_")}
    nemo_gym_root = str(Path(nemo_gym.__file__).parent.parent)
    env.update(BUILD_LOG=str(build_log), PYTHONPATH=os.pathsep.join([str(tmp_path), nemo_gym_root]))

    subprocess.run([sys.executable, str(package / "app.py")], cwd=tmp_path, env=env, check=True, timeout=120)

    # Spawn re-runs the parent's script as `__mp_main__` before the worker imports `fake_server.app`.
    # Only the import builds the server.
    assert build_log.read_text().splitlines() == ["fake_server.app"]


def test_the_entrypoint_check_holds_for_the_imported_module(monkeypatch) -> None:
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME, "fake_server/app.py")
    check = "from nemo_gym.utils.server_process import is_nemo_gym_fastapi_entrypoint\n"
    check += "result = is_nemo_gym_fastapi_entrypoint('/repo/fake_server/app.py')\n"
    for module_name, expected in [("fake_server.app", True), ("__mp_main__", False)]:
        namespace = {"__name__": module_name}
        exec(check, namespace)
        assert namespace["result"] is expected, module_name

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from responses_api_agents.hermes_agent.sandbox_runner import FileModelRelay, _write_atomic


def test_file_model_relay_exchanges_one_chat_completion(tmp_path) -> None:
    relay = FileModelRelay(tmp_path)
    completed = []

    thread = threading.Thread(target=lambda: completed.append(relay.call({"model": "policy_model"})))
    thread.start()

    request_path = tmp_path / "model-request-0.json"
    for _ in range(100):
        if request_path.exists():
            break
        time.sleep(0.01)
    assert json.loads(request_path.read_text()) == {"model": "policy_model"}

    _write_atomic(
        tmp_path / "model-response-0.json",
        {
            "response": {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "done", "role": "assistant"},
                    }
                ],
                "created": 0,
                "model": "model",
                "object": "chat.completion",
            }
        },
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert completed[0].id == "chatcmpl-test"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")
@pytest.mark.parametrize("ending", ["normal", "cancel"])
def test_supervisor_reaps_detached_tools_before_acknowledging_close(tmp_path, ending):
    worker = (
        "import subprocess,sys,pathlib,time,signal; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); " + ("time.sleep(60)" if ending == "cancel" else "pass")
    )
    root = str(Path(__file__).resolve().parents[3])
    supervisor = (
        "import sys,json,pathlib; "
        f"sys.path.insert(0, {root!r}); "
        "import responses_api_agents; "
        f"responses_api_agents.__path__ = [{str(Path(root) / 'responses_api_agents')!r}]; "
        "from responses_api_agents.hermes_agent.sandbox_runner import _supervise; "
        "receipt=_supervise(json.loads(sys.argv[1]),cleanup_timeout=2); "
        "pathlib.Path('cleanup.json').write_text(json.dumps(receipt))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", supervisor, json.dumps([sys.executable, "-c", worker])],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        if ending == "cancel":
            for _ in range(500):
                if (tmp_path / "child.pid").exists():
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            assert (tmp_path / "child.pid").exists()
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        receipt = json.loads((tmp_path / "cleanup.json").read_text())
        assert receipt == {"cleanup_confirmed": True, "error": None}
        pid = int((tmp_path / "child.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        # A failing regression must not leave the test's detached child running.
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

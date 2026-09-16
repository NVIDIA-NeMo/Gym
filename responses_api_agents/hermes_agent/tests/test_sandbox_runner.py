# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time

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

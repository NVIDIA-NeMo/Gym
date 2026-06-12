# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from responses_api_agents.swe_agents.openclaw.trajectory_reconstruction import (
    strip_token_ids_in_proxy_log,
)


def _make_log(tmp_path: Path) -> Path:
    fpath = tmp_path / "openclaw_proxy.jsonl"
    lines = [
        json.dumps(
            {
                "turn": 0,
                "endpoint": "/v1/responses",
                "request": {"input": []},
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi"}],
                            "prompt_token_ids": [1, 2, 3],
                            "generation_token_ids": [10],
                            "generation_log_probs": [-0.1],
                        },
                    ]
                },
                "upstream_status": 200,
            }
        ),
        json.dumps(
            {
                "turn": 1,
                "endpoint": "/v1/responses",
                "request": {"input": []},
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "x",
                            "call_id": "c1",
                            "arguments": "{}",
                            "prompt_token_ids": [4, 5],
                            "generation_token_ids": [11, 12],
                        },
                    ]
                },
                "upstream_status": 200,
            }
        ),
    ]
    fpath.write_text("\n".join(lines) + "\n")
    return fpath


def test_strip_removes_token_id_fields(tmp_path):
    fpath = _make_log(tmp_path)
    strip_token_ids_in_proxy_log(str(fpath))
    contents = fpath.read_text().splitlines()
    for line in contents:
        entry = json.loads(line)
        for item in entry["response"]["output"]:
            assert "prompt_token_ids" not in item
            assert "generation_token_ids" not in item
            assert "generation_log_probs" not in item


def test_strip_preserves_other_fields(tmp_path):
    fpath = _make_log(tmp_path)
    strip_token_ids_in_proxy_log(str(fpath))
    e0 = json.loads(fpath.read_text().splitlines()[0])
    assert e0["turn"] == 0
    assert e0["upstream_status"] == 200
    assert e0["response"]["output"][0]["content"][0]["text"] == "hi"


def test_strip_preserves_entry_when_token_ids_absent(tmp_path):
    # With no token-id fields to remove, the entry's surviving fields are preserved through the
    # rewrite (the file is re-serialized, so this is not a byte-identical no-op).
    fpath = tmp_path / "p.jsonl"
    fpath.write_text(
        json.dumps(
            {
                "turn": 0,
                "request": {},
                "response": {"output": [{"type": "message"}]},
                "upstream_status": 200,
            }
        )
        + "\n"
    )
    strip_token_ids_in_proxy_log(str(fpath))
    assert json.loads(fpath.read_text().splitlines()[0])["response"]["output"][0]["type"] == "message"


def test_strip_removes_token_id_fields_from_request_input(tmp_path):
    # The shim injects the prior turn's token IDs onto the request's last assistant item
    # (for NeMo-RL's on-policy splice). Those large arrays must also be stripped from the
    # log by default, or the request side bloats the file just like the response side did.
    fpath = tmp_path / "openclaw_proxy.jsonl"
    fpath.write_text(
        json.dumps(
            {
                "turn": 1,
                "endpoint": "/v1/responses",
                "request": {
                    "input": [
                        {"role": "user", "content": "u"},
                        {
                            "type": "function_call",
                            "name": "x",
                            "call_id": "c1",
                            "arguments": "{}",
                            "prompt_token_ids": [1, 2, 3],
                            "generation_token_ids": [4, 5],
                            "generation_log_probs": [-0.1],
                        },
                        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                    ]
                },
                "response": {"output": []},
                "upstream_status": 200,
            }
        )
        + "\n"
    )
    strip_token_ids_in_proxy_log(str(fpath))
    entry = json.loads(fpath.read_text().splitlines()[0])
    for item in entry["request"]["input"]:
        assert "prompt_token_ids" not in item
        assert "generation_token_ids" not in item
        assert "generation_log_probs" not in item
    # Non-token fields on the same item are preserved.
    assert entry["request"]["input"][1]["name"] == "x"
    assert entry["request"]["input"][1]["arguments"] == "{}"


def test_strip_handles_refused_entry_with_none_response(tmp_path):
    fpath = tmp_path / "p.jsonl"
    fpath.write_text(
        json.dumps(
            {
                "turn": 1,
                "request": None,
                "response": None,
                "upstream_status": None,
                "error": "max_iteration",
            }
        )
        + "\n"
    )
    strip_token_ids_in_proxy_log(str(fpath))  # must not raise
    entry = json.loads(fpath.read_text().splitlines()[0])
    assert entry["error"] == "max_iteration"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from resources_servers.browser_gym.prepare_data import fetch_gym_tasks, write_jsonl


def _mock_urlopen(payload: dict):
    """Return a context-manager mock for urllib.request.urlopen."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


class TestFetchGymTasks:
    def test_happy_path_with_task_statement_and_viewport_list(self) -> None:
        payload = {
            "verifiers": {
                "TASK-001": {
                    "task_statement": "Click the button",
                    "start_url": "http://app.test/page1",
                    "viewport_size": [1920, 1080],
                }
            }
        }
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test")

        assert len(rows) == 1
        assert rows[0]["responses_create_params"]["input"][0]["content"] == "Click the button"
        assert rows[0]["verifier_metadata"]["start_url"] == "http://app.test/page1"
        assert rows[0]["verifier_metadata"]["viewport"] == {"width": 1920, "height": 1080}

    def test_prompt_fallback_when_no_task_statement(self) -> None:
        payload = {
            "verifiers": {
                "TASK-002": {
                    "prompt": "Fill the form",
                    "start_url": "http://app.test/form",
                }
            }
        }
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test/")

        assert rows[0]["responses_create_params"]["input"][0]["content"] == "Fill the form"

    def test_viewport_dict_passthrough(self) -> None:
        payload = {
            "verifiers": {
                "TASK-003": {
                    "task_statement": "test",
                    "viewport_size": {"width": 800, "height": 600},
                }
            }
        }
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test")

        assert rows[0]["verifier_metadata"]["viewport"] == {"width": 800, "height": 600}

    def test_viewport_default_when_missing(self) -> None:
        payload = {"verifiers": {"TASK-004": {"task_statement": "test"}}}
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test")

        assert rows[0]["verifier_metadata"]["viewport"] == {"width": 1280, "height": 720}

    def test_non_dict_details_uses_gym_url_fallback(self) -> None:
        payload = {"verifiers": {"TASK-005": "just a string"}}
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test")

        assert rows[0]["responses_create_params"]["input"][0]["content"] == ""
        assert rows[0]["verifier_metadata"]["start_url"] == "http://gym.test"
        assert rows[0]["verifier_metadata"]["viewport"] == {"width": 1280, "height": 720}

    def test_start_url_defaults_to_gym_url(self) -> None:
        payload = {"verifiers": {"TASK-006": {"task_statement": "test"}}}
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test")

        assert rows[0]["verifier_metadata"]["start_url"] == "http://gym.test"

    def test_filter_by_task_ids(self) -> None:
        payload = {
            "verifiers": {
                "A": {"task_statement": "a"},
                "B": {"task_statement": "b"},
                "C": {"task_statement": "c"},
            }
        }
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test", task_ids=["A", "C"])

        assert len(rows) == 2
        task_ids = [r["verifier_metadata"]["task_id"] for r in rows]
        assert task_ids == ["A", "C"]

    def test_missing_task_ids_raises(self) -> None:
        payload = {"verifiers": {"A": {"task_statement": "a"}}}
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            with pytest.raises(ValueError, match="Task ID.*not found"):
                fetch_gym_tasks("http://gym.test", task_ids=["MISSING"])

    def test_empty_verifiers_raises(self) -> None:
        payload = {"verifiers": {}}
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            with pytest.raises(ValueError, match="No verifiers found"):
                fetch_gym_tasks("http://gym.test")

    def test_http_error_raises(self) -> None:
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("http://gym.test", 500, "Internal Error", {}, BytesIO(b"")),
        ):
            with pytest.raises(ValueError, match="HTTP 500"):
                fetch_gym_tasks("http://gym.test")

    def test_url_error_raises(self) -> None:
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(ValueError, match="Could not connect"):
                fetch_gym_tasks("http://gym.test")

    def test_gym_url_trailing_slash_stripped(self) -> None:
        payload = {"verifiers": {"T1": {"task_statement": "test"}}}
        with patch(
            "resources_servers.browser_gym.prepare_data.urllib.request.urlopen",
            return_value=_mock_urlopen(payload),
        ):
            rows = fetch_gym_tasks("http://gym.test///")

        assert rows[0]["verifier_metadata"]["gym_url"] == "http://gym.test///"
        assert rows[0]["verifier_metadata"]["task_id"] == "T1"


class TestWriteJsonl:
    def test_writes_correct_jsonl(self, tmp_path: Path) -> None:
        rows = [
            {"responses_create_params": {"input": [{"role": "user", "content": "task 1"}]}},
            {"responses_create_params": {"input": [{"role": "user", "content": "task 2"}]}},
        ]
        output = tmp_path / "out.jsonl"
        count = write_jsonl(rows, str(output))

        assert count == 2
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["responses_create_params"]["input"][0]["content"] == "task 1"
        assert json.loads(lines[1])["responses_create_params"]["input"][0]["content"] == "task 2"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "dir" / "out.jsonl"
        write_jsonl([{"key": "value"}], str(output))
        assert output.exists()

    def test_empty_rows(self, tmp_path: Path) -> None:
        output = tmp_path / "empty.jsonl"
        count = write_jsonl([], str(output))
        assert count == 0
        assert output.read_text() == ""

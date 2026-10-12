# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import MagicMock

from resources_servers.gdpval import comparison
from resources_servers.gdpval.judge_telemetry import (
    JudgeTelemetrySink,
    classify_judge_error,
    profile_judge_request,
    summarize_events,
)


def _messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<REFERENCES_FILES_START>\n"},
                {"type": "text", "text": "\ninputs/source.mp4:\n"},
                {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,YWJj"}},
                {"type": "text", "text": "\n<REFERENCES_FILES_END>\n\n"},
                {"type": "text", "text": "<SUBMISSION_A_START>\n"},
                {"type": "text", "text": "\nreport.pdf:\n"},
                {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,ZGVmZw=="}},
                {"type": "text", "text": "\n<SUBMISSION_A_END>\n\n"},
            ],
        }
    ]


def _reply(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def test_request_profile_records_shape_without_payload_contents() -> None:
    profile = profile_judge_request(_messages())

    assert profile["file_count"] == 2
    assert profile["section_file_counts"] == {"reference_files": 1, "submission_a": 1}
    assert profile["extension_counts"] == {"mp4": 1, "pdf": 1}
    assert profile["mime_type_counts"] == {"application/pdf": 1, "video/mp4": 1}
    assert profile["raw_attachment_bytes"] == 7
    encoded = json.dumps(profile)
    assert "YWJj" not in encoded
    assert "ZGVmZw" not in encoded


def test_retry_failure_and_recovery_are_correlated(tmp_path, monkeypatch) -> None:
    sink = JudgeTelemetrySink(tmp_path)
    client = MagicMock()
    client.chat.completions.create.side_effect = [RuntimeError("Error code: 429 - rate limit"), _reply("BOXED[A]")]
    monkeypatch.setattr(comparison.time, "sleep", lambda _seconds: None)

    result = comparison.send_judge_request(
        client,
        "gemini",
        _messages(),
        telemetry=sink,
        telemetry_context={"task_id": "task-1", "judge": "gemini"},
        transport_receipt={"serialized_request_bytes": 1234, "video_file_count": 1},
    )

    assert result == "BOXED[A]"
    events = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["judge_attempt_failed", "judge_request_completed"]
    assert events[0]["error"]["kind"] == "http_429_rate_limit"
    assert events[0]["request"]["files"][0]["name"] == "inputs/source.mp4"
    assert events[1]["outcome"] == "recovered"
    assert events[1]["attempts"] == 2
    assert events[0]["request_id"] == events[1]["request_id"]


def test_error_description_redacts_credentials() -> None:
    error = RuntimeError("status 401 api_key=secret-value Bearer another-secret")

    description = classify_judge_error(error, retryable=False)

    assert description["kind"] == "authentication"
    assert "secret-value" not in description["message"]
    assert "another-secret" not in description["message"]


def test_summarizer_reports_rates_and_failure_modes(tmp_path) -> None:
    sink = JudgeTelemetrySink(tmp_path)
    sink.emit(
        "judge_attempt_failed",
        task_id="task-1",
        judge="gemini",
        error={"kind": "http_429_rate_limit", "http_status": 429},
        request={"serialized_request_bytes": 60 * 1024 * 1024, "extension_counts": {"mp4": 2}},
    )
    sink.emit("judge_request_completed", task_id="task-1", judge="gemini", outcome="recovered")
    sink.emit("judge_request_completed", task_id="task-2", judge="gemini", outcome="failed")
    sink.emit(
        "judge_preflight_excluded",
        task_id="task-3",
        judge="gemini",
        request={"reasons": ["provider_video_count_cap"]},
    )

    summary = summarize_events([tmp_path])

    assert summary["completed_requests"] == 2
    assert summary["terminal_failures"] == 1
    assert summary["terminal_failure_rate"] == 0.5
    assert summary["recovered_requests"] == 1
    assert summary["failure_kinds"] == {"http_429_rate_limit": 1}
    assert summary["preflight_reasons"] == {"provider_video_count_cap": 1}
    assert summary["failure_request_size_buckets"] == {"50-100 MiB": 1}


def test_preflight_exclusion_records_the_actual_request(tmp_path) -> None:
    sink = JudgeTelemetrySink(tmp_path)
    judge = comparison.Judge(
        name="gemini",
        client=MagicMock(),
        model="gemini",
        max_video_files=0,
    )
    sections = {
        "refs": _messages()[0]["content"][1:3],
        "submission_a": [],
        "submission_b": [],
    }

    receipt = comparison.preflight_judge_transport(
        judge,
        "task",
        sections,
        telemetry=sink,
        telemetry_context={"task_id": "task-1"},
    )

    assert receipt["eligible"] is False
    event = json.loads(sink.path.read_text())
    assert event["event"] == "judge_preflight_excluded"
    assert event["request"]["reasons"] == ["provider_video_count_cap"]
    assert event["request"]["files"][0]["name"] == "inputs/source.mp4"


def test_invalid_response_is_recorded_without_response_contents(tmp_path) -> None:
    sink = JudgeTelemetrySink(tmp_path)
    client = MagicMock()
    client.chat.completions.create.return_value = _reply("private malformed response")
    judge = comparison.Judge(name="judge", client=client, model="judge-model")

    try:
        comparison.run_trials(
            judges=[judge],
            task_prompt="task",
            refs=[],
            submission_a=[],
            submission_b=[],
            num_trials=1,
            telemetry=sink,
            telemetry_context={"task_id": "task-1"},
        )
    except ValueError as error:
        assert "responses were invalid" in str(error)
    else:
        raise AssertionError("invalid-only trial should fail")

    events = [json.loads(line) for line in sink.path.read_text().splitlines()]
    invalid = next(event for event in events if event["event"] == "judge_response_invalid")
    assert invalid["response_chars"] == len("private malformed response")
    assert "private malformed response" not in sink.path.read_text()

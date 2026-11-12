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
"""Unit tests for SWE Bench Multimodal row preparation (network-free)."""

import base64
import json
import urllib.request

import pytest

from benchmarks.swebench.multimodal import prepare
from resources_servers.swebench.task_data import TaskData


PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-body"
GIF = b"GIF89a" + b"gif-body"
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"webp-body"
HTML = b"<!doctype html><html></html>"

TEMPLATE = "<issue_details>\n{{ instance.problem_statement }}\n</issue_details>\nWork in {{ workspace_path }}."


def _row(image_assets) -> dict:
    return {
        "instance_id": "chartjs__Chart.js-1",
        "repo": "chartjs/Chart.js",
        "problem_statement": "Legend overlaps the chart ![shot](https://example.com/a.png)",
        "image_assets": image_assets,
        "FAIL_TO_PASS": ["legend does not overlap"],
        "PASS_TO_PASS": ["legend renders"],
    }


def _decode(data_uri: str) -> tuple[str, bytes]:
    header, payload = data_uri.split(",", 1)
    return header, base64.b64decode(payload, validate=True)


@pytest.mark.parametrize(
    ("content", "expected"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp"), (HTML, None), (b"", None)],
)
def test_sniff_image_media_type(content: bytes, expected: str | None) -> None:
    assert prepare.sniff_image_media_type(content) == expected


@pytest.mark.parametrize(
    "image_assets",
    [
        {"problem_statement": ["https://example.com/a.png"], "patch": []},
        json.dumps({"problem_statement": ["https://example.com/a.png"], "patch": []}),
    ],
)
def test_issue_image_urls_reads_dict_and_json_string(image_assets) -> None:
    assert prepare.issue_image_urls(image_assets) == ["https://example.com/a.png"]


@pytest.mark.parametrize("image_assets", [None, "", {}, {"problem_statement": []}])
def test_issue_image_urls_empty(image_assets) -> None:
    assert prepare.issue_image_urls(image_assets) == []


def test_build_row_attaches_issue_images_in_order_and_skips_non_images() -> None:
    assets = {
        "a.png": PNG,
        "page.html": HTML,
        "b.gif": GIF,
    }
    row = _row({"problem_statement": [f"https://example.com/{name}" for name in assets], "patch": []})

    out = prepare.build_row(row, TEMPLATE, fetch=lambda url: assets[url.rsplit("/", 1)[1]])

    [message] = out["responses_create_params"]["input"]
    assert message["role"] == "user"
    text, *images = message["content"]
    assert text == {
        "type": "input_text",
        "text": (
            "<issue_details>\nLegend overlaps the chart ![shot](https://example.com/a.png)\n</issue_details>\n"
            "Work in /testbed."
        ),
    }
    assert [part["type"] for part in images] == ["input_image", "input_image"]
    assert [part["detail"] for part in images] == ["high", "high"]
    assert [_decode(part["image_url"]) for part in images] == [
        ("data:image/png;base64", PNG),
        ("data:image/gif;base64", GIF),
    ]


def test_build_row_without_images_keeps_a_text_prompt() -> None:
    out = prepare.build_row(_row({"problem_statement": [], "patch": []}), TEMPLATE, fetch=pytest.fail)

    [message] = out["responses_create_params"]["input"]
    assert isinstance(message["content"], str)
    assert "Legend overlaps the chart" in message["content"]


def test_build_row_matches_the_swebench_row_format() -> None:
    out = prepare.build_row(_row({"problem_statement": []}), TEMPLATE, fetch=pytest.fail)

    assert json.loads(out["FAIL_TO_PASS"]) == ["legend does not overlap"]
    assert json.loads(out["PASS_TO_PASS"]) == ["legend renders"]
    assert (out["subset"], out["split"]) == ("multimodal", "dev")
    assert (out["environment_setup_commit"], out["difficulty"]) == ("", "")
    assert out["instance_id"] == "chartjs__Chart.js-1"


def test_build_row_stores_the_test_patch_files_for_the_verifier() -> None:
    files = {"https://example.com/ref-a.png": PNG, "https://example.com/ref-b.gif": GIF}
    test_assets = [{"path": f"test/fixtures/{url.rsplit('/', 1)[1]}", "url": url} for url in files]
    image_assets = {"problem_statement": [], "patch": [], "test_patch": test_assets}
    row = _row(json.dumps(image_assets)) | {
        "base_commit": "abc123",
        "patch": "",
        "test_patch": "",
        "hints_text": "",
        "created_at": "",
        "version": "3.7",
    }

    out = prepare.build_row(row, TEMPLATE, fetch=files.__getitem__)

    stored = out["image_assets"]["test_patch"]
    assert [(asset["path"], asset["url"]) for asset in stored] == [(a["path"], a["url"]) for a in test_assets]
    assert [base64.b64decode(asset["content_base64"], validate=True) for asset in stored] == [PNG, GIF]
    assert out["image_assets"]["patch"] == [] and out["image_assets"]["problem_statement"] == []
    # The swebench resources server accepts the row and reads the same bytes.
    assert [asset.content for asset in TaskData.model_validate(out).image_assets.test_patch] == [PNG, GIF]


def test_fetch_url_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return PNG

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise OSError("connection reset")
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prepare.time, "sleep", lambda seconds: None)

    assert prepare.fetch_url("https://example.com/a.png") == PNG
    assert calls == ["https://example.com/a.png"] * 3


def test_fetch_url_raises_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):
        raise OSError("not found")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(prepare.time, "sleep", lambda seconds: None)

    with pytest.raises(OSError, match="not found"):
        prepare.fetch_url("https://example.com/a.png")

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
"""Network-free tests of the MM-BrowseComp row mapping on a synthetic, self-encrypted row."""

import base64
import hashlib

import pytest

from benchmarks.mm_browsecomp.prepare import image_data_url, image_file_name, map_mm_browsecomp_row


CANARY = "test-canary"


def _encrypt(plaintext: str) -> str:
    data = plaintext.encode()
    key = hashlib.sha256(CANARY.encode()).digest()
    key = key * (len(data) // len(key)) + key[: len(data) % len(key)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def _row(images: list[str], checklist_property: str = "1,0") -> dict:
    return {
        "id": 7,
        "images": images,
        "question": _encrypt("Please answer ... Question: Which tower?"),
        "answer": _encrypt("Eiffel Tower"),
        "checklist": [_encrypt("Identify the city"), _encrypt("Identify the tower")],
        "checklist_property": checklist_property,
        "source": ["https://example.com/evidence"],
        "category": "Geography",
        "subtask": "Landmarks",
        "level": 1,
        "canary": CANARY,
    }


IMAGE_URL = "https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/refs/heads/main/MMBC_images/7.png"


def test_row_decrypts_and_inlines_images_after_the_text() -> None:
    out = map_mm_browsecomp_row(
        _row([IMAGE_URL]), paper_ids={7}, images={"7.png": "data:image/png;base64,AA"}, date="d"
    )
    system, user = out["responses_create_params"]["input"]
    assert "view_image" in system["content"]
    assert user["content"][0]["type"] == "input_text"
    assert user["content"][0]["text"].startswith("Please answer ... Question: Which tower?")
    assert "Exact Answer:" in user["content"][0]["text"]
    assert user["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high"}
    assert out["ground_truth"] == "Eiffel Tower"
    assert out["checklist"] == ["Identify the city", "Identify the tower"]
    assert out["in_paper_set"] is True and out["needs_video"] is False
    # The evidence URLs point at the answer: they never reach the row.
    assert "source" not in out and "example.com/evidence" not in str(out)


def test_text_only_row_keeps_a_string_prompt_and_flags_video() -> None:
    out = map_mm_browsecomp_row(_row([], checklist_property="0, 2"), paper_ids=set(), images={}, date="d")
    assert isinstance(out["responses_create_params"]["input"][1]["content"], str)
    assert out["in_paper_set"] is False and out["needs_video"] is True


def test_view_image_sits_before_bash_command() -> None:
    out = map_mm_browsecomp_row(_row([]), paper_ids=set(), images={}, date="d")
    names = [tool["name"] for tool in out["responses_create_params"]["tools"]]
    assert names == ["search", "browse", "view_image", "bash_command"]


def test_image_type_comes_from_the_bytes_not_the_extension() -> None:
    assert image_data_url(b"\xff\xd8\xff\xe0rest").startswith("data:image/jpeg;base64,")
    assert image_data_url(b"\x89PNG\r\n\x1a\nrest").startswith("data:image/png;base64,")
    assert image_data_url(b"RIFF\x00\x00\x00\x00WEBPrest").startswith("data:image/webp;base64,")
    with pytest.raises(ValueError):
        image_data_url(b"<html>")


def test_image_file_name_is_read_from_the_moving_branch_url() -> None:
    assert image_file_name(IMAGE_URL) == "7.png"
    with pytest.raises(ValueError):
        image_file_name("https://example.com/7.png")


def test_no_tools_row_is_upstreams_tool_free_prompt() -> None:
    out = map_mm_browsecomp_row(
        _row([IMAGE_URL]), paper_ids={7}, images={"7.png": "data:image/png;base64,AA"}, date="d", no_tools=True
    )
    params = out["responses_create_params"]
    assert [m["role"] for m in params["input"]] == ["user"]
    assert params["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "Please answer ... Question: Which tower?",
    }
    assert params["input"][0]["content"][1]["type"] == "input_image"
    assert "tools" not in params
    assert (params["max_output_tokens"], params["temperature"], params["top_p"]) == (16384, 1.0, 1.0)
    assert out["ground_truth"] == "Eiffel Tower" and out["in_paper_set"] is True

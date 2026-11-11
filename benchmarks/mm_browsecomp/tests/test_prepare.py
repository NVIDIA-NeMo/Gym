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
"""Unit tests for MM-BrowseComp row construction (network-free)."""

import base64
import hashlib
import json

from pytest import raises

from benchmarks.browsecomp.prepare import QUERY_SUFFIX, TOOLS
from benchmarks.mm_browsecomp import prepare as mm
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import ResponsesConverter


CANARY = "mmbrowsecomp:test-canary"
PNG = b"\x89PNG\r\n\x1a\n-test-image"
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()


def _encrypt(plaintext: str, password: str) -> str:
    data = plaintext.encode()
    key = hashlib.sha256(password.encode()).digest()
    key = key * (len(data) // len(key)) + key[: len(data) % len(key)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def _row(question: str, images: list[str]) -> dict:
    return {
        "id": 7,
        "images": images,
        "question": _encrypt(question, CANARY),
        "answer": _encrypt("Counter-clockwise", CANARY),
        "checklist": [_encrypt("Confirm the hotel", CANARY), _encrypt("Get the answer", CANARY)],
        "category": "Geography",
        "subtask": "Visual Geolocalization & Reasoning",
        "level": 2,
        "canary": CANARY,
    }


def test_the_official_prompt_prefix_is_stripped():
    row = _row(
        "Please answer the following question and also provide your problem-solving roadmap. Question: Where?", []
    )
    assert mm.build_question(row) == "Where?"


def test_a_question_with_an_image_gets_a_multimodal_user_turn():
    row = mm.map_row(_row("Please answer ... Question: Where is this?", ["u/7.png"]), [PNG_URI], "2026-10-30")
    system, user = row["responses_create_params"]["input"]
    assert system["role"] == "system" and "2026-10-30" in system["content"]
    assert user["content"] == [
        {"type": "input_text", "text": "Where is this?" + QUERY_SUFFIX},
        {"type": "input_image", "image_url": PNG_URI, "detail": "high"},
    ]
    assert row["responses_create_params"]["tools"] == TOOLS
    assert (row["question"], row["ground_truth"], row["num_images"]) == ("Where is this?", "Counter-clockwise", 1)
    assert row["checklist"] == ["Confirm the hotel", "Get the answer"]
    json.dumps(row)


def test_a_question_without_images_stays_plain_text():
    row = mm.map_row(_row("Please answer ... Question: Which video?", []), [], "2026-10-30")
    assert row["responses_create_params"]["input"][1]["content"] == "Which video?" + QUERY_SUFFIX
    assert row["num_images"] == 0


def test_two_images_keep_their_order():
    uris = [PNG_URI, "data:image/jpeg;base64,/9j/"]
    row = mm.map_row(_row("Please answer ... Question: Compare.", ["a.png", "b.jpg"]), uris, "2026-10-30")
    parts = row["responses_create_params"]["input"][1]["content"]
    assert [p["image_url"] for p in parts[1:]] == uris


def test_the_image_reaches_the_chat_request_unchanged():
    row = mm.map_row(_row("Please answer ... Question: Where?", ["u/7.png"]), [PNG_URI], "2026-10-30")
    create_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    chat = ResponsesConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        create_params
    )
    user = [m for m in chat.messages if m["role"] == "user"][0]
    assert [p["type"] for p in user["content"]] == ["text", "image_url"]
    assert user["content"][1]["image_url"]["url"] == PNG_URI


def test_images_are_read_at_the_pinned_revision_not_the_branch_in_the_row():
    row_url = "https://raw.githubusercontent.com/MMBrowseComp/MM-BrowseComp/refs/heads/main/MMBC_images/12.png"
    assert mm.pinned_image_url(row_url) == f"{mm.REPO_RAW}/MMBC_images/12.png"
    assert mm.REPO_REVISION in mm.pinned_image_url(row_url)


def test_image_data_uri_follows_the_bytes_not_the_extension(tmp_path):
    jpeg_named_png = tmp_path / "a.png"
    jpeg_named_png.write_bytes(b"\xff\xd8\xffjpeg")
    png = tmp_path / "b.jpg"
    png.write_bytes(PNG)
    assert mm._data_uri(jpeg_named_png) == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xffjpeg").decode()
    assert mm._data_uri(png) == PNG_URI
    junk = tmp_path / "c.png"
    junk.write_bytes(b"GIF89a")
    with raises(ValueError):
        mm._data_uri(junk)

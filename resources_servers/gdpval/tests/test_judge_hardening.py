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
"""Judge error classification, loss markers, AV identity and judge-name guards."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.gdpval.comparison import (
    _is_lossy_transport_marker,
    build_file_section,
)
from resources_servers.gdpval.scoring import is_permanent_judge_error


class TestThrottleVeto:
    def test_429_body_with_permanent_phrase_is_not_permanent(self) -> None:
        assert is_permanent_judge_error("Error code: 429 - you have sent too many tokens this minute") is False
        error = RuntimeError("upstream failure")
        error.status = 429
        error.body = "too many tokens"
        assert is_permanent_judge_error(error) is False

    def test_413_and_context_overflow_remain_permanent(self) -> None:
        assert is_permanent_judge_error("HTTP 413: request entity too large") is True
        error = RuntimeError("bad request")
        error.status_code = 413
        assert is_permanent_judge_error(error) is True
        assert is_permanent_judge_error("maximum context length exceeded: input is too long") is True


class TestLossMarkers:
    def test_new_marker_forms_are_detected(self) -> None:
        assert _is_lossy_transport_marker("[attachment unavailable for clip.mp4]")
        assert _is_lossy_transport_marker("[page 3 omitted for deck.pdf: raster dimensions too large]")
        assert not _is_lossy_transport_marker("[page 3] ordinary document text")
        assert not _is_lossy_transport_marker("regular text")

    def test_loss_marker_survives_exhausted_text_budget(self, tmp_path: Path, monkeypatch) -> None:
        import resources_servers.gdpval.comparison as comparison

        monkeypatch.setattr(comparison, "MAX_SECTION_TEXT_CHARS_FOR_JUDGE", 10)
        (tmp_path / "a_filler.txt").write_text("x" * 100)
        (tmp_path / "b_huge.mp4").write_bytes(b"0" * 64)
        monkeypatch.setattr(comparison, "MAX_FILE_BYTES_FOR_JUDGE", 8)

        clean_up: list = []
        section = build_file_section(
            str(tmp_path),
            clean_up,
            media_mode="native_pdf",
            render_dpi=144,
            max_pages=50,
            include_text=True,
            audio_capable=False,
            video_capable=False,
        )
        texts = [str(block.get("text", "")) for block in section if block.get("type") == "text"]
        assert any(_is_lossy_transport_marker(text) for text in texts)


class TestAvIdentityGuard:
    def test_oversize_av_is_not_hashed_and_fs_fault_degrades_to_marker(self, tmp_path: Path, monkeypatch) -> None:
        import resources_servers.gdpval.comparison as comparison

        monkeypatch.setattr(comparison, "MAX_FILE_BYTES_FOR_JUDGE", 4)
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"0" * 64)

        hashed: list = []
        real_sha = comparison.hashlib.sha256

        class _Spy:
            def __init__(self) -> None:
                hashed.append(1)
                self._inner = real_sha()

            def update(self, chunk) -> None:
                self._inner.update(chunk)

            def hexdigest(self) -> str:
                return self._inner.hexdigest()

        monkeypatch.setattr(comparison.hashlib, "sha256", _Spy)
        section = build_file_section(
            str(tmp_path),
            [],
            media_mode="native_pdf",
            render_dpi=144,
            max_pages=50,
            include_text=True,
            audio_capable=True,
            video_capable=True,
        )
        assert not hashed, "over-cap AV must not be hashed"
        assert section is not None


class TestJudgeNameCollision:
    def test_duplicate_resolved_names_raise(self, monkeypatch) -> None:
        from resources_servers.gdpval import app as gdpval_app

        monkeypatch.setattr(gdpval_app, "get_server_url", lambda name: "http://judge")

        def member(media_mode: str) -> SimpleNamespace:
            return SimpleNamespace(
                name=None,
                model="gemini-2.5-pro",
                create_params_overrides=None,
                weight=1.0,
                handles_audio=False,
                handles_video=False,
                media_mode=media_mode,
                max_native_pdf_pages=None,
                max_native_pdf_documents=None,
                max_native_pdf_bytes=None,
                max_native_pdf_bytes_per_document=None,
                raster_dpi_tiers=[],
                max_serialized_request_bytes=None,
                max_image_base64_bytes=None,
                max_total_image_base64_bytes=None,
                max_video_files=None,
                model_server=SimpleNamespace(name="judge_server"),
            )

        fake_server = SimpleNamespace(
            config=SimpleNamespace(
                judge_responses_create_params_overrides=None,
                judge_model_server=None,
                judge_media_mode="native_pdf",
            ),
            _effective_panel=lambda: [member("native_pdf"), member("images_and_text")],
        )
        with pytest.raises(ValueError, match="unique names"):
            gdpval_app.GDPValResourcesServer._resolve_judges(fake_server)

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
import zipfile
from unittest.mock import MagicMock

import pytest

from resources_servers.gdpval.comparison import Judge, run_trials
from resources_servers.gdpval.judge_panel import (
    ResolvedJudge,
    dir_contains_audio_video,
    dir_media_modalities,
    is_audio_file,
    is_audio_video_file,
    is_video_file,
    make_rng,
    merge_create_kwargs,
    panel_summary,
    sample_judge,
    select_av_judges,
)


class TestMakeRng:
    def test_same_identity_is_deterministic(self) -> None:
        a = make_rng(None, "task-1", "rubric")
        b = make_rng(None, "task-1", "rubric")
        assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]

    def test_different_identity_diverges(self) -> None:
        a = make_rng(None, "task-1", "rubric")
        b = make_rng(None, "task-2", "rubric")
        assert [a.random() for _ in range(5)] != [b.random() for _ in range(5)]

    def test_seed_shifts_stream(self) -> None:
        a = make_rng(0, "task-1", "rubric")
        b = make_rng(1, "task-1", "rubric")
        assert [a.random() for _ in range(5)] != [b.random() for _ in range(5)]


class TestSampleJudge:
    def _panel(self):
        return [
            ResolvedJudge(name="a", base_url="u", model="m-a"),
            ResolvedJudge(name="b", base_url="u", model="m-b"),
        ]

    def test_single_member_always_returned(self) -> None:
        only = [ResolvedJudge(name="solo", base_url="u", model="m")]
        assert sample_judge(only, make_rng(None, "x")).name == "solo"

    def test_empty_panel_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_judge([], make_rng(None, "x"))

    def test_seeded_selection_is_reproducible(self) -> None:
        panel = self._panel()
        picks_1 = [sample_judge(panel, make_rng(None, "t", i)).name for i in range(10)]
        picks_2 = [sample_judge(panel, make_rng(None, "t", i)).name for i in range(10)]
        assert picks_1 == picks_2

    def test_zero_weight_member_never_picked(self) -> None:
        panel = [
            ResolvedJudge(name="live", base_url="u", model="m", weight=1.0),
            ResolvedJudge(name="dead", base_url="u", model="m", weight=0.0),
        ]
        rng = make_rng(0, "t")
        names = {sample_judge(panel, rng).name for _ in range(50)}
        assert names == {"live"}

    def test_all_zero_weights_fall_back_to_uniform(self) -> None:
        panel = [
            ResolvedJudge(name="a", base_url="u", model="m", weight=0.0),
            ResolvedJudge(name="b", base_url="u", model="m", weight=0.0),
        ]
        rng = make_rng(0, "t")
        names = {sample_judge(panel, rng).name for _ in range(50)}
        assert names == {"a", "b"}


class TestMergeCreateKwargs:
    def test_override_wins_and_none_drops(self) -> None:
        base = {"model": "m", "temperature": 1.0, "max_tokens": 100}
        merged = merge_create_kwargs(base, {"temperature": None, "max_tokens": 200, "reasoning_effort": "high"})
        assert merged == {"model": "m", "max_tokens": 200, "reasoning_effort": "high"}
        # base is not mutated
        assert base == {"model": "m", "temperature": 1.0, "max_tokens": 100}

    def test_none_overrides_is_noop(self) -> None:
        base = {"model": "m"}
        assert merge_create_kwargs(base, None) == {"model": "m"}


class TestPanelSummary:
    def test_summary_shape(self) -> None:
        panel = [
            ResolvedJudge(name="a", base_url="u", model="m-a", weight=1.0),
            ResolvedJudge(name="b", base_url="u", model="m-b", weight=2.0),
        ]
        assert panel_summary(panel) == [
            {"name": "a", "model": "m-a", "weight": 1.0},
            {"name": "b", "model": "m-b", "weight": 2.0},
        ]


class TestAudioVideoDetection:
    def test_is_audio_video_file(self) -> None:
        assert is_audio_video_file("clip.MP4")
        assert is_audio_video_file("song.wav")
        assert not is_audio_video_file("report.pdf")
        assert not is_audio_video_file("notes.txt")

    def test_missing_path_is_false(self, tmp_path) -> None:
        assert dir_contains_audio_video(tmp_path / "nope") is False
        assert dir_contains_audio_video(None) is False

    def test_detects_nested_audio_video(self, tmp_path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "clip.mov").write_bytes(b"\x00")
        assert dir_contains_audio_video(tmp_path) is True

    def test_text_only_dir_is_false(self, tmp_path) -> None:
        (tmp_path / "a.docx").write_text("x")
        (tmp_path / "b.pdf").write_text("y")
        assert dir_contains_audio_video(tmp_path) is False

    def test_detects_audio_video_inside_zip(self, tmp_path) -> None:
        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner/voice.mp3", b"\x00")
        assert dir_contains_audio_video(tmp_path) is True

    def test_zip_without_av_is_false(self, tmp_path) -> None:
        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner/report.txt", "hello")
        assert dir_contains_audio_video(tmp_path) is False

    def test_single_file_path(self, tmp_path) -> None:
        f = tmp_path / "clip.webm"
        f.write_bytes(b"\x00")
        assert dir_contains_audio_video(f) is True

    def test_per_modality_classifiers(self) -> None:
        assert is_audio_file("song.MP3") and not is_video_file("song.MP3")
        assert is_video_file("clip.mov") and not is_audio_file("clip.mov")
        assert not is_audio_video_file("report.pdf")

    def test_dir_media_modalities_splits_audio_and_video(self, tmp_path) -> None:
        (tmp_path / "a.wav").write_bytes(b"\x00")
        (tmp_path / "b.mp4").write_bytes(b"\x00")
        (tmp_path / "c.pdf").write_text("x")
        assert dir_media_modalities(tmp_path) == {"audio", "video"}

    def test_dir_media_modalities_video_only(self, tmp_path) -> None:
        (tmp_path / "b.mkv").write_bytes(b"\x00")
        (tmp_path / "c.txt").write_text("x")
        assert dir_media_modalities(tmp_path) == {"video"}

    def test_dir_media_modalities_peeks_zip(self, tmp_path) -> None:
        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner/voice.mp3", b"\x00")
        assert dir_media_modalities(tmp_path) == {"audio"}

    def test_dir_media_modalities_empty(self, tmp_path) -> None:
        assert dir_media_modalities(tmp_path / "nope") == set()
        assert dir_media_modalities(None) == set()


class TestSelectAvJudges:
    def test_narrows_to_flagged_members(self) -> None:
        panel = [
            ResolvedJudge(name="gpt", base_url="u", model="m"),
            ResolvedJudge(name="gemini", base_url="u", model="m", handles_audio=True, handles_video=True),
            ResolvedJudge(name="claude", base_url="u", model="m"),
        ]
        assert [j.name for j in select_av_judges(panel)] == ["gemini"]

    def test_falls_back_to_full_panel_when_none_flagged(self) -> None:
        panel = [
            ResolvedJudge(name="gpt", base_url="u", model="m"),
            ResolvedJudge(name="claude", base_url="u", model="m"),
        ]
        assert [j.name for j in select_av_judges(panel)] == ["gpt", "claude"]

    def test_video_only_member_is_selected(self) -> None:
        panel = [
            ResolvedJudge(name="gpt", base_url="u", model="m"),
            ResolvedJudge(name="minimax", base_url="u", model="m", handles_video=True),
        ]
        assert [j.name for j in select_av_judges(panel)] == ["minimax"]


def _judge_returning(name: str, verdict: str) -> Judge:
    """A Judge whose sync OpenAI client always returns *verdict* text."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=verdict))])
    return Judge(name=name, client=client, model=f"model-{name}")


class TestRunTrialsPanel:
    def test_requires_non_empty_panel(self) -> None:
        with pytest.raises(ValueError):
            run_trials(judges=[], task_prompt="p", refs=[], submission_a=[], submission_b=[], num_trials=2)

    def test_records_per_judge_and_trial_judges(self) -> None:
        # Both members always vote BOXED[B]; with position swapping this means
        # eval (submission_b) wins the unswapped trials and loses swapped ones,
        # but we only assert bookkeeping shape here.
        panel = [_judge_returning("a", "BOXED[B]"), _judge_returning("b", "BOXED[B]")]
        result = run_trials(
            judges=panel,
            task_prompt="p",
            refs=[],
            submission_a=[{"role": "user", "content": "ref"}],
            submission_b=[{"role": "user", "content": "eval"}],
            num_trials=6,
            rng=make_rng(0, "task", "repeat_0"),
        )
        assert len(result["trial_judges"]) == 6
        assert set(result["per_judge"]) <= {"a", "b"}
        total_trials = sum(jc["trials"] for jc in result["per_judge"].values())
        assert total_trials == 6

    def test_single_member_panel_matches_all_trials(self) -> None:
        panel = [_judge_returning("solo", "BOXED[TIE]")]
        result = run_trials(
            judges=panel,
            task_prompt="p",
            refs=[],
            submission_a=[],
            submission_b=[],
            num_trials=4,
            rng=make_rng(0, "t"),
        )
        assert result["trial_judges"] == ["solo"] * 4
        assert result["per_judge"]["solo"]["trials"] == 4
        assert result["tie_count"] == 4

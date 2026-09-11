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
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from openai import OpenAI

from nemo_gym import server_utils
from resources_servers.gdpval import app as gdpval_app
from resources_servers.gdpval import setup_libreoffice
from resources_servers.gdpval.comparison import (
    B_WIN_RESPONSE,
    FILE_TYPE_MAP,
    TIE_RESPONSE,
    Judge,
    parse_judgement,
    run_trials,
    send_judge_request,
)
from resources_servers.gdpval.judge_panel import (
    AUDIO_EXTS,
    VIDEO_EXTS,
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
from responses_api_models.openai_model import app as proxy_app


@pytest.mark.parametrize("gemini_key", [None, "gemini-fixture-key"])
@pytest.mark.parametrize("custom_models", [False, True])
def test_benchmark_panel_reaches_intended_upstream_models_and_credentials(monkeypatch, gemini_key, custom_models):
    """Resolve the benchmark, then traverse the real judge request and fixed-model proxy."""
    monkeypatch.setattr(setup_libreoffice, "ensure_libreoffice", lambda: True)
    shared_key = "shared-fixture-key"
    monkeypatch.setenv("JUDGE_API_KEY", shared_key)
    monkeypatch.setenv("JUDGE_BASE_URL", "https://upstream.invalid/v1")
    if gemini_key is None:
        monkeypatch.delenv("JUDGE_GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("JUDGE_GEMINI_API_KEY", gemini_key)
    expected_models = {
        "gpt-5.5": ("JUDGE_GPT_MODEL", "openai/openai/gpt-5.5"),
        "gemini-3.1-pro": ("JUDGE_GEMINI_MODEL", "gcp/google/gemini-3.1-pro-preview"),
        "claude-opus-4.8": ("JUDGE_CLAUDE_MODEL", "aws/anthropic/bedrock-claude-opus-4-8"),
    }
    for name, (variable, model) in list(expected_models.items()):
        monkeypatch.delenv(variable, raising=False)
        if custom_models:
            model += "-fixture-override"
            monkeypatch.setenv(variable, model)
        expected_models[name] = model
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)
    monkeypatch.delenv("GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS", raising=False)
    config = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).resolve().parents[3] / "benchmarks/gdpval/config.yaml"), resolve=True
    )

    class CapturingClient:
        def __init__(self, **connection):
            self.connection = connection
            self.requests = []

        async def create_chat_completion(self, **body):
            self.requests.append(body)
            return {
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "BOXED[B]"}}
                ],
            }

    monkeypatch.setattr(proxy_app, "NeMoGymAsyncOpenAI", CapturingClient)
    proxies = {}
    for name, block in config.items():
        if not isinstance(block, dict) or "responses_api_models" not in block:
            continue
        fields = block["responses_api_models"]["openai_model"]
        fields.update(host="127.0.0.1", port=12000 + len(proxies), name=name)
        proxy = proxy_app.SimpleModelServer(
            config=proxy_app.SimpleModelServerConfig(**fields),
            server_client=MagicMock(spec=server_utils.ServerClient, global_config_dict=config),
        )
        proxies[f"http://{fields['host']}:{fields['port']}/v1"] = proxy
    monkeypatch.setattr(server_utils, "get_global_config_dict", lambda: config)
    resource = gdpval_app.GDPValResourcesServer.model_construct(
        config=gdpval_app.GDPValResourcesServerConfig(
            host="127.0.0.1",
            port=13000,
            name="gdpval_resources_server",
            **config["gdpval_resources_server"]["resources_servers"]["gdpval"],
        )
    )
    panel = resource._resolve_judges()
    assert {judge.name for judge in panel} == set(expected_models)
    for judge in panel:
        proxy = proxies[judge.base_url]
        with TestClient(proxy.setup_webserver()) as transport:
            with OpenAI(
                base_url=judge.base_url, api_key=judge.api_key, http_client=transport, max_retries=0
            ) as client:
                assert (
                    send_judge_request(
                        client,
                        judge.model,
                        [{"role": "user", "content": "fixture"}],
                        max_output_tokens=16,
                        create_overrides=judge.create_overrides,
                    )
                    == "BOXED[B]"
                )
        body = proxy._client.requests[-1]
        assert body["model"] == expected_models[judge.name]
        assert proxy._client.connection["base_url"] == "https://upstream.invalid/v1"
        expected_key = (gemini_key or shared_key) if judge.name == "gemini-3.1-pro" else shared_key
        assert proxy._client.connection["api_key"] == expected_key
        if judge.name == "gpt-5.5":
            assert body["reasoning_effort"] == "medium"
        elif judge.name == "gemini-3.1-pro":
            assert body["reasoning_effort"] == "high"
        else:
            assert body["thinking"] == {"type": "adaptive"}
            assert body["output_config"] == {"effort": "high"}
            assert body["timeout"] == body["request_timeout"] == 900
            assert "temperature" not in body


class TestMediaExtensionSetsAgree:
    """Detection, routing and emission must share one definition of AV.

    Routing decides judge capability from judge_panel's sets. Any extension it
    detects but an emitter can't build a block for is silently stubbed/dropped,
    so the judge scores a task whose media it never saw -- wrong votes, no error.
    """

    def test_comparison_emitter_covers_every_audio_ext(self) -> None:
        missing = {e for e in AUDIO_EXTS if e.lstrip(".") not in FILE_TYPE_MAP}
        assert not missing, f"FILE_TYPE_MAP is missing audio extensions: {sorted(missing)}"

    def test_comparison_emitter_covers_every_video_ext(self) -> None:
        missing = {e for e in VIDEO_EXTS if e.lstrip(".") not in FILE_TYPE_MAP}
        assert not missing, f"FILE_TYPE_MAP is missing video extensions: {sorted(missing)}"

    def test_comparison_emitter_classifies_media_correctly(self) -> None:
        for ext in AUDIO_EXTS:
            assert FILE_TYPE_MAP[ext.lstrip(".")]["type"] == "AUDIO", ext
        for ext in VIDEO_EXTS:
            assert FILE_TYPE_MAP[ext.lstrip(".")]["type"] == "VIDEO", ext

    def test_comparison_emitter_has_no_dotted_media_keys(self) -> None:
        """Guards the old 'x-flv' / '3gpp' typos, which keyed on the mime subtype.

        get_file_content_block looks up the bare extension, so a key like 'x-flv'
        can never match and the file falls through to the DOC/PDF branch.
        """
        media_keys = {k for k, v in FILE_TYPE_MAP.items() if v["type"] in ("AUDIO", "VIDEO")}
        expected = {e.lstrip(".") for e in AUDIO_EXTS | VIDEO_EXTS}
        assert media_keys == expected

    def test_rubric_emitter_covers_every_media_ext(self) -> None:
        from responses_api_agents.stirrup_agent.file_reader import MIME_TYPES

        missing = (AUDIO_EXTS | VIDEO_EXTS) - set(MIME_TYPES)
        assert not missing, f"file_reader.MIME_TYPES is missing: {sorted(missing)}"

    def test_rubric_emitter_reuses_canonical_sets(self) -> None:
        from responses_api_agents.stirrup_agent import file_reader

        assert file_reader.AUDIO_EXTS is AUDIO_EXTS
        assert file_reader.VIDEO_EXTS is VIDEO_EXTS


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


def _judge_returning_sequence(name: str, verdicts: list[str]) -> Judge:
    """A Judge whose sync OpenAI client returns each response in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=verdict))]) for verdict in verdicts
    ]
    return Judge(name=name, client=client, model=f"model-{name}")


class TestParseJudgement:
    def test_last_boxed_verdict_wins(self) -> None:
        assert parse_judgement("not BOXED[A]; final answer BOXED[B]") == B_WIN_RESPONSE

    @pytest.mark.parametrize("response", ["", "A wins", "BOXED[C]"])
    def test_missing_valid_verdict_is_invalid(self, response: str) -> None:
        assert parse_judgement(response) is None

    def test_explicit_tie_is_valid(self) -> None:
        assert parse_judgement("Final answer: BOXED[TIE]") == TIE_RESPONSE


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

    def test_invalid_response_is_excluded_from_counts(self) -> None:
        panel = [_judge_returning_sequence("solo", ["BOXED[B]", "malformed", "BOXED[B]"])]
        result = run_trials(
            judges=panel,
            task_prompt="p",
            refs=[],
            submission_a=[],
            submission_b=[],
            num_trials=3,
            return_raw_responses=True,
        )

        assert result["winner"] == B_WIN_RESPONSE
        assert result["win_count_b"] == 2
        assert result["tie_count"] == 0
        assert result["task_count"] == 2
        assert result["invalid_count"] == 1
        assert result["per_judge"]["solo"]["trials"] == 2
        assert result["per_judge"]["solo"]["invalid_count"] == 1
        assert len(result["raw_responses"]) == 3

    def test_all_invalid_responses_fail_matchup(self) -> None:
        panel = [_judge_returning("solo", "malformed")]
        with pytest.raises(ValueError, match="All 2 pairwise judge responses were invalid"):
            run_trials(
                judges=panel,
                task_prompt="p",
                refs=[],
                submission_a=[],
                submission_b=[],
                num_trials=2,
            )

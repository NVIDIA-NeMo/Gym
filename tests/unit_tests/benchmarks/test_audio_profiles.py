# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile file loading, repository-relative path resolution and the source-row digest."""

import json

import pytest

from resources_servers.audio_eval.profiles import REPO_ROOT, load_profiles, resolve_repo_path, source_hash


def test_load_profiles_reads_and_refreshes_when_the_file_changes(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": {"a": {}}}))
    assert list(load_profiles(path)["profiles"]) == ["a"]
    path.write_text(json.dumps({"profiles": {"a": {}, "b": {}}}))
    assert list(load_profiles(path)["profiles"]) == ["a", "b"]


def test_load_profiles_rejects_a_file_without_profiles(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Invalid audio benchmark profiles"):
        load_profiles(path)


def test_defaults_merge_one_level_deep_and_judge_templates_resolve_by_name(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "request": {"preset": "greedy", "max_output_tokens": 1024},
                    "prompt_override": {"system_prompt": "", "user_prompt": "u"},
                    "skills": {"METRICS_TYPE": "audio"},
                },
                "judge_prompt_templates": {"j": {"user": "grade"}},
                "profiles": {
                    "plain": {},
                    "long": {"request": {"max_output_tokens": 16384}},
                    "preset": {"request": "reasoning"},
                    "judged": {"judge_prompt_template": "j", "prompt_override": {"user_prompt": None}},
                },
            }
        )
    )
    profiles = load_profiles(path)["profiles"]
    assert profiles["plain"] == {
        "request": {"preset": "greedy", "max_output_tokens": 1024},
        "prompt_override": {"system_prompt": "", "user_prompt": "u"},
        "skills": {"METRICS_TYPE": "audio"},
    }
    assert profiles["long"]["request"] == {"preset": "greedy", "max_output_tokens": 16384}
    assert profiles["preset"]["request"] == "reasoning"
    assert profiles["judged"]["judge_prompt_template"] == {"user": "grade"}
    assert profiles["judged"]["prompt_override"] == {"system_prompt": "", "user_prompt": None}
    assert "judge_prompt_template" not in profiles["plain"]
    profiles["plain"]["skills"]["METRICS_TYPE"] = "changed"
    assert profiles["long"]["skills"]["METRICS_TYPE"] == "audio"


def test_repo_relative_paths_resolve_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        resolve_repo_path("resources_servers/audio_eval/profiles.py")
        == REPO_ROOT / "resources_servers/audio_eval/profiles.py"
    )
    absolute = tmp_path / "profiles.json"
    assert resolve_repo_path(str(absolute)) == absolute
    (tmp_path / "local.json").write_text("{}")
    assert resolve_repo_path("local.json").resolve() == tmp_path / "local.json"


def test_source_hash_is_order_independent_and_content_sensitive():
    assert source_hash({"a": 1, "b": "é"}) == source_hash({"b": "é", "a": 1})
    assert source_hash({"a": 1}) != source_hash({"a": 2})

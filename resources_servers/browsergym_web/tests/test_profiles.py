# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web.models import WebBenchmark, WebObservationProfile, WebTask
from resources_servers.browsergym_web.profiles import resolve_browsergym_profile


def test_webarena_and_visualwebarena_resolve_distinct_profiles():
    webarena = resolve_browsergym_profile(WebTask(benchmark=WebBenchmark.WEBARENA, task_id=310))
    visual = resolve_browsergym_profile(WebTask(benchmark=WebBenchmark.VISUALWEBARENA, task_id=721))

    assert webarena.env_id == "browsergym/webarena.310"
    assert webarena.action_subsets == ("webarena",)
    assert webarena.observation_profile == WebObservationProfile.A11Y
    assert visual.env_id == "browsergym/visualwebarena.721"
    assert visual.action_subsets == ("visualwebarena",)
    assert visual.observation_profile == WebObservationProfile.SOM


def test_webvoyager_uses_openended_task_and_external_verifier():
    profile = resolve_browsergym_profile(
        WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Allrecipes--0",
            intent="Find a recipe",
            start_urls=["https://www.allrecipes.com/"],
        )
    )

    assert profile.env_id == "browsergym/openended"
    assert profile.task_kwargs == {
        "start_url": "https://www.allrecipes.com/",
        "goal": "Find a recipe",
    }
    assert profile.external_verifier is True


def test_webvoyager_requires_a_start_url():
    with pytest.raises(ValueError, match="start URL"):
        resolve_browsergym_profile(WebTask(benchmark=WebBenchmark.WEBVOYAGER, task_id="missing-url"))

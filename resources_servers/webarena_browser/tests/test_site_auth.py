# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web.models import WebTask
from resources_servers.webarena_browser.site_auth import (
    configured_site_urls,
    resolve_site_templates,
    resolve_start_urls,
)


def _task(**updates) -> WebTask:
    values = {
        "benchmark": "webarena",
        "task_id": "0",
        "runtime_profile": "visual_browser",
        "action_profile": "computer_use",
        "sites": ["gitlab", "reddit"],
        "start_urls": ["__GITLAB__/group/project", "__REDDIT__ |AND| __GITLAB_SSH__"],
    }
    values.update(updates)
    return WebTask.model_validate(values)


def test_site_urls_are_resolved_from_task_scoped_environment(monkeypatch) -> None:
    monkeypatch.setenv("WA_GITLAB", "http://sites.test:8023/")
    monkeypatch.setenv("WA_REDDIT", "http://sites.test:9999")
    task = _task()

    urls = configured_site_urls(task)

    assert urls == {
        "gitlab": "http://sites.test:8023",
        "reddit": "http://sites.test:9999",
    }
    assert resolve_start_urls(task, urls) == [
        "http://sites.test:8023/group/project",
        "http://sites.test:9999 |AND| sites.test:2222",
    ]


def test_missing_required_site_url_fails_before_browser_launch(monkeypatch) -> None:
    monkeypatch.delenv("WA_GITLAB", raising=False)
    monkeypatch.setenv("WA_REDDIT", "http://sites.test:9999")

    with pytest.raises(ValueError, match="WA_GITLAB"):
        configured_site_urls(_task())


def test_evaluator_placeholders_require_and_resolve_nested_site_urls(monkeypatch) -> None:
    monkeypatch.setenv("WA_GITLAB", "http://sites.test:8023")
    task = _task(
        sites=[],
        start_urls=[],
        original_metadata={
            "eval": {
                "program_html": [
                    {
                        "url": "__GITLAB__/group/project/-/issues/1",
                        "locator": "document.querySelector('body').innerText",
                    }
                ]
            }
        },
    )

    urls = configured_site_urls(task)
    resolved = resolve_site_templates(task.original_metadata, urls)

    assert urls == {"gitlab": "http://sites.test:8023"}
    assert resolved["eval"]["program_html"][0]["url"] == "http://sites.test:8023/group/project/-/issues/1"

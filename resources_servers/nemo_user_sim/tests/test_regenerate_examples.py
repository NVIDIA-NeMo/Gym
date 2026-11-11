# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from resources_servers.nemo_user_sim.scripts.regenerate_examples import regenerate


def _example() -> dict:
    return {
        "task_id": "example-tool-calling",
        "resolved_row": {
            "probe_type": "tool_calling",
            "locale": "en_US",
            "persona": "Synthetic persona",
            "persona_name": "Avery",
            "conversation_language": "English",
        },
    }


def _prepared(locale: str, language: str) -> dict:
    return {
        "task_id": f"prepared-{locale}",
        "resolved_row": {
            "probe_type": "tool_calling",
            "locale": locale,
            "persona": f"Nemotron persona from {locale}",
            "persona_name": f"Name from {locale}",
            "conversation_language": language,
            "usersim_config": {"locale": locale},
            "trajectory_id": f"prepared-{locale}",
            "usersim_provenance": json.dumps({"code_sha": "e6282672a5e5", "nemotron_personas_version": "0.0.2"}),
        },
    }


@pytest.mark.parametrize("order", [("en_US", "ja_JP"), ("ja_JP", "en_US")])
def test_regenerate_uses_the_prepared_row_of_the_example_locale(order: tuple[str, str]) -> None:
    languages = {"en_US": "English", "ja_JP": "Japanese"}
    prepared = [_prepared(locale, languages[locale]) for locale in order]

    [rebuilt] = regenerate([_example()], prepared)

    row = rebuilt["resolved_row"]
    assert rebuilt["task_id"] == "example-tool-calling"
    assert row["locale"] == "en_US"
    assert row["conversation_language"] == "English"
    assert row["usersim_config"] == {"locale": "en_US"}
    assert row["persona"] == "Synthetic persona"
    assert row["persona_name"] == "Avery"
    assert row["trajectory_id"] == "example-tool-calling"
    assert json.loads(row["usersim_provenance"]) == {
        "code_sha": "e6282672a5e5",
        "nemotron_personas_version": "synthetic",
    }


def test_regenerate_fails_without_a_prepared_row_in_the_example_locale() -> None:
    with pytest.raises(ValueError, match="'tool_calling' in locale 'en_US'"):
        regenerate([_example()], [_prepared("ja_JP", "Japanese")])

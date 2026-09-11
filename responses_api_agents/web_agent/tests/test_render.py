# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web.models import (
    WebBenchmark,
    WebImage,
    WebObservation,
    WebObservationProfile,
    WebTask,
)
from responses_api_agents.web_agent.render import (
    TASK_INPUT_IMAGE_REDACTION_NOTICE,
    compact_som_text,
    render_observation,
)


def _block_types(message):
    return [
        block.get("type") if isinstance(block, dict) else getattr(block, "type", None) for block in message.content
    ]


def _block_text(block):
    return block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")


def test_visual_browser_render_is_screenshot_first_and_has_no_text_action_grammar():
    task = WebTask(
        benchmark="webvoyager",
        task_id="Allrecipes--0",
        intent="Find a recipe",
        runtime_profile="visual_browser",
        observation_profile="screenshot",
        action_profile="computer_use",
    )
    observation = WebObservation(
        goal=[{"type": "text", "text": "Find a recipe"}],
        screenshot=WebImage(data_url="data:image/png;base64,AA=="),
        url="https://example.com",
    )

    message = render_observation(observation, task, step_index=0)

    assert _block_types(message) == ["input_image", "input_text"]
    text = _block_text(message.content[1])
    assert "# Task Instruction:" in text
    assert "Step 1" in text
    assert "Action:" not in text


def test_visual_runtime_keeps_screenshot_when_optional_a11y_text_is_present():
    task = WebTask(
        benchmark=WebBenchmark.WEBARENA,
        task_id="0",
        observation_profile=WebObservationProfile.A11Y,
    )
    observation = WebObservation(
        goal=[{"type": "text", "text": "Find the answer"}],
        axtree_text="[a1] button 'Search'",
        screenshot=WebImage(data_url="data:image/png;base64,abc"),
    )

    message = render_observation(observation, task, step_index=0)

    assert _block_types(message) == ["input_image", "input_text"]
    assert "[a1] button" in _block_text(message.content[1])


def test_som_profile_includes_page_and_goal_images():
    task = WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id="0",
        observation_profile=WebObservationProfile.SOM,
    )
    observation = WebObservation(
        goal=[
            {"type": "text", "text": "Find the matching item"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,goal"}},
        ],
        screenshot=WebImage(data_url="data:image/png;base64,page"),
    )

    message = render_observation(observation, task, step_index=0)

    assert _block_types(message) == [
        "input_image",
        "input_text",
        "input_text",
        "input_image",
        "input_text",
    ]


def test_visual_browser_reference_image_is_loaded_from_explicit_root(tmp_path):
    image = tmp_path / "images" / "reference.png"
    image.parent.mkdir()
    image.write_bytes(b"png-payload")
    task = WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id="0",
        input_images=["images/reference.png"],
        runtime_profile="visual_browser",
        observation_profile="screenshot",
        action_profile="computer_use",
    )

    message = render_observation(
        WebObservation(),
        task,
        step_index=0,
        task_image_root=str(tmp_path),
    )

    assert _block_types(message) == ["input_text", "input_text", "input_image", "input_text"]
    assert message.content[1]["text"] == "Task image 1 of 1:"
    assert message.content[2]["image_url"] == "data:image/png;base64,cG5nLXBheWxvYWQ="

    later = render_observation(
        WebObservation(),
        task,
        step_index=1,
        task_image_root=str(tmp_path),
    )
    assert _block_types(later) == ["input_text"]
    assert "# Task Instruction:" in later.content[0]["text"]
    assert TASK_INPUT_IMAGE_REDACTION_NOTICE in later.content[0]["text"]


def test_visual_browser_reference_image_cannot_escape_explicit_root(tmp_path):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"not-readable-through-task-metadata")
    task = WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id="0",
        input_images=["../outside.png"],
        runtime_profile="visual_browser",
        action_profile="computer_use",
    )

    with pytest.raises(ValueError, match="outside task_image_root"):
        render_observation(
            WebObservation(),
            task,
            step_index=0,
            task_image_root=str(tmp_path),
        )


def test_som_only_text_keeps_only_labelled_interactive_elements():
    task = WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id="ArXiv--13",
        observation_profile=WebObservationProfile.SOM,
    )
    observation = WebObservation(
        axtree_text=(
            "RootWebArea 'Example'\n"
            "\t[10] link 'Search', som, expanded=False\n"
            "\t[11] heading 'News'\n"
            "\t\tStaticText 'body text'\n"
            "\t[12] textbox 'Query', som\n"
        ),
        screenshot=WebImage(data_url="data:image/png;base64,page"),
    )

    message = render_observation(
        observation,
        task,
        step_index=0,
        visual_observation_text="som_only",
    )
    text = _block_text(message.content[1])

    assert "[10] link 'Search', expanded=False" in text
    assert "[12] textbox 'Query'" in text
    assert "heading 'News'" not in text
    assert "body text" not in text
    assert "Accessibility tree" not in text


def test_compact_som_text_has_a_hard_character_budget():
    axtree = "\n".join(f"[{index}] link '{'x' * 200}', som" for index in range(100))

    compact = compact_som_text(axtree, max_chars=500)

    assert len(compact) < 550
    assert compact.endswith("[Additional labelled elements omitted.]")

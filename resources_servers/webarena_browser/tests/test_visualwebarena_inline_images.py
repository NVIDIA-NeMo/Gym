# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote_from_bytes

import pytest
from PIL import Image

from nemo_gym.web import VisualBrowserEvaluationContext, WebObservation
from resources_servers.webarena_browser import reference_evaluation as reference
from resources_servers.webarena_browser.evaluators import WebArenaTaskEvaluator
from resources_servers.webarena_browser.reference_evaluation import visualwebarena_evaluation as visual
from resources_servers.webarena_browser.tests.test_contract import _config, _visualwebarena_task


def _image_url(color, *, image_format="PNG", encoding="base64"):
    content = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(content, format=image_format)
    if encoding == "base64":
        return f"data:image/{image_format.lower()};base64," + base64.b64encode(content.getvalue()).decode()
    return f"data:image/{image_format.lower()}," + quote_from_bytes(content.getvalue())


SVG_PLACEHOLDER = "data:image/svg+xml," + quote_from_bytes(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'
)


@pytest.mark.parametrize("image_format", ["PNG", "GIF"])
@pytest.mark.parametrize("encoding", ["base64", "percent"])
def test_inline_bitmap_pixels_are_decoded_without_http(monkeypatch, image_format, encoding):
    fetch = Mock(side_effect=AssertionError("inline images must not use HTTP"))
    monkeypatch.setattr(visual.httpx, "get", fetch)

    image = visual._fetch_image(_image_url((9, 8, 7), image_format=image_format, encoding=encoding))

    assert image.mode == "RGB"
    assert image.size == (8, 8)
    assert image.getpixel((0, 0)) == (9, 8, 7)
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        SVG_PLACEHOLDER,
        "data:image/png;base64,not-base64!",
        "data:image/png;base64",
        "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
        "data:image/png," + quote_from_bytes(b"not an image"),
    ],
)
def test_unusable_inline_candidates_have_the_same_absence_semantics_as_http_images(monkeypatch, source):
    fetch = Mock(side_effect=AssertionError("inline images must not use HTTP"))
    monkeypatch.setattr(visual.httpx, "get", fetch)

    with pytest.raises(visual._UnavailableTaskImage):
        visual._fetch_image(source)

    fetch.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["sync", "async", "public"])
@pytest.mark.parametrize("color,expected", [(None, 0), ("red", 1), ("blue", 0)])
@pytest.mark.asyncio
async def test_inline_placeholders_do_not_mask_a_conclusive_image_verdict(monkeypatch, entrypoint, color, expected):
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret
    fetch = Mock(side_effect=AssertionError("inline images must not use HTTP"))
    monkeypatch.setattr(visual.httpx, "get", fetch)
    sources = [SVG_PLACEHOLDER]
    if color:
        sources.append(_image_url(color))
    method = AsyncMock if entrypoint == "async" else Mock
    elements = [
        SimpleNamespace(
            get_attribute=method(side_effect=lambda name, source=source: source if name == "src" else None)
        )
        for source in sources
    ]
    page = SimpleNamespace(
        url="https://example.test/page",
        is_closed=Mock(return_value=False),
        query_selector_all=method(return_value=elements),
    )
    page.context = SimpleNamespace(pages=[page])
    calls = []

    def judge(images, question, expected_answer, judge_log_path, image_log_paths=None):
        assert question == "Is the image red?"
        assert expected_answer == "yes"
        assert len(images) == 1
        calls.append(images[0].getpixel((0, 0)))
        return float(calls[-1] == (255, 0, 0))

    monkeypatch.setattr(visual, "_vlm_vqa_score", judge)
    task_config = {
        "eval": {
            "eval_types": ["page_image_query"],
            "page_image_query": [
                {
                    "eval_image_url": "last",
                    "eval_image_class": "",
                    "eval_vqa": [{"question": "Is the image red?", "answer": "yes"}],
                }
            ],
        }
    }
    if entrypoint == "public":
        monkeypatch.setattr(reference, "collect_snapshots", lambda *_args: {})
        monkeypatch.setattr(reference, "collect_browser_snapshots_sync", lambda *_args: {})
        task = _visualwebarena_task(original_metadata=task_config, task_kwargs={"collision_plan": {}})
        evaluator = WebArenaTaskEvaluator(config=_config())
        context = VisualBrowserEvaluationContext(page=page, browser_context=page.context, evidence=())
        evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
        result = evaluator.evaluate(
            task=task, observation=WebObservation(), final_answer="done", browser_context=context
        )
        assert result.valid_sample
        assert not result.failure_kind
        assert result.reward == expected
    else:
        if entrypoint == "async":
            score, message = await visual.evaluate_visualwebarena_task(task_config, {"answer": "done"}, page)
        else:
            score, message = visual.evaluate_visualwebarena_task_sync(task_config, {"answer": "done"}, page)
        assert not message.startswith("VisualWebArena evaluation error:")
        assert score == expected
    assert len(calls) == int(color is not None)
    fetch.assert_not_called()

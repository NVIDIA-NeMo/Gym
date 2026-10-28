# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation mode ``webdev_eval_v1``: render in the pod, one vision call, absolute score.

    score = (visual + query_fulfillment + premium_assets) / 3

This is the **evaluation** reward path, not a training one. It runs as verl's rollout-only
mode -- ``trainer.val_only=True`` AND ``trainer.val_before_train=True``, both required -- and
the number lands in the validation reward mean. The rubric and the reasoning behind its three
terms are in ``eval_rubric.py``.

**Why evaluation cannot reuse the training grader.** Training's ``design_group_v1`` ships the
workspace to the grading service, which returns a group-relative rank. A relative rank has no
absolute meaning for a single arm and is not comparable across arms, so it cannot evaluate.
This path therefore renders the site itself (``shot.py``, playwright inside the pod) and calls
the judge itself (``_llm_client.py``, any OpenAI-compatible vision endpoint). The cost is a
second rendering implementation; what it buys is an evaluation that does not need the grading
service to be running, which matters because evaluation is usually done where it is not.

**Three behaviours to know before reading a number out of this:**

1. **A render failure is not a bad page.** ``render_failed`` / ``screenshot_empty`` /
   ``render_env_failed`` return ``reward=None`` so the caller MASKS the trajectory instead of
   averaging a 0 into the result. Counting infra failures as zeros misreads the mean
   outright: one measured run had 57.9% judge failures and reported 0.351 for an arm whose
   real mean was 0.677 -- which reversed the conclusion.
2. **A judge failure returns None** for the same reason. Only "the judge answered and the
   page is genuinely poor" is a 0.
3. **The judge rejects oversized screenshots permanently** -- 23.0 MP was accepted and
   46.5 MP refused, and retrying does not help -- so every shot is scaled proportionally at
   ``MAX_MEGAPIXELS`` before being sent. Deliberately NOT resized to a fixed width: a page
   that overflows horizontally has a width that is itself the defect ``layout_integrity``
   exists to catch, and normalising it would hide the fault on the page's behalf.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time


MODE = "webdev_eval_v1"

MAX_MEGAPIXELS = float(os.getenv("WEBDEV_EVAL_MAX_MP", "20"))
JUDGE_RETRIES = int(os.getenv("WEBDEV_EVAL_JUDGE_RETRIES", "4"))
JUDGE_TIMEOUT_S = int(os.getenv("WEBDEV_EVAL_JUDGE_TIMEOUT", "180"))
SHOT_TIMEOUT_S = int(os.getenv("WEBDEV_EVAL_SHOT_TIMEOUT", "150"))
QUERY_CAP = int(os.getenv("WEBDEV_EVAL_QUERY_CAP", "1500"))

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _drop(reason: str, extra: dict | None = None) -> dict:
    """Infra failure -> ``reward=None``, so the caller masks the trajectory, not scores it 0."""
    g = {"mode": MODE, "verdict": "drop", "drop_reason": reason}
    if extra:
        g.update(extra)
    return {"grading": g, "grader_reward": None, "grader_turn_scores": [], "grader_cost": {}}


def _shrink(jpeg_b64: str) -> tuple[str, bool]:
    """Scale proportionally if above ``MAX_MEGAPIXELS``. Returns ``(b64, was_shrunk)``.

    Passes the image through unchanged when PIL is unavailable: a missing optional dependency
    must not turn into a drop.
    """
    try:
        from PIL import Image
    except ImportError:
        return jpeg_b64, False
    raw = base64.b64decode(jpeg_b64)
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(io.BytesIO(raw))
    if im.width * im.height <= MAX_MEGAPIXELS * 1e6:
        return jpeg_b64, False
    k = ((MAX_MEGAPIXELS * 1e6) / (im.width * im.height)) ** 0.5
    im = im.convert("RGB").resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)  # same quality the in-pod screenshot uses
    return base64.b64encode(buf.getvalue()).decode(), True


def parse_verdict(raw_text: str) -> dict:
    """Judge JSON -> per-dimension scores, ``visual``, ``score``, ``reason``.

    Returns ``{"_failed": why}`` when it cannot be parsed. A missing dimension counts as a
    parse failure rather than defaulting to 0: defaulting would turn "the judge forgot to
    mention a dimension" into "the page scores 0 on it", inventing a low score out of
    nothing.
    """
    from .eval_rubric import ALL_KEYS, VISUAL_KEYS

    m = _JSON_RE.search(raw_text or "")
    if not m:
        return {"_failed": f"no json in judge output: {(raw_text or '')[:160]}"}
    try:
        data = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        return {"_failed": f"json error: {type(e).__name__}: {e}"}
    dims: dict[str, float] = {}
    for k in ALL_KEYS:
        v = data.get(k)
        if v is None:
            return {"_failed": f"judge omitted dimension {k!r}"}
        try:
            dims[k] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return {"_failed": f"dimension {k!r} is not a number: {v!r}"}
    visual = sum(dims[k] for k in VISUAL_KEYS) / len(VISUAL_KEYS)
    score = (visual + dims["query_fulfillment"] + dims["premium_assets"]) / 3.0
    return {
        "dims": dims,
        "visual": round(visual, 4),
        "score": round(score, 4),
        "reason": str(data.get("reason", ""))[:500],
    }


def _judge(image_b64: str, query: str, log=print) -> tuple[dict, int]:
    """One vision call, with the outer retry. Returns ``(parse_verdict result, tries used)``."""
    from ._llm_client import RequestsChatModel
    from .eval_rubric import build_prompt

    prompt = build_prompt().format(query=(query or "")[:QUERY_CAP])
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
        {"type": "text", "text": prompt},
    ]
    try:
        client = RequestsChatModel(
            model_name=os.getenv("WEBDEV_EVAL_JUDGE_MODEL"),
            base_url=os.getenv("WEBDEV_EVAL_JUDGE_BASE_URL") or os.getenv("LLM_JUDGE_BASE_URL"),
            api_key=os.getenv("WEBDEV_EVAL_JUDGE_API_KEY") or os.getenv("LLM_JUDGE_API_KEY", ""),
            temperature=1.0,
            timeout=JUDGE_TIMEOUT_S,
        )
    except ValueError as e:
        return {"_failed": str(e)}, 0

    last = None
    for attempt in range(JUDGE_RETRIES):
        if attempt:
            time.sleep(min(60.0, 3.0 * (3 ** (attempt - 1))) * (0.5 + random.random()))
        try:
            raw = client.completion([{"role": "user", "content": content}])
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:200]
            continue
        v = parse_verdict(raw)
        if "_failed" not in v:
            return v, attempt + 1
        last = v["_failed"]
    return {"_failed": f"judge failed after {JUDGE_RETRIES} tries: {last}"}, JUDGE_RETRIES


def grade_eval(env, deliver_dir: str, query: str, cfg: dict, *, log=print) -> dict:
    """Render in the pod, judge, score. ``grade()`` routes here when the mode is ``MODE``."""
    from .shot import _build_shot_cmd, _parse_render_env, _parse_shot_b64

    t0 = time.perf_counter()
    page = (cfg or {}).get("entry_page") or "index.html"
    proxy = os.getenv("POD_PROXY", "")
    url = f"file://{deliver_dir.rstrip('/')}/{page.lstrip('/')}"
    r = env.execute(_build_shot_cmd(url, proxy=proxy), "/", SHOT_TIMEOUT_S)
    shot_s = time.perf_counter() - t0
    if r.get("reason") != "ok":
        return _drop(f"pod exec failed: {r.get('reason')!r}", {"render_failed": True})
    shot_b64, console_errors = _parse_shot_b64(r)
    render_env = _parse_render_env(r)
    if render_env.get("proxy_failed"):
        return _drop(
            "render env failed: external assets unreachable on every route: "
            + "; ".join(render_env.get("fatal") or [])[:240],
            {"render_env": render_env},
        )
    if not shot_b64:
        return _drop("screenshot produced no image", {"console_errors": console_errors[:300]})

    shot_b64, shrunk = _shrink(shot_b64)
    t1 = time.perf_counter()
    v, tries = _judge(shot_b64, query, log=log)
    judge_s = time.perf_counter() - t1
    if "_failed" in v:
        return _drop(f"judge: {v['_failed']}", {"judge_tries": tries})

    from .eval_rubric import RUBRIC_ID

    grading = {
        "mode": MODE,
        "rubric": RUBRIC_ID,
        "score": v["score"],
        "visual": v["visual"],
        "dims": v["dims"],
        "reason": v["reason"],
        "reasoning": (
            f"score {v['score']:.3f} = mean(visual {v['visual']:.2f}, "
            f"query {v['dims']['query_fulfillment']:.2f}, "
            f"asset {v['dims']['premium_assets']:.2f})"
        ),
        "judge_tries": tries,
        "shot_downscaled": shrunk,
        "console_errors": console_errors[:300],
        "shot_s": round(shot_s, 2),
        "judge_s": round(judge_s, 2),
        "grade_total_s": round(time.perf_counter() - t0, 2),
    }
    if render_env.get("recovered"):
        grading["render_env_recovered"] = True
    log(f"[webdev eval] {grading['reasoning']} tries={tries} shot={shot_s:.1f}s judge={judge_s:.1f}s")
    return {
        "grading": grading,
        "grader_reward": float(v["score"]),
        "grader_turn_scores": [],
        "grader_cost": {},
        "shot_jpg_b64": shot_b64,
    }


__all__ = ["MODE", "grade_eval", "parse_verdict"]

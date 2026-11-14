# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade a Harbor answer in a separate verifier container, without a Gym server."""

import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from upstream import (
    _JUDGE_SYSTEM,
    _MULTIMODAL_SYSTEM,
    FreeFormJudgeResponse,
    _build_judge_user_message,
    _salvage_json,
    extract_answer_block,
)


def evaluate(*, system: str, content: str | list[dict], base_url: str, api_key: str, model: str) -> dict:
    """Call an OpenAI-compatible judge; failed requests do not become wrong answers."""
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
        raise ValueError("Judge base URL must be HTTP(S), without embedded credentials")
    schema = FreeFormJudgeResponse.model_json_schema()
    schema["additionalProperties"] = False
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "max_completion_tokens": 512,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "vials_verdict", "schema": schema, "strict": True},
        },
    }
    total_in = total_out = 0
    last_error = None
    received = False
    for attempt in range(3):
        request = Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            received = True
            usage = result.get("usage") or {}
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
            message = result["choices"][0]["message"]
            parsed = _salvage_json(message.get("content") or "")
            if parsed is not None:
                return {
                    "correct": parsed.equivalent,
                    "judge_reasoning": parsed.reasoning,
                    "judge_unverifiable": parsed.reasoning.lstrip().upper().startswith("UNVERIFIABLE:"),
                    "judge_model": result.get("model", model),
                    "judge_input_tokens": total_in,
                    "judge_output_tokens": total_out,
                }
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            # Avoid serializing provider responses or headers, which can contain credentials.
            last_error = type(error).__name__
            if isinstance(error, HTTPError) and error.code not in {408, 429} and error.code < 500:
                raise RuntimeError(f"Judge HTTP request failed with status {error.code}") from None
        if attempt < 2:
            time.sleep(attempt + 1)
    if last_error:
        raise RuntimeError(f"Judge requests exhausted: {last_error}") from None
    if not received:
        raise RuntimeError("Judge did not return a response")
    return {
        "correct": False,
        "judge_reasoning": "Could not parse a judge verdict after three attempts.",
        "judge_unparseable": True,
        "judge_model": model,
        "judge_input_tokens": total_in,
        "judge_output_tokens": total_out,
    }


def grade(*, answer_path: Path, input_path: Path, output_dir: Path, base_url: str, api_key: str, model: str) -> dict:
    """Write Harbor rewards and diagnostics using the original VIALS judge contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward.json").unlink(missing_ok=True)
    (output_dir / "info.json").unlink(missing_ok=True)
    raw = answer_path.read_text() if answer_path.is_file() else ""
    visible = re.sub(r"<(think|thinking)>.*?(?:</\1>|$)", "", raw, flags=re.I | re.S)
    candidate = extract_answer_block(visible)
    verdict = {"correct": False, "judge_reasoning": "No final answer was extracted."}
    if candidate is not None:
        if not api_key:
            raise ValueError("Set VIALS_JUDGE_API_KEY on the Harbor worker")
        payload = json.loads(input_path.read_text())
        item = {
            "Question": payload["question"],
            "GTFA": payload["expected_answer"],
            "LeadBoundLower": payload.get("answer_lower_bound"),
            "LeadBoundUpper": payload.get("answer_upper_bound"),
        }
        content = _build_judge_user_message(item, candidate)
        verdict = evaluate(system=_JUDGE_SYSTEM, content=content, base_url=base_url, api_key=api_key, model=model)
        if verdict.get("judge_unverifiable"):
            paths = payload.get("images", [])
            if not paths:
                raise ValueError("Image-assisted grading requires original image panels")
            images = []
            for name in paths:
                path = (input_path.parent / name).resolve()
                if not path.is_relative_to((input_path.parent / "images").resolve()):
                    raise ValueError("Verifier image must be under its private images directory")
                media_type = mimetypes.guess_type(path.name)[0] or "image/png"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}", "detail": "auto"},
                    }
                )
            text = content.rsplit("Are the candidate and ground-truth final answers equivalent?", 1)[0]
            text += "Using the image only to resolve labelling ambiguities, are the candidate and ground-truth final answers equivalent?"
            fallback = evaluate(
                system=_MULTIMODAL_SYSTEM,
                content=[*images, {"type": "text", "text": text}],
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
            fallback["judge_input_tokens"] += verdict["judge_input_tokens"]
            fallback["judge_output_tokens"] += verdict["judge_output_tokens"]
            verdict = {**fallback, "judge_multimodal": True}
    result = {**verdict, "candidate_answer": candidate, "reward": float(verdict["correct"])}
    (output_dir / "info.json").write_text(json.dumps(result) + "\n")
    (output_dir / "reward.json").write_text(json.dumps({"reward": result["reward"]}) + "\n")
    return result


if __name__ == "__main__":
    grade(
        answer_path=Path("/task/answer.txt"),
        input_path=Path("/tests/verifier_input.json"),
        output_dir=Path("/logs/verifier"),
        base_url=os.environ.get("VIALS_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("VIALS_JUDGE_API_KEY", ""),
        model=os.environ.get("VIALS_JUDGE_MODEL", "gpt-5-mini"),
    )

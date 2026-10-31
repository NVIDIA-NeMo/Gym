# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image evidence checks against independently witnessed model requests."""

import base64
import binascii
import hashlib
import json
from collections import Counter
from collections.abc import Callable

from .checks import BehavioralCheck
from .results import Results


def _image(part: dict) -> dict:
    value = part.get("image_url")
    detail = part.get("detail")
    if isinstance(value, dict):
        detail = value.get("detail")
        value = value.get("url")
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("expected an inline image data URI")
    header, encoded = value.split(";base64,", 1)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid image base64") from exc
    if not data:
        raise ValueError("empty image payload")
    return {"mime_type": header[5:], "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "detail": detail}


def image_payloads(content: object) -> list[dict]:
    """Return ordered inline-image identities from Chat or Responses content.

    Only structured content counts; a data URI embedded in a text string does not.
    Invalid/unavailable image payloads raise ValueError rather than disappearing.
    """
    if isinstance(content, list):
        return [image for item in content for image in image_payloads(item)]
    if isinstance(content, dict):
        if content.get("type") in ("input_image", "image_url"):
            return [_image(content)]
        return [
            image for key in ("input", "messages", "content", "output") for image in image_payloads(content.get(key))
        ]
    return []


def _request_content(request: dict) -> object:
    return request.get("messages", request.get("input"))


def _fingerprint(content: object) -> str:
    # Keep roles, call IDs, text, ordering and image metadata, while comparing
    # decoded image bytes instead of their transport spelling.
    def normalize(value: object) -> object:
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            if value.get("type") in ("input_image", "image_url"):
                return {**value, "image_url": _image(value)}
            return {key: normalize(item) for key, item in value.items()}
        return value

    return json.dumps(normalize(content), sort_keys=True)


def inspect_media(
    witness: dict | None,
    record: dict | None,
    *,
    prompt_images: tuple[dict, ...] = (),
    tool_image: dict | None = None,
    applies: bool = False,
) -> list[dict]:
    """Compare fixture delivery and saved image evidence without granting exemptions.

    Expectations are supplied by the scenario, never inferred from missing images.
    Each saved-evidence check uses one canonical trajectory location.
    """
    witness = witness or {}
    attempts = witness.get("attempts", [])
    trajectory = (record or {}).get("ng_trajectory") or {}
    calls = trajectory.get("model_calls") or []
    tool = witness.get("media_tool") or {}
    results = Results()

    def run(
        check_id: str, location: str, reason: str, predicate: Callable[[], bool], *, available: bool = True
    ) -> None:
        def valid() -> bool:
            try:
                return bool(predicate())
            except (ValueError, TypeError, KeyError, AttributeError):
                return False

        results.run(
            BehavioralCheck(
                id=check_id,
                tier="P1",
                evidence=("TE-4", "TE-5", "TE-7"),
                location=location,
                reason=reason,
                predicate=valid,
                available=available,
                applies=applies,
            )
        )

    def identities(images: list[dict] | tuple[dict, ...]) -> list[dict]:
        # Harness APIs may add their default detail setting. Its fidelity is
        # checked against the actual wire request in media.saved_requests.
        return [{k: v for k, v in image.items() if k != "detail"} for image in images]

    run(
        "media.prompt_delivery",
        "witness.json:attempts[0].request",
        "first model request must carry the two prescribed prompt images in order",
        lambda: len(prompt_images) == 2
        and identities(image_payloads(attempts[0]["request"])) == identities(prompt_images),
        available=bool(attempts),
    )
    run(
        "media.tool_delivery",
        "witness.json:attempts[1].request",
        "next model request must retain the prompt images and add the prescribed tool image",
        lambda: len(attempts) == 2
        and tool_image is not None
        and identities(image_payloads(attempts[1]["request"])) == identities([*prompt_images, tool_image]),
        available=len(attempts) >= 2,
    )
    run(
        "media.backend_delivery",
        "witness.json:backend_requests",
        "scripted backend must receive every image after any Responses-to-Chat conversion",
        lambda: len(witness.get("backend_requests", [])) == len(attempts)
        and all(
            image_payloads(backend) == image_payloads(attempt["request"])
            for backend, attempt in zip(witness["backend_requests"], attempts, strict=True)
        ),
        available=bool(attempts) and bool(witness.get("backend_requests")),
    )
    run(
        "media.saved_requests",
        "$.ng_trajectory.model_calls[*].request",
        "saved request content differs from witnessed text, images, order, metadata or tool associations",
        lambda: Counter(_fingerprint(_request_content(a["request"])) for a in attempts)
        == Counter(_fingerprint(_request_content(c["request"])) for c in calls),
        available=record is not None and bool(calls) and bool(attempts),
    )

    def retained_tool() -> dict:
        matches = [t for t in trajectory.get("tool_calls", []) if t.get("tool_call_id") == tool.get("id")]
        if len(matches) != 1 or not tool.get("id"):
            raise ValueError("expected one witnessed image tool")
        return matches[0]

    run(
        "media.tool_output",
        "$.ng_trajectory.tool_calls[*].output",
        "saved image-tool output must retain the prescribed structured image payload",
        lambda: tool_image is not None
        and identities(image_payloads(retained_tool().get("output"))) == identities([tool_image]),
        available=record is not None and bool(tool),
    )

    def joined() -> bool:
        saved = retained_tool()
        requests = [
            item
            for inv in trajectory.get("invocations", [])
            if inv.get("invocation_id") == saved.get("invocation_id")
            for item in inv.get("conversation", [])
            if item.get("type") in ("function_call", "custom_tool_call") and item.get("call_id") == tool["id"]
        ]
        if len(requests) != 1:
            return False
        request = requests[0]
        name = request.get("name")
        if request.get("namespace"):
            name = f"{request['namespace']}__{name}"
        return (
            name == tool["name"]
            and json.loads(request["arguments"]) == tool["arguments"]
            and saved.get("status") == "completed"
            and saved.get("tool_name") == request.get("name")
        )

    run(
        "media.tool_join",
        "$.ng_trajectory.tool_calls",
        "saved image tool must join to the witnessed invocation, name, arguments and successful result",
        joined,
        available=record is not None and bool(tool),
    )
    return results.dump()

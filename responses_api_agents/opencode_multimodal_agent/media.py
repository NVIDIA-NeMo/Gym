# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image staging and native OpenCode artifact validation."""

import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from PIL import Image

from nemo_gym.core import (
    MediaFile,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    split_text_and_media,
)
from nemo_gym.trajectory import AgentInvocation
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations, parse_opencode_session
from responses_api_agents.opencode_multimodal_agent.rewards import final_answer as final_answer


def prepare_image(image: MediaFile, *, max_source_pixels: int, preserve_format: bool) -> MediaFile:
    """Apply benchmark pixel limits and optional RGB conversion to decoded media."""
    try:
        with Image.open(io.BytesIO(image.data)) as source:
            if source.width * source.height > max_source_pixels:
                raise ValueError("Visual input exceeds max_source_pixels")
            if preserve_format:
                return image
            output = io.BytesIO()
            source.convert("RGB").save(output, format="PNG")
            return MediaFile(media_type="image/png", data=output.getvalue())
    except OSError as error:
        raise ValueError("Invalid or truncated input image") from error


def stage_inputs(
    body: NeMoGymResponseCreateParamsNonStreaming,
    directory: Path,
    *,
    max_source_pixels: int,
    max_images: int | None,
    preserve_format: bool = True,
) -> tuple[str, str | None, list[Path]]:
    """Stage initial images without exposing verifier metadata to the runtime.

    This entrypoint starts fresh sessions. Explicitly reject replay/history rather
    than silently treating a later user message as a new independent task.
    """
    if isinstance(body.input, str):
        raise ValueError("A multimodal task requires input_image parts")
    texts, systems, paths = [], [], []
    for item in body.model_dump()["input"]:
        if item.get("role") not in {"user", "system", "developer"}:
            raise ValueError("OpenCodeMultimodalAgent accepts initial prompts, not replayed trajectories")
        text, media = split_text_and_media(item["content"])
        if media and item["role"] != "user":
            raise ValueError("Only user images and text prompts are supported")
        (texts if item["role"] == "user" else systems).append(text)
        for image in media:
            if max_images is not None and len(paths) >= max_images:
                raise ValueError("Initial image count exceeds max_images")
            image = prepare_image(image, max_source_pixels=max_source_pixels, preserve_format=preserve_format)
            path = directory / f"input-{len(paths)}.{image.extension}"
            path.write_bytes(image.data)
            paths.append(path)
    if not paths:
        raise ValueError("A multimodal task requires an input image")
    return "\n\n".join(texts), "\n\n".join(systems) or None, paths


def tool_images(part: dict) -> list[dict]:
    """Read embedded image attachments from one OpenCode tool part before teardown."""
    images = []
    for attachment in (part.get("state") or {}).get("attachments", []):
        if not attachment.get("mime", "").startswith("image/"):
            continue
        url = attachment.get("url", "")
        if not url.startswith("data:image/"):
            raise ValueError("Tool images must be inline data URLs before sandbox teardown")
        MediaFile.from_data_uri(url)
        images.append({"type": "input_image", "image_url": url, "detail": "auto"})
    return images


def restore_tool_images(output: list[NeMoGymResponseOutputItem], attachments: dict[str, list[dict]]) -> None:
    """Restore embedded tool images without duplicating existing image parts."""
    for item in output:
        if isinstance(item, NeMoGymFunctionCallOutput) and item.call_id in attachments:
            text = item.output if isinstance(item.output, list) else [{"type": "input_text", "text": item.output}]
            existing = {
                part.get("image_url") for part in text if isinstance(part, dict) and part.get("type") == "input_image"
            }
            item.output = [
                *text,
                *(part for part in attachments.pop(item.call_id) if part["image_url"] not in existing),
            ]
    if attachments:
        raise ValueError("Image attachment has no matching tool result")


def parse_visual_session(
    database: Path, *, expected_images: int, max_turns: int | None
) -> tuple[list[NeMoGymResponseOutputItem], dict[str, int], int]:
    """Use the parent parser, restoring image tool outputs and failing on runtime errors."""
    if not database.is_file():
        raise ValueError("OpenCode did not produce its native session database")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as con:
        roots = con.execute("SELECT id FROM session WHERE parent_id IS NULL").fetchall()
        if len(roots) != 1:
            raise ValueError("Expected exactly one root OpenCode session")
        root = roots[0][0]
        message_rows = con.execute(
            "SELECT id, data FROM message WHERE session_id=? ORDER BY time_created, id", (root,)
        ).fetchall()
        messages = [json.loads(row[1]) for row in message_rows]
        # A failed child also means the full capture may contain a failed model call.
        all_messages = [json.loads(row[0]) for row in con.execute("SELECT data FROM message")]
        if any(message.get("error") for message in all_messages):
            raise ValueError("OpenCode recorded a model/runtime error; do not score this rollout")
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if not assistants or (max_turns is not None and len(assistants) > max_turns):
            raise ValueError("Missing assistant turns or exceeded max_turns")
        if any(not message.get("time", {}).get("completed") for message in assistants):
            raise ValueError("Unfinished OpenCode assistant turn")
        parts = [
            json.loads(row[0])
            for row in con.execute("SELECT data FROM part WHERE session_id=? ORDER BY time_created, id", (root,))
        ]
        roles = {
            row[0]: json.loads(row[1]).get("role")
            for row in con.execute("SELECT id, data FROM message WHERE session_id=?", (root,))
        }
        initial_images = con.execute("SELECT message_id, data FROM part WHERE session_id=?", (root,)).fetchall()
        last_assistant_id = next(
            mid for mid, data in reversed(message_rows) if json.loads(data).get("role") == "assistant"
        )
        terminal_parts = [json.loads(data) for mid, data in initial_images if mid == last_assistant_id]
    count = sum(
        roles.get(mid) == "user"
        and (part := json.loads(data)).get("type") == "file"
        and part.get("mime", "").startswith("image/")
        for mid, data in initial_images
    )
    if count != expected_images:
        raise ValueError("OpenCode dropped initial images")
    # The visible answer belongs to the root, but usage includes child model calls.
    _, usage = parse_opencode_session(database)
    bundle = parse_opencode_observations(database, root)
    invocation = next(
        record for record in bundle.records if isinstance(record, AgentInvocation) and record.invocation_id == root
    )
    # Unlike the parent's convenience output parser, its native-observation
    # parser retains reasoning. This prevents verifying an earlier answer when
    # the final turn contains only reasoning.
    output = [item for item in invocation.conversation if getattr(item, "role", None) != "user"]
    attachments = {}
    for part in parts:
        if part.get("type") != "tool":
            continue
        images = tool_images(part)
        if images:
            attachments[part["callID"]] = images
    restore_tool_images(output, attachments)
    if not any(part.get("type") in {"text", "reasoning", "tool"} for part in terminal_parts):
        output.append(
            NeMoGymResponseOutputMessage(
                id="empty-final",
                type="message",
                role="assistant",
                status="completed",
                content=[NeMoGymResponseOutputText(text="", annotations=[])],
            )
        )
    return output, usage, len(assistants)

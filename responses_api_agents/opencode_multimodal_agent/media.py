# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image staging and native OpenCode artifact validation."""

import base64
import io
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from PIL import Image

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_observability import AgentInvocation
from responses_api_agents.opencode_agent.app import _parse_opencode_session, parse_opencode_session


def stage_inputs(
    body: NeMoGymResponseCreateParamsNonStreaming,
    directory: Path,
    *,
    max_source_pixels: int,
    max_images: int | None,
    preserve_format: bool = False,
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
        content = item["content"]
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            if part["type"] == "input_text":
                (texts if item["role"] == "user" else systems).append(part["text"])
                continue
            if part["type"] != "input_image" or item["role"] != "user":
                raise ValueError("Only user images and text prompts are supported")
            if max_images is not None and len(paths) >= max_images:
                raise ValueError("Initial image count exceeds max_images")
            source = part.get("image_url")
            if not isinstance(source, str):
                raise ValueError("Images require image_url, not file_id")
            if source.startswith("data:"):
                header, payload = source.split(",", 1)
                if not header.startswith("data:image/") or not header.endswith(";base64"):
                    raise ValueError("Expected a base64 image data URL")
                image_source = io.BytesIO(base64.b64decode(payload, validate=True))
            else:
                if "://" in source and not source.startswith("file://"):
                    raise ValueError("Stage remote image URLs locally before rollout")
                image_source = Path(source.removeprefix("file://"))
            with Image.open(image_source) as image:
                if image.width * image.height > max_source_pixels:
                    raise ValueError("Visual input exceeds max_source_pixels")
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(image.format)
                if preserve_format and suffix:
                    path = directory / f"input-{len(paths)}{suffix}"
                    raw = (
                        image_source.getvalue() if isinstance(image_source, io.BytesIO) else image_source.read_bytes()
                    )
                    path.write_bytes(raw)
                else:
                    path = directory / f"input-{len(paths)}.png"
                    image.convert("RGB").save(path)
            paths.append(path)
    if not paths:
        raise ValueError("A multimodal task requires an input image")
    return "\n\n".join(texts), "\n\n".join(systems) or None, paths


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
    bundle = _parse_opencode_session(database, root)
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
        images = []
        for attachment in part.get("state", {}).get("attachments", []):
            if not attachment.get("mime", "").startswith("image/"):
                continue
            url = attachment.get("url", "")
            if not url.startswith("data:image/"):
                raise ValueError("Tool images must be inline data URLs before sandbox teardown")
            images.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if images:
            attachments[part["callID"]] = images
    for item in output:
        if isinstance(item, NeMoGymFunctionCallOutput) and item.call_id in attachments:
            item.output = [{"type": "input_text", "text": item.output}, *attachments.pop(item.call_id)]
    if attachments:
        raise ValueError("Native image attachment has no matching tool output")
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


def final_answer(output: list[dict], *, strip_reasoning: bool = False) -> str:
    """Extract the scoring copy; optional training cleanup never mutates history."""
    for item in reversed(output):
        if item.get("type") in {"reasoning", "function_call"}:
            break
        if item.get("type") == "message" and item.get("role") == "assistant":
            text = "".join(part["text"] for part in item["content"] if part["type"] == "output_text")
            if not strip_reasoning:
                return text
            text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
            text = re.split(r"<(?:think|thinking)>", text, maxsplit=1)[0]
            return re.split(r"</(?:think|thinking)>", text)[-1].strip()
    return ""

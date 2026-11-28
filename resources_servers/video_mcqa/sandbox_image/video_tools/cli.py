# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One tool call: the plugin writes ``{"tool": ..., "arguments": ...}`` to standard input and reads one JSON reply.

The configuration path comes from the plugin, never from the model. A reply is ``{"output", "attachments",
"metadata"}``, or ``{"error", "output"}`` for an error the model can act on (an invalid argument, a missing file, a
limit); anything else exits non-zero, and the plugin reports that as a failed tool call. ``--check`` validates the
configuration and finds FFmpeg, for the resources server's check at seed.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contract import MediaContractError, ToolConfig, VideoSpec
from .media import view_image, watch_video


# Tool arguments are small JSON; a larger request is refused before it is parsed.
MAX_REQUEST_BYTES = 64 << 10


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tool: Literal["view_image", "watch_video"]
    arguments: dict[str, object] = Field(default_factory=dict)


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str = Field(min_length=1)


def execute_tool(request: ToolRequest, config: ToolConfig) -> dict[str, object]:
    """Run one tool; it never resizes, drops or adds frames on its own."""
    workspace_root = Path(config.workspace_root)
    scratch_roots = tuple(Path(root) for root in config.scratch_roots)
    if request.tool == "view_image":
        return view_image(
            ImageRequest.model_validate(request.arguments).path,
            workspace_root=workspace_root,
            scratch_roots=scratch_roots,
        )
    return watch_video(
        VideoSpec.model_validate(request.arguments),
        workspace_root=workspace_root,
        cache_dir=Path(config.cache_dir),
        profile=config.profile,
        scratch_roots=scratch_roots,
    )


def check(config: ToolConfig) -> dict[str, object]:
    """The configuration is valid and FFmpeg runs."""
    versions = {}
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise FileNotFoundError(f"{program} is not on PATH")
        result = subprocess.run([program, "-version"], capture_output=True, check=True, timeout=30)
        versions[program] = result.stdout.decode(errors="replace").splitlines()[0]
    if not Path(config.workspace_root).is_dir():
        raise FileNotFoundError(f"workspace_root {config.workspace_root} is not a directory")
    return {"ok": True, **versions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = ToolConfig.model_validate_json(args.config.read_bytes())
    if args.check:
        print(json.dumps(check(config)))
        return
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    try:
        if len(payload) > MAX_REQUEST_BYTES:
            raise MediaContractError("tool_input_too_large", "tool request exceeds 64 KiB", details={})
        result = execute_tool(ToolRequest.model_validate_json(payload), config)
    except MediaContractError as error:
        result = {"error": error.to_dict(), "output": str(error)}
    except (ValidationError, FileNotFoundError, PermissionError) as error:
        result = {"error": {"code": "invalid_media_request", "message": str(error)}, "output": str(error)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()

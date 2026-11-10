# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-local MCP tools. No target STEP or verifier credentials are available."""

import base64
import io
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path("/workspace")
HERE = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def workspace_path(path: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Path must stay inside /workspace")
    return resolved


def view_image(path: str) -> list[dict]:
    from PIL import Image

    p = workspace_path(path)
    if not p.is_file() or p.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Image missing or exceeds 8 MiB")
    with Image.open(p) as im:
        if im.width * im.height > 16_777_216:
            raise ValueError("Image exceeds 16 megapixels")
        im = im.convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return [
        {"type": "text", "text": str(p)},
        {"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mimeType": "image/png"},
    ]


def call_tool(name: str, arguments: dict) -> dict:
    try:
        if name == "cad_view":
            return {"content": view_image(arguments["path"])}
        if name != "cad_render":
            raise ValueError(f"Unknown tool: {name}")
        model = workspace_path(arguments.get("path", "model.py"))
        # Recreate previews: a failed render must never return a previous successful image.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="cad-preview-", dir=ROOT) as directory:
            p = subprocess.Popen(
                [sys.executable, str(HERE / "cad_render.py"), str(model), "--output", directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = p.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.communicate()
                raise TimeoutError("CAD render exceeded 120 seconds") from None
            if p.returncode:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": stderr.decode(errors="replace")[-12000:]}],
                }
            paths = json.loads(stdout.decode(errors="replace").splitlines()[-1])
            content = [{"type": "text", "text": "Current model.py preview, same cameras as target; no annotations."}]
            for path in paths:
                preview = view_image(path)
                preview[0]["text"] = f"{Path(path).name} (inline preview; temporary render file)"
                content.extend(preview)
            return {"content": content}
    except Exception:
        return {"isError": True, "content": [{"type": "text", "text": traceback.format_exc()[-12000:]}]}


TOOLS = [
    {
        "name": "cad_render",
        "description": "Execute model.py and inspect its matching CAD camera views. Returns images or a traceback.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "default": "model.py"}}},
    },
    {
        "name": "cad_view",
        "description": "View a target image or an image you created with your own rendering code. Returns the actual image to the model.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
]


def main() -> None:
    # MCP stdio uses one JSON-RPC message per line. stdout is exclusively protocol data.
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request["method"]
        if method == "initialize":
            result = {
                "protocolVersion": request["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "cad-workspace", "version": "1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = call_tool(request["params"]["name"], request["params"].get("arguments", {}))
        else:
            print(
                json.dumps(
                    {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": "Method not found"}}
                ),
                flush=True,
            )
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


if __name__ == "__main__":
    main()

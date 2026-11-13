# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs only inside the namespace sandbox, with no host data or credentials."""

import base64
import io
import json
import re
import socket
import sys

from PIL import Image


LIMIT = 64 * 1024 * 1024


def main():
    from tools.code_interpreter import CodeInterpreter, _kill_kernels_and_subprocesses

    tool = CodeInterpreter({"work_dir": "/workspace/kernel", "output_timeout": 30})
    with socket.socket(socket.AF_UNIX) as server:
        server.bind("/channel/tool.sock")
        server.listen(1)
        try:
            with server.accept()[0] as connection:
                stream = connection.makefile("rwb")
                stream.write(b'{"isolation":"bubblewrap-namespaces-v1"}\n')
                stream.flush()
                while raw := stream.readline(LIMIT + 1):
                    if len(raw) > LIMIT:
                        raise ValueError("Oversized tool request")
                    try:
                        request = json.loads(raw)
                        images = {
                            name: Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
                            for name, data in request["images"].items()
                        }
                        result = tool.call(request["params"], image_map=images)
                        files = {}
                        for _, reference in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", result):
                            if len(files) >= 32:
                                break
                            try:
                                image = Image.open(reference)
                                out = io.BytesIO()
                                image.save(out, format="PNG")
                                if len(out.getvalue()) <= 16 * 1024 * 1024:
                                    files[reference] = base64.b64encode(out.getvalue()).decode()
                            except (OSError, ValueError):
                                pass
                        reply = {"result": result, "images": files}
                    except Exception as exc:
                        reply = {"error": type(exc).__name__}
                    payload = json.dumps(reply).encode() + b"\n"
                    if len(payload) > LIMIT:
                        payload = b'{"error":"Oversized tool response"}\n'
                    stream.write(payload)
                    stream.flush()
        finally:
            _kill_kernels_and_subprocesses()


if __name__ == "__main__":
    # Original-tool diagnostics must never corrupt the socket protocol.
    sys.stdout = sys.stderr
    main()

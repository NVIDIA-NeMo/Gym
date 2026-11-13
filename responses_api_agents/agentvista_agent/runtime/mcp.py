# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose the four original tools, with an episode-local kernel/image registry."""

import asyncio
import contextlib
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image
from utils.context_utils import pil_to_base64_data_uri

from responses_api_agents.agentvista_agent.runtime.code_sandbox import CodeSandbox
from responses_api_agents.agentvista_agent.runtime.paths import output_path


class ToolSession:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.output = output_path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.image_map = {}
        for index, path in enumerate(config["images"]):
            name = "original_image" + (f"_{index}" if index else "")
            self.image_map[name] = Image.open(path).convert("RGB")
        self.instances, self.counts = {}, {}
        self.code_sandbox = None
        self.image_aliases = {}
        self.lock = asyncio.Lock()

    def close(self):
        if self.code_sandbox is not None:
            self.code_sandbox.close()

    def schema(self) -> list[dict]:
        from tools import get_tool_info, list_tools

        names = ["web_search", "image_search", "visit", "code_interpreter"]
        if set(names) - set(list_tools()):
            raise RuntimeError("Original AgentVista tool dependencies are missing")
        return [get_tool_info(name) for name in names]

    def execute(self, name: str, arguments: dict) -> list[dict]:
        from tools import get_tool

        if name not in {entry["name"] for entry in self.schema()}:
            raise ValueError("Unknown benchmark tool")
        limit = self.config.get(name + "_max_calls", 100) if name in {"web_search", "image_search"} else None
        if limit is not None and self.counts.get(name, 0) >= limit:
            return [{"type": "text", "text": f"Maximum number of '{name}' calls ({limit}) exceeded."}]
        if name not in self.instances:
            import yaml

            configs = yaml.safe_load((Path(os.environ["AGENTVISTA_ROOT"]) / "configs/tool_configs.yaml").read_text())
            settings = dict(configs.get(name, {}))
            settings.update(save_dir=str(self.output), work_dir=str(self.output / "kernel"))
            # A fixed auxiliary endpoint preserves visit's existing summarizer without
            # calling a trainable policy outside Gym's token-capture path.
            if name == "visit":
                for key in ("api_key", "api_endpoint", "model_name"):
                    value = os.environ.get("AGENTVISTA_VISIT_" + key.upper())
                    if value:
                        settings[key] = value
            if name == "code_interpreter":
                self.code_sandbox = CodeSandbox(self.config, self.output)
                self.instances[name] = self.code_sandbox
            else:
                self.instances[name] = get_tool(name)(settings)
        params = dict(arguments)
        if name == "image_search" and params.get("image_url"):
            reference = params["image_url"]
            if not reference.startswith(("http://", "https://")):
                key = Path(reference).stem
                if key in self.image_map:
                    path = self.output / (key + ".jpg")
                    self.image_map[key].save(path, quality=95)
                else:
                    path = self.image_aliases.get(reference) or output_path(self.output / reference)
                    if not path.resolve(strict=True).is_relative_to(self.output.resolve(strict=True)):
                        raise ValueError("Image reference outside episode")
                params["image_url"] = str(path)
        kwargs = {"image_map": self.image_map.copy()} if name == "code_interpreter" else {"save_dir": str(self.output)}
        result = self.instances[name].call(params, **kwargs)
        self.counts[name] = self.counts.get(name, 0) + 1
        content = []
        processed = result
        # Match the original handler: image_search and visit return text only.
        if name not in {"image_search", "visit"}:
            for label, reference in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", result):
                path = Path(reference)
                if not path.is_absolute():
                    path = self.output / path
                image = Image.open(path).convert("RGB")
                key = Path(label).stem
                if not re.fullmatch(r"tool_image_\d+|observation_\d+", key):
                    key = f"tool_image_{1 + sum(k.startswith('tool_image_') for k in self.image_map)}"
                self.image_map[key] = image
                self.image_aliases[path.name] = output_path(path)
                uri = pil_to_base64_data_uri(image)
                content.append({"type": "image", "mimeType": "image/jpeg", "data": uri.split(",", 1)[1]})
                processed = processed.replace(f"![{label}]({reference})", f"[Image: {key}, file: {path.name}]")
        content.insert(0, {"type": "text", "text": json.dumps({"result": processed})})
        with (self.output / "tools.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "result": processed,
                        "images": [k for k in self.image_map if k.startswith("tool_image_")],
                    }
                )
                + "\n"
            )
        return content


async def serve(config: dict) -> None:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import ImageContent, TextContent, Tool

    server = Server("agentvista")
    # Existing tools print diagnostics; stdio stdout is reserved for MCP messages.
    with contextlib.redirect_stdout(sys.stderr):
        session = ToolSession(config)
        schemas = session.schema()

    @server.list_tools()
    async def list_tools():
        tools = [
            Tool(name=item["name"], description=item["description"], inputSchema=item["parameters"])
            for item in schemas
        ]
        (session.output / "tools_listed").write_text(json.dumps([tool.name for tool in tools]))
        return tools

    @server.call_tool()
    async def call_tool(name, arguments):
        async with session.lock:
            with contextlib.redirect_stdout(sys.stderr):
                content = await asyncio.to_thread(session.execute, name, arguments)
            return [ImageContent(**item) if item["type"] == "image" else TextContent(**item) for item in content]

    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        session.close()
        with contextlib.redirect_stdout(sys.stderr):
            from tools.code_interpreter import _kill_kernels_and_subprocesses

            _kill_kernels_and_subprocesses()


if __name__ == "__main__":
    asyncio.run(serve(json.loads(Path(sys.argv[1]).read_text())))

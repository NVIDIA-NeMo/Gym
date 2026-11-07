# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build ``tool_catalog.json`` from ``/list-tools`` answers of agent-environment sandboxes.

A sandbox with every server stops if one server fails to start, so export one server at a time (with
``ENABLED_SERVERS=<server>,calculator`` to keep the name prefixes) and pass all answers here::

    python export_tool_catalog.py --template mcp_server_template.json --out tool_catalog.json answers/*.json

``--template`` is ``/agent-environment/src/agent_environment/mcp_server_template.json`` from the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_catalog(servers: list[str], answers: list[list[dict]]) -> dict:
    by_server: dict[str, dict[str, dict]] = {server: {} for server in servers}
    for answer in answers:
        for tool in answer:
            server = next((s for s in servers if tool["name"].startswith(f"{s}_")), None)
            if server is not None:
                by_server[server].setdefault(
                    tool["name"],
                    {
                        "name": tool["name"],
                        "server": server,
                        "description": tool.get("description") or "",
                        "inputSchema": tool.get("inputSchema") or {"type": "object", "properties": {}},
                    },
                )
    return {
        "source": "ghcr.io/scaleapi/mcp-atlas:1.2.7 /list-tools, one server per sandbox",
        "servers": servers,
        "tools": [tool for server in servers for tool in by_server[server].values()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("answers", type=Path, nargs="+")
    args = parser.parse_args()
    servers = list(json.loads(args.template.read_text())["mcpServers"])
    catalog = build_catalog(servers, [json.loads(path.read_text()) for path in args.answers])
    # One tool per line keeps the file small and its diffs readable.
    tools = ",\n".join(json.dumps(tool, ensure_ascii=False) for tool in catalog["tools"])
    header = {key: value for key, value in catalog.items() if key != "tools"}
    args.out.write_text(json.dumps(header, ensure_ascii=False)[:-1] + ', "tools": [\n' + tools + "\n]}\n")
    exported = {tool["server"] for tool in catalog["tools"]}
    print(f"{len(catalog['tools'])} tools; servers without tools: {[s for s in servers if s not in exported]}")


if __name__ == "__main__":
    main()

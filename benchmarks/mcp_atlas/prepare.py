# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the MCP-Atlas public set (ScaleAI/MCP-Atlas, 500 tasks) as Gym benchmark rows.

Each row gives the model the task prompt and, as function tools, the task's ``ENABLED_TOOLS`` that the
agent-environment image serves (schemas from ``resources_servers/mcp_atlas/tool_catalog.json``). Tools of
servers the image does not ship are dropped, as upstream does. Claims are cleaned with upstream's
``score_claims.py`` rules and stored for the verifier.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "mcp_atlas_benchmark.jsonl"
TOOL_CATALOG = BENCHMARK_DIR.parent.parent / "resources_servers" / "mcp_atlas" / "tool_catalog.json"
REVISION = "8c563b55d7c967755f474299848049834d624617"
PARQUET_URL = f"https://huggingface.co/datasets/ScaleAI/MCP-Atlas/resolve/{REVISION}/MCP-Atlas.parquet"
EXPECTED_TASKS = 500


def clean_claim_text(text: str) -> str:
    """Upstream ``clean_claim_text``: strip list markers, normalize quotes and dashes."""
    text = text.strip()
    text = re.sub(r"^[-*•·◦‣⁃]\s*", "", text)
    text = re.sub(r"^\d+[.)]\s*", "", text)
    text = text.replace("“", '"')
    text = re.sub(r'[”"]', '"', text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    text = re.sub(r"[.\s]*[\"']+ $", "", text)
    text = re.sub(r"[\"']+\.*$", "", text)
    return text.strip(" \t\n\r")


def _cleaned(items: list[Any]) -> list[str]:
    claims = [clean_claim_text(str(item)) for item in items]
    return [claim for claim in claims if len(claim) > 3]


def extract_claims(blob: str) -> list[str]:
    """Upstream ``extract_claims`` for a string blob: a JSON or Python list, else separator splitting."""
    blob = (blob or "").strip()
    if not blob:
        return []
    if blob.startswith("[") and blob.endswith("]"):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(blob)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                return _cleaned(parsed)
    for separator in ("\n•", "\n-", "\n*", "\n1.", "\n2.", ";", "||"):
        if separator in blob:
            claims = _cleaned(blob.split(separator))
            if claims:
                return claims
    return _cleaned(blob.split("\n"))


def enabled_tool_names(blob: str) -> list[str]:
    """``ENABLED_TOOLS`` holds names, and in a few tasks ``{"name": ...}`` objects."""
    return [item["name"] if isinstance(item, dict) else item for item in json.loads(blob)]


def build_row(record: dict[str, str], catalog: dict[str, dict[str, Any]], servers: list[str]) -> dict[str, Any]:
    names = enabled_tool_names(record["ENABLED_TOOLS"])
    exported = {tool["server"] for tool in catalog.values()}
    unexported = sorted(
        {
            s
            for name in names
            for s in servers
            if name.startswith(f"{s}_") and s not in exported and name not in catalog
        }
    )
    if unexported:
        # A server the image ships but the catalog lacks: dropping its tools would change the task.
        raise ValueError(f"task {record['TASK']}: tool_catalog.json has no tools of {unexported}; export them first")
    served = [name for name in catalog if name in set(names)]
    tools = [
        {
            "type": "function",
            "name": name,
            "description": catalog[name].get("description") or "",
            "parameters": catalog[name].get("inputSchema") or {"type": "object", "properties": {}},
            "strict": False,
        }
        for name in served
    ]
    return {
        "task_id": record["TASK"],
        "responses_create_params": {"input": [{"role": "user", "content": record["PROMPT"]}], "tools": tools},
        "verifier_metadata": {
            "task_id": record["TASK"],
            "enabled_tools": served,
            "servers": sorted({catalog[name]["server"] for name in served}),
            "claims": extract_claims(record["GTFA_CLAIMS"]),
            "dropped_tools": sorted(set(names) - set(served)),
        },
    }


def load_catalog(path: Path = TOOL_CATALOG) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Tool name to its catalog entry, in the order the sandbox lists them, and the servers of the image."""
    catalog = json.loads(path.read_text())
    return {tool["name"]: tool for tool in catalog["tools"]}, catalog["servers"]


def _download(url: str, target: Path) -> None:
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
        target.write_bytes(response.read())


def prepare() -> Path:
    import pandas

    catalog, servers = load_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        parquet = Path(tmp) / "MCP-Atlas.parquet"
        _download(PARQUET_URL, parquet)
        records = pandas.read_parquet(parquet).to_dict(orient="records")
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks at revision {REVISION}, got {len(records)}")
    rows = [build_row(record, catalog, servers) for record in records]
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()

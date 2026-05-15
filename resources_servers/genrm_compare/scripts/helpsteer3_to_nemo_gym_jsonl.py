# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Hugging Face `nvidia/HelpSteer3` (preference subset) to NeMo Gym JSONL.

Each output row matches the format expected by the GenRM compare agent
(`resources_servers/genrm_compare/data/example.jsonl`): OpenAI-style messages in
`responses_create_params.input`, optional empty `tools`, and `agent_ref` pointing
at `genrm_simple_agent`.

Human preference columns (`response1`, `response2`, `overall_preference`, etc.)
are not copied: GRPO generates fresh completions and the GenRM assigns rewards.

Usage::

    python resources_servers/genrm_compare/scripts/helpsteer3_to_nemo_gym_jsonl.py \\
        --output-dir data/helpsteer3_gym

    # Smoke test without writing full splits
    python resources_servers/genrm_compare/scripts/helpsteer3_to_nemo_gym_jsonl.py \\
        --output-dir /tmp/hs3 --max-samples 100
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type")
                if t in ("text", "output_text") and "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(_content_to_str(block["content"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def _normalize_message(msg: Any) -> dict[str, str] | None:
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if role not in ("user", "assistant", "system"):
        return None
    text = html.unescape(_content_to_str(msg.get("content")))
    return {"role": role, "content": text}


def parse_context(context: Any) -> list[dict[str, str]]:
    if context is None:
        return []
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            return []
    if not isinstance(context, list):
        return []
    out: list[dict[str, str]] = []
    for item in context:
        m = _normalize_message(item)
        if m and m["content"].strip():
            out.append(m)
    return out


def trim_to_last_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """GenRM compare expects history whose last turn is from the user."""
    last_user = -1
    for i, m in enumerate(messages):
        if m["role"] == "user":
            last_user = i
    if last_user < 0:
        return []
    return messages[: last_user + 1]


def row_to_gym_record(
    row: dict[str, Any],
    idx: int,
    *,
    agent_ref_name: str,
    dataset_tag: str,
) -> dict[str, Any] | None:
    messages = trim_to_last_user(parse_context(row.get("context")))
    if not messages:
        logger.debug("skip idx=%s: empty or invalid context after trim", idx)
        return None
    return {
        "id": row.get("id", idx),
        "responses_create_params": {
            "input": messages,
            "tools": [],
            "parallel_tool_calls": False,
        },
        "agent_ref": {"type": "responses_api_agents", "name": agent_ref_name},
        "dataset": dataset_tag,
    }


def convert_split(
    split_name: str,
    output_path: Path,
    *,
    repo_id: str,
    config_name: str,
    agent_ref_name: str,
    dataset_tag: str,
    max_samples: int | None,
    hf_token: str | None,
) -> tuple[int, int]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, name=config_name, split=split_name, token=hf_token)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if max_samples is not None and written >= max_samples:
                break
            rec = row_to_gym_record(
                dict(row),
                i,
                agent_ref_name=agent_ref_name,
                dataset_tag=dataset_tag,
            )
            if rec is None:
                skipped += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for train.jsonl and validation.jsonl",
    )
    p.add_argument("--repo-id", default="nvidia/HelpSteer3", help="Hugging Face dataset id")
    p.add_argument(
        "--config-name",
        default="preference",
        help="HelpSteer3 subset (use `preference` for pairwise-prompt conversations)",
    )
    p.add_argument(
        "--agent-ref-name",
        default="genrm_simple_agent",
        help="Must match responses_api_agents block loaded in NeMo Gym config",
    )
    p.add_argument(
        "--dataset-tag",
        default="helpsteer3_preference",
        help="Optional tag stored on each JSONL row under `dataset`",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap rows per split (for debugging)",
    )
    p.add_argument("--hf-token", default=None, help="Hugging Face token if the dataset is gated")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_w = total_s = 0
    for split in ("train", "validation"):
        out = args.output_dir / f"{split}.jsonl"
        try:
            w, s = convert_split(
                split,
                out,
                repo_id=args.repo_id,
                config_name=args.config_name,
                agent_ref_name=args.agent_ref_name,
                dataset_tag=args.dataset_tag,
                max_samples=args.max_samples,
                hf_token=args.hf_token,
            )
        except ValueError as e:
            # Older `datasets` versions may not support missing splits the same way
            logger.warning("split %s: %s — skipping", split, e)
            continue
        logger.info("wrote %s rows to %s (skipped %s)", w, out, s)
        total_w += w
        total_s += s
    if total_w == 0:
        raise SystemExit(
            "No rows written. Check repo_id/config_name, network access, and that "
            "`context` fields parse to messages ending with a user turn."
        )


if __name__ == "__main__":
    main()

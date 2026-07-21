# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Evaluation logic adapted from StructEval (Apache 2.0, TIGER-Lab)
# Dataset: TIGER-Lab/StructEval (MIT License)
# https://github.com/TIGER-Lab/StructEval
import codecs
import csv
import io
import json
import re
from collections import defaultdict
from enum import StrEnum
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import toml
import xmltodict
import yaml
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class StructEvalResourcesServerConfig(BaseResourcesServerConfig):
    pass


class OutputType(StrEnum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    CSV = "csv"


class StructEvalVerifyRequest(BaseVerifyRequest):
    task_id: str
    task_name: str
    input_type: str
    output_type: str
    raw_output_metric: List[str]
    rendering: bool = False


class StructEvalVerifyResponse(BaseVerifyResponse):
    task_id: str
    task_name: str
    input_type: str
    output_type: str
    raw_output_metric: List[str]
    rendering: bool = False
    render_score: float = 0.0
    key_validation_score: float = 0.0
    raw_output_score: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class StructEvalResourcesServer(SimpleResourcesServer):
    config: StructEvalResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        by_output: Dict[str, List[float]] = defaultdict(list)
        by_input: Dict[str, List[float]] = defaultdict(list)
        by_task_name: Dict[str, List[float]] = defaultdict(list)
        render_scores: List[float] = []
        key_scores: List[float] = []
        render_by_output: Dict[str, List[float]] = defaultdict(list)
        key_by_output: Dict[str, List[float]] = defaultdict(list)

        for rollouts in tasks:
            for r in rollouts:
                reward = r.get("reward", 0.0)
                ot = r.get("output_type", "unknown")
                it = r.get("input_type", "unknown")
                tn = r.get("task_name", "unknown")
                rs = r.get("render_score", 0.0)
                ks = r.get("key_validation_score", 0.0)

                by_output[ot].append(reward)
                by_input[it].append(reward)
                by_task_name[tn].append(reward)
                render_scores.append(rs)
                key_scores.append(ks)
                render_by_output[ot].append(rs)
                key_by_output[ot].append(ks)

        metrics: Dict[str, Any] = {}
        if render_scores:
            metrics["mean/render_score"] = mean(render_scores)
        if key_scores:
            metrics["mean/key_validation_score"] = mean(key_scores)

        for k, v in by_output.items():
            if v:
                metrics[f"mean/reward_output_{k}"] = mean(v)
        for k, v in by_input.items():
            if v:
                metrics[f"mean/reward_input_{k}"] = mean(v)
        for k, v in by_task_name.items():
            if v:
                metrics[f"mean/reward_{k}"] = mean(v)
        for k, v in render_by_output.items():
            if v:
                metrics[f"mean/render_score_{k}"] = mean(v)
        for k, v in key_by_output.items():
            if v:
                metrics[f"mean/key_validation_score_{k}"] = mean(v)

        return metrics

    async def verify(self, body: StructEvalVerifyRequest) -> StructEvalVerifyResponse:
        # Extract assistant text from response.
        assistant_text = _extract_assistant_text(body.response)

        if not assistant_text or not assistant_text.strip():
            return StructEvalVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                render_score=0.0,
                key_validation_score=0.0,
                error_type="empty_response",
                error_message="No assistant response text",
            )

        output_type = body.output_type.lower()

        # Step 1: Extract code from generation.
        code = _extract_code(assistant_text, output_type)

        # Step 2: Raw output eval — keyword matching (matches eval_utils.py::raw_output_eval).
        raw_output_score = _raw_output_eval(assistant_text, body.raw_output_metric)

        # Step 3: Parse extracted code (render_score).
        parsed, render_score, error_type, error_message = _parse_content(code, output_type)

        # Step 4: Validate paths (key_validation_score).
        key_validation_score = 0.0
        if render_score == 1.0 and body.raw_output_metric:
            matched = sum(1 for path in body.raw_output_metric if _path_exists(parsed, path))
            key_validation_score = matched / len(body.raw_output_metric)

        # Step 5: Final reward (matches eval_engine/main.py::calculate_final_score for non-renderable).
        reward = round(0.2 * render_score + 0.8 * key_validation_score, 2)

        return StructEvalVerifyResponse(
            **body.model_dump(),
            reward=reward,
            render_score=render_score,
            key_validation_score=key_validation_score,
            raw_output_score=raw_output_score,
            error_type=error_type,
            error_message=error_message,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_assistant_text(response) -> str:
    """Extract concatenated assistant text from a NeMoGymResponse."""
    parts: List[str] = []
    for output_item in response.output:
        if output_item.type != "message":
            continue
        for content_item in output_item.content:
            if content_item.type != "output_text":
                continue
            parts.append(content_item.text)
    return "".join(parts)


def _extract_code(text: str, output_type: str) -> str:
    """Extract code from <|BEGIN_CODE|>...<|END_CODE|> or fenced blocks.

    Verbatim logic from StructEval render_utils.py::extract_code_and_save,
    including unicode decode preprocessing and two-pass fence re-scan.
    """
    # Decode unicode escape sequences (matches extract_code_and_save lines 134-139).
    try:
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("<think>\n\n</think>\n\n", "")
        text = codecs.decode(text, "unicode_escape")
    except Exception:
        pass

    # 1) <|BEGIN_CODE|> ... <|END_CODE|> — closing tag optional
    begin_end_pat = (
        r"<\|BEGIN_CODE\|\>[ \t]*\n?"
        r"(?P<payload1>.*?)"
        r"(?:<\|END_CODE\|\>|$)"
    )

    # 2) ``` fenced block — closing fence optional
    fence_pat = (
        rf"```(?:{re.escape(output_type)}|[^\n]*)[ \t]*\n"
        r"(?P<payload2>.*?)"
        r"(?:```|$)"
    )

    pattern = rf"(?:{begin_end_pat})|(?:{fence_pat})"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if m:
        payload = m.group("payload1") or m.group("payload2")
        code = payload.strip()
    else:
        code = text.strip()

    # Second pass: re-scan for fence pattern (matches extract_code_and_save lines 166-170).
    m = re.search(fence_pat, text, re.DOTALL | re.IGNORECASE)
    if m:
        payload = m.group("payload2")
        code = payload.strip()

    return code


def _parse_content(code: str, output_type: str) -> Tuple[Any, float, Optional[str], Optional[str]]:
    """Parse extracted code string into a structure.

    Matches StructEval score_non_renderable (syntax validation) and
    load_file_structure (for path checking). Uses `if result:` truthiness
    check to match StructEval's behavior.

    Returns (parsed_obj, render_score, error_type, error_message).
    """
    try:
        if output_type == "json":
            result = json.loads(code)
        elif output_type == "yaml":
            result = yaml.safe_load(code)
        elif output_type == "toml":
            result = toml.loads(code)
        elif output_type == "xml":
            result = xmltodict.parse(code)
        elif output_type == "csv":
            reader = csv.DictReader(io.StringIO(code))
            # For render_score: StructEval checks truthiness of a DictReader (always True).
            # For path validation: need materialized structure with csv_headers.
            result = {"csv_headers": reader.fieldnames, "csv_rows": list(reader)}
        else:
            return None, 0.0, "unsupported_type", f"Unsupported output type: {output_type}"

        # Matches StructEval score_non_renderable: `if result:` (truthiness check).
        if not result:
            return None, 0.0, "parse_error", "Parsed result is empty"
        return result, 1.0, None, None
    except Exception as e:
        return None, 0.0, "parse_error", f"{type(e).__name__}: {str(e)[:200]}"


def _raw_output_eval(generation: str, raw_output_metric: List[str]) -> float:
    """Keyword matching against generation text.

    Verbatim from StructEval eval_engine/eval_utils.py::raw_output_eval.
    """
    if not raw_output_metric:
        return 0.0
    generation_lower = generation.lower()
    matches = sum(1 for keyword in raw_output_metric if keyword.lower() in generation_lower)
    return matches / len(raw_output_metric)


# ---------------------------------------------------------------------------
# Path validation (verbatim from StructEval eval_engine/eval_utils.py)
# ---------------------------------------------------------------------------


def _tokenize_path(path: str) -> List[str]:
    """Tokenize a dot-notation path, handling backticks and array indices."""
    if path.startswith("csv::"):
        return [path]

    tokens: List[str] = []
    buf = ""
    in_bt = False
    i, n = 0, len(path)

    while i < n:
        ch = path[i]
        if ch == "`":
            in_bt = not in_bt
            i += 1
            continue
        if ch == "." and not in_bt:
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
            continue
        if ch == "[" and not in_bt:
            if buf:
                tokens.append(buf)
                buf = ""
            j = path.find("]", i)
            if j == -1:
                raise ValueError(f"Unclosed '[' in path: {path}")
            tokens.append(path[i : j + 1])
            i = j + 1
            continue
        buf += ch
        i += 1

    if buf:
        tokens.append(buf)
    return tokens


def _path_exists(data: Any, path: str) -> bool:
    """Check if a dot-notation path exists in a parsed data structure."""
    tokens = _tokenize_path(path)

    def walk(node: Any, toks: List[str]) -> bool:
        if not toks:
            return True
        tok, *rest = toks

        # CSV header rule.
        if isinstance(node, dict) and "csv_headers" in node and tok.startswith("csv::"):
            header = tok[5:]
            return header in (node["csv_headers"] or []) and not rest

        # Wildcard.
        if tok == "*":
            if isinstance(node, list):
                return any(walk(item, rest) for item in node)
            return False

        # Fixed index [n].
        if tok.startswith("[") and tok.endswith("]"):
            try:
                idx = int(tok[1:-1])
            except ValueError:
                return False
            return isinstance(node, list) and 0 <= idx < len(node) and walk(node[idx], rest)

        # Dict key.
        if isinstance(node, dict):
            if tok in node:
                return walk(node[tok], rest)
            # XML attribute fallback: @id -> id.
            if tok.startswith("@"):
                attr = tok[1:]
                if attr in node:
                    return walk(node[attr], rest)

        return False

    return walk(data, tokens)


if __name__ == "__main__":
    StructEvalResourcesServer.run_webserver()

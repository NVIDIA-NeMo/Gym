# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
RDKit Chemistry — Nemo-Gym Resources Server

Verifiable chemistry question answering with optional Python tool-use.

The agent receives a natural-language chemistry question paired with a SMILES
string and must respond with a numeric value in the requested answer format.

Questions are drawn from a stratified sample of the ChEMBL database and cover
RDKit-computable molecular properties (ring counts, hydrogen bond
donor/acceptor counts, fragment presence, continuous descriptors, etc.).

Two question methods are supported (selected per-row via the ``method`` field):

* **direct** — the model answers from parametric knowledge alone.
* **mcp-python** — the model may call a Python tool (via ``ns_tools`` wrapper)
  to compute the answer using RDKit.

This server is a pure verifier: it only implements ``verify()``.  When tool-use
is needed, pair this server with ``ns_tools`` via
``rdkit_chemistry.yaml`` — ``ns_tools`` handles tool execution and
delegates verification here.

Reward signal
-------------
Discrete property types use rounded exact match:
reward = 1.0 iff round(predicted) == round(actual), else 0.0.
Float properties use tight numeric equality.
When no numeric value can be extracted from the response, reward = 0.0.

Dataset format (JSONL)
----------------------
Each row carries:
  responses_create_params.input  — user message (prompt + format instruction)
  responses_create_params.tools  — [] for direct, [stateful_python_code_exec] for mcp-python
  expected_answer                — ground-truth numeric value
  property_type                  — "count" | "bool" | "presence" | "fragment" | "float"
  property                       — RDKit property name, e.g. "NumValenceElectrons"
  chembl_id                      — ChEMBL molecule identifier
  smiles                         — canonical SMILES string
  method                         — "direct" | "mcp-python"
  answer_format                  — optional answer-format key (fmt_00 through fmt_30)
"""

from __future__ import annotations

import ipaddress
import math
import re
import socket
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_first_server_config_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_PROPERTY_TYPES = {"count", "bool", "presence", "fragment", "float"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_ANSWER_FORMAT_REGEXES: dict[str, re.Pattern[str]] = {
    "fmt_00": re.compile(r"\(\((.*?)\)\)", re.S),
    "fmt_01": re.compile(r"\(Answer:\s*(.+?)\)", re.S),
    "fmt_02": re.compile(r"Final answer:\s*\((.+?)\)", re.S),
    "fmt_03": re.compile(r"Answer is\s*\[(.+?)\]", re.S),
    "fmt_04": re.compile(r"\[Answer:\s*(.+?)\]", re.S),
    "fmt_05": re.compile(r"\[\[(.+?)\]\]", re.S),
    "fmt_06": re.compile(r"Correct Answer:\s*\[(.+?)\]", re.S),
    "fmt_07": re.compile(r"\\boxed\{(.+?)\}", re.S),
    "fmt_08": re.compile(r"\\boxed\{(.+?)\}", re.S),
    "fmt_09": re.compile(r"\{\{(.+?)\}\}", re.S),
    "fmt_10": re.compile(r"Answer Value:\s*\{(.+?)\}", re.S),
    "fmt_11": re.compile(r"<<(.+?)>>", re.S),
    "fmt_12": re.compile(r"<<(.+?)>>", re.S),
    "fmt_13": re.compile(r"<(.+?)>", re.S),
    "fmt_14": re.compile(r"<Answer:\s*(.+?)>", re.S),
    "fmt_15": re.compile(r"<final_answer>\s*(.+?)\s*</final_answer>", re.S),
    "fmt_16": re.compile(r"Final Answer:\s*\|\|(.+?)\|\|", re.S),
    "fmt_17": re.compile(r"The answer is:\s*\|(.+?)\|", re.S),
    "fmt_18": re.compile(r"\*\*Answer:\s*(.+?)\*\*", re.S),
    "fmt_19": re.compile(r"\*\*Final answer is:\s*(.+?)\*\*", re.S),
    "fmt_20": re.compile(r"Answer:\s*\*(.+?)\*", re.S),
    "fmt_21": re.compile(r"## Answer:\s*(.+?)\s*##", re.S),
    "fmt_22": re.compile(r"ANSWER IS\s*(.+)"),
    "fmt_23": re.compile(r"Response:\s*(.+)"),
    "fmt_24": re.compile(r"Final Answer\s*->\s*(.+)"),
    "fmt_25": re.compile(r"Final value is:\s*(.+)"),
    "fmt_26": re.compile(r"Correct Answer >>\s*(.+)"),
    "fmt_27": re.compile(r"Answer Value:\s*(.+)"),
    "fmt_28": re.compile(r"Final Answer\s*=\s*(.+)"),
    "fmt_29": re.compile(r"Correct answer is\s*(.+)"),
    "fmt_30": re.compile(r"Final Answer:\s*(.+)"),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RDKitChemistryConfig(BaseResourcesServerConfig):
    sandbox_venv_path: str = ""
    sandbox_proxy_port: int | None = 6001
    sandbox_proxy_max_concurrency: int = 128
    sandbox_proxy_request_timeout_s: float = 120.0
    sandbox_proxy_connect_retries: int = 3
    sandbox_proxy_retry_backoff_s: float = 0.25
    sandbox_startup_probe_enabled: bool = True
    sandbox_startup_probe_timeout_s: float = 15.0
    sandbox_extra_packages: list[str] = ["rdkit", "flask", "wcwidth"]
    sandbox_discovery_path: str = ""
    require_local_ns_tools_colocation: bool = False
    ns_tools_server_name: str = "rdkit_chemistry_ns_tools"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChemistryRunRequest(BaseRunRequest):
    expected_answer: Union[str, float, int]
    property_type: str
    property: str
    chembl_id: Optional[str] = None
    smiles: Optional[str] = None
    method: Optional[str] = None
    answer_format: Optional[str] = None
    use_box_format: bool = False


class ChemistryVerifyRequest(ChemistryRunRequest, BaseVerifyRequest):
    pass


class ChemistryVerifyResponse(BaseVerifyResponse):
    predicted_value: Optional[float] = None
    correct: bool = False
    property: str = ""
    property_type: str = ""
    chembl_id: Optional[str] = None
    method: Optional[str] = None
    answer_format: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: response text extraction
# ---------------------------------------------------------------------------


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract the final assistant text from a Responses API output trajectory."""
    texts: list[str] = []
    for output_item in body.response.output:
        if getattr(output_item, "type", None) == "message" and getattr(output_item, "role", None) == "assistant":
            content = getattr(output_item, "content", None)
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Helpers: value extraction
# ---------------------------------------------------------------------------


def _parse_numeric_capture(inner: str) -> Optional[float]:
    """Parse a numeric value from a regex capture."""
    inner = inner.strip()
    try:
        return float(inner)
    except (ValueError, TypeError):
        pass

    nums = _NUMBER_RE.findall(inner)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def _extract_from_answer_format(text: str, answer_format: str) -> Optional[float]:
    """Extract a numeric value using the requested answer-format regex."""
    pattern = _ANSWER_FORMAT_REGEXES.get(answer_format)
    if pattern is None:
        raise ValueError(f"Unsupported answer_format={answer_format!r}")

    matches = pattern.findall(text)
    if not matches:
        return None

    match = matches[-1]
    if isinstance(match, tuple):
        match = next((group for group in match if group), "")
    return _parse_numeric_capture(match)


def extract_predicted_value(
    response: str,
    property_type: str,
    *,
    answer_format: Optional[str] = None,
    use_box_format: bool = False,
) -> Optional[float]:
    """
    Extract a predicted numeric value from the model's response text.

    When *answer_format* is present, the answer must match the corresponding
    ``fmt_XX`` regex. Only the last match is considered. Unknown formats raise
    ``ValueError`` so bad data fails loudly.

    Legacy rows without *answer_format* still use *use_box_format*: boxed when
    true, double parentheses when false.

    Returns None if no value can be extracted.
    """
    if not isinstance(response, str):
        return None

    text = response.strip()
    if answer_format is not None:
        return _extract_from_answer_format(text, answer_format)

    legacy_answer_format = "fmt_07" if use_box_format else "fmt_00"
    return _extract_from_answer_format(text, legacy_answer_format)


# ---------------------------------------------------------------------------
# Helpers: reward computation
# ---------------------------------------------------------------------------


def compute_reward(
    predicted: Optional[float],
    actual: float,
    property_type: str = "",
) -> float:
    """Compute reward for numeric RDKit properties."""
    if predicted is None or math.isnan(predicted):
        return 0.0
    if property_type == "float":
        return 1.0 if math.isclose(predicted, actual, rel_tol=1e-6, abs_tol=1e-6) else 0.0
    return 1.0 if round(predicted) == round(actual) else 0.0


# ---------------------------------------------------------------------------
# Resources server
# ---------------------------------------------------------------------------


class RDKitChemistryResourcesServer(SimpleResourcesServer):
    config: RDKitChemistryConfig

    def _resolve_host_for_compare(self, host: str) -> str:
        if host in {"localhost", "127.0.0.1", "::1"}:
            return "127.0.0.1"
        if host in {"0.0.0.0", "::"}:
            return socket.gethostbyname(socket.gethostname())
        return socket.gethostbyname(host)

    def _is_loopback_host(self, host: str) -> bool:
        try:
            return ipaddress.ip_address(self._resolve_host_for_compare(host)).is_loopback
        except ValueError:
            return host == "localhost"

    def _validate_local_ns_tools_colocation(self) -> None:
        if not self.config.require_local_ns_tools_colocation:
            return

        if not self.config.sandbox_venv_path:
            raise RuntimeError(
                "require_local_ns_tools_colocation=true requires sandbox_venv_path "
                "to be set so the local sandbox can be started."
            )

        ns_tools_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.ns_tools_server_name,
        )
        sandbox_host = ns_tools_config.get("sandbox_host", "127.0.0.1")
        expected_sandbox_port = self.config.sandbox_proxy_port or 6000
        if not self._is_loopback_host(sandbox_host):
            raise RuntimeError(
                "require_local_ns_tools_colocation=true requires the paired ns_tools "
                f"server to use a loopback sandbox_host, but got {sandbox_host!r}."
            )
        if int(ns_tools_config.get("sandbox_port", expected_sandbox_port)) != expected_sandbox_port:
            raise RuntimeError(
                "require_local_ns_tools_colocation=true requires the paired ns_tools "
                f"server to use sandbox_port={expected_sandbox_port}, but got "
                f"{ns_tools_config.get('sandbox_port')!r}."
            )

        rdkit_host = self._resolve_host_for_compare(self.config.host)
        ns_tools_host = self._resolve_host_for_compare(ns_tools_config["host"])
        if rdkit_host != ns_tools_host:
            raise RuntimeError(
                "Local sandbox mode requires rdkit_chemistry and its paired ns_tools "
                "server to be colocated on the same host, "
                f"but rdkit_chemistry resolved to {rdkit_host} and "
                f"{self.config.ns_tools_server_name!r} resolved to {ns_tools_host}."
            )

    def setup_webserver(self) -> FastAPI:
        if self.config.sandbox_venv_path:
            import sandbox_launcher

            self._validate_local_ns_tools_colocation()
            sandbox_launcher.start_sandbox(
                venv_path=self.config.sandbox_venv_path,
                proxy_port=self.config.sandbox_proxy_port,
                proxy_max_concurrency=self.config.sandbox_proxy_max_concurrency,
                proxy_request_timeout_s=self.config.sandbox_proxy_request_timeout_s,
                proxy_connect_retries=self.config.sandbox_proxy_connect_retries,
                proxy_retry_backoff_s=self.config.sandbox_proxy_retry_backoff_s,
                startup_probe_enabled=self.config.sandbox_startup_probe_enabled,
                startup_probe_timeout_s=self.config.sandbox_startup_probe_timeout_s,
                extra_packages=self.config.sandbox_extra_packages,
                discovery_path=self.config.sandbox_discovery_path or None,
            )

        return super().setup_webserver()

    async def verify(
        self,
        body: ChemistryVerifyRequest,
    ) -> ChemistryVerifyResponse:
        if body.property_type not in _SUPPORTED_PROPERTY_TYPES:
            raise ValueError(f"Unsupported property_type={body.property_type!r}")

        text = _extract_last_assistant_text(body)
        predicted = extract_predicted_value(
            text,
            body.property_type,
            answer_format=body.answer_format,
            use_box_format=body.use_box_format,
        )
        actual = float(body.expected_answer)
        reward = compute_reward(predicted, actual, property_type=body.property_type)
        correct = reward == 1.0

        return ChemistryVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_value=predicted,
            correct=correct,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rollouts = [r for task in tasks for r in task]

        grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in rollouts:
            method = r.get("method", "unknown") or "unknown"
            ptype = r.get("property_type", "unknown") or "unknown"
            grouped[method][ptype].append(r)

        def _ptype_stats(group: list) -> Dict[str, Any]:
            rewards = [r["reward"] for r in group]
            corrects = [int(r.get("correct", False)) for r in group]
            return {
                "count": len(group),
                "accuracy": statistics.mean(corrects),
                "mean_reward": statistics.mean(rewards),
            }

        result: Dict[str, Any] = {}
        for method in sorted(grouped):
            method_rollouts = [r for ptype_group in grouped[method].values() for r in ptype_group]
            method_rewards = [r["reward"] for r in method_rollouts]
            method_corrects = [int(r.get("correct", False)) for r in method_rollouts]
            by_ptype = {ptype: _ptype_stats(g) for ptype, g in sorted(grouped[method].items())}
            result[method] = {
                "count": len(method_rollouts),
                "accuracy": statistics.mean(method_corrects),
                "mean_reward": statistics.mean(method_rewards),
                "by_property_type": by_ptype,
            }
        return result

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = {"mean/reward", "mean/correct"}
        return {k: v for k, v in agent_metrics.items() if k in keys or k in ("direct", "mcp-python")}


if __name__ == "__main__":
    RDKitChemistryResourcesServer.run_webserver()

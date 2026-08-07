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
"""AgentIF resources server — agentic instruction-following benchmark scoring.

Ports THU-KEG AgentIF (707 agentic scenarios) to a NeMo Gym resources server.
Each row carries a list of constraints scored by a typed checker pipeline:
``llm`` / ``llm_conditional_check`` steps call an LLM judge; ``code`` steps
exec a dataset-provided ``check_following(response)`` in a fresh globals dict.

Two headline metrics follow upstream ``1.evaluation_api.py``:

* **ISR** (Instruction Success Rate) — 1.0 when every scored constraint in the
  row passed, else 0.0; returned by ``verify()`` as the per-row reward.
* **CSR** (Constraint Success Rate) — fraction of scored constraints that
  passed; reported corpus-wide by ``compute_metrics``.

Per-row dimension (unconditional / conditional / example_driven) and type
(formatting / semantic / resource) tallies let ``compute_metrics`` reconstruct
the corpus breakdowns. ``code`` checkers use a fresh per-call globals dict so
concurrent rollouts cannot interfere.

Build datasets with ``python resources_servers/agentif/prepare_agentif.py``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json


LOG = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_IMPORT_RE = re.compile(r"^\s*(import\s+\S+|from\s+\S+\s+import\s+\S+)", re.MULTILINE)
_DIMENSIONS = frozenset({"unconditional", "conditional", "example_driven"})
_TYPES = frozenset({"formatting", "semantic", "resource"})
_DIM_MAP = {"unconditional": "vanilla", "conditional": "condition", "example_driven": "example"}
_TYPE_MAP = {"formatting": "formatting", "semantic": "semantic", "resource": "tool"}

_UNSET = object()


def _strip_think(text: str) -> str:
    if not text or "</think>" not in text:
        return text or ""
    cleaned = _THINK_RE.sub("", text)
    if cleaned == text:
        cleaned = text.split("</think>", 1)[-1]
    return cleaned.strip()


def _coerce_text(content: Any) -> str:
    """Flatten Responses-API message content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
                continue
            t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
        return "".join(parts)
    return "" if content is None else str(content)


def _response_text(response: Optional[NeMoGymResponse]) -> str:
    """Best-effort extraction of the assistant text from a NeMoGymResponse."""
    if response is None:
        return ""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        parts.append(_coerce_text(getattr(item, "content", "")))
    return "".join(parts)


def _normalize_checker_source(function: str) -> str:
    """Insert ``def`` when a checker source has ``check_following(...):`` without it."""
    if not function or not isinstance(function, str):
        return function
    if re.search(r"^\s*def\s+check_following\s*\(", function, re.MULTILINE):
        return function
    return re.sub(
        r"^(\s*)check_following\s*\(",
        r"\1def check_following(",
        function,
        count=1,
        flags=re.MULTILINE,
    )


def _format_judge_prompt(prompt_template: str, student_text: str) -> str:
    """Render the judge prompt, appending a ``{response}`` slot when absent."""
    p = prompt_template
    if "{response}" not in p:
        p += "\n\nHere is model response: {response}"
    return p.replace("{response}", student_text)


def _run_code_check(exec_src: str, working: str) -> Tuple[Any, Optional[str]]:
    """Execute a dataset ``check_following`` in a fresh globals dict (async-safe)."""
    if not working:
        return None, "Empty response"
    try:
        exec_src = _normalize_checker_source(exec_src)
        shared_globals: dict = {}
        local_vars: dict = {"response": working}
        for stmt in _IMPORT_RE.findall(exec_src):
            exec(stmt, shared_globals)  # noqa: S102
        exec(exec_src, shared_globals, local_vars)  # noqa: S102
        if "check_following" not in local_vars:
            return None, "check_following not defined"
        return local_vars["check_following"](local_vars["response"]), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _bump_breakdowns(
    constraint: Dict[str, Any],
    success: bool,
    by_dimension: Dict[str, Dict[str, int]],
    by_type: Dict[str, Dict[str, int]],
) -> None:
    """Tally a scored constraint into its dimension and type buckets."""
    key = "n_true" if success else "n_false"
    dim = constraint.get("dimension")
    if dim in _DIMENSIONS:
        by_dimension.setdefault(dim, {"n_true": 0, "n_false": 0})[key] += 1
    raw_type = constraint.get("type")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    for t in types:
        if t in _TYPES:
            by_type.setdefault(t, {"n_true": 0, "n_false": 0})[key] += 1


class AgentIFResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the agentif resources server (LLM judge + code exec scorers)."""

    name: str = "agentif"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 32


class AgentIFRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[Dict[str, Any]] = None


class AgentIFVerifyRequest(AgentIFRunRequest, BaseVerifyRequest):
    pass


class AgentIFVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    n_true: int = 0
    n_false: int = 0
    n_null: int = 0
    isr_pass: int = 0
    isr_counted: int = 0
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_type: Dict[str, Dict[str, int]] = {}
    query_id: Optional[int] = None
    turn_id: Optional[int] = None


class AgentIFResourcesServer(SimpleResourcesServer):
    config: AgentIFResourcesServerConfig

    _semaphore: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        mc = self.config.judge_endpoint_max_concurrency
        self._semaphore = asyncio.Semaphore(mc) if mc is not None else nullcontext()

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def _call_judge(self, judge_prompt: str) -> Optional[str]:
        """Call the judge model on one prompt; return None on any failure."""
        params = self.config.judge_responses_create_params.model_dump()
        params["input"] = [{"role": "user", "content": judge_prompt}]
        try:
            async with self._semaphore:
                resp = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
                result = await get_response_json(resp)
            judge_text = _strip_think(_response_text(NeMoGymResponse(**result)))
            return judge_text or None
        except Exception as exc:  # noqa: BLE001
            LOG.warning("agentif judge call failed: %s: %s", type(exc).__name__, exc)
            return None

    async def _score_constraint(self, constraint: Dict[str, Any], response: str) -> Any:
        """Run one constraint's checker pipeline to a True / False / None verdict."""
        working: Any = response
        score: Any = _UNSET
        for ev in constraint.get("evaluation", []):
            etype = ev.get("type")
            if working is None and etype in ("llm", "llm_conditional_check"):
                # Preceding code check failed; propagate None without calling the judge (upstream parity).
                score = None
                break
            if etype == "llm_conditional_check":
                cond = await self._call_judge(_format_judge_prompt(ev.get("exec", ""), working or ""))
                if cond and "YES" in cond:
                    continue
                score = None
                break
            elif etype == "llm":
                judge_text = await self._call_judge(_format_judge_prompt(ev.get("exec", ""), working or ""))
                if not judge_text:
                    working = None
                    break
                working = judge_text
            elif etype == "code":
                loop = asyncio.get_running_loop()
                working, err = await loop.run_in_executor(None, _run_code_check, ev.get("exec", ""), working or "")
                if err:
                    LOG.debug("agentif code check error: %s", err)
                    working = None

        if score is _UNSET:
            if isinstance(working, str):
                if "YES" in working:
                    score = True
                elif "NO" in working:
                    score = False
                else:
                    score = None
            else:
                score = working
        return score

    async def verify(self, body: AgentIFVerifyRequest) -> AgentIFVerifyResponse:
        meta = body.verifier_metadata or {}
        constraints = meta.get("constraints") or []
        response = _strip_think(_response_text(body.response))

        n_true = n_false = n_null = 0
        by_dimension: Dict[str, Dict[str, int]] = {}
        by_type: Dict[str, Dict[str, int]] = {}

        for constraint in constraints:
            score = await self._score_constraint(constraint, response)
            if score is None:
                n_null += 1
            elif score is True:
                n_true += 1
                _bump_breakdowns(constraint, True, by_dimension, by_type)
            else:
                n_false += 1
                _bump_breakdowns(constraint, False, by_dimension, by_type)

        n_scored = n_true + n_false
        csr = n_true / n_scored if n_scored else 0.0
        isr_counted = 1 if n_scored else 0
        isr_pass = 1 if (n_scored and n_false == 0) else 0

        LOG.info(
            "agentif verify query_id=%s isr=%d csr=%.3f n_scored=%d",
            meta.get("query_id"),
            isr_pass,
            csr,
            n_scored,
        )
        return AgentIFVerifyResponse(
            **body.model_dump(),
            # Per-row reward is ISR (all-or-nothing); CSR is reported by compute_metrics.
            reward=float(isr_pass),
            n_true=n_true,
            n_false=n_false,
            n_null=n_null,
            isr_pass=isr_pass,
            isr_counted=isr_counted,
            by_dimension=by_dimension,
            by_type=by_type,
            query_id=meta.get("query_id"),
            turn_id=meta.get("turn_id"),
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task_rollouts in tasks for r in task_rollouts]
        metrics: Dict[str, Any] = {}
        if not rows:
            return metrics

        total_true = sum(int(r.get("n_true", 0)) for r in rows)
        total_false = sum(int(r.get("n_false", 0)) for r in rows)
        n_scored = total_true + total_false
        if n_scored > 0:
            metrics["csr"] = total_true / n_scored

        isr_pass = sum(int(r.get("isr_pass", 0)) for r in rows)
        isr_counted = sum(int(r.get("isr_counted", 0)) for r in rows)
        if isr_counted > 0:
            metrics["isr"] = isr_pass / isr_counted

        for dim in _DIMENSIONS:
            dt = sum(int(r.get("by_dimension", {}).get(dim, {}).get("n_true", 0)) for r in rows)
            df = sum(int(r.get("by_dimension", {}).get(dim, {}).get("n_false", 0)) for r in rows)
            if dt + df > 0:
                metrics[f"by_dimension/{_DIM_MAP[dim]}/accuracy"] = dt / (dt + df)

        for t in _TYPES:
            tt = sum(int(r.get("by_type", {}).get(t, {}).get("n_true", 0)) for r in rows)
            tf = sum(int(r.get("by_type", {}).get(t, {}).get("n_false", 0)) for r in rows)
            if tt + tf > 0:
                metrics[f"by_type/{_TYPE_MAP[t]}/accuracy"] = tt / (tt + tf)

        rewards = [float(r["reward"]) for r in rows if isinstance(r.get("reward"), (int, float))]
        if rewards:
            metrics["mean_reward"] = sum(rewards) / len(rewards)
        metrics["count"] = len(rows)
        metrics["n_null_total"] = sum(int(r.get("n_null", 0)) for r in rows)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {k: agent_metrics[k] for k in ("isr", "csr", "mean_reward") if k in agent_metrics}


if __name__ == "__main__":
    AgentIFResourcesServer.run_webserver()

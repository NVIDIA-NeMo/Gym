# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LogScale (CQL) NL-to-query resources server for NeMo Gym.

Scoring rubric (3 components):
  1. Validity    – is the CQL syntactically valid?        (0 or 1, from LogScale validateQuery)
  2. Semantic    – does the query capture user intent?     (1-5, LLM judge)
  3. Execution   – do the results look correct?            (1-5, LLM judge)

  reward = (validity + semantic/5 + execution/5) / 3      (0.0 – 1.0)

Quick start:
    bash setup.sh          # start container, ingest data
    python app.py          # start the resources server
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import requests as http_requests
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status

import ray

from container_engine import LogScaleContainerEngine


@ray.remote(
    num_cpus=1,
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def verify_cql_remote(
    logscale_url: str,
    repository: str | None,
    cql: str,
) -> dict:
    """Ray-remote CQL validation + execution on a SPREAD-scheduled worker."""
    from container_engine import LogScaleContainerEngine as _Engine
    engine = _Engine(base_url=logscale_url, repository=repository)
    validation = engine.validate_query(cql)
    exec_result = engine.execute(cql)
    return {"validation": validation, "exec_result": exec_result}

# ---------------------------------------------------------------------------
# Judge prompts — loaded from prompt_templates/ for easy tracking and reuse
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).resolve().parent / "prompt_templates"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text().strip()


SEMANTIC_JUDGE_PROMPT = _load_prompt("semantic_judge.txt")
EXECUTION_JUDGE_PROMPT = _load_prompt("execution_judge.txt")
QUESTION_GEN_PROMPT = _load_prompt("question_generation.txt")

DIFFICULTY_GUIDES = {
    1: "EASY: Simple single-step queries. One filter + one aggregation. Basic count, top, groupBy on a single field.",
    2: "MEDIUM: Two-step pipelines. Filter + aggregate + sort, or filter + table with field selection.",
    3: "HARD: Multi-step pipelines with 3+ stages. Combine filters, groupBy with aliases, sort, regex, cidr, multiple event types.",
    4: "EXPERT: Complex queries requiring joins, temporal correlation, case statements, nested aggregations, or multiple rename/transform steps.",
    5: "ADVERSARIAL: Edge cases, ambiguous phrasing, trick questions, queries that test CQL quirks (OR precedence, tag vs field, regex escaping).",
}


# ---------------------------------------------------------------------------
# CQL extraction
# ---------------------------------------------------------------------------

def extract_cql_from_response(response: NeMoGymResponse) -> str:
    """Extract CQL from model output (strip markdown, take first block or full text)."""
    texts = []
    for o in getattr(response, "output", []) or []:
        if getattr(o, "type", None) != "message":
            continue
        for c in getattr(o, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                texts.append(getattr(c, "text", "") or "")
    raw = "".join(texts).strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.split("\n") if not line.strip().startswith("```")).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Query execution helpers
# ---------------------------------------------------------------------------

def _execute_container(engine: LogScaleContainerEngine, cql: str) -> dict:
    return engine.execute(cql)


def _execute_local(engine, cql: str) -> dict:
    try:
        result_df = engine.execute(cql)
        return {
            "success": True,
            "n_rows": len(result_df),
            "preview": result_df.head(15).to_string(index=False),
        }
    except Exception as e:
        return {
            "success": False,
            "n_rows": 0,
            "preview": "",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Judge response parsers
# ---------------------------------------------------------------------------

def _parse_semantic_judge(raw: str) -> dict:
    sm = re.search(r"SEMANTIC_SCORE:\s*(\d)", raw)
    rm = re.search(r"SEMANTIC_REASONING:\s*(.+)", raw)
    return {
        "semantic_score": int(sm.group(1)) if sm else None,
        "semantic_reasoning": rm.group(1).strip() if rm else "",
    }


def _parse_execution_judge(raw: str) -> dict:
    sm = re.search(r"EXECUTION_SCORE:\s*(\d)", raw)
    rm = re.search(r"EXECUTION_REASONING:\s*(.+)", raw)
    return {
        "execution_score": int(sm.group(1)) if sm else None,
        "execution_reasoning": rm.group(1).strip() if rm else "",
    }


def compute_reward(validity: int, semantic: int | None, execution: int | None) -> float:
    """Combine the three rubric components into a single 0-1 reward.

    reward = (validity + semantic/5 + execution/5) / 3
    Falls back gracefully when judge scores are unavailable.
    """
    sem = (semantic / 5.0) if semantic is not None else 0.0
    exe = (execution / 5.0) if execution is not None else 0.0
    return (float(validity) + sem + exe) / 3.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LogScaleCQLResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "logscale_cql"

    engine: str = "container"
    container_runtime: str = "docker"
    logscale_url: str = "http://localhost:8080"
    repository: Optional[str] = None

    csv_path: Optional[str] = None
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None

    judge_api_key: Optional[str] = None
    judge_api_url: str = "https://inference-api.nvidia.com"
    judge_model: str = "aws/anthropic/bedrock-claude-opus-4-6"


class LogScaleCQLVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    question: str = ""
    metadata: Optional[dict[str, Any]] = None


class GenerateQuestionsRequest(BaseModel):
    count: int = 10
    difficulty: int = 2


class GenerateQuestionsResponse(BaseModel):
    questions: List[dict]
    fields: List[str]
    event_types: List[str]


class LogScaleCQLVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    question: str = ""
    extracted_cql: Optional[str] = None

    validity_score: int = 0
    validity_diagnostics: Optional[list[dict]] = None

    semantic_score: Optional[int] = None
    semantic_reasoning: Optional[str] = None

    execution_score: Optional[int] = None
    execution_reasoning: Optional[str] = None
    exec_success: bool = False
    exec_n_rows: int = 0
    exec_preview: Optional[str] = None
    exec_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Resources server
# ---------------------------------------------------------------------------

class LogScaleCQLResourcesServer(SimpleResourcesServer):
    config: LogScaleCQLResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.engine == "container":
            self._run_setup_sh()
            self._container_engine = LogScaleContainerEngine(
                base_url=self.config.logscale_url,
                repository=self.config.repository,
            )
            self._local_engine = None
        else:
            self._container_engine = None
            self._init_local_engine()

        api_key = self.config.judge_api_key or os.getenv("INFERENCE_API_KEY", "")
        if not api_key:
            api_key = self._load_key_from_dotenv("INFERENCE_API_KEY")
        self._judge_api_key = api_key
        self._judge_api_url = (os.getenv("JUDGE_API_URL") or self.config.judge_api_url).rstrip("/")
        self._judge_model = os.getenv("JUDGE_MODEL") or self.config.judge_model

    def _run_setup_sh(self):
        """Start LogScale container, install license, and ingest data automatically."""
        server_dir = Path(__file__).resolve().parent
        setup_script = server_dir / "setup.sh"
        data_dir = server_dir / "data"
        logscale_url = self.config.logscale_url

        env = os.environ.copy()
        env["LOGSCALE_URL"] = logscale_url
        env["CONTAINER_RUNTIME"] = self.config.container_runtime
        env.setdefault("HUMIO_LICENSE",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzUxMiJ9.eyJpc09lbSI6ZmFsc2UsImF1ZCI6Ikh1bWlvLWxpY2Vuc2UtY2hlY2siLCJzdWIiOiJOdmlkaWFTaGFyZWRSZXNlYXJjaCIsInVpZCI6IjVOS1ZjTEJFRFB3cVVIR0ciLCJtYXhVc2VycyI6MTAwMCwiYWxsb3dTQUFTIjpmYWxzZSwibWF4Q29yZXMiOjk5OTk5OSwidmFsaWRVbnRpbCI6MTgwMzgxOTYwMCwiZXhwIjoxODY2NDcwMzQ5LCJpc1RyaWFsIjpmYWxzZSwiaWF0IjoxNzcxODYyMzQ5LCJtYXhJbmdlc3RHYlBlckRheSI6NjR9.Ad0BFe5rCCOKEo0WFYNdsLyFB7iOSrNGRMIMLQppmoFBU1xACh3V5P_XKmVV1NqK0PGZ4FeODQjuGkLHteWWVsywASCz8C-HYpw9kgRPjbocuikn1p0YuzoxTf53OA0-MD0HH_RafmJ98E2CXlXRkvNlLUB3LsgbayP2PEhebhVMfqu4"
        )

        # Step 1: Run setup.sh (starts container, waits for healthy, discovers repo)
        if setup_script.exists():
            logger.info("Running LogScale container setup: %s", setup_script)
            result = subprocess.run(
                ["bash", str(setup_script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=1200,
            )
            if result.returncode != 0:
                logger.warning("setup.sh returned rc=%d (non-fatal, container may already be running)", result.returncode)
            logger.info("LogScale container setup complete")

        # Step 2: Install license via GraphQL (idempotent)
        license_key = env.get("HUMIO_LICENSE", "")
        if license_key:
            try:
                import requests as _req
                resp = _req.post(
                    f"{logscale_url}/graphql",
                    json={
                        "query": "mutation($l: String!) { updateLicenseKey(license: $l) { expiresAt } }",
                        "variables": {"l": license_key},
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                if resp.ok:
                    logger.info("LogScale license installed")
                else:
                    logger.warning("License install returned %d", resp.status_code)
            except Exception as e:
                logger.warning("Could not install license: %s", e)

        # Step 3: Ingest datasets if not already ingested
        datasets_dir = data_dir / "datasets_v2"
        ingest_script = data_dir / "ingest_datasets.py"
        if datasets_dir.exists() and ingest_script.exists():
            try:
                import requests as _req
                # Quick check: if data already ingested, skip
                resp = _req.post(
                    f"{logscale_url}/graphql",
                    json={"query": "{ searchDomains { name } }"},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                if resp.ok:
                    repos = resp.json().get("data", {}).get("searchDomains", [])
                    sandbox = next((r["name"] for r in repos if r["name"].startswith("sandbox_")), None)
                    if sandbox:
                        count_resp = _req.post(
                            f"{logscale_url}/api/v1/repositories/{sandbox}/queryjobs",
                            json={"queryString": "count()", "start": "365days", "isLive": False},
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                            timeout=10,
                        )
                        if count_resp.ok:
                            import time
                            job_id = count_resp.json().get("id")
                            for _ in range(20):
                                time.sleep(0.3)
                                pr = _req.get(
                                    f"{logscale_url}/api/v1/repositories/{sandbox}/queryjobs/{job_id}",
                                    headers={"Accept": "application/json"}, timeout=10,
                                ).json()
                                if pr.get("done"):
                                    count = int(pr.get("events", [{}])[0].get("_count", 0))
                                    if count > 0:
                                        logger.info("Data already ingested (%d events), skipping", count)
                                        return
                                    break

                logger.info("Ingesting datasets from %s", datasets_dir)
                result = subprocess.run(
                    [sys.executable, str(ingest_script), "--datasets-dir", str(datasets_dir)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    logger.info("Dataset ingestion complete")
                else:
                    logger.warning("Ingestion returned rc=%d: %s", result.returncode, result.stderr[-300:])
            except Exception as e:
                logger.warning("Could not auto-ingest datasets: %s", e)

    @staticmethod
    def _load_key_from_dotenv(key_name: str) -> str:
        env_file = Path.home() / ".env"
        if not env_file.exists():
            return ""
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == key_name:
                    return v.strip()
        return ""

    def _init_local_engine(self):
        import pandas as pd
        import sys

        _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        from logscale_evaluations.cql_engine import CQLEngine

        csv_path = self.config.csv_path
        if not csv_path:
            csv_path = Path(__file__).resolve().parent / "data" / "synthetic_falcon_events.csv"
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found (needed for engine=local): {csv_path}")

        df = pd.read_csv(csv_path, dtype=str)
        self._local_engine = CQLEngine(df)

    def _execute(self, cql: str) -> dict:
        if self._container_engine is not None:
            return _execute_container(self._container_engine, cql)
        return _execute_local(self._local_engine, cql)

    def _validate(self, cql: str) -> dict:
        """Score 1: is the query syntactically valid? Returns {is_valid, diagnostics}."""
        if self._container_engine is not None:
            return self._container_engine.validate_query(cql)
        # Local mode: no validation available, assume valid if it parses
        return {"is_valid": True, "diagnostics": []}

    async def _judge_call(self, prompt_text: str) -> str:
        """Send a prompt to the judge LLM and return the raw text response."""
        judge_params = self.config.judge_responses_create_params.model_copy(deep=True)
        judge_params.input = [NeMoGymEasyInputMessage(role="user", content=prompt_text)]
        response = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_params.model_dump(exclude_none=True),
        )
        await raise_for_status(response)
        judge_json = await get_response_json(response)
        judge_response = NeMoGymResponse.model_validate(judge_json)
        text = ""
        for o in getattr(judge_response, "output", []) or []:
            if getattr(o, "type", None) != "message":
                continue
            for c in getattr(o, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    text += getattr(c, "text", "") or ""
        return text

    def _direct_llm_call(self, prompt_text: str, max_tokens: int = 512, temperature: float = 0) -> str:
        """Call the LLM directly via OpenAI-compatible API (no Gym model server needed)."""
        for attempt in range(3):
            resp = http_requests.post(
                f"{self._judge_api_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._judge_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._judge_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert LogScale CQL query evaluator."},
                        {"role": "user", "content": prompt_text},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            if resp.status_code in (429, 502, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        resp.raise_for_status()
        return ""

    def _discover_schema(self) -> tuple[list[str], list[str]]:
        """Query LogScale for field names and event types."""
        if self._container_engine is None:
            return [], []

        fields_result = self._container_engine.execute("head(1)")
        fields: list[str] = []
        if fields_result.get("success"):
            preview = fields_result.get("preview", "")
            if preview and preview != "(no results)":
                header_line = preview.strip().split("\n")[0]
                fields = sorted(h.strip() for h in re.split(r"\s{2,}", header_line) if h.strip())

        et_result = self._container_engine.execute(
            'rename("#event_simpleName", as=eSN) | groupBy(eSN, function=count()) | sort(_count, order=desc) | head(50)'
        )
        event_types: list[str] = []
        if et_result.get("success"):
            preview = et_result.get("preview", "")
            for line in preview.strip().split("\n")[1:]:
                # Format: "  7505          ProcessRollup2" — count then name
                parts = line.strip().split()
                if len(parts) >= 2:
                    event_types.append(parts[-1])
                elif len(parts) == 1 and not parts[0].isdigit():
                    event_types.append(parts[0])
        return fields, event_types

    def _generate_questions(self, count: int = 10, difficulty: int = 2) -> list[dict]:
        """Generate evaluation questions using the LLM."""
        if not self._judge_api_key:
            raise ValueError("No LLM API key configured (set INFERENCE_API_KEY)")
        fields, event_types = self._discover_schema()
        difficulty = max(1, min(5, difficulty))
        guide = DIFFICULTY_GUIDES[difficulty]
        prompt = QUESTION_GEN_PROMPT.format(
            count=count,
            fields=", ".join(fields) if fields else "(discover via head(1))",
            event_types=", ".join(event_types) if event_types else "(discover via top(#event_simpleName))",
            difficulty=difficulty,
            guide=guide,
        )
        raw = self._direct_llm_call(prompt, max_tokens=2048, temperature=0.7)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```")).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [
                    {"question": str(x.get("question", "")).strip(),
                     "category": str(x.get("category", "semantic")).strip()}
                    for x in parsed if isinstance(x, dict) and x.get("question")
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return [
                    {"question": str(x.get("question", "")).strip(),
                     "category": str(x.get("category", "semantic")).strip()}
                    for x in parsed if isinstance(x, dict) and x.get("question")
                ]
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.post("/generate_questions")
        async def generate_questions_endpoint(body: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
            fields, event_types = self._discover_schema()
            questions = self._generate_questions(count=body.count, difficulty=body.difficulty)
            return GenerateQuestionsResponse(
                questions=questions, fields=fields, event_types=event_types,
            )

        return app

    async def verify(self, body: LogScaleCQLVerifyRequest) -> LogScaleCQLVerifyResponse:
        question = (body.question or "").strip()
        if not question:
            return LogScaleCQLVerifyResponse(
                **body.model_dump(exclude={"question"}), reward=0.0, question=question,
            )

        cql = extract_cql_from_response(body.response)
        if not cql:
            return LogScaleCQLVerifyResponse(
                **body.model_dump(exclude={"question"}), reward=0.0, question=question,
                extracted_cql=None, validity_score=0,
                semantic_reasoning="No CQL extracted from model output.",
            )

        # -- Score 1 & execution via Ray SPREAD when container engine is active
        if self._container_engine is not None and ray.is_initialized():
            ref = verify_cql_remote.remote(
                self.config.logscale_url,
                self.config.repository,
                cql,
            )
            remote_result = await asyncio.to_thread(ray.get, ref)
            validation = remote_result["validation"]
            exec_result = remote_result["exec_result"]
        else:
            validation = self._validate(cql)
            exec_result = self._execute(cql)

        validity_score = 1 if validation["is_valid"] else 0
        exec_success = exec_result.get("success", False)
        results_str = exec_result["preview"] if exec_success else f"ERROR: {exec_result.get('error', 'unknown')}"

        # -- Scores 2 & 3: LLM judge (if configured) -----------------------
        semantic_score = None
        semantic_reasoning = ""
        execution_score = None
        execution_reasoning = ""

        has_gym_judge = self.config.judge_model_server and self.config.judge_responses_create_params
        has_direct_judge = bool(self._judge_api_key)

        if has_gym_judge:
            try:
                sem_raw = await self._judge_call(
                    SEMANTIC_JUDGE_PROMPT.format(question=question, query=cql)
                )
                parsed_sem = _parse_semantic_judge(sem_raw)
                semantic_score = parsed_sem["semantic_score"]
                semantic_reasoning = parsed_sem["semantic_reasoning"]
            except Exception as e:
                semantic_reasoning = f"Semantic judge error: {e}"

            try:
                exe_raw = await self._judge_call(
                    EXECUTION_JUDGE_PROMPT.format(question=question, query=cql, results=results_str)
                )
                parsed_exe = _parse_execution_judge(exe_raw)
                execution_score = parsed_exe["execution_score"]
                execution_reasoning = parsed_exe["execution_reasoning"]
            except Exception as e:
                execution_reasoning = f"Execution judge error: {e}"

        elif has_direct_judge:
            try:
                sem_raw = self._direct_llm_call(
                    SEMANTIC_JUDGE_PROMPT.format(question=question, query=cql)
                )
                parsed_sem = _parse_semantic_judge(sem_raw)
                semantic_score = parsed_sem["semantic_score"]
                semantic_reasoning = parsed_sem["semantic_reasoning"]
            except Exception as e:
                semantic_reasoning = f"Semantic judge error: {e}"

            try:
                exe_raw = self._direct_llm_call(
                    EXECUTION_JUDGE_PROMPT.format(question=question, query=cql, results=results_str)
                )
                parsed_exe = _parse_execution_judge(exe_raw)
                execution_score = parsed_exe["execution_score"]
                execution_reasoning = parsed_exe["execution_reasoning"]
            except Exception as e:
                execution_reasoning = f"Execution judge error: {e}"

        else:
            execution_score = 5 if exec_success else 1
            execution_reasoning = "Execution success" if exec_success else "Execution failed"

        reward = compute_reward(validity_score, semantic_score, execution_score)

        return LogScaleCQLVerifyResponse(
            **body.model_dump(exclude={"question"}),
            reward=reward,
            question=question,
            extracted_cql=cql,
            validity_score=validity_score,
            validity_diagnostics=validation.get("diagnostics"),
            semantic_score=semantic_score,
            semantic_reasoning=semantic_reasoning,
            execution_score=execution_score,
            execution_reasoning=execution_reasoning,
            exec_success=exec_success,
            exec_n_rows=int(exec_result.get("n_rows", 0) or 0),
            exec_preview=exec_result.get("preview") if exec_success else None,
            exec_error=exec_result.get("error") if not exec_success else None,
        )


if __name__ == "__main__":
    LogScaleCQLResourcesServer.run_webserver()

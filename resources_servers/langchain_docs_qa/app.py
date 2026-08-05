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

import json
import os
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.langchain_docs_qa.retriever import BM25
from resources_servers.langchain_docs_qa.scoring import (
    citation_match,
    mcqa_match,
    parse_answer,
)


class LangchainDocsQAConfig(BaseResourcesServerConfig):
    name: str = "langchain_docs_qa"
    corpus_path: str = "environments/langchain_docs_qa/data/chunks.jsonl"
    top_k: int = 5
    w_answer: float = 0.7
    w_cite: float = 0.3
    reward_mode: str = "mcqa"  # mcqa | judge
    retrieval_backend: str = "bm25"  # bm25 | mcp
    mcp_url: str = "https://docs.langchain.com/mcp"
    mcp_tools: List[str] = [
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    ]
    judge_base_url: str = ""
    judge_model: str = "judge"
    judge_api_key: str = "x"
    judge_cred_file: str = ""


class SearchDocsRequest(BaseModel):
    query: str


class SearchDocsResponse(BaseModel):
    output: str


class DocsQARunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    gold_answer: str = ""
    gold_page: str = ""
    gold_letter: Optional[str] = None


class DocsQAVerifyRequest(DocsQARunRequest, BaseVerifyRequest):
    pass


class DocsQAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    extracted_answer: Optional[str]
    cited_pages: List[str]
    answer_match: float
    citation_match: float


class LangchainDocsQAResourcesServer(SimpleResourcesServer):
    config: LangchainDocsQAConfig

    _bm25: Optional[BM25] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.config.retrieval_backend == "mcp":
            return
        if not os.path.exists(self.config.corpus_path):
            raise FileNotFoundError(
                f"BM25 corpus not found at {self.config.corpus_path}. Build it with "
                "environments/langchain_docs_qa/prepare.py, or set retrieval_backend=mcp."
            )
        with open(self.config.corpus_path) as handle:
            chunks = [json.loads(line) for line in handle]
        chunks = [c for c in chunks if _substantive(c.get("content", ""))]
        object.__setattr__(self, "_bm25", BM25(chunks))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        if self.config.retrieval_backend == "mcp":
            app.post("/search_docs")(self._make_mcp_handler(self.config.mcp_tools[0]))
            for tool in self.config.mcp_tools:
                app.post(f"/{tool}")(self._make_mcp_handler(tool))
        else:
            app.post("/search_docs")(self.search_docs)
        return app

    async def search_docs(self, body: SearchDocsRequest) -> SearchDocsResponse:
        hits = self._bm25.search(body.query, k=self.config.top_k)
        blocks = [
            f"Title: {h['title']}\nLink: {h['link']}\nPage: {h['page']}\nContent: {h['content'][:1200]}" for h in hits
        ]
        return SearchDocsResponse(output="\n\n".join(blocks) if blocks else "No results.")

    def _make_mcp_handler(self, tool: str):
        async def handler(body: Dict[str, Any]):
            return {"output": await self._mcp().call_tool(tool, body or {})}

        return handler

    def _mcp(self):
        client = getattr(self, "_mcp_client", None)
        if client is None:
            from resources_servers.langchain_docs_qa.mcp_retriever import MCPClient

            client = MCPClient(self.config.mcp_url)
            object.__setattr__(self, "_mcp_client", client)
        return client

    async def verify(self, body: DocsQAVerifyRequest) -> DocsQAVerifyResponse:
        generation = _collect_text(body.response.output)
        pred_answer, cited_pages = parse_answer(generation)
        cite = citation_match(body.gold_page, cited_pages)

        if self.config.reward_mode == "judge":
            answer = await self._judge_score(body, pred_answer or generation)
            reward = self.config.w_answer * answer + self.config.w_cite * cite
        else:
            answer = mcqa_match(body.gold_letter or "", generation)
            reward = answer

        return DocsQAVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=pred_answer or None,
            cited_pages=cited_pages,
            answer_match=answer,
            citation_match=cite,
        )

    async def _judge_score(self, body: DocsQAVerifyRequest, model_answer: str) -> float:
        """1.0 if the answer is equivalent to gold, else 0.0."""
        base = self.config.judge_base_url or os.environ.get("OPENAI_BASE_URL", "")
        if not base:
            return 0.0
        api_key = self.config.judge_api_key
        if api_key == "x":
            api_key = os.environ.get("OPENAI_API_KEY", "x")
        if getattr(self.config, "judge_cred_file", ""):
            try:
                with open(self.config.judge_cred_file) as handle:
                    api_key = handle.read().strip()
            except OSError:
                pass
        model = self.config.judge_model
        if model == "judge":
            model = os.environ.get("NRL_JUDGE_MODEL", "judge")
        prompt = (
            "You are grading a LangChain documentation QA answer for equivalence. "
            "Reply with ONLY one word: CORRECT or INCORRECT.\n"
            f"Question: {_question_of(body)}\n"
            f"Reference answer: {body.gold_answer}\n"
            f"Model answer: {str(model_answer)[:600]}\n"
            "CORRECT if the model answer conveys the same fact as the reference, "
            "even if phrased differently. INCORRECT otherwise."
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1024,
        }
        try:
            session = getattr(self, "_judge_session", None)
            if session is None:
                session = aiohttp.ClientSession()
                object.__setattr__(self, "_judge_session", session)
            async with session.post(
                base.rstrip("/") + "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = json.loads(await resp.text())
            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            text = content or (message.get("reasoning_content") or "")
            upper = text.upper()
            last_correct = upper.rfind("CORRECT")
            last_incorrect = upper.rfind("INCORRECT")
            if last_correct == -1:
                return 0.0
            return 1.0 if last_correct != last_incorrect + 2 else 0.0
        except Exception:
            return 0.0


def _collect_text(outputs) -> str:
    parts: List[str] = []
    for output in outputs:
        for item in getattr(output, "content", None) or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        for attr in ("text", "reasoning", "summary"):
            value = getattr(output, attr, None)
            if isinstance(value, str) and value:
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    text = item if isinstance(item, str) else getattr(item, "text", None)
                    if text:
                        parts.append(text)
    return "\n".join(parts)


def _question_of(body) -> str:
    try:
        for message in body.responses_create_params["input"]:
            if message.get("role") == "user":
                return str(message.get("content", ""))[:300]
    except Exception:
        pass
    return ""


def _substantive(content: str) -> bool:
    import re

    stripped = re.sub(r"<[^>]+>", "", content).strip()
    return not (content.strip().startswith("<div") or content.count("<") > 15 or len(stripped) < 120)


if __name__ == "__main__":
    LangchainDocsQAResourcesServer.run_webserver()

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
"""GenRM training inside an agent harness.

The task and reward are generative_reward_model's: a judging prompt with two candidate responses in,
one JSON verdict out, negative L1 against the human labels. The policy answers from inside a sandboxed
agent (opencode_sandboxed_agent) that can run code and search the web before it commits to a verdict.
Three things differ from the single-turn server:

- Only the final assistant message is scored. An agentic trajectory carries many assistant messages,
  and the strict parser fails on more than one JSON dict, so a draft verdict mid-run plus the final
  one would otherwise score as a parse failure.
- A response that stopped on the token limit is a parse failure even if a verdict made it out, as a
  truncated single-turn verdict would be.
- /seed_session saves each response's fenced code blocks as sandbox files, so the judge can run them
  without copying them out of the prompt. The prose stays in the prompt only: plain-text copies of the
  conversation and responses mostly got re-read, costing turns without adding anything.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseSeedSessionRequest, BaseSeedSessionResponse
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.generative_reward_model.app import (
    GenerativeRewardModelResourcesServer,
    GenerativeRewardModelResourcesServerConfig,
    GenerativeRewardModelVerifyRequest,
    GenerativeRewardModelVerifyResponse,
)


# Section markers of the GenRM judging prompt (every row of the training blend carries each exactly once).
_CONTEXT_START = "#### Conversation Context ####"
_RESPONSES_START = "#### Responses to be Scored ####"
_PLAN_START = "#### Evaluation Plan ####"
_RESPONSE_RE = r"\[The Start of Response {n}\]\n?(.*?)\n?\[The End of Response {n}\]"
_CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)
_CODE_EXTENSIONS = {
    "python": "py",
    "py": "py",
    "python3": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "csharp": "cs",
    "cs": "cs",
    "c#": "cs",
    "go": "go",
    "golang": "go",
    "rust": "rs",
    "rs": "rs",
    "ruby": "rb",
    "rb": "rb",
    "php": "php",
    "sql": "sql",
    "bash": "sh",
    "sh": "sh",
    "shell": "sh",
    "zsh": "sh",
    "kotlin": "kt",
    "swift": "swift",
    "r": "r",
    "html": "html",
    "css": "css",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
}


def split_genrm_prompt(prompt: str) -> Optional[Tuple[str, str, str]]:
    """Return (conversation context, response 1, response 2) from a GenRM judging prompt, or None."""
    ctx_start = prompt.find(_CONTEXT_START)
    resp_start = prompt.find(_RESPONSES_START)
    if ctx_start < 0 or resp_start < ctx_start:
        return None
    responses = prompt[resp_start:]
    plan_start = responses.find(_PLAN_START)
    if plan_start >= 0:
        responses = responses[:plan_start]
    parts = []
    for n in (1, 2):
        match = re.search(_RESPONSE_RE.format(n=n), responses, flags=re.DOTALL)
        if match is None:
            return None
        parts.append(match.group(1))
    context = prompt[ctx_start + len(_CONTEXT_START) : resp_start].strip("\n")
    return context, parts[0], parts[1]


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Fenced code blocks of `text` as (file extension, code)."""
    return [
        (_CODE_EXTENSIONS.get(lang.lower(), "txt"), code)
        for lang, code in _CODE_BLOCK_RE.findall(text)
        if code.strip()
    ]


def final_assistant_text(response: NeMoGymResponse) -> Optional[str]:
    """Text of the assistant messages after the last tool call, or None when the run ended on a tool call."""
    last_tool_idx = -1
    for idx, item in enumerate(response.output):
        if item.type in ("function_call", "function_call_output"):
            last_tool_idx = idx
    texts = []
    for item in response.output[last_tool_idx + 1 :]:
        if item.type != "message":
            continue
        for content in item.content:
            if content.type == "output_text":
                texts.append(content.text)
    return "".join(texts) if texts else None


def tool_call_counts(response: NeMoGymResponse) -> Dict[str, float]:
    names = [item.name for item in response.output if item.type == "function_call"]
    # OpenCode names MCP tools <server>_<tool>, e.g. genrm_search_search / genrm_search_browse.
    num_search = sum(name.endswith("search") for name in names)
    num_browse = sum(name.endswith("browse") for name in names)
    num_bash = sum(name == "bash" for name in names)
    return {
        "num_tool_calls": float(len(names)),
        "num_bash_calls": float(num_bash),
        "num_search_calls": float(num_search),
        "num_browse_calls": float(num_browse),
        "num_other_tool_calls": float(len(names) - num_search - num_browse - num_bash),
        "used_tools": float(bool(names)),
        "used_web": float(num_search + num_browse > 0),
    }


class AgenticGenRMResourcesServerConfig(GenerativeRewardModelResourcesServerConfig):
    # Each response's fenced code blocks go to <dir>/response_<n>_code/block_<k>.<ext> in the agent's
    # sandbox; null writes none.
    sandbox_files_dir: Optional[str] = "/workspace/genrm"


class AgenticGenRMSeedSessionRequest(BaseSeedSessionRequest):
    # The agent posts the whole dataset row.
    model_config = ConfigDict(extra="allow")


class AgenticGenRMSeedSessionResponse(BaseSeedSessionResponse):
    # Remote path -> text, written into the agent-created sandbox before the run.
    sandbox_files: Dict[str, str] = {}


class AgenticGenRMVerifyRequest(GenerativeRewardModelVerifyRequest):
    pass


class AgenticGenRMVerifyResponse(GenerativeRewardModelVerifyResponse):
    final_message_found: float
    response_incomplete: float
    num_tool_calls: float
    num_bash_calls: float
    num_search_calls: float
    num_browse_calls: float
    num_other_tool_calls: float
    used_tools: float
    used_web: float


class AgenticGenRMResourcesServer(GenerativeRewardModelResourcesServer):
    config: AgenticGenRMResourcesServerConfig

    async def seed_session(
        self, request: Request, body: AgenticGenRMSeedSessionRequest
    ) -> AgenticGenRMSeedSessionResponse:
        return AgenticGenRMSeedSessionResponse(sandbox_files=self._sandbox_files(body.model_extra or {}))

    def _sandbox_files(self, row: Dict[str, Any]) -> Dict[str, str]:
        root = self.config.sandbox_files_dir
        if not root:
            return {}
        try:
            content = row["responses_create_params"]["input"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            return {}
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            return {}
        split = split_genrm_prompt(content)
        if split is None:
            return {}
        _, response_1, response_2 = split
        root = root.rstrip("/")
        files = {}
        for n, text in ((1, response_1), (2, response_2)):
            for k, (ext, code) in enumerate(extract_code_blocks(text), start=1):
                files[f"{root}/response_{n}_code/block_{k:02d}.{ext}"] = code
        return files

    async def verify(self, body: AgenticGenRMVerifyRequest) -> AgenticGenRMVerifyResponse:
        final_text = final_assistant_text(body.response)
        incomplete = body.response.status == "incomplete"
        scored_output = []
        if final_text is not None and not incomplete:
            # Score the final message alone; the base verify concatenates every message item.
            scored_output = [
                {
                    "type": "message",
                    "id": "msg_agentic_genrm_final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": final_text, "annotations": []}],
                }
            ]
        scored_response = NeMoGymResponse.model_validate(body.response.model_dump() | {"output": scored_output})
        result = await super().verify(body.model_copy(update={"response": scored_response}))
        return AgenticGenRMVerifyResponse(
            **(result.model_dump() | {"response": body.response.model_dump()}),
            final_message_found=float(final_text is not None),
            response_incomplete=float(incomplete),
            **tool_call_counts(body.response),
        )


if __name__ == "__main__":
    AgenticGenRMResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgenticGenRMResourcesServer.run_webserver()  # noqa: F401

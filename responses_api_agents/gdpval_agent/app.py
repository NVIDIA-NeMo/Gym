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
import json
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from urllib.parse import quote

import anyio
from fastapi import HTTPException, Request, Response
from openai.types.responses.response_usage import ResponseUsage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.global_config import REASON_TO_SKIP_KEY_NAME
from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.bash_sandbox.app import (
    SavedFile,
    SeedSessionRequest,
    UploadedFile,
    UploadFilesRequest,
)


logger = logging.getLogger(__name__)

TASK_TIMEOUT = 60 * 15  # 15 minutes — cap on the /v1/responses agent loop

MAX_TOOL_OUTPUT_CHARS = 20_000


def _estimate_input_tokens(messages: list) -> int:
    """Rough token estimate: 4 chars per token, based on JSON-serialised message text."""
    return sum(len(m.model_dump_json()) for m in messages) // 4


FINISH_TOOL_NAME = "finish"
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gdpval_user_prompt.txt"

# Files written by infrastructure/sentinels; excluded when enumerating task outputs on resume
_RESUME_SKIP_NAMES = frozenset({"finish_params.json", "reward.json", "history.json", "metadata.json", "log.txt"})


def _task_dir_for(body: "GDPValAgentRunRequest") -> Path:
    name = f"task_{body.task_id}"
    if body.repeat_index is not None:
        name += f"_r{body.repeat_index}"
    return Path(body.output_dir) / name


MESSAGE_SUMMARIZER_PATH = Path(__file__).parent / "prompts" / "message_summarizer.txt"
MESSAGE_SUMMARIZER_BRIDGE_PATH = Path(__file__).parent / "prompts" / "message_summarizer_bridge.txt"
MESSAGE_SUMMARIZER = MESSAGE_SUMMARIZER_PATH.read_text()
MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE = MESSAGE_SUMMARIZER_BRIDGE_PATH.read_text()


@dataclass
class Cookies:
    """Tracks cookies for outgoing requests to the model and resources servers."""

    model_server: dict[str, str] = field(default_factory=dict)
    resources_server: dict[str, str] = field(default_factory=dict)

    def model_dump(self):
        return {
            "model_server": self.model_server,
            "resources_server": self.resources_server,
        }


class GDPValAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 100
    context_window_tokens: int = 128000  # model's total context window (e.g. 128k for Llama-3.1-70B)
    context_summarization_cutoff: float = 0.7
    remaining_step_warning_threshold: int | None = 20


class GDPValAgentRunRequest(BaseRunRequest):
    task_prompt: str | NeMoGymEasyInputMessage
    system_prompt: str | NeMoGymEasyInputMessage
    instruction_prompt_template: str | None = None
    output_dir: str
    reference_file_urls: List[str] = []
    reference_files_to_save: List[str] = []
    uploaded_reference_files: List[UploadedFile] = []
    task_id: str | None = None
    session_id: str | None = None
    task_dir: Path | None = None
    repeat_index: int | None = None


class GDPValAgentVerifyRequest(BaseVerifyRequest):
    task_id: str
    output_files: List[SavedFile] = Field(default_factory=list)


class GDPValAgentVerifyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    responses_create_params: dict = Field(default_factory=dict)
    response: dict = Field(default_factory=dict)
    reward: float = 0.0
    task_id: str | None = None
    output_files: List[SavedFile] = Field(default_factory=list)
    committee_verdicts: list = Field(default_factory=list)


class GDPValAgentResponse(BaseModel):
    """GDPVal agent response including permanent save locations of output files."""

    output: list
    saved_files: List[SavedFile] = Field(default_factory=list)
    session_id: str | None = None


class GDPValAgent(SimpleResponsesAPIAgent):
    config: GDPValAgentConfig

    async def download_reference_files(self, body: GDPValAgentRunRequest) -> list[str] | None:
        reference_dir = os.path.join(body.task_dir, "reference_files")
        os.makedirs(reference_dir, exist_ok=True)
        reference_files = []

        for url, filename in zip(body.reference_file_urls, body.reference_files_to_save):
            path = os.path.join(reference_dir, os.path.basename(filename))
            if os.path.exists(path):
                print(f"Reference file: [{filename}] already exists at path: [{path}]. Skipping download.")
                reference_files.append(path)
                continue
            try:
                safe_url = quote(url, safe=":/%")
                urllib.request.urlretrieve(safe_url, path)
                print(f"Successfully downloaded reference file: [{filename}] to path: [{path}].")
                reference_files.append(path)
            except Exception as e:
                print(f"Failed to download reference file: [{filename}] to path: [{path}] with error: [{e}].")
                continue

        return reference_files if reference_files else None

    async def prepare_reference_files(
        self, body: GDPValAgentRunRequest, cookies: Cookies
    ) -> tuple[List[UploadedFile] | None, Cookies]:
        """
        Downloads reference files from the URLs and uploads them to the resources
        server session. Returns the uploaded reference files and the updated cookies.
        """

        if body.task_dir is None:
            body.task_dir = Path(body.output_dir).joinpath(f"task_{body.task_id}")
        body.task_dir.mkdir(parents=True, exist_ok=True)

        reference_files = await self.download_reference_files(body)
        uploaded_reference_files = None

        # Upload reference files to the resources server session
        if reference_files is not None:
            upload_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/upload_files",
                json=UploadFilesRequest(
                    session_id=body.session_id,
                    paths=reference_files,
                    dest_dir="reference_files",
                ).model_dump(),
                cookies=cookies.resources_server,
                affinity_key=body.session_id,
            )
            await raise_for_status(upload_response)
            upload_response_json = await get_response_json(upload_response)
            uploaded_reference_files = [UploadedFile(**f) for f in upload_response_json["uploaded"]]
            cookies.resources_server = upload_response.cookies

        return uploaded_reference_files, cookies

    async def init_message_history(self, body: GDPValAgentRunRequest) -> NeMoGymResponseInput:
        if isinstance(body.task_prompt, str):
            # Load the prompt template from disk if not provided in the request.
            instruction_template = body.instruction_prompt_template
            if instruction_template is None and PROMPT_TEMPLATE_PATH.exists():
                instruction_template = PROMPT_TEMPLATE_PATH.read_text()

            if instruction_template is not None and body.uploaded_reference_files is not None:
                reference_files_str = "\n".join([f"[{file.dest_path}]" for file in body.uploaded_reference_files])
                task_prompt = instruction_template.format(task=body.task_prompt, references=reference_files_str)
            elif instruction_template is not None:
                task_prompt = instruction_template.format(task=body.task_prompt, references="None")
            else:
                task_prompt = body.task_prompt

            task_prompt = NeMoGymEasyInputMessage(role="user", content=task_prompt)

        else:
            task_prompt = body.task_prompt

        if isinstance(body.system_prompt, str):
            system_prompt = NeMoGymEasyInputMessage(role="system", content=body.system_prompt)
        else:
            system_prompt = body.system_prompt

        return [system_prompt, task_prompt]

    async def single_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Cookies,
    ) -> tuple[NeMoGymResponse, Cookies]:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=cookies.model_server,
        )
        # We raise for status here since we expect model calls to always work.
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)
        cookies.model_server = model_response.cookies

        try:
            model_response = NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {json.dumps(model_response_json)}"
            ) from e

        return model_response, cookies

    async def run_tool(
        self,
        call: NeMoGymResponseFunctionToolCall,
        session_id: str,
        output_dir: str | None,
        cookies: Cookies,
    ) -> tuple[NeMoGymFunctionCallOutput, Cookies]:
        """Execute a tool call on the resources server.

        Injects session_id into the call arguments so the resources server
        knows which sandbox session to operate on. For the finish tool,
        also injects output_dir so files are saved to the right location.
        """
        KNOWN_TOOLS = {"run_command", "web_search", "web_fetch", "finish", "upload_files", "save_files"}
        if call.name not in KNOWN_TOOLS:
            tool_response = NeMoGymFunctionCallOutput(
                type="function_call_output",
                call_id=call.call_id,
                output=json.dumps(
                    {
                        "error": f"Unknown tool '{call.name}'. Available tools: {', '.join(sorted(KNOWN_TOOLS))}. "
                        "Use 'run_command' to execute bash commands (including python scripts)."
                    }
                ),
            )
            return tool_response, cookies

        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError as e:
            tool_response = NeMoGymFunctionCallOutput(
                type="function_call_output",
                call_id=call.call_id,
                output=json.dumps(
                    {
                        "error": f"Invalid JSON in tool arguments: {e}. "
                        "Please ensure your tool call arguments are valid JSON."
                    }
                ),
            )
            return tool_response, cookies
        args["session_id"] = session_id

        if call.name == FINISH_TOOL_NAME and output_dir is not None:
            args.setdefault("output_dir", output_dir)

        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{call.name}",
            json=args,
            cookies=cookies.resources_server,
            affinity_key=session_id,
        )
        cookies.resources_server = api_response.cookies

        raw_output = (await api_response.content.read()).decode()
        if len(raw_output) > MAX_TOOL_OUTPUT_CHARS:
            raw_output = (
                raw_output[:MAX_TOOL_OUTPUT_CHARS]
                + f"\n[...output truncated at {MAX_TOOL_OUTPUT_CHARS} chars; "
                f"total output was {len(raw_output)} chars. Use head/tail or redirect to a file to avoid truncation...]"
            )
        tool_response = NeMoGymFunctionCallOutput(
            type="function_call_output",
            call_id=call.call_id,
            output=raw_output,
        )

        return tool_response, cookies

    async def step(
        self,
        model_params: NeMoGymResponseCreateParamsNonStreaming,
        session_id: str,
        output_dir: str | None,
        cookies: Cookies,
    ) -> tuple[
        List[NeMoGymResponseOutputMessage],
        List[NeMoGymFunctionCallOutput],
        List[SavedFile] | None,
        Cookies,
        ResponseUsage | None,
    ]:
        """Execute one agent step: generate assistant message and run any requested tool calls.

        Args:
            model_params: The model request parameters (input messages, tools, etc.)
            session_id: The sandbox session ID to inject into tool calls.
            output_dir: Directory for permanently saving output files (injected into finish).
            cookies: Current cookies for server communication.

        Returns:
            Tuple of (model_outputs, tool_outputs, finished, cookies, usage).
            finished is None if the agent has not finished, or a list of SavedFile
            objects representing the permanently saved output files when finished.
            usage is the ResponseUsage object from the model response (may be None).

        """
        model_outputs = []
        tool_outputs = []
        finished = None

        model_response, cookies = await self.single_response(model_params, cookies)
        output = model_response.output
        model_outputs.extend(output)

        if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
            return model_outputs, tool_outputs, finished, cookies, model_response.usage

        function_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]

        for call in function_calls:
            tool_response, cookies = await self.run_tool(call, session_id, output_dir, cookies)
            tool_outputs.append(tool_response)

            if call.name == FINISH_TOOL_NAME:
                finished = []
                try:
                    finish_data = json.loads(tool_response.output)
                    for file_info in finish_data.get("saved", []):
                        finished.append(
                            SavedFile(
                                source_path=file_info.get("source_path", ""),
                                output_path=file_info["output_path"],
                                size=file_info["size"],
                            )
                        )
                    if finished:
                        print("Output files saved to permanent locations:")
                        for saved_file in finished:
                            print(f"  {saved_file.output_path} ({saved_file.size} bytes)")
                    else:
                        print("Finish tool called but no output files were saved.")
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not parse finish tool response for saved files: {e}")

        return model_outputs, tool_outputs, finished, cookies, model_response.usage

    def _get_steps_remaining_msg(self, remaining_steps: int):
        """Create a user message warning the agent about remaining turns before max_steps is reached."""
        if remaining_steps == 1:
            return NeMoGymEasyInputMessage(
                role="user", content="This is the last turn. Please finish the task by calling the finish tool."
            )

        return NeMoGymEasyInputMessage(
            role="user",
            content=(
                f"You have {remaining_steps} turns remaining to complete the task. "
                "Please continue. Remember you will need a separate turn to finish the task."
            ),
        )

    async def summarize_messages(
        self,
        messages: NeMoGymResponseInput,
        model_params: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Cookies,
        n_task_context: int,
    ) -> tuple[NeMoGymResponseInput, Cookies]:
        """Condense message history using LLM to stay within context window."""
        # Pin task_context to the original initial messages (system prompt + task description +
        # uploaded reference files). Using a loop like stirrup does (takewhile role=user/system)
        # causes bridge/ack messages (also role=user) to accumulate in task_context on each
        # re-summarization, creating nested summaries and growing rather than shrinking context.
        task_context = list(messages[:n_task_context])

        summarizer_input = list(messages) + [NeMoGymEasyInputMessage(role="user", content=MESSAGE_SUMMARIZER)]
        summary_params = model_params.model_copy(
            update={
                "input": summarizer_input,
                "tools": [],
                "max_output_tokens": None,
            }
        )
        summary_response, cookies = await self.single_response(summary_params, cookies)

        summary_text = ""
        for item in summary_response.output:
            if hasattr(item, "content") and item.content:
                for block in item.content:
                    if hasattr(block, "text"):
                        summary_text += block.text

        bridge_content = MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary_text)
        bridge_msg = NeMoGymEasyInputMessage(role="user", content=bridge_content)
        acknowledgement_msg = NeMoGymEasyInputMessage(role="user", content="Got it, thanks!")

        print(
            f"Context summarization complete. History condensed from {len(messages)} to {len(task_context) + 2} messages."
        )
        return list(task_context) + [bridge_msg, acknowledgement_msg], cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: GDPValAgentRunRequest = Body(),
    ) -> GDPValAgentResponse:
        cookies = Cookies()
        if request.cookies:
            cookies.resources_server = dict(request.cookies)
            cookies.model_server = dict(request.cookies)

        # 1. Create session in this handler so the same worker owns the session for the
        #    whole conversation (avoids run() and responses() being on different agent workers).
        if body.session_id is None:
            if body.task_dir is None:
                body.task_dir = _task_dir_for(body)
            body.task_dir.mkdir(parents=True, exist_ok=True)
            if body.task_id and body.repeat_index is not None:
                session_id = f"{body.task_id}_r{body.repeat_index}"
            else:
                session_id = body.task_id or str(uuid.uuid4())
            seed_session_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=SeedSessionRequest(session_id=session_id, repeat_index=body.repeat_index).model_dump(),
                cookies=cookies.resources_server,
                affinity_key=session_id,
            )
            await raise_for_status(seed_session_response)
            seed_session_json = await get_response_json(seed_session_response)
            body.session_id = seed_session_json["session_id"]
            cookies.resources_server = seed_session_response.cookies

        # 2. Prepare reference files (download + upload to sandbox) only if not already uploaded.
        if not body.uploaded_reference_files:
            if body.task_dir is None:
                body.task_dir = _task_dir_for(body)
            body.task_dir.mkdir(parents=True, exist_ok=True)
            uploaded_reference_files, cookies = await self.prepare_reference_files(body, cookies)
            if uploaded_reference_files:
                body.uploaded_reference_files = uploaded_reference_files

        # 3. Build initial message history from task/system prompts
        input_messages = await self.init_message_history(body)
        n_task_context = len(input_messages)

        # 4. Build model params from the responses_create_params, overriding input
        model_params = body.responses_create_params.model_copy(update={"input": input_messages})

        session_id = body.session_id
        output_dir = str(body.task_dir)
        max_steps = self.config.max_steps
        warning_threshold = self.config.remaining_step_warning_threshold
        summary_cutoff = self.config.context_summarization_cutoff
        finished = None
        outputs = []

        # 5. Execute the task
        for step_num in range(max_steps):
            # Add warning message if max steps is near
            if warning_threshold is not None and max_steps - step_num <= warning_threshold and step_num != 0:
                num_steps_remaining_msg = self._get_steps_remaining_msg(max_steps - step_num)
                model_params = model_params.model_copy(
                    update={"input": model_params.input + [num_steps_remaining_msg]}
                )
                outputs.append(num_steps_remaining_msg)

            model_outputs, tool_outputs, finished, cookies, usage = await self.step(
                model_params, session_id, output_dir, cookies
            )
            model_params = model_params.model_copy(update={"input": model_params.input + model_outputs + tool_outputs})
            outputs.extend(model_outputs)
            outputs.extend(tool_outputs)

            if finished is not None:
                if finished:
                    print(f"Task completed. {len(finished)} output file(s) saved:")
                    for saved_file in finished:
                        print(f"  {saved_file.output_path} ({saved_file.size} bytes)")
                else:
                    print("Task completed. No output files were saved.")
                break

            if summary_cutoff is not None and step_num < max_steps - 1:
                estimated = _estimate_input_tokens(model_params.input)
                usage_total = usage.total_tokens if usage is not None else 0
                max_seen = max(estimated, usage_total)
                if max_seen / self.config.context_window_tokens >= summary_cutoff:
                    pct_used = max_seen / self.config.context_window_tokens
                    print(
                        f"Context at {pct_used:.1%} of context_window_tokens "
                        f"(char-estimate={estimated}, usage={usage_total}, "
                        f"limit={self.config.context_window_tokens}). Summarizing..."
                    )
                    condensed_input, cookies = await self.summarize_messages(
                        model_params.input, model_params, cookies, n_task_context
                    )
                    model_params = model_params.model_copy(update={"input": condensed_input})

        # 6. Use cookies to set response cookies
        if cookies.resources_server:
            for k, v in cookies.resources_server.items():
                response.set_cookie(k, v)
        if cookies.model_server:
            for k, v in cookies.model_server.items():
                response.set_cookie(k, v)

        final_response = GDPValAgentResponse(
            output=outputs,
            saved_files=finished if finished is not None else [],
            session_id=body.session_id,
        )
        return final_response

    async def run(self, request: Request, body: GDPValAgentRunRequest) -> GDPValAgentVerifyResponse:
        cookies = Cookies()
        if request.cookies:
            cookies.resources_server = dict(request.cookies)

        assert len(body.reference_files_to_save) == len(body.reference_file_urls), (
            "Number of reference files and URLs must match."
        )

        # Session is created inside responses() so the same handler (and thus same
        # agent worker) owns the session for the whole conversation.
        if body.task_dir is None:
            body.task_dir = _task_dir_for(body)
        body.task_dir.mkdir(parents=True, exist_ok=True)

        reward_sentinel = body.task_dir / "reward.json"
        finish_sentinel = body.task_dir / "finish_params.json"
        timeout_sentinel = body.task_dir / "timeout.json"

        if timeout_sentinel.exists():
            logger.info(
                "Task %s (repeat=%s) previously timed out, retrying from scratch",
                body.task_id, body.repeat_index,
            )

        # Fully done: return cached result without calling /v1/responses or /verify
        if reward_sentinel.exists():
            logger.info("Task %s (repeat=%s) already verified, skipping", body.task_id, body.repeat_index)
            data = json.loads(reward_sentinel.read_text())
            data[REASON_TO_SKIP_KEY_NAME] = "GDPVal: already collected in previous run"
            return GDPValAgentVerifyResponse.model_validate(data)

        if finish_sentinel.exists():
            # Agent done but not verified: synthesize response_json from existing files, skip to /verify
            logger.info("Task %s (repeat=%s) agent done, skipping to verify", body.task_id, body.repeat_index)
            existing_files = [
                SavedFile(source_path=str(p), output_path=p, size=p.stat().st_size)
                for p in body.task_dir.iterdir()
                if p.is_file() and p.name not in _RESUME_SKIP_NAMES
            ]
            history_file = body.task_dir / "history.json"
            history_output = json.loads(history_file.read_text()) if history_file.exists() else []
            response_json = {"saved_files": [f.model_dump(mode="json") for f in existing_files], "output": history_output}
        else:
            # Normal path: call /v1/responses to run the agent
            try:
                with anyio.fail_after(TASK_TIMEOUT):
                    response = await self.server_client.post(
                        server_name=self.config.name,
                        url_path="/v1/responses",
                        json=body.model_dump(mode="json"),
                        cookies=cookies.resources_server,
                    )
                    await raise_for_status(response)
                    response_json = await get_response_json(response)
                    cookies = response.cookies
                    body.session_id = response_json.get("session_id") or body.session_id
                    history_file = body.task_dir / "history.json"
                    history_file.write_text(json.dumps(response_json.get("output", [])))
            except TimeoutError:
                logger.warning(
                    "Task %s (repeat=%s) timed out after %ds",
                    body.task_id, body.repeat_index, TASK_TIMEOUT,
                )
                timeout_sentinel.write_text(json.dumps({
                    "task_id": body.task_id,
                    "repeat_index": body.repeat_index,
                    "timeout_seconds": TASK_TIMEOUT,
                }))
                raise HTTPException(status_code=408, detail=f"Task timed out after {TASK_TIMEOUT}s")

        # Extract saved files from the agent response
        saved_files = []
        for f in response_json.get("saved_files", []):
            try:
                saved_files.append(SavedFile(**f))
            except Exception:
                pass

        # TODO: Implement real verification logic.
        # For now call /verify as a placeholder — the bash_sandbox verify
        # returns reward=1.0 unconditionally.
        task_prompt_str = body.task_prompt if isinstance(body.task_prompt, str) else body.task_prompt.content
        # Use JSON-serializable values: Path must be str for orjson.dumps()
        output_files_serializable = [{"path": str(f.output_path), "size": f.size} for f in saved_files]
        verify_request = {
            "responses_create_params": body.responses_create_params.model_dump(),
            "response": {
                "id": "placeholder",
                "created_at": 0,
                "model": "placeholder",
                "object": "response",
                "output": response_json.get("output", []),
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            "task_id": body.task_id,
            "task_prompt": task_prompt_str,
            "output_files": output_files_serializable,
            "session_id": body.session_id or "",
            "paths": [str(f.output_path) for f in saved_files],
            "output_dir": str(body.task_dir),
        }

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request,
            cookies=cookies,
            affinity_key=body.session_id or "",
        )
        await raise_for_status(verify_response)
        verify_json = await get_response_json(verify_response)

        result = GDPValAgentVerifyResponse(
            responses_create_params=body.responses_create_params.model_dump(),
            response={
                "id": "placeholder",
                "created_at": 0,
                "model": "placeholder",
                "object": "response",
                "output": response_json.get("output", []),
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            reward=verify_json.get("reward", 1.0),
            task_id=body.task_id,
            output_files=saved_files,
            committee_verdicts=verify_json.get("committee_verdicts", []),
        )

        # Write verify sentinel so future runs can skip directly to return
        reward_sentinel.write_text(result.model_dump_json())
        (body.task_dir / "history.json").unlink(missing_ok=True)
        return result


if __name__ == "__main__":
    GDPValAgent.run_webserver()

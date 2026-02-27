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
import os
import urllib.request
from dataclasses import dataclass, field
from typing import List

from pathlib import Path
from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from urllib.parse import quote

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
)
from resources_servers.bash_sandbox.app import (
    SavedFile,
    UploadedFile,
    UploadFilesRequest,
    SeedSessionRequest,
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

FINISH_TOOL_NAME = "finish"
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gdpval_user_prompt.txt"


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
    max_tokens: int = 10000
    context_summarization_cutoff: float = 0.7
    step_warning_threshold: int | None = 80


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


class GDPValAgentVerifyRequest(BaseVerifyRequest):
    task_id: str
    output_files: List[SavedFile] = Field(default_factory=list)


class GDPValAgentVerifyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    reward: float = 0.0
    task_id: str | None = None
    output_files: List[SavedFile] = Field(default_factory=list)


class GDPValAgentResponse(BaseModel):
    """GDPVal agent response including permanent save locations of output files."""
    output: list
    saved_files: List[SavedFile] = Field(default_factory=list)


class GDPValAgent(SimpleResponsesAPIAgent):
    config: GDPValAgentConfig

    async def download_reference_files(self, body: GDPValAgentRunRequest) -> list[str] | None:
        reference_dir = os.path.join(body.task_dir, "reference_files")
        os.makedirs(reference_dir, exist_ok=True)
        reference_files = []

        for url, filename in zip(body.reference_file_urls, body.reference_files_to_save):
            path = os.path.join(reference_dir, os.path.basename(filename))
            if os.path.exists(path):
                print(
                    f"Reference file: [{filename}] already exists at path: [{path}]. Skipping download."
                )
                continue
            try:
                safe_url = quote(url, safe=":/%")
                urllib.request.urlretrieve(safe_url, path)
                print(f"Successfully downloaded reference file: [{filename}] to path: [{path}].")
                reference_files.append(path)
            except Exception as e:
                print(
                    f"Failed to download reference file: [{filename}] to path: "
                    f"[{path}] with error: [{e}]."
                )
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
            )
            await raise_for_status(upload_response)
            upload_response_json = await get_response_json(upload_response)
            uploaded_reference_files = [
                UploadedFile(**f) for f in upload_response_json["uploaded"]
            ]
            cookies.resources_server = upload_response.cookies

        return uploaded_reference_files, cookies


    async def init_message_history(self, body: GDPValAgentRunRequest) -> NeMoGymResponseInput:
        if isinstance(body.task_prompt, str):
            # Load the prompt template from disk if not provided in the request.
            instruction_template = body.instruction_prompt_template
            if instruction_template is None and PROMPT_TEMPLATE_PATH.exists():
                instruction_template = PROMPT_TEMPLATE_PATH.read_text()

            if instruction_template is not None and body.uploaded_reference_files is not None:
                reference_files_str = "\n".join(
                    [f"[{file.dest_path}]" for file in body.uploaded_reference_files]
                )
                task_prompt = instruction_template.format(
                    task=body.task_prompt, references=reference_files_str
                )
            elif instruction_template is not None:
                task_prompt = instruction_template.format(
                    task=body.task_prompt, references="None"
                )
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
        args = json.loads(call.arguments)
        args["session_id"] = session_id

        # For finish, inject output_dir so files are saved to the permanent location
        if call.name == FINISH_TOOL_NAME and output_dir is not None:
            args.setdefault("output_dir", output_dir)

        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{call.name}",
            json=args,
            cookies=cookies.resources_server,
        )
        # We don't raise for status here since it's a valid return for the API to error e.g. 
        # if the model outputs an invalid call or something.
        cookies.resources_server = api_response.cookies

        tool_response = NeMoGymFunctionCallOutput(
            type="function_call_output",
            call_id=call.call_id,
            output=(await api_response.content.read()).decode(),
        )

        return tool_response, cookies

    async def step(
        self,
        model_params: NeMoGymResponseCreateParamsNonStreaming,
        session_id: str,
        output_dir: str | None,
        cookies: Cookies,
    ) -> tuple[List[NeMoGymResponseOutputMessage], List[NeMoGymFunctionCallOutput], List[SavedFile] | None, Cookies]:
        """Execute one agent step: generate assistant message and run any requested tool calls.

        Args:
            model_params: The model request parameters (input messages, tools, etc.)
            session_id: The sandbox session ID to inject into tool calls.
            output_dir: Directory for permanently saving output files (injected into finish).
            cookies: Current cookies for server communication.

        Returns:
            Tuple of (model_outputs, tool_outputs, finished, cookies).
            finished is None if the agent has not finished, or a list of SavedFile
            objects representing the permanently saved output files when finished.

        """
        model_outputs = []
        tool_outputs = []
        finished = None

        model_response, cookies = await self.single_response(model_params, cookies)
        output = model_response.output
        model_outputs.extend(output)

        if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
            return model_outputs, tool_outputs, finished, cookies

        function_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
    
        for call in function_calls:
            tool_response, cookies = await self.run_tool(call, session_id, output_dir, cookies)
            tool_outputs.append(tool_response)

            if call.name == FINISH_TOOL_NAME:
                # Parse saved file locations from the finish tool response
                finished = []
                try:
                    finish_data = json.loads(tool_response.output)
                    for file_info in finish_data.get("saved", []):
                        finished.append((file_info["output_path"], file_info["size"]))
                    if finished:
                        print("Output files saved to permanent locations:")
                        for output_path, size in finished:
                            print(f"  {output_path} ({size} bytes)")
                    else:
                        print("Finish tool called but no output files were saved.")
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not parse finish tool response for saved files: {e}")

        return model_outputs, tool_outputs, finished, cookies

    def _get_steps_remaining_msg(self, remaining_steps: int):
        """Create a user message warning the agent about remaining turns before max_steps is reached."""
        if remaining_steps == 1:
            return NeMoGymEasyInputMessage(
                role="user",
                content="This is the last turn. Please finish the task by calling the finish tool."
            )

        return NeMoGymEasyInputMessage(
            role="user",
            content=(
                f"You have {remaining_steps} turns remaining to complete the task. "
                "Please continue. Remember you will need a separate turn to finish the task."
            )
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: GDPValAgentRunRequest = Body(),
    ) -> GDPValAgentResponse:

        cookies = Cookies()

        # 1. Prepare reference files (download + upload to sandbox)
        uploaded_reference_files, cookies = await self.prepare_reference_files(body, cookies)
        if uploaded_reference_files:
            body.uploaded_reference_files = uploaded_reference_files

        # 2. Build initial message history from task/system prompts
        input_messages = await self.init_message_history(body)

        # 3. Build model params from the responses_create_params, overriding input
        model_params = body.responses_create_params.model_copy(
            update={"input": input_messages}
        )

        session_id = body.session_id
        output_dir = str(body.task_dir)
        max_steps = self.config.max_steps
        warning_threshold = self.config.step_warning_threshold
        summary_cutoff = self.config.context_summarization_cutoff
        finished = None
        outputs = []

        # 4. Execute the task
        for step_num in range(max_steps):
            # Add warning message if max steps is near
            if warning_threshold is not None and \
                max_steps - step_num <= warning_threshold and step_num != 0:
                num_steps_remaining_msg = self._get_steps_remaining_msg(max_steps - step_num)
                model_params = model_params.model_copy(
                    update={"input": model_params.input + [num_steps_remaining_msg]}
                )
                outputs.append(num_steps_remaining_msg)

            # TODO: Add text only tool call message functionality
            model_outputs, tool_outputs, finished, cookies = await self.step(
                model_params, session_id, output_dir, cookies
            )
            model_params = model_params.model_copy(
                update={"input": model_params.input + model_outputs + tool_outputs}
            )
            outputs.extend(model_outputs)
            outputs.extend(tool_outputs)

            if finished is not None:
                if finished:
                    print(f"Task completed. {len(finished)} output file(s) saved:")
                    for output_path, size in finished:
                        print(f"  {output_path} ({size} bytes)")
                else:
                    print("Task completed. No output files were saved.")
                break

            if summary_cutoff is not None:
                continue # TODO

        # 5. Use cookies to set response cookies
        if cookies.resources_server:
            for k, v in cookies.resources_server.items():
                response.set_cookie(k, v)
        if cookies.model_server:
            for k, v in cookies.model_server.items():
                response.set_cookie(k, v)

        final_response = GDPValAgentResponse(
            output=outputs,
            saved_files=finished if finished is not None else [],
        )
        return final_response


    async def run(self, request: Request, body: GDPValAgentRunRequest) -> GDPValAgentVerifyResponse:
        cookies = request.cookies

        assert len(body.reference_files_to_save) == len(body.reference_file_urls), \
            "Number of reference files and URLs must match."

        # Start a unique session for the task
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=SeedSessionRequest(session_id=body.task_id).model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        seed_session_json = await get_response_json(seed_session_response)
        cookies = seed_session_response.cookies

        # Update the session ID in the request body to propagate into task execution.
        body.session_id = seed_session_json["session_id"]

        # Execute the task by calling the self.responses endpoint
        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(response)
        response_json = await get_response_json(response)
        cookies = response.cookies

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
        verify_request = {
            "task_id": body.task_id,
            "output_files": [f.model_dump() for f in saved_files],
            "response": response_json,
        }

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_json = await get_response_json(verify_response)

        return GDPValAgentVerifyResponse(
            reward=verify_json.get("reward", 1.0),
            task_id=body.task_id,
            output_files=saved_files,
        )


if __name__ == "__main__":
    GDPValAgent.run_webserver()

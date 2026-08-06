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
import sys
from pathlib import Path
from shlex import quote
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from openai.types.responses import ResponseInputTextParam
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, get_server_url, raise_for_status


class OpenCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    opencode_version: str
    opencode_config: Dict[str, Any] = Field(default_factory=dict)

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

    debug: bool = False


class OpenCodeSandboxedAgentRunRequest(BaseRunRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class OpenCodeSandboxedAgentVerifyRequest(BaseVerifyRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class OpenCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    opencode_results_fpath: str
    opencode_run_stdout: str
    opencode_run_stderr: str
    opencode_no_export_found: bool


class OpenCodeSandboxedAgent(SimpleResponsesAPIAgent):
    config: OpenCodeSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _start_sandbox(self, sandbox_id: Optional[str] = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        if sandbox_id:
            sandbox = await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=resolved_sandbox_provider)
            return sandbox

        if self.config.debug:
            print("Creating new sandbox since one wasn't provided", file=sys.stderr)

        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        sandbox_spec = SandboxSpec(
            image="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907",  # This is just the first SWE Bench Verified image for now
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=dict(),
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {})),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )

        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    def _create_opencode_config(self) -> Dict[str, Any]:
        return {
            "model": "nemo_gym/dummy_model",
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "nemo_gym": {
                    # TODO @bxyu-nvidia: We should use @ai-sdk/openai here but there is some /v1/responses streaming error.
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": f"{get_server_url(self.config.model_server.name)}/v1",
                        "apiKey": "dummy_key",
                    },
                    "models": {
                        "dummy_model": {
                            "limit": {
                                "context": 0,
                                "input": 0,
                                # @bxyu-nvidia: OpenCode defaults to 32k here https://github.com/anomalyco/opencode/blob/58a99916bb96edf5cf605dc03e1be1e4bacf9ff7/packages/opencode/src/provider/transform.ts#L21
                                # and there is no way to set it to null.
                                # We set it here to explicitly acknowledge that this parameter is set.
                                "output": 32_000,
                            },
                        },
                    },
                }
            },
            **self.config.opencode_config,
        }

    def _opencode_export_to_usages(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseUsage]:
        usages: List[NeMoGymResponseUsage] = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] != "assistant":
                continue

            token_info = message["info"].get("tokens")
            if not token_info:
                continue

            usage = NeMoGymResponseUsage(
                input_tokens=token_info["input"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=token_info["cache"]["read"]),
                output_tokens=token_info["output"],
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=token_info["reasoning"]),
                total_tokens=token_info.get("total", 0),  # Somehow total may be missing
            )
            usages.append(usage)

        return usages

    def _opencode_export_to_output_items(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        messages = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] == "user":
                message_parts = []
                for part in message["parts"]:
                    if part["type"] != "text":
                        continue

                    message_parts.append(ResponseInputTextParam(text=part["text"], type="input_text"))

                messages.append(NeMoGymEasyInputMessage(content=message_parts, role="user"))
            elif message["info"]["role"] == "assistant":
                from nemo_gym.responses_converter import ResponsesConverter

                converter = ResponsesConverter(return_token_id_information=True)
                for part in message["parts"]:
                    if part["type"] == "text":
                        output_items = converter.postprocess_assistant_message_dict(
                            message_dict={
                                "content": part["text"],
                                "role": "assistant",
                            }
                        )
                        messages.extend(output_items)
                    elif part["type"] == "tool":
                        messages.append(
                            NeMoGymResponseFunctionToolCall(
                                arguments=json.dumps(part["state"]["input"]),
                                call_id=part["callID"],
                                name=part["tool"],
                            )
                        )
                        messages.append(
                            NeMoGymFunctionCallOutput(
                                call_id=part["callID"],
                                # @bxyu-nvidia: Somehow the output here may be missing...
                                output=part["state"].get("output", ""),
                            )
                        )
                    elif part["type"] in ("step-finish", "step-start", "patch"):
                        pass
                    else:
                        # @bxyu-nvidia: Defensive raise in case we're missing something.
                        raise NotImplementedError(part)
            else:
                # @bxyu-nvidia: Defensive raise in case we're missing something.
                raise NotImplementedError(message)

        return messages

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]

        query = None
        # This can be modified to handle system/developer prompts too.
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]

        assert query, body.input

        opencode_debug_str = ""
        if self.config.debug:
            opencode_debug_str = "--print-logs --log-level DEBUG"

        # TODO @bxyu-nvidia: We need to manually activate the conda env here for SWE Verified
        # Eventually this will only be present on the SWE Bench resources server side
        # For now, the activation is put on the harness side.
        conda_activate_command_str = "{ source /opt/miniconda3/bin/activate && conda activate testbed || true; }"

        opencode_thinking_str = "--thinking"

        # --auto is to approve not explicitly denied requests.
        command = f"""
        echo "Shell: $SHELL" \
        && {conda_activate_command_str} \
        && curl -fsSL https://opencode.ai/install | VERSION={self.config.opencode_version} bash \
        && export PATH=$HOME/.opencode/bin:$PATH \
        && opencode run {opencode_debug_str} {opencode_thinking_str} {quote(query)}
        """

        opencode_config_content = json.dumps(self._create_opencode_config())

        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\n", file=sys.stderr)
            print(f"OpenCode config JSON str: {opencode_config_content}", file=sys.stderr)
        result = await sandbox.exec(
            command=command,
            timeout_s=self.config.sandbox_timeout,
            env={"OPENCODE_CONFIG_CONTENT": opencode_config_content},
        )
        if self.config.debug:
            print("OpenCode install and run stdout:\n", result.stdout, file=sys.stderr)
            print("OpenCode install and run stderr:\n", result.stderr, file=sys.stderr)

        export_fname = "export.json"
        export_result = await sandbox.exec(
            command=f"""export PATH=$HOME/.opencode/bin:$PATH \
        && session_id=$(opencode session list --format json | jq -r '.[0].id') \
        && opencode export $session_id > {export_fname}"""
        )
        if self.config.debug:
            print("Export stdout:\n", export_result.stdout, file=sys.stderr)
            print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        pwd_result = await sandbox.exec(command="pwd")
        results_remote_fpath = Path(pwd_result.stdout) / export_fname

        results_dir: Path = Path(__file__).parent / "results" / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        results_local_fpath = results_dir / export_fname
        if self.config.debug:
            print(f"Downloading results from {results_remote_fpath} to {results_local_fpath}", file=sys.stderr)
        try:
            await sandbox.download(str(results_remote_fpath), results_local_fpath)
        except:
            print(f"Failed to download export results to {results_local_fpath}", file=sys.stderr)

        opencode_export = dict()
        if results_local_fpath.exists():
            opencode_export = json.loads(results_local_fpath.read_text().strip() or "{}")

        output = []
        usage = None
        if opencode_export:
            # Assume only one input message. May change with a system/developer message later on.
            output = self._opencode_export_to_output_items(opencode_export)[1:]
            usage = NeMoGymResponseUsage.sum_from_list(self._opencode_export_to_usages(opencode_export))

        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = {
            "opencode_results_fpath": str(results_local_fpath),
            "opencode_run_stdout": result.stdout or "",
            "opencode_run_stderr": result.stderr or "",
            "opencode_no_export_found": not bool(opencode_export),
        }

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def run(
        self, request: Request, body: OpenCodeSandboxedAgentRunRequest
    ) -> OpenCodeSandboxedAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies

        # @bxyu-nvidia: "sandbox_handle" comes from resources_servers/swebench/app.py
        # Once we graduate to use the sandbox server, this will be in a generic seed_session type that can be model validated.
        seed_session_result = await seed_session_response.json()
        sandbox = await self._start_sandbox(sandbox_id=seed_session_result.get("sandbox_handle"))
        self._sandbox_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox

        # Propagating the sandbox handle
        cookies["sandbox_id"] = request.session[SESSION_ID_KEY]

        # Stop the sandbox even when the trial fails; a raised exec error
        # otherwise leaks it until its multi-hour TTL, and a crashed run
        # leaks one per in-flight trial.
        try:
            response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(response)
            cookies = cookies | response.cookies

            verify_request = OpenCodeSandboxedAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": await get_response_json(response)}
            )

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
        finally:
            # TODO @bxyu-nvidia: Check if sandbox stop is idempotent
            try:
                await sandbox.stop()
            except Exception as e:
                print(f"Failed to stop sandbox after run: {e!r}", file=sys.stderr)
            self._sandbox_id_to_sandbox.pop(request.session[SESSION_ID_KEY], None)

        # @bxyu-nvidia: This is scraped from the raw create params. Later on we can dynamically set this if OpenCode exports this :rofl:
        opencode_system_prompt = "You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\n\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\n\nIf the user asks for help or wants to give feedback inform them of the following:\n- /help: Get help with using opencode\n- To give feedback, users should report the issue at https://github.com/anomalyco/opencode/issues\n\nWhen the user directly asks about opencode (eg 'can opencode do...', 'does opencode have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from opencode docs at https://opencode.ai\n\n# Tone and style\nYou should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).\nRemember that your output will be displayed on a command line interface. Your responses can use GitHub-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.\nIf you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.\nOnly use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.\nIMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\nIMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as \"The answer is <answer>.\", \"Here is the content of the file...\" or \"Based on the information provided, the answer is...\" or \"Here is what I will do next...\". Here are some examples to demonstrate appropriate verbosity:\n<example>\nuser: what is 2+2?\nassistant: 4\n</example>\n\n<example>\nuser: is 11 a prime number?\nassistant: Yes\n</example>\n\n<example>\nuser: what command should I run to list files in the current directory?\nassistant: ls\n</example>\n\n<example>\nuser: what command should I run to watch files in the current directory?\nassistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]\nnpm run dev\n</example>\n\n<example>\nuser: what files are in the directory src/?\nassistant: [runs ls and sees foo.c, bar.c, baz.c]\nuser: which file contains the implementation of foo?\nassistant: src/foo.c\n</example>\n\n<example>\nuser: write tests for new feature\nassistant: [uses grep and glob search tools to find where similar tests are defined, uses concurrent read file tool use blocks in one tool call to read relevant files at the same time, uses edit file tool to write new tests]\n</example>\n\n# Proactiveness\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\n1. Doing the right thing when asked, including taking actions and follow-up actions\n2. Not surprising the user with actions you take without asking\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\n3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\n\n# Following conventions\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\n\n# Code style\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\n\n# Doing tasks\nThe user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:\n- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.\n- Implement the solution using all tools available to you\n- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.\n- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (e.g. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to AGENTS.md so that you will know to run it next time.\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\n\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\n\n# Tool usage policy\n- When doing file search, prefer to use the Task tool in order to reduce context usage.\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run \"git status\" and \"git diff\", send a single message with two tool calls to run the calls in parallel.\n\nYou MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.\n\nIMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames directory structure.\n\n# Code References\n\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\n\n<example>\nuser: Where are errors from the client handled?\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\nYou are powered by the model named dummy_model. The exact model ID is nemo_gym/dummy_model\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: /testbed\n  Workspace root folder: /testbed\n  Is directory a git repo: yes\n  Platform: linux\n  Today's date: Tue Aug 04 2026\n</env>\nSkills provide specialized instructions and workflows for specific tasks.\nUse the skill tool to load a skill when a task matches its description.\n<available_skills>\n  <skill>\n    <name>customize-opencode</name>\n    <description>Use ONLY when the user is editing or creating opencode's own configuration: opencode.json, opencode.jsonc, files under .opencode/, or files under ~/.config/opencode/. Also use when creating or fixing opencode agents, subagents, skills, plugins, MCP servers, or permission rules. Do not use for the user's own application code, or for any project that is not configuring opencode itself.</description>\n    <location>file:///testbed/%3Cbuilt-in%3E</location>\n  </skill>\n</available_skills>"

        response_dict = await get_response_json(verify_response)
        response_dict |= self._sandbox_id_to_run_result[cookies["sandbox_id"]]
        response_dict["responses_create_params"]["input"].insert(
            0, {"content": opencode_system_prompt, "role": "system"}
        )
        return OpenCodeSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    OpenCodeSandboxedAgent.run_webserver()

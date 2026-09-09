# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, Request
from pydantic import ConfigDict, Field, SecretStr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, raise_for_status


JUDGE_PROMPT = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

* **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".

* **Process:**

* Identify the "Prompt Type": "<prompt_type>".

* Refer to the "Correct Answer": "<answer>".
* Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).

* **’Single Answer’**: Check if the response provides the answer that addresses the user’s question. It does not have to match the exact wording of the provided answer.
* **’Set Answer’**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.

* **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
* **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.

* For ’Set Answer’, this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
* **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there’s no excessive answers in the response.

**Output Format:**
Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: ‘"Answer Correctness"‘. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for ‘"Answer Correctness"‘ should be a dictionary containing ‘"Explanation"‘ (a string), ‘"Correctness Details"‘ (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and ‘"Excessive Answers"‘ (a list of strings indicating the excessive answers).
Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.

Grader Partial Output Example

**Example (Partial):**

```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": ["Italy"]
  }}
}}
```

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):

<prompt>
{problem}
</prompt>

--------------------

** Correct Answer (Wrapped in <answer> and </answer>):

Prompt Type: {answer_type}

<answer>
{answer}
</answer>

--------------------

AI assistant response (Wrapped in <response> and </response>):

<response>
{response}
</response>
--------------------

Rating:
"""


class DeepSearchQAConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    harness_module: str
    harness_class: str
    harness_config_class: str
    harness_kwargs: dict[str, Any] = Field(default_factory=dict)
    image: str
    python: str = "python3"
    setup_command: str | None = None
    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    sandbox_model_base_url: str | None = None
    exa_api_key: SecretStr | None = None


class DeepSearchQARunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    example_id: str
    problem: str
    answer: str
    answer_type: str
    problem_category: str


class DeepSearchQAResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    precision: float
    recall: float
    f1: float
    fully_correct: float
    fully_incorrect: float
    correct_with_extraneous: float
    judge_output: dict[str, Any]


def response_text(response: NeMoGymResponse) -> str:
    texts = []
    for output in response.output:
        if output.type == "message" and output.role == "assistant":
            texts.extend(part.text for part in output.content if getattr(part, "text", None))
    return "\n".join(texts).strip()


def parse_judge(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].removesuffix("```").strip()
    result = json.loads(text)["Answer Correctness"]
    details = result["Correctness Details"]
    excessive = result.get("Excessive Answers", [])
    if (
        not isinstance(details, dict)
        or not details
        or not all(isinstance(k, str) and isinstance(v, bool) for k, v in details.items())
    ):
        raise ValueError("judge Correctness Details must be a non-empty string-to-boolean object")
    if not isinstance(excessive, list) or not all(isinstance(item, str) for item in excessive):
        raise ValueError("judge Excessive Answers must be a list of strings")
    return result


class DeepSearchQAAgent(SimpleResponsesAPIAgent):
    config: DeepSearchQAConfig

    def model_post_init(self, context: Any) -> None:
        model = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        self._model_url = self.server_client._build_server_base_url(model)
        self._provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        self._metadata = resolve_provider_metadata(self.config.sandbox_provider, self.server_client.global_config_dict)
        super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        root = f"/tmp/nemo-gym-deepsearchqa-{uuid.uuid4().hex}"
        input_path, output_path = f"{root}/input.json", f"{root}/response.json"
        runner_path, config_path = f"{root}/agent_runner.py", f"{root}/runner.json"
        values = dict(self.config.sandbox_spec)
        spec = SandboxSpec(
            image=self.config.image.removeprefix("docker://"),
            ttl_s=values.pop("ttl_s", None),
            ready_timeout_s=values.pop("ready_timeout_s", 1200),
            workdir=values.pop("workdir", root),
            env=values.pop("env", {}),
            metadata={**self._metadata, **values.pop("metadata", {}), "nemo_gym_agent": "deepsearchqa"},
            resources=SandboxResources.from_mapping(values.pop("resources", {})),
            entrypoint=values.pop("entrypoint", None),
            provider_options=values.pop("provider_options", {}),
        )
        if values:
            raise ValueError(f"unknown sandbox_spec keys: {sorted(values)}")
        runner_config = {
            "harness_module": self.config.harness_module,
            "harness_class": self.config.harness_class,
            "harness_config_class": self.config.harness_config_class,
            "harness_kwargs": self.config.harness_kwargs,
            "model_url": (self.config.sandbox_model_base_url or self._model_url).rstrip("/")
            + self.url_path_for_request("", request).rstrip("/"),
            "input_path": input_path,
            "output_path": output_path,
            "exa_api_key": self.config.exa_api_key.get_secret_value() if self.config.exa_api_key else None,
        }
        sandbox = AsyncSandbox(self._provider, spec)
        try:
            await sandbox.start()
            with tempfile.TemporaryDirectory() as temporary:
                local = Path(temporary)
                (local / "input.json").write_text(body.model_dump_json())
                (local / "runner.json").write_text(json.dumps(runner_config))
                await sandbox.upload(Path(__file__).with_name("agent_runner.py"), runner_path)
                await sandbox.upload(local / "input.json", input_path)
                await sandbox.upload(local / "runner.json", config_path)
                command = f"{shlex.quote(self.config.python)} {runner_path} {config_path}"
                if self.config.setup_command:
                    command = f"{self.config.setup_command} && {command}"
                result = await sandbox.exec(command, timeout_s=None)
                if result.return_code != 0:
                    raise RuntimeError(f"sandboxed harness failed: {(result.stderr or result.stdout or '')[-2000:]}")
                await sandbox.download(output_path, local / "response.json")
                return NeMoGymResponse.model_validate_json((local / "response.json").read_text())
        finally:
            await sandbox.stop()

    async def verify(self, body: DeepSearchQARunRequest, response: NeMoGymResponse) -> DeepSearchQAResponse:
        prompt = JUDGE_PROMPT.format(
            problem=body.problem, answer_type=body.answer_type, answer=body.answer, response=response_text(response)
        )
        params = self.config.judge_responses_create_params.model_copy(
            update={"input": [NeMoGymEasyInputMessage(role="user", content=prompt)]}, deep=True
        )
        judged = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        judge_output = parse_judge(response_text(judged))
        matched = sum(judge_output["Correctness Details"].values())
        expected = len(judge_output["Correctness Details"])
        excessive = len(judge_output.get("Excessive Answers", []))
        precision = matched / (matched + excessive) if matched + excessive else 0.0
        recall = matched / expected
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return DeepSearchQAResponse(
            **body.model_dump(),
            response=response,
            reward=f1,
            precision=precision,
            recall=recall,
            f1=f1,
            fully_correct=float(matched == expected and excessive == 0),
            fully_incorrect=float(matched == 0),
            correct_with_extraneous=float(matched == expected and excessive > 0),
            judge_output=judge_output,
        )

    async def run(self, body: DeepSearchQARunRequest = Body()) -> DeepSearchQAResponse:
        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
        )
        await raise_for_status(response)
        return await self.verify(body, NeMoGymResponse.model_validate(await get_response_json(response)))


if __name__ == "__main__":
    DeepSearchQAAgent.run_webserver()

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

"""SciCode multi-step generation agent.

SciCode problems are solved one sub-step at a time: for each sub-step the agent builds a prompt
from the problem description, the code it generated for previous sub-steps, and the current
function header, calls the model, extracts the Python code, and accumulates it. After all
sub-steps it sends the accumulated per-step solutions to the resources server's /verify endpoint.

The full step loop — building each step's prompt, extracting the Python code block,
prefilled-steps handling, context-window-exhaustion handling, and the accumulation/verify
call — is not yet implemented (see run()).

Templated on responses_api_agents/proof_refinement_agent (the multi-turn run() skeleton).
"""

import logging
import statistics
from typing import Any, Dict, List

from fastapi import Request, Response
from pydantic import ConfigDict
from step_utils import (
    OUT_OF_CONTEXT,
    PREFILLED_STEPS_CODE,
    extract_python_script,
    is_context_window_error,
    process_problem_steps,
)

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.prompt import PromptConfig, load_prompt_config
from nemo_gym.server_utils import raise_for_status


LOG = logging.getLogger(__name__)


class ScicodeAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for the SciCode multi-step agent."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Per-sub-step user prompt template (PromptConfig YAML) the agent fills each step.
    prompt_fpath: str
    # Inject each sub-step's scientific background into the prompt context.
    with_background: bool = True


class ScicodeAgentRunRequest(BaseRunRequest):
    # extra="allow" so passthrough fields (e.g. uuid) survive to the resources server.
    model_config = ConfigDict(extra="allow")
    problem_id: str
    sub_steps: List[dict]
    required_dependencies: str


def _empty_response() -> dict:
    """Minimal NeMoGymResponse for the degenerate case where no sub-step was generated."""
    return NeMoGymResponse(
        id="scicode",
        created_at=0.0,
        model="",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump()


def _across_run_stats(tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Across-run (repeat-to-repeat) variability of the two headline metrics.

    Run i is "take repeat i of every problem" (repeats aligned by their rollout
    index, run count = minimum repeat count so the matrix stays rectangular).
    Per run, problem accuracy is the mean of problem_accuracy over problems and
    subtask accuracy is the sub-step-weighted pool — the same definitions as
    the headline mean/problem_accuracy and subtask_accuracy, so the std-dev
    over the per-run values (sample std, ddof=1) is the run-to-run variability
    of exactly those metrics. Single-repeat collections emit nothing.
    """
    max_k = min((len(t) for t in tasks if t), default=0)
    if max_k < 2:
        return {}

    rows = [sorted(t, key=lambda r: r.get(ROLLOUT_INDEX_KEY_NAME, 0))[:max_k] for t in tasks if len(t) >= max_k]

    problem_runs: List[float] = []
    subtask_runs: List[float] = []
    for i in range(max_k):
        acc = [float(r[i]["problem_accuracy"]) for r in rows if r[i].get("problem_accuracy") is not None]
        if acc:
            problem_runs.append(sum(acc) / len(acc))
        passed = sum(r[i].get("num_steps_passed", 0) for r in rows)
        total = sum(r[i].get("num_steps_total", 0) for r in rows)
        if total:
            subtask_runs.append(passed / total)

    metrics: Dict[str, Any] = {}
    for key, runs in (("mean/problem_accuracy", problem_runs), ("subtask_accuracy", subtask_runs)):
        if len(runs) < 2:
            continue
        std_dev = 0.0 if all(v == runs[0] for v in runs) else statistics.stdev(runs)
        metrics[f"{key}/std_dev_across_runs"] = std_dev
    return metrics


class ScicodeAgent(SimpleResponsesAPIAgent):
    """Agent that drives the SciCode per-sub-step generation + code-accumulation loop."""

    config: ScicodeAgentConfig

    def model_post_init(self, context):
        self._prompt: PromptConfig = load_prompt_config(self.config.prompt_fpath)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        model_response_json = await model_response.json()

        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def run(self, request: Request, body: ScicodeAgentRunRequest):
        """Generate code for each sub-step (accumulating prior code as context), then verify."""
        cookies = request.cookies
        sub_steps = body.sub_steps
        total = len(sub_steps)
        previous_llm_code = [None] * total
        solutions: Dict[str, str] = {}
        out_of_context = False
        last_response_json = None

        for cur_step in range(total):
            # Prefilled steps provide context for later steps but are not scored (no solution entry).
            if (body.problem_id, cur_step) in PREFILLED_STEPS_CODE:
                previous_llm_code[cur_step] = PREFILLED_STEPS_CODE[(body.problem_id, cur_step)]
                continue
            if out_of_context:
                solutions[f"{body.problem_id}.{cur_step + 1}"] = OUT_OF_CONTEXT
                continue

            problem_steps_str, next_step_str, previous_code_str = process_problem_steps(
                sub_steps, cur_step, previous_llm_code, self.config.with_background
            )
            dependencies = body.required_dependencies
            previous_code = f"{dependencies}\n{previous_code_str}\n" if previous_code_str else f"{dependencies}\n"
            user_content = self._prompt.user.format(
                problem_steps_str=problem_steps_str, next_step_str=next_step_str, dependencies=dependencies
            )

            try:
                gen_response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path="/v1/responses",
                    json={"input": [{"role": "user", "content": user_content}]},
                    cookies=cookies,
                )
                await raise_for_status(gen_response)
            except Exception as error:
                if is_context_window_error(error):
                    LOG.warning("SciCode step %s: context window exceeded; failing remaining steps.", cur_step)
                    out_of_context = True
                    solutions[f"{body.problem_id}.{cur_step + 1}"] = OUT_OF_CONTEXT
                    continue
                raise

            cookies = gen_response.cookies
            last_response_json = await gen_response.json()
            generation = NeMoGymResponse.model_validate(last_response_json).output_text
            extracted = extract_python_script(generation)
            previous_llm_code[cur_step] = extracted
            solutions[f"{body.problem_id}.{cur_step + 1}"] = f"{previous_code}\n{extracted}"

        verify_request_data = body.model_dump()
        verify_request_data["solutions"] = solutions
        # /verify requires a response; record the last sub-step's generation (empty if none ran).
        verify_request_data["response"] = last_response_json if last_response_json is not None else _empty_response()
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_data,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return await verify_response.json()

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Headline SciCode metric: sub-step-weighted accuracy = total passed / total over all rollouts."""
        passed = sum(r.get("num_steps_passed", 0) for task in tasks for r in task)
        total = sum(r.get("num_steps_total", 0) for task in tasks for r in task)
        metrics = {"subtask_accuracy": passed / total if total else 0.0}
        metrics.update(_across_run_stats(tasks))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in agent_metrics.items() if k.startswith("mean/") or k.startswith("subtask_accuracy")}


if __name__ == "__main__":
    ScicodeAgent.run_webserver()

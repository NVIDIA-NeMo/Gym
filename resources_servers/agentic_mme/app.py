# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic-MME task-only scoring; process judgments are deliberately not rewards.

Two settings decide what a harness gets, and each can be on or off:

- With ``sandbox_provider`` set, each episode's seed lends the harness a sandbox. ``task_images_in_sandbox`` decides
  whether it holds the task images, for a harness that inspects them with its own tools, or nothing.
- With ``serve_image_tools`` set, the server serves the 14 image tools: each episode's seed loads the task images
  into a workspace on this server, and each tool route returns its new image under ``tool_output_images``.
"""

import asyncio
import json
import math
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, ClassVar, Literal, Optional

import cv2
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BeforeValidator, ConfigDict, Field, PrivateAttr, TypeAdapter, ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.agents import HARNESS_STEP_LIMIT_METADATA_KEY
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import TOOL_OUTPUT_IMAGES_KEY, VERIFIER_ERROR, EpisodeId, NeMoGymChatCompletion, TaskId
from nemo_gym.sandbox import SandboxConfig
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    ResourcesSessionLostError,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
    observe_llm_judge_grading,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError, VerifierFixture, call_judge, record_parsed_judge_verdict
from resources_servers.agentic_mme.image_tools import IMAGE_TOOLS, ImageArgs, ImageWorkspace
from resources_servers.agentic_mme.task_data import GoldenAnswer, ImageFile
from resources_servers.agentic_mme.verifier_fixture import create_server


# Where a lent sandbox holds the task images; the OpenCode prompt names files relative to it.
WORKDIR = "/workspace"


def judge_tolerance(metadata: dict[str, Any], golden: GoldenAnswer) -> float:
    """The numeric tolerance shown to the judge: the task's answer_tolerance (the author's final answer
    check), else a numeric golden answer's own tolerance."""
    tolerance = metadata.get("answer_tolerance")
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and math.isfinite(tolerance):
        return max(float(tolerance), 0.0)
    return golden.tolerance if golden.match_type == "numeric" else 0.0


def extract_answer(text: str) -> str:
    """Extract the public atomic answer format, excluding private reasoning.

    With several <answer> tags the last one wins: a model that quotes the format before answering
    (e.g. <answer>YOUR_SHORT_ANSWER</answer>) is scored on the answer it gives afterwards.
    """
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.S | re.I)
    text = re.split(r"</(?:think|thinking)>", text, flags=re.I)[-1]
    text = re.split(r"<(?:think|thinking)>", text, flags=re.I)[0]
    answers = re.findall(r"<answer>(.*?)(?:</answer>|$)", text, re.S | re.I)
    return (answers[-1] if answers else text).strip()


def matches_answer(answer: str, golden: GoldenAnswer) -> bool:
    """Match the released evaluator's exact/contains/numeric task semantics.

    The paper says normalized exact match, but the released dataset/evaluator
    default to contains. Do not silently substitute Gym's soft numeric grader.
    """
    target = str(golden.value).strip()
    if not answer:
        return False
    if golden.match_type == "exact":
        return answer == target
    if golden.match_type == "numeric":
        try:
            actual, expected = float(answer), float(target)
        except ValueError:
            return False
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= golden.tolerance
    numeric = target.replace(".", "").replace("-", "").replace("+", "").isdigit()
    if numeric and len(target) <= 2:
        return re.search(r"\b" + re.escape(target) + r"\b", answer) is not None
    return target.lower() in answer.lower()


JUDGE_SYSTEM = """You are a strict grader. You decide ONE thing: whether the model's FINAL \
answer matches the reference answer.

Judge only the final answer the model commits to (normally inside <answer>...</answer>), not its \
reasoning. Differences that do NOT matter: units, casing, whitespace, LaTeX wrappers, \
spelled-out vs numeric digits, ordering where the question does not ask for an order \
(e.g. groups of matching items), trailing punctuation, extra prose around the answer.

These make it WRONG: a different number (outside the stated tolerance, when one is given), \
a different option letter, a different coordinate, a different colour, a different count, \
a different set of items, or several conflicting final answers.

You are NOT solving the problem and you cannot see any image. If the model gives no final \
answer, or you cannot tell what it asserts, answer "unsure".

Reply with ONLY a JSON object:
{"verdict": "equivalent" | "different" | "unsure", "confidence": 0.0-1.0, "reason": "<12 words"}
("equivalent" = the final answer is correct.)"""

JUDGE_USER = """Question:
{question}

Reference answer (gold):
{expected}
{tolerance}
What the extractor pulled out of the model's response:
{extracted}

The model's final response text (may be truncated):
<<<
{text}
>>>

Is the model's final answer correct?"""


def parse_verdict(content: str) -> dict[str, Any]:
    """Read the judge's JSON verdict. A reply without one is a judge failure, not a wrong answer.

    Every JSON object in the reply that has a verdict counts, so a brace ahead of it (a quoted \\boxed{7}) is skipped.
    Objects that name different verdicts, as when a reasoning judge quotes one from the model's text, stay a judge
    failure.
    """
    text = content or ""
    decoder = json.JSONDecoder()
    verdicts: list[dict[str, Any]] = []
    end = 0
    for brace in re.finditer(r"\{", text):
        if brace.start() < end:
            continue
        try:
            data, end = decoder.raw_decode(text, brace.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "verdict" in data:
            verdicts.append(data)
    if len({str(data["verdict"]).lower().strip() for data in verdicts}) != 1:
        raise JudgeError(f"The judge reply has no single verdict: {text[:200]!r}")
    data = verdicts[-1]
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("equivalent", "different", "unsure"):
        raise JudgeError(f"The judge reply has no verdict: {(content or '')[:200]!r}")
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:200]}


def question_text(params: Any) -> str:
    """Text of the last user message; images are not sent to the judge."""
    items = params.input if not isinstance(params.input, str) else [{"role": "user", "content": params.input}]
    for item in reversed(list(items)):
        data = item if isinstance(item, dict) else item.model_dump()
        if data.get("role") != "user":
            continue
        content = data.get("content")
        if isinstance(content, str):
            return content
        return "\n".join(part.get("text", "") for part in content or [] if part.get("type") == "input_text")
    return ""


class AgenticMMESandboxConfig(SandboxConfig):
    # An OCI image built from sandbox_image/Dockerfile: Python 3 with OpenCV, NumPy and Pillow.
    image: Optional[str] = None


_IMAGE_FILES = TypeAdapter(list[ImageFile])


class AgenticMMEConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # With sandbox_provider set, each episode gets a sandbox that is lent to the harness. Unset keeps the server
    # stateless, as the agent's own image tools need.
    sandbox_provider: Optional[str] = None
    sandbox_config: AgenticMMESandboxConfig = Field(default_factory=AgenticMMESandboxConfig)
    # Egress of the lent sandbox. model_only reaches only sandbox_model_server's host; model_and_tools also reaches this
    # server, for a harness that calls the image tools over MCP. inherit keeps the provider's network.
    sandbox_network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    sandbox_model_server: Optional[ModelServerRef] = None
    # Whether the lent sandbox holds the task images. Turn it off with serve_image_tools to lend an empty sandbox, so
    # that the harness reaches the images only through this server's tools.
    task_images_in_sandbox: bool = True
    # With serve_image_tools set, the 14 image tools are routes on this server (Pattern B) and each episode's seed
    # loads the task images into a workspace here. Unset leaves the tools to the Agentic-MME agent.
    serve_image_tools: bool = False
    max_image_pixels: int = Field(default=16_000_000, ge=1)
    max_total_image_pixels: int = Field(default=64_000_000, ge=1)
    # The folder on this host that holds the files a row's image_files names. A relative path is relative to the Gym
    # checkout, not to the working directory of this server's process.
    image_root: Optional[str] = None
    # When set, an LLM judge decides correctness and string match is kept only as a diagnostic.
    # The judge's model server picks the judge model (openai_model's openai_model), so this server names none.
    judge_model_server: Optional[ModelServerRef] = None
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024
    # Per worker: with num_workers > 1, each worker allows this many judge calls at once.
    judge_max_concurrency: int = Field(default=32, ge=1)
    judge_max_text_chars: int = Field(default=8000, ge=1)
    # Whether an answer given at or after a harness's own step limit counts. The harness marks such a response
    # (harness_step_limit_reached, set by opencode_agent when the prompt made agent.build.steps model calls). False
    # scores it 0, the way the Agentic-MME agent scores an episode that runs out of tool rounds; true grades the
    # answer like any other. A policy trained with the answer at the limit scored learns to use every step.
    score_step_limit_answer: bool = False


class AgenticMMEVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticMMEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    # False when no grade was possible (no usable gold answer, or a session this server already closed). Such a row
    # is also masked, so it never counts as a wrong answer. A judge failure never reaches this model: the base class's
    # judge_failsafe masks it and sends the row to the failures sidecar.
    evaluation_completed: bool = True
    extracted_answer: str = ""
    failure_reason: str | None = None
    evaluation_track: str = "task_only"
    string_match_reward: float | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None
    # 1.0 when the harness marked the response as reaching its step limit, so the share of such rollouts shows in
    # the per-key metrics whether or not the answer is scored.
    step_limit_reached: float = 0.0


@dataclass
class _Episode:
    episode_id: EpisodeId
    task_id: TaskId
    workspace: ImageWorkspace | None
    # Tool calls of one episode change its workspace one at a time.
    tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class AgenticMMEServer(SimpleResourcesServer):
    config: AgenticMMEConfig
    _judge_slots: asyncio.Semaphore | None = None
    _sessions: dict[str, _Episode] = PrivateAttr(default_factory=dict)
    _locks: ResourcesSessionLocks = PrivateAttr()
    _fences: ResourcesSessionFences[EpisodeId] = PrivateAttr()
    _lent_sandboxes: LentSandboxes | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._locks = ResourcesSessionLocks()
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        if self.config.sandbox_provider is not None:
            if not self.config.sandbox_config.image:
                raise ValueError(
                    "Set sandbox_config.image to an image built from resources_servers/agentic_mme/sandbox_image"
                )
            if self.config.sandbox_network_access == "model_and_tools" and not self.config.expose_tools_over_mcp:
                raise ValueError("sandbox_network_access model_and_tools needs expose_tools_over_mcp: true")
            model_server = self.config.sandbox_model_server
            self._lent_sandboxes = LentSandboxes(
                self.config.sandbox_provider,
                self.config.sandbox_config,
                image=self.config.sandbox_config.image,
                workdir=WORKDIR,
                network_access=self.config.sandbox_network_access,
                model_server=model_server.name if model_server is not None else None,
                tool_servers=[self.config.name] if self.config.sandbox_network_access == "model_and_tools" else (),
            )
        elif self.config.sandbox_network_access != "inherit":
            raise ValueError("sandbox_network_access applies to a lent sandbox; set sandbox_provider")
        elif not self.config.task_images_in_sandbox:
            raise ValueError("task_images_in_sandbox: false needs sandbox_provider, which lends the sandbox")
        if self._loads_task_images and self.config.image_root is None:
            raise ValueError("Set image_root to the folder that benchmarks/agentic_mme/prepare.py wrote")

    @property
    def _loads_task_images(self) -> bool:
        """Whether a seed reads the task images: into the lent sandbox, the tool workspace, or both."""
        sandbox_images = self.config.sandbox_provider is not None and self.config.task_images_in_sandbox
        return sandbox_images or self.config.serve_image_tools

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        if self.config.serve_image_tools:
            for name, (args_model, description) in IMAGE_TOOLS.items():
                app.post(f"/{name}", description=description)(self._image_tool_route(name, args_model))
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Load the task images into the episode's workspace, lend it a sandbox with or without them, or both.

        A malformed typed seed is refused (422) instead of being taken as a cookie seed. A server that keeps
        per-episode state (a lent sandbox or the image tools' workspace) refuses a cookie seed too: it would get
        neither, and an agent's own /run, which seeds that way, would run its harness on its own host.
        """
        if self.config.sandbox_provider is None and not self.config.serve_image_tools:
            return await super().seed_session(body)
        if not isinstance(body, ResourcesSeedSessionRequest):
            raise ResourcesSeedRejectedError(
                "This server lends a sandbox or serves image tools only to an Environment Server's episode seed"
            )
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            session = self._sessions.get(session_id)
            if session is not None:
                if (session.episode_id, session.task_id) != (body.episode_id, body.task_id):
                    raise ResourcesSessionConflictError(
                        "resources_session_id is already bound to another episode or task"
                    )
                request.session[SESSION_ID_KEY] = session_id
                return await self._seed_response(session_id, body.task_id)
            images = await asyncio.to_thread(self._task_images, body.task_data) if self._loads_task_images else []
            workspace = await asyncio.to_thread(self._workspace, images) if self.config.serve_image_tools else None
            files = (
                {str(PurePosixPath(WORKDIR) / target): data for target, data in images}
                if self.config.task_images_in_sandbox
                else {}
            )
            response = await self._seed_response(session_id, body.task_id, files=files)
            self._sessions[session_id] = _Episode(body.episode_id, body.task_id, workspace)
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop the episode's sandbox and drop its workspace. Safe to repeat, and for a session never seeded."""
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            closed_episode = self._fences.get(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            session = self._sessions.get(session_id)
            if session is not None and session.episode_id != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            # Release by resources_session_id, which keys the lent sandboxes, not by the episode record. A failed stop
            # raises before the fence, so the caller can close again.
            if self._lent_sandboxes is not None:
                await self._lent_sandboxes.release(session_id)
            self._sessions.pop(session_id, None)
            self._fences.fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    def _task_images(self, task_data: Mapping[str, Any]) -> list[tuple[PurePosixPath, bytes]]:
        """Read the task images that task_data.image_files names, in prompt order, with their sandbox targets."""
        try:
            image_files = _IMAGE_FILES.validate_python(task_data.get("image_files"))
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"task_data.image_files is missing or malformed: {error}") from error
        if not image_files:
            raise ResourcesSeedRejectedError("task_data.image_files is empty; the episode needs the task images")
        root = (PARENT_DIR / self.config.image_root).resolve()
        images = []
        for image in image_files:
            source = (root / image.source).resolve()
            target = PurePosixPath(image.target)
            if not source.is_relative_to(root):
                raise ResourcesSeedRejectedError(f"image file {image.source!r} is outside image_root")
            if target.is_absolute() or ".." in target.parts:
                raise ResourcesSeedRejectedError(f"image target {image.target!r} must stay in the working directory")
            if not source.is_file():
                raise ResourcesSeedRejectedError(f"image file {image.source!r} is not under image_root")
            images.append((target, source.read_bytes()))
        return images

    def _workspace(self, images: list[tuple[PurePosixPath, bytes]]) -> ImageWorkspace:
        """Load the task images as images 0, 1, ... in prompt order, as the Agentic-MME agent does."""
        workspace = ImageWorkspace(
            max_pixels=self.config.max_image_pixels, max_total_pixels=self.config.max_total_image_pixels
        )
        for target, data in images:
            try:
                with Image.open(BytesIO(data)) as image:
                    workspace.add(image)
            except (OSError, ValueError) as error:
                raise ResourcesSeedRejectedError(f"task image {str(target)!r} cannot be loaded: {error}") from error
        return workspace

    def _image_tool_route(self, name: str, args_model: type[ImageArgs]):
        async def image_tool(request: Request, body: args_model) -> dict[str, Any]:
            session_id = request.session.get(SESSION_ID_KEY, "")
            if self._fences.get(session_id) is not None:
                # A late call must not reach a workspace that close dropped.
                raise ResourcesSessionLostError(f"Resources session {session_id} is closed")
            session = self._sessions.get(session_id)
            if session is None or session.workspace is None:
                raise HTTPException(409, "No image workspace for this session; seed it with serve_image_tools set")
            async with session.tool_lock:
                try:
                    result = await asyncio.to_thread(session.workspace.apply, name, body.model_dump())
                except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
                    # The same model-visible error the Agentic-MME agent returns for a failed image tool.
                    return {"ok": False, "error": str(error)}
            image_url = result.pop("image_url")
            label = f"Image {result['new_image_index']}"
            return {**result, TOOL_OUTPUT_IMAGES_KEY: [{"image_url": image_url, "label": label}]}

        image_tool.__name__ = name
        return image_tool

    def mcp_tools(self, harvested: list[Any], catchall: Optional[Any]) -> Optional[list[Any]]:
        """Expose only the image tools over MCP."""
        return [tool for tool in harvested if tool.name in IMAGE_TOOLS]

    async def _seed_response(
        self, session_id: str, task_id: TaskId, *, files: dict[str, bytes] | None = None
    ) -> ResourcesSeedSessionResponse:
        """Lend the session's sandbox, started with ``files`` on the first seed, when the server lends one."""
        if self._lent_sandboxes is None:
            return ResourcesSeedSessionResponse(resources_session_id=session_id)
        sandbox_access = await self._lent_sandboxes.lend(session_id, task_id, files=files)
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=sandbox_access)

    async def shutdown(self) -> None:
        if self._lent_sandboxes is not None:
            await self._lent_sandboxes.release_all()
        self._sessions.clear()

    async def judge(
        self, question: str, expected: str, extracted: str, text: str, tolerance: float = 0.0
    ) -> dict[str, Any]:
        if self._judge_slots is None:
            self._judge_slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        prompt = JUDGE_USER.format(
            question=question,
            expected=expected,
            tolerance=f"Tolerance: a number within ±{tolerance:g} of the reference is correct.\n" if tolerance else "",
            extracted=extracted or "(nothing)",
            text=text[-self.config.judge_max_text_chars :],
        )
        params = {
            "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
        }
        async with self._judge_slots:
            completion = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=params,
                response_model=NeMoGymChatCompletion,
            )
        if not completion.choices:
            raise JudgeError("The judge reply has no choices")
        return parse_verdict(completion.choices[0].message.content or "")

    async def verify(self, request: Request, body: AgenticMMEVerifyRequest) -> AgenticMMEVerifyResponse:
        """Score the final answer, by string match or, when configured, by the judge.

        A judge failure propagates: the base class wraps this method in judge_failsafe, which masks the row and sends
        it to the failures sidecar instead of counting it as a wrong answer.
        """
        if self._fences.get(request.session.get(SESSION_ID_KEY, "")) is not None:
            # Masked although the score reads only the response, so that every server under the resources session
            # contract treats a closed session alike. single_agent_turn verifies before it closes, so only an
            # out-of-order verify lands here. A session that no close fenced, such as an agent's /run, is graded.
            lost = lost_session_verification("This agentic_mme session is already closed")
            return AgenticMMEVerifyResponse(**(body.model_dump() | lost | {"evaluation_completed": False}))
        result = AgenticMMEVerifyResponse(**body.model_dump(), reward=0.0)
        result.step_limit_reached = float(
            (body.response.metadata or {}).get(HARNESS_STEP_LIMIT_METADATA_KEY) == "true"
        )
        try:
            golden = GoldenAnswer.model_validate(body.verifier_metadata.get("golden_answer"))
        except ValidationError:
            # A task without a usable gold answer measures nothing about the policy.
            result.mask_sample = True
            result.evaluation_completed = False
            result.failure_kind = VERIFIER_ERROR
            result.failure_reason = "invalid_golden_answer"
            return result
        # Only the terminal turn may answer; never score an earlier tool-turn message.
        terminal = []
        for item in body.response.output:
            if item.type in ("function_call", "function_call_output"):
                terminal = []
            elif item.type == "message" and item.role == "assistant":
                terminal = [part.text for part in item.content if part.type == "output_text"]
        if body.response.incomplete_details:
            result.failure_reason = "incomplete_response"
            return result
        if result.step_limit_reached and not self.config.score_step_limit_answer:
            result.failure_reason = "step_limit_reached"
            return result
        final_text = "\n".join(terminal)
        result.extracted_answer = extract_answer(final_text)
        result.string_match_reward = float(matches_answer(result.extracted_answer, golden))
        result.reward = result.string_match_reward
        if self.config.judge_model_server is not None and result.extracted_answer:
            # A failed judge call raises JudgeError; the judge_failsafe wrapper around /verify turns it
            # into a masked judge_failed row in the failures sidecar, regraded later, not a wrong answer.
            return await observe_llm_judge_grading(lambda: self._grade_with_judge(body, golden, result, final_text))
        if not result.reward:
            result.failure_reason = "incorrect_answer" if result.extracted_answer else "missing_answer"
        return result

    async def _grade_with_judge(
        self, body: AgenticMMEVerifyRequest, golden: GoldenAnswer, result: AgenticMMEVerifyResponse, final_text: str
    ) -> AgenticMMEVerifyResponse:
        judged = await self.judge(
            question_text(body.responses_create_params),
            str(golden.value),
            result.extracted_answer,
            final_text,
            tolerance=judge_tolerance(body.verifier_metadata, golden),
        )
        # judge() raises on a reply without a valid verdict, so reaching here means the verdict parsed.
        record_parsed_judge_verdict()
        result.judge_verdict = judged["verdict"]
        result.judge_reason = judged["reason"]
        result.reward = float(judged["verdict"] == "equivalent")
        if not result.reward:
            result.failure_reason = "incorrect_answer"
        return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=AgenticMMEVerifyRequest,
    # A request without a session cookie, as from an agent's /run before any close.
    invoke=lambda server, body: server.verify(Request({"type": "http", "session": {}}), body),
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    AgenticMMEServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AgenticMMEServer.run_webserver()  # noqa: F401

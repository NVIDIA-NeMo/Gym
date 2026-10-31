# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VCQA agent.

A `responses_api_agent` for the Verified Code QA benchmark. Each task gives
the model a per-task working tree (a repo snapshot or a git bundle checked
out at a specific commit) plus a problem statement; the agent runs a tool
loop in a sandboxed subprocess, then grades the model's final answer
against a must-have rubric via an LLM judge.

Per-rollout pipeline (`/run`):

1. Validate the row's `verifier_metadata` (`dataset_kind`, repo info, rubric).
2. Allocate a per-rollout scratch dir and fetch+extract the task's working
   tree there (tarball to extract, or git bundle to clone+checkout).
3. Start the configured sandbox backend, exposing the working tree at
   `/codebase:ro` plus a writable `/tmp/scratch` for `write_todos`.
4. Run the model+tool loop (mirrors `simple_agent.responses` line-by-line so
   token-ID-bearing output items propagate verbatim across turns and
   `prompt_token_ids` / `generation_token_ids` / `generation_log_probs`
   never get dropped).
5. Pull the last assistant text and grade it against the rubric's must-have
   items via an LLM judge.
6. Stop the sandbox, clean scratch state, and return the verify response.

Any failure inside steps 2-5 is caught and surfaced as `reward=0.0` plus an
`error` field on the verify response; a single bad row never crashes the
agent server. Failures of the environment rather than the policy (artifact
fetch, sandbox start, model-server errors, judge outages) also set
`mask_sample=True`, `evaluation_completed=False` and a `failure_kind`, so a
trainer can drop them from the loss instead of learning from them.

Model calls go through `url_path_for_run`, so evaluation model-call capture
and training token capture see every turn of the rollout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, Request, Response
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
    failure_kinds,
)
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from responses_api_agents.vcqa_agent.judge import extract_must_have_items
from responses_api_agents.vcqa_agent.judge import grade as judge_grade
from responses_api_agents.vcqa_agent.materialize import (
    SharedWorkingTrees,
    fetch_artifact,
    local_artifact_path,
    materialize_working_tree,
)
from responses_api_agents.vcqa_agent.sandbox import (
    ApptainerDirectSandbox,
    ApptainerSandbox,
    LocalSandbox,
    ProviderSandbox,
    Sandbox,
    SandboxInfraError,
    make_instance_name,
    provider_setup_commands,
)
from responses_api_agents.vcqa_agent.tools import build_tool_definitions, dispatch_tool


LOG = logging.getLogger(__name__)


########################################
# Config
########################################


class VcqaAgentConfig(BaseResponsesAPIAgentConfig):
    """Static config: every knob comes from YAML, never from env vars."""

    model_server: ModelServerRef
    sandbox_backend: Literal["apptainer", "apptainer_exec", "local", "provider"] = Field(
        default="apptainer",
        description="Where to run tool commands. `apptainer` = Apptainer instance per rollout (production). "
        "`apptainer_exec` = direct Apptainer exec per tool call, for nested-container environments "
        "where instance exec cannot enter its namespace. "
        "`local` = `cwd=working_tree`, no container, intended for dev / macOS testing only. "
        "the host filesystem outside the working tree is reachable to anything that escapes "
        "path containment. "
        "`provider` = one sandbox per rollout on the NeMo Gym sandbox provider named by `sandbox_provider` "
        "(e.g. OpenSandbox); the repo is materialized inside it (`sandbox_*` settings).",
    )
    sandbox_provider: str = Field(
        default="sandbox", description="Provider instance name in the global config (sandbox_backend=provider)."
    )
    sandbox_image: str = Field(
        default="docker.io/library/debian:bookworm-slim",
        description="Provider sandbox image (sandbox_backend=provider).",
    )
    sandbox_resources: Dict[str, Any] = Field(
        default_factory=lambda: {"cpu": 2, "memory_mib": 8192, "disk_gib": 20},
        description="SandboxSpec.resources for provider sandboxes.",
    )
    sandbox_ttl_s: int = Field(default=14400, description="Provider sandbox time to live.")
    sandbox_ready_timeout_s: int = Field(
        default=600,
        description="Provider sandbox readiness timeout. With the shipped OpenSandbox create budget (1500 s) a create "
        "attempt needs this twice, so it stays below 750 s.",
    )
    sandbox_setup_timeout_s: int = Field(
        default=900, description="Per setup command (tools, user, repo extraction or clone) in a provider sandbox."
    )
    sandbox_provider_options: Dict[str, Any] = Field(
        default_factory=lambda: {"network_policy": {"defaultAction": "deny"}},
        description="SandboxSpec.provider_options. The default is OpenSandbox's deny-all egress policy (the task is "
        "offline, and the agent server drives every command through the provider API); other providers ignore it "
        "and need their own network setting, e.g. `sandbox.docker.create.network: none`.",
    )
    sandbox_tools_archive: Optional[str] = Field(
        default=None,
        description="Agent-side path of a tarball with the tools (rg, fd, git, tree) built for the sandbox image; "
        "uploaded and extracted over `/` at sandbox start.",
    )
    sandbox_tools_command: Optional[str] = Field(
        default=None,
        description="Instead of sandbox_tools_archive: a root command that installs the tools inside the sandbox, e.g. "
        "extracting an archive from a mounted path.",
    )
    sandbox_artifact_root: Optional[str] = Field(
        default=None,
        description="Sandbox-side directory holding the artifacts under their `artifact_key` (e.g. a mounted bucket). "
        "Unset: each rollout's artifact is uploaded from `artifact_url_prefix`.",
    )
    sandbox_user: Optional[str] = Field(
        default="vcqa",
        description="Non-root user for tool commands in provider sandboxes: the root-owned /codebase is then read-only, "
        "as with the Apptainer `:ro` bind. None runs them as root.",
    )
    sandbox_hidden_paths: List[str] = Field(
        default_factory=lambda: ["/mnt"],
        description="Provider-sandbox paths made root-only (chmod 700) after setup, so `sandbox_user` cannot read them. "
        "The default hides cell volumes such as a shared bucket mount, which can hold every task's artifacts. "
        "Ignored when sandbox_user is None.",
    )
    sandbox_upload_concurrency: int = Field(
        default=16, ge=1, description="Artifact and tools uploads in flight at once from this server process."
    )
    container_image: str = Field(
        default="docker://debian:bookworm-slim",
        description="Apptainer image URI. Pulled+cached on first use. Ignored when sandbox_backend='local'.",
    )
    apptainer_setup_command: str = Field(
        default=(
            "apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
            "ripgrep git fd-find tree ca-certificates "
            "&& ln -sf $(command -v fdfind) /usr/local/bin/fd"
        ),
        description="One-shot bash command run inside the container before the tool loop.",
    )
    apptainer_exec_timeout_s: int = Field(
        default=30, description="Default timeout for `apptainer exec` calls (per tool)."
    )
    artifact_url_prefix: str = Field(
        description="Base URL prefix that, joined with `verifier_metadata.artifact_key`, "
        "gives a fetchable URL for the per-task tarball or git bundle."
    )
    artifact_fetch_timeout_s: int = Field(
        default=120,
        description="Hard ceiling for a single artifact fetch. Past this the rollout fails "
        "with reward=0 instead of letting nemo_gym's global retry loop run forever on a "
        "broken artifact URL.",
    )
    artifact_request_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Extra HTTP headers attached to every artifact fetch (tarball / git "
        "bundle). Use this to authenticate against private hosts. Example for a private "
        "Hugging Face dataset repo:\n"
        "  artifact_request_headers:\n"
        "    Authorization: Bearer ${oc.env:HF_TOKEN}\n"
        "Values are passed through verbatim, so OmegaConf interpolation (incl. env-var "
        "lookup) works as usual.",
    )
    judge_base_url: str = Field(description="OpenAI-compatible judge endpoint base URL.")
    judge_api_key: str = Field(
        default="dummy", description="Bearer token for the judge endpoint."
    )  # pragma: allowlist secret
    judge_api_key_env: Optional[str] = Field(
        default=None,
        description="Name of an environment variable holding the judge API key, read when the judge is "
        "called. Takes precedence over `judge_api_key` and keeps the key out of the resolved config, "
        "which training drivers print and log.",
    )
    judge_model_name: str = Field(description="Model name to send to the judge endpoint.")
    judge_timeout_s: int = Field(default=60, description="Per-criterion judge call timeout.")
    judge_max_completion_tokens: int = Field(
        default=2048,
        description="Per-call output budget for the judge. For reasoning models (gpt-5 / o-series) "
        "this needs to be large enough to cover hidden reasoning + the final YES/NO token.",
    )
    judge_reasoning_effort: Optional[str] = Field(
        default=None,
        description="Forwarded as `reasoning_effort` to the judge endpoint. Set to `minimal` "
        "when judging with a gpt-5 / o-series reasoning model so the YES/NO actually fits "
        "inside `judge_max_completion_tokens`. Leave None for non-reasoning models.",
    )
    judge_extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra fields merged into every judge request body, e.g. "
        "`{chat_template_kwargs: {enable_thinking: false}}` for a hybrid-thinking judge.",
    )
    judge_max_attempts: int = Field(
        default=4,
        ge=1,
        description="Attempts per criterion. Timeouts, transport errors, HTTP 408/429/5xx and "
        "non-YES/NO replies are retried with exponential backoff.",
    )
    judge_retry_backoff_s: float = Field(
        default=2.0, ge=0.0, description="Backoff before the first retry; doubles on each further retry."
    )
    judge_max_concurrency: int = Field(
        default=256,
        ge=1,
        description="Judge calls in flight at once from this server process, across all rollouts. "
        "Keeps a burst of rollouts finishing together under the judge endpoint's rate limit.",
    )
    max_turns: int = Field(default=150, description="Hard cap on tool-loop turns.")
    apptainer_contain: bool = Field(
        default=True,
        description="Run Apptainer sandboxes with `--containall` / `--cleanenv`: no host environment "
        "(API keys), no host $HOME, /tmp or working directory inside the container. Only the "
        "working tree (read-only) and the rollout scratch dir are visible.",
    )
    apptainer_network_none: bool = Field(
        default=False,
        description="Give each Apptainer sandbox an empty network namespace (`--net --network none`), so the "
        "model's commands cannot reach the internet (the task is offline by design). Judge calls and artifact "
        "fetches run in the agent server and are unaffected.",
    )
    scratch_root: Optional[str] = Field(
        default=None,
        description="Directory for per-rollout scratch dirs (extracted working trees). Defaults to "
        "the system temp dir; point it at node-local disk for large repos.",
    )
    share_working_trees: bool = Field(
        default=True,
        description="Concurrent rollouts of the same task (same artifact and commit) share one "
        "materialized working tree instead of each extracting or cloning its own. Apptainer sandboxes mount "
        "it read-only, so sharing is not observable to the model; `local` sandboxes never share (their tree "
        "is writable), and `provider` sandboxes build their own tree.",
    )
    inject_tools: bool = Field(
        default=True,
        description="If true, the agent injects the VCQA tool definitions into "
        "responses_create_params.tools before each model call. Set false to let "
        "the row's tools field passthrough unchanged.",
    )
    include_distractor_tools: bool = Field(
        default=True,
        description="If true, surface the four distractor tools (install_package, "
        "send_pr_review, websearch, ask_user) alongside the six real tools. They "
        "exercise the model's investigator persona by surfacing tools that are "
        "off-task for a read-only code-investigation question (mutation, web "
        "search, asking the user). The grader does not currently penalize "
        "distractor usage. Set false for an A/B against the leaner "
        "real-tools-only surface. Ignored when `inject_tools` is false.",
    )


########################################
# Request / response models
########################################


class VcqaVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = None
    dataset_kind: Literal["fileset", "githistory"]
    repo_full_name: Optional[str] = None
    pre_merge_sha: Optional[str] = None
    head_sha: Optional[str] = None
    artifact_key: str
    rubric: Optional[Dict[str, Any]] = None
    verifiers: Optional[Dict[str, Any]] = None
    max_turns: Optional[int] = None


class VcqaAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: VcqaVerifierMetadata


class VcqaAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    error: Optional[str] = None
    judge_per_item: Optional[List[Dict[str, Any]]] = None
    must_pass: Optional[int] = None
    must_total: Optional[int] = None
    final_answer: Optional[str] = None
    num_turns: Optional[int] = None
    # False when the reward is not a measurement of the policy (infra or judge failure).
    evaluation_completed: bool = True
    failure_stage: Optional[str] = None
    judge_errors: Optional[int] = None


########################################
# Agent
########################################


_STAGE_FAILURE_KINDS = {
    "materialize": "vcqa_agent:artifact_unavailable",
    "sandbox_start": failure_kinds.PROVIDER_UNAVAILABLE,
    "model": failure_kinds.AGENT_REQUEST_FAILED,
    "judge": failure_kinds.JUDGE_FAILED,
    "sandbox": failure_kinds.PROVIDER_UNAVAILABLE,
}


class VcqaAgent(SimpleResponsesAPIAgent):
    config: VcqaAgentConfig

    _judge_semaphore_obj: Optional[asyncio.Semaphore] = PrivateAttr(default=None)
    _shared_trees_obj: Optional[SharedWorkingTrees] = PrivateAttr(default=None)
    _provider_obj: Any = PrivateAttr(default=None)
    _provider_metadata: Dict[str, str] = PrivateAttr(default_factory=dict)
    _upload_semaphore_obj: Optional[asyncio.Semaphore] = PrivateAttr(default=None)

    def _sandbox_provider(self) -> Any:
        if self._provider_obj is None:
            from nemo_gym.sandbox import create_provider, resolve_provider_config, resolve_provider_metadata

            global_config = self.server_client.global_config_dict
            self._provider_obj = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
            self._provider_metadata = dict(
                resolve_provider_metadata(self.config.sandbox_provider, global_config) or {}
            )
        return self._provider_obj

    def _upload_semaphore(self) -> asyncio.Semaphore:
        if self._upload_semaphore_obj is None:
            self._upload_semaphore_obj = asyncio.Semaphore(self.config.sandbox_upload_concurrency)
        return self._upload_semaphore_obj

    async def _build_provider_sandbox(
        self, *, meta: "VcqaVerifierMetadata", artifact_url: str, scratch_dir: Path
    ) -> ProviderSandbox:
        """A provider sandbox that materializes this task's tree itself (uploaded, or from sandbox_artifact_root)."""
        from nemo_gym.sandbox import SandboxSpec

        provider = self._sandbox_provider()
        uploads: List[tuple] = []
        tools_command = self.config.sandbox_tools_command
        if self.config.sandbox_tools_archive:
            uploads.append((Path(self.config.sandbox_tools_archive), "/tmp/vcqa-tools.tar.gz"))
            # Merged-/usr images: never replace the /lib, /bin symlinks with directories from the archive.
            tools_command = (
                "tar -xzf /tmp/vcqa-tools.tar.gz -C / --keep-directory-symlink --no-overwrite-dir"
                " && rm -f /tmp/vcqa-tools.tar.gz"
            )
        if self.config.sandbox_artifact_root:
            artifact_path = f"{self.config.sandbox_artifact_root.rstrip('/')}/{meta.artifact_key.lstrip('/')}"
            remove_artifact = False
        else:
            local = local_artifact_path(artifact_url)
            if local is None:
                local = scratch_dir / "artifact"
                await fetch_artifact(
                    artifact_url,
                    local,
                    timeout_s=self.config.artifact_fetch_timeout_s,
                    headers=self.config.artifact_request_headers,
                )
            artifact_path = "/tmp/vcqa-artifact"
            uploads.append((local, artifact_path))
            remove_artifact = True
        spec = SandboxSpec(
            image=self.config.sandbox_image,
            ttl_s=self.config.sandbox_ttl_s,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            resources=dict(self.config.sandbox_resources),
            metadata={**self._provider_metadata, "vcqa-task": str(meta.task_id or "")[:63]},
            provider_options=dict(self.config.sandbox_provider_options),
        )
        return ProviderSandbox(
            provider=provider,
            spec=spec,
            uploads=uploads,
            setup_commands=provider_setup_commands(
                dataset_kind=meta.dataset_kind,
                artifact_path=artifact_path,
                head_sha=meta.head_sha,
                tools_command=tools_command,
                run_as=self.config.sandbox_user,
                remove_artifact=remove_artifact,
                hidden_paths=list(self.config.sandbox_hidden_paths),
            ),
            run_as=self.config.sandbox_user,
            exec_timeout_s=self.config.apptainer_exec_timeout_s,
            setup_timeout_s=self.config.sandbox_setup_timeout_s,
            upload_semaphore=self._upload_semaphore(),
        )

    def _shared_trees(self) -> SharedWorkingTrees:
        if self._shared_trees_obj is None:
            root = Path(self.config.scratch_root or tempfile.gettempdir()) / "vcqa_shared_trees"
            self._shared_trees_obj = SharedWorkingTrees(root)
        return self._shared_trees_obj

    def _judge_api_key(self) -> str:
        name = self.config.judge_api_key_env
        if not name:
            return self.config.judge_api_key
        key = os.environ.get(name)
        if not key:
            raise RuntimeError(f"judge_api_key_env={name!r} is not set in the agent server's environment")
        return key

    def _judge_semaphore(self) -> asyncio.Semaphore:
        if self._judge_semaphore_obj is None:
            self._judge_semaphore_obj = asyncio.Semaphore(self.config.judge_max_concurrency)
        return self._judge_semaphore_obj

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Standalone /v1/responses is not supported because VCQA needs a per-row sandbox."""
        raise NotImplementedError("vcqa_agent does not support /v1/responses standalone. Post a row to /run instead.")

    async def run(
        self,
        request: Request,
        response: Response,
        body: VcqaAgentRunRequest = Body(),
    ) -> VcqaAgentVerifyResponse:
        meta = body.verifier_metadata
        rubric = self._select_rubric(meta)
        max_turns = meta.max_turns or self.config.max_turns

        if self.config.scratch_root:
            Path(self.config.scratch_root).mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix="vcqa_", dir=self.config.scratch_root))
        # Training token capture and eval model-call capture key every model call by this run's rollout.
        model_url_path = self.url_path_for_run("/v1/responses", body)
        sandbox: Optional[Sandbox] = None
        model_response: Optional[NeMoGymResponse] = None
        final_answer = ""
        num_turns = 0
        run_error: Optional[str] = None
        judge_per_item: Optional[List[Dict[str, Any]]] = None
        must_pass: Optional[int] = None
        must_total: Optional[int] = None
        judge_errors: Optional[int] = None
        reward = 0.0
        evaluation_completed = True
        failure_stage: Optional[str] = None
        failure_kind: Optional[str] = None
        stage = "materialize"
        trees = AsyncExitStack()

        try:
            artifact_url = self._build_artifact_url(meta.artifact_key)
            if self.config.sandbox_backend == "provider":
                # The sandbox builds the tree itself; nothing is extracted on this host.
                stage = "sandbox_start"
                sandbox = await self._build_provider_sandbox(
                    meta=meta, artifact_url=artifact_url, scratch_dir=scratch_dir
                )
                await sandbox.start()

            async def build_tree(directory: Path) -> Path:
                return await materialize_working_tree(
                    dataset_kind=meta.dataset_kind,
                    artifact_url=artifact_url,
                    head_sha=meta.head_sha,
                    scratch_dir=directory,
                    fetch_timeout_s=self.config.artifact_fetch_timeout_s,
                    fetch_headers=self.config.artifact_request_headers,
                )

            if sandbox is None:
                # Only read-only (Apptainer) trees are shared: a local sandbox's tree is writable.
                if self.config.share_working_trees and self.config.sandbox_backend != "local":
                    tree_key = f"{meta.dataset_kind}:{meta.artifact_key}:{meta.head_sha or ''}"
                    working_tree = await trees.enter_async_context(self._shared_trees().acquire(tree_key, build_tree))
                else:
                    working_tree = await build_tree(scratch_dir)

                stage = "sandbox_start"
                sandbox = self._build_sandbox(working_tree=working_tree, scratch_dir=scratch_dir / "sandbox_scratch")
                await sandbox.start()

            stage = "model"
            model_response, num_turns = await self._tool_loop(
                request=request,
                response=response,
                body=body.responses_create_params,
                sandbox=sandbox,
                max_turns=max_turns,
                model_url_path=model_url_path,
            )

            final_answer = _extract_final_assistant_text(model_response)
            problem_statement = _extract_problem_statement(body.responses_create_params)

            stage = "judge"
            judge_result = await judge_grade(
                rubric=rubric,
                problem_statement=problem_statement,
                model_answer=final_answer,
                base_url=self.config.judge_base_url,
                # Only a call needs the key: grade() scores an empty answer 0, and rejects a rubric without
                # must-have criteria, before calling the judge.
                api_key=self._judge_api_key() if final_answer.strip() and extract_must_have_items(rubric) else "",
                model_name=self.config.judge_model_name,
                timeout_s=self.config.judge_timeout_s,
                cookies=dict(request.cookies),
                max_completion_tokens=self.config.judge_max_completion_tokens,
                reasoning_effort=self.config.judge_reasoning_effort,
                extra_body=self.config.judge_extra_body,
                max_attempts=self.config.judge_max_attempts,
                retry_backoff_s=self.config.judge_retry_backoff_s,
                semaphore=self._judge_semaphore(),
            )
            reward = judge_result.reward
            judge_per_item = judge_result.per_item
            must_pass = judge_result.must_pass
            must_total = judge_result.must_total
            judge_errors = judge_result.judge_errors
            run_error = judge_result.error
            if judge_result.judge_errors:
                # A criterion the judge never answered is not a measured NO.
                evaluation_completed, failure_stage, failure_kind = False, "judge", failure_kinds.JUDGE_FAILED
            elif judge_result.must_total == 0:
                # A row without must-have items cannot be scored at all.
                evaluation_completed, failure_stage, failure_kind = False, "judge", failure_kinds.VERIFIER_ERROR

        except Exception as e:
            run_error = f"{type(e).__name__}: {e}"
            reward = 0.0
            evaluation_completed = False
            # A sandbox that dies mid-rollout is the environment's failure, whichever stage it hit.
            failure_stage = "sandbox" if isinstance(e, SandboxInfraError) else stage
            failure_kind = _STAGE_FAILURE_KINDS.get(failure_stage, failure_kinds.AGENT_RUN_ERROR)
        finally:
            if sandbox is not None:
                await _stop_sandbox(sandbox)
            # Release the shared tree only after the sandbox that mounts it has stopped.
            await trees.aclose()
            shutil.rmtree(scratch_dir, ignore_errors=True)

        if model_response is None:
            model_response = _empty_response(body.responses_create_params)

        return VcqaAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=model_response,
            reward=reward,
            error=run_error,
            judge_per_item=judge_per_item,
            must_pass=must_pass,
            must_total=must_total,
            final_answer=final_answer,
            num_turns=num_turns,
            evaluation_completed=evaluation_completed,
            mask_sample=not evaluation_completed,
            failure_stage=failure_stage,
            failure_kind=failure_kind,
            failure_reason=run_error if not evaluation_completed else None,
            judge_errors=judge_errors,
        )

    ########################################
    # Internals
    ########################################

    def _build_artifact_url(self, artifact_key: str) -> str:
        prefix = self.config.artifact_url_prefix.rstrip("/")
        key = artifact_key.lstrip("/")
        return f"{prefix}/{key}"

    def _build_sandbox(self, *, working_tree: Path, scratch_dir: Path) -> Sandbox:
        if self.config.sandbox_backend == "local":
            return LocalSandbox(
                working_tree=working_tree,
                scratch_dir=scratch_dir,
                exec_timeout_s=self.config.apptainer_exec_timeout_s,
            )
        if self.config.sandbox_backend == "apptainer_exec":
            return ApptainerDirectSandbox(
                container_image=self.config.container_image,
                working_tree=working_tree,
                scratch_dir=scratch_dir,
                exec_timeout_s=self.config.apptainer_exec_timeout_s,
                extra_setup_command=self.config.apptainer_setup_command or None,
                contain=self.config.apptainer_contain,
                network_none=self.config.apptainer_network_none,
            )
        return ApptainerSandbox(
            instance_name=make_instance_name(),
            container_image=self.config.container_image,
            working_tree=working_tree,
            scratch_dir=scratch_dir,
            exec_timeout_s=self.config.apptainer_exec_timeout_s,
            extra_setup_command=self.config.apptainer_setup_command or None,
            contain=self.config.apptainer_contain,
            network_none=self.config.apptainer_network_none,
        )

    @staticmethod
    def _select_rubric(meta: VcqaVerifierMetadata) -> Optional[Dict[str, Any]]:
        """Fileset rows put the rubric under `rubric`; githistory rows under `verifiers`."""
        if meta.rubric:
            return meta.rubric
        return meta.verifiers

    async def _tool_loop(
        self,
        *,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming,
        sandbox: Sandbox,
        max_turns: int,
        model_url_path: str = "/v1/responses",
    ) -> tuple[NeMoGymResponse, int]:
        """Mirror of `simple_agent.responses`, with local sandbox-backed tool dispatch.

        Output items are appended verbatim to the running input list so any
        token-ID/log-prob fields the model server stamped on them propagate
        unchanged into subsequent turns (NeMo Gym guideline 2).
        """
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        if self.config.inject_tools:
            existing = list(body.tools) if body.tools else []
            body.tools = existing + build_tool_definitions(include_distractors=self.config.include_distractor_tools)

        new_outputs: List[Any] = []
        usage = None
        step = 0
        model_server_cookies = None
        latest_response: Optional[NeMoGymResponse] = None
        context_exceeded = False

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response_raw = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            if model_response_raw.status == 400:
                content = await model_response_raw.read()
                if _is_context_refusal(content):
                    # The prompt no longer fits the context window. This ends the episode as a measurement, as when
                    # the model server answers such a call itself: what the policy produced so far is graded.
                    context_exceeded = True
                    break
                await raise_for_status(model_response_raw, content)
            await raise_for_status(model_response_raw)
            model_server_cookies = model_response_raw.cookies
            model_response_json = await get_response_json(model_response_raw)
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(f"Invalid response from model server: {json.dumps(model_response_json)}") from e
            latest_response = model_response

            output = model_response.output
            new_outputs.extend(output)

            usage = accumulate_response_usage(usage, model_response.usage)
            model_response.usage = None

            if model_response.incomplete_details:
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in output if isinstance(o, NeMoGymResponseFunctionToolCall)
            ]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if isinstance(o, NeMoGymResponseOutputMessage)
            ]
            if not all_fn_calls and all_output_messages:
                break

            for fn_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(fn_call.arguments) if fn_call.arguments else {}
                except (json.JSONDecodeError, TypeError) as e:
                    new_outputs.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=fn_call.call_id,
                            output=json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
                        )
                    )
                    continue

                tool_result = await dispatch_tool(fn_call.name, parsed_arguments, sandbox)
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=fn_call.call_id,
                        output=json.dumps(tool_result),
                    )
                )

            if step >= max_turns:
                break

        for k, v in (model_server_cookies or {}).items():
            response.set_cookie(k, v)

        if latest_response is None:  # only a refusal of the first call gets here without one
            latest_response = _empty_response(body)
        latest_response.output = new_outputs
        latest_response.usage = usage
        if context_exceeded:
            latest_response.status = "incomplete"
            latest_response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
            latest_response.metadata = {**(latest_response.metadata or {}), "nemo_gym_context_window_exceeded": "true"}
        return latest_response, step


########################################
# Helpers
########################################


def _extract_final_assistant_text(resp: NeMoGymResponse) -> str:
    """Last assistant message's concatenated `output_text` in the response."""
    for item in reversed(resp.output):
        if isinstance(item, NeMoGymResponseOutputMessage) and item.role == "assistant":
            text_parts: List[str] = []
            for content in item.content:
                text = getattr(content, "text", None)
                if text:
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts).strip()
    return ""


def _extract_problem_statement(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Best-effort: concatenate user-message contents from the input."""
    if isinstance(params.input, str):
        return params.input
    parts: List[str] = []
    for item in params.input:
        if isinstance(item, (NeMoGymEasyInputMessage,)):
            if item.role == "user":
                if isinstance(item.content, str):
                    parts.append(item.content)
                else:
                    for c in item.content:
                        text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                        if text:
                            parts.append(text)
    return "\n".join(parts).strip()


async def _stop_sandbox(sandbox: Any, *, attempts: int = 3) -> None:
    """Stop a rollout's sandbox, retrying a failed stop; one that never stops is left to the provider's TTL."""
    for attempt in range(1, attempts + 1):
        try:
            await sandbox.stop()
            return
        except Exception as e:
            if attempt == attempts:
                LOG.warning(
                    "stopping a VCQA sandbox failed %d times, leaving it to its TTL: %s: %s",
                    attempts,
                    type(e).__name__,
                    e,
                )
                return
            await asyncio.sleep(attempt)


def _is_context_refusal(content: bytes) -> bool:
    """Whether a model server's HTTP 400 body is a refusal for exceeding the context window (OpenAI's envelope with
    ``code: context_length_exceeded``, which Gym's model servers answer with when they propagate the refusal)."""
    try:
        error = json.loads(content).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and error.get("code") == "context_length_exceeded"


def _empty_response(params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
    """Build a minimal NeMoGymResponse for the failure-before-model case."""
    return NeMoGymResponse(
        id="vcqa-empty",
        created_at=0,
        model=params.model or "unknown",
        object="response",
        output=[],
        parallel_tool_calls=params.parallel_tool_calls,
        tool_choice=params.tool_choice,
        tools=[],
    )


if __name__ == "__main__":
    VcqaAgent.run_webserver()

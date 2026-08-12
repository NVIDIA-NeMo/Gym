# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hidden final-answer verifier for BioMysteryBench.

BioMysteryBench is method agnostic: the policy agent may use any analysis path,
but receives binary credit only when its final answer satisfies the task rubric
and it did not actively reverse-identify the source study.  The rubric remains
host-side and is never included in ``responses_create_params``.
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
import tempfile
from asyncio import to_thread
from pathlib import Path
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.frontierscience_judge.app import (
    FrontierScienceJudgeConfig,
    FrontierScienceJudgeServer,
    extract_text_from_response,
    parse_judgement,
)


_ACCESSION_RE = re.compile(
    r"\b(?:GSE|GSM|SRR|SRX|SRS|ERP|ERR|ERX|ERS|DRP|DRR|DRX|DRS|PRJNA|PRJEB|PRJDB)\d+\b",
    re.IGNORECASE,
)
_FORBIDDEN_LOOKUP_RE = re.compile(
    r"(?:geo/query/acc\.cgi|ena/browser/(?:view|api)|(?:db|database)\s*[=:]\s*(?:sra|gds|bioproject))",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s\\\"'<>]+", re.IGNORECASE)
_INTERNAL_HOSTNAMES = {"localhost", "host.docker.internal"}


class BioMysteryBenchJudgeConfig(FrontierScienceJudgeConfig):
    """Binary rubric judge plus conservative trajectory anti-cheat checks."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    detect_forbidden_accession_lookups: bool = Field(
        default=True,
        description=(
            "Fail attempts whose captured tool-call arguments actively use GEO/SRA/ENA/BioProject accessions "
            "or known study-lookup endpoints. Reference-genome accessions such as GCF_/NC_ remain allowed."
        ),
    )
    enforce_allowed_domains: bool = Field(
        default=True,
        description="Fail captured tool calls whose explicit URLs target a domain outside allowed_domains.",
    )
    sandbox_provider: Optional[str] = None
    sandbox_config: Dict[str, Any] = Field(default_factory=dict)


class BioMysteryBenchSeedSessionRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BioMysteryBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: Dict[str, Any]


class BioMysteryBenchJudgeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    human_solvable: Optional[str] = None
    allowed_domains: Optional[List[str]] = None


class BioMysteryBenchJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    verdict: Optional[str] = None
    judge_output: Optional[str] = None
    invalid_judge_response: bool = False
    cheat_detected: bool = False
    cheat_evidence: List[str] = Field(default_factory=list)


def _tool_call_arguments(body: BioMysteryBenchJudgeVerifyRequest) -> list[str]:
    """Return captured tool-call arguments without considering final-answer text.

    Recalling an accession from model memory is explicitly allowed by the v11
    grading rule.  We therefore inspect only active tool calls, not assistant
    prose or tool output.
    """

    arguments: list[str] = []
    for item in body.response.output:
        if getattr(item, "type", None) != "function_call":
            continue
        raw = getattr(item, "arguments", "")
        if isinstance(raw, str):
            arguments.append(raw)
        elif raw is not None:
            arguments.append(json.dumps(raw, sort_keys=True))
    return arguments


def detect_forbidden_lookup(body: BioMysteryBenchJudgeVerifyRequest) -> list[str]:
    """Return short, non-secret evidence labels for prohibited active lookups."""

    evidence: list[str] = []
    for index, arguments in enumerate(_tool_call_arguments(body)):
        accessions = sorted(set(match.upper() for match in _ACCESSION_RE.findall(arguments)))
        if accessions:
            evidence.append(f"tool_call[{index}] accession lookup: {', '.join(accessions[:5])}")
        if _FORBIDDEN_LOOKUP_RE.search(arguments):
            evidence.append(f"tool_call[{index}] prohibited study-lookup endpoint")
    return evidence


def detect_disallowed_domains(body: BioMysteryBenchJudgeVerifyRequest) -> list[str]:
    """Audit explicit URLs against allowed domains and their subdomains."""

    allowed = {domain.lower().rstrip(".") for domain in body.allowed_domains or []}
    if not allowed:
        return []

    evidence: list[str] = []
    for index, arguments in enumerate(_tool_call_arguments(body)):
        for url in _URL_RE.findall(arguments):
            try:
                hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
            except ValueError:
                # This audit covers explicit URLs only. Malformed shell fragments
                # are neither a resolvable hostname nor evidence of network use.
                continue
            if any(marker in hostname for marker in ("$", "{", "}", "%")):
                # Shell-expanded hosts such as https://$d/path are not explicit
                # destinations. Treating "$d" as a literal DNS name creates a
                # false cheating verdict and cannot establish where the command
                # actually connected.
                continue
            if hostname in _INTERNAL_HOSTNAMES or hostname.endswith(".localhost"):
                # Docker's host gateway carries the model/proxy connection but
                # is not the external destination being audited. The proxied
                # URL, when explicit, is checked independently.
                continue
            hostname_allowed = any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed)
            if hostname and not hostname_allowed:
                label = f"tool_call[{index}] disallowed domain: {hostname}"
                if label not in evidence:
                    evidence.append(label)
    return evidence


class BioMysteryBenchJudgeServer(FrontierScienceJudgeServer):
    config: BioMysteryBenchJudgeConfig

    def model_post_init(self, context: Any, /) -> None:
        self._sandboxes: Dict[str, AsyncSandbox] = {}
        super().model_post_init(context)

    @staticmethod
    def _request_metadata(body: BioMysteryBenchSeedSessionRequest) -> dict[str, Any]:
        return dict(body.responses_create_params.metadata or {})

    @staticmethod
    def _archive_data(data_dir: Path) -> Path:
        archive_dir = data_dir.parent.parent / "sandbox_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        marker = data_dir / ".biomysterybench-extracted.json"
        try:
            marker_data = json.loads(marker.read_text()) if marker.is_file() else {}
        except (OSError, ValueError):
            marker_data = {}
        archive_hash = str(marker_data.get("archive_sha256") or "unknown")
        archive = archive_dir / f"{data_dir.name}-{archive_hash}.tar.gz"
        if archive.is_file():
            return archive
        with tempfile.NamedTemporaryFile(dir=archive_dir, suffix=".tar.gz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with tarfile.open(temporary, "w:gz") as bundle:
                bundle.add(data_dir, arcname=".")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        return archive

    async def _create_sandbox(self, metadata: dict[str, Any]) -> AsyncSandbox:
        if not self.config.sandbox_provider:
            raise ValueError("sandbox_provider is required for BioMysteryBench rollout collection")
        global_config = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        config = self.config.sandbox_config
        resources = dict(config.get("resources") or {})
        if metadata.get("cpus") is not None:
            resources["cpu"] = float(metadata["cpus"])
        if metadata.get("memory_mb") is not None:
            resources["memory_mib"] = int(float(metadata["memory_mb"]))
        if metadata.get("storage_mb") is not None:
            storage_mb = int(float(metadata["storage_mb"]))
            resources["disk_gib"] = max(1, (storage_mb + 1023) // 1024)

        spec = SandboxSpec(
            image=str(metadata.get("docker_image") or "ubuntu:22.04").removeprefix("docker://"),
            ttl_s=config.get("ttl_s"),
            ready_timeout_s=config.get("ready_timeout_s"),
            workdir=str(metadata.get("workdir") or "/workspace"),
            metadata=provider_metadata
            | dict(config.get("metadata") or {})
            | {
                "benchmark": "biomysterybench",
                "instance_id": str(metadata.get("instance_id") or "unknown")[:63],
            },
            resources=SandboxResources.from_mapping(resources),
            provider_options=dict(config.get("provider_options") or {}),
        )
        sandbox = AsyncSandbox(provider_config, spec)
        await sandbox.start()

        data_dir = Path(str(metadata["data_dir"])).expanduser().resolve()
        if not data_dir.is_dir():
            await sandbox.stop()
            raise FileNotFoundError(f"BioMysteryBench data directory does not exist: {data_dir}")
        archive = await to_thread(self._archive_data, data_dir)
        await sandbox.upload(archive, "/tmp/biomysterybench-data.tar.gz")
        result = await sandbox.exec(
            "rm -rf /data && mkdir -p /data && "
            "tar -xzf /tmp/biomysterybench-data.tar.gz -C /data && chmod -R a-w /data",
            timeout_s=1800,
            user="root",
        )
        if result.return_code != 0:
            await sandbox.stop()
            raise RuntimeError(result.stderr or "failed to stage BioMysteryBench data")
        return sandbox

    async def seed_session(
        self,
        request: Request,
        body: BioMysteryBenchSeedSessionRequest,
    ) -> BioMysteryBenchSeedSessionResponse:
        sandbox = await self._create_sandbox(self._request_metadata(body))
        self._sandboxes[request.session[SESSION_ID_KEY]] = sandbox
        return BioMysteryBenchSeedSessionResponse(sandbox_handle=await sandbox.serialize(scope="operate"))

    async def _stop_sandbox(self, request: Request | None) -> None:
        if request is None:
            return
        sandbox = self._sandboxes.pop(request.session.get(SESSION_ID_KEY), None)
        if sandbox is not None:
            try:
                await sandbox.stop()
            except Exception:
                print("Failed to stop BioMysteryBench sandbox", format_exc(), file=sys.stderr)

    async def _call_judge(self, judge_prompt: str) -> str:
        """Call NVIDIA-hosted Anthropic models with one sampling control.

        Anthropic's chat-completions contract rejects requests containing both
        ``temperature`` and ``top_p``.  The shared FrontierScience olympiad
        helper always adds both, so BioMysteryBench uses the same call path but
        deliberately sends only temperature.
        """

        if not self.config.use_chat_completions_for_judge:
            return await super()._call_judge(judge_prompt)

        chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=self.config.judge_responses_create_params.max_output_tokens or 2048,
            temperature=self.config.judge_responses_create_params.temperature
            if self.config.judge_responses_create_params.temperature is not None
            else 0.0,
        )
        chat_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/chat/completions",
            json=chat_params,
            response_model=NeMoGymChatCompletion,
        )
        content = chat_response.choices[0].message.content if chat_response.choices else None
        return content.strip() if content else ""

    @staticmethod
    def _score_fn(result: dict) -> dict:
        return {"accuracy": float(result.get("reward", 0.0))}

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_answer",
        )
        metrics.update(
            compute_subset_metrics(
                tasks,
                subset_key="human_solvable",
                score_fn=self._score_fn,
                answer_key="extracted_answer",
            )
        )
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        key: dict = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        for split in ("yes", "no"):
            key.update(
                highest_k_metrics(
                    agent_metrics,
                    f"{split}/pass@1[avg-of-{{k}}]",
                    score_names=["accuracy"],
                )
            )
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))
        return key

    async def verify(
        self,
        body: BioMysteryBenchJudgeVerifyRequest,
        request: Request = None,
    ) -> BioMysteryBenchJudgeVerifyResponse:
        raw_text = extract_text_from_response(body.response, strip_thinking=False)
        generation = extract_text_from_response(body.response)
        has_open = "<think>" in raw_text or "<thinking>" in raw_text
        has_close = "</think>" in raw_text or "</thinking>" in raw_text
        if has_open and not has_close:
            generation = ""

        expected_answer = body.expected_answer or ""
        evidence = detect_forbidden_lookup(body) if self.config.detect_forbidden_accession_lookups else []
        if self.config.enforce_allowed_domains:
            evidence.extend(detect_disallowed_domains(body))

        await self._stop_sandbox(request)

        judge_text = ""
        verdict = None
        invalid_judge_response = False
        if generation and not evidence:
            judge_prompt = self._judge_prompt_template.format(
                question=body.question or "",
                expected_answer=expected_answer,
                rubric=expected_answer,
                generation=generation,
            )
            judge_text = await self._call_judge(judge_prompt)
            verdict = parse_judgement(judge_text)
            invalid_judge_response = verdict is None

        reward = 1.0 if verdict == "YES" and not evidence else 0.0
        return BioMysteryBenchJudgeVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            extracted_answer=generation or None,
            expected_answer=expected_answer,
            verdict=verdict,
            judge_output=judge_text,
            invalid_judge_response=invalid_judge_response,
            cheat_detected=bool(evidence),
            cheat_evidence=evidence,
        )


if __name__ == "__main__":
    BioMysteryBenchJudgeServer.run_webserver()

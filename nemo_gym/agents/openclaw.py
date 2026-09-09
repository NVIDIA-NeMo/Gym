# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import copy
import json
import logging
import os
import shlex
import shutil
import signal
from pathlib import Path
from time import time
from typing import Any, Callable, ClassVar, Optional
from uuid import uuid4

import psutil

from nemo_gym.agents.config import AgentHarnessConfig
from nemo_gym.agents.openclaw_observability import (
    OPENCLAW_OBSERVATION_SOURCE,
    OpenClawSessionTree,
    build_openclaw_observation_tree,
    build_openclaw_observations,
    discover_openclaw_session_tree,
)
from nemo_gym.agents.responses import extract_instruction as _extract_instruction
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentObservationBundle,
    ObservationGap,
)


LOG = logging.getLogger(__name__)


def _decode_last_json_dict_suffix(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and not text[start + consumed :].strip():
            return obj
    return None


def _text_from_openclaw_payloads(envelope: dict[str, Any]) -> str:
    payloads = envelope.get("payloads")
    if not isinstance(payloads, list):
        payloads = []
    parts = [p["text"].strip() for p in payloads if isinstance(p, dict) and (p.get("text") or "").strip()]
    if parts:
        return "\n\n".join(parts)
    final = (envelope.get("meta") or {}).get("finalAssistantVisibleText")
    return final.strip() if isinstance(final, str) else ""


def parse_openclaw_output(stdout: str) -> tuple[list[Any], dict[str, int]]:
    envelope = _decode_last_json_dict_suffix(stdout)
    if not envelope:
        return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    text = _text_from_openclaw_payloads(envelope)
    output_items: list[Any] = []
    if text:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id="msg-0",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    cache_read = int(usage.get("cacheRead") or 0)
    input_tokens = int(usage.get("input") or 0) + cache_read
    output_tokens = int(usage.get("output") or 0)
    return output_items, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cache_read,
    }


def parse_openclaw_session_items(events: list[dict[str, Any]], *, include_input: bool = False) -> list[Any]:
    output_items: list[Any] = []
    for event in events:
        event_id = event.get("id")
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if include_input and role in {"user", "system", "developer"}:
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"]
                )
            if text:
                output_items.append(NeMoGymEasyInputMessage(role=role, content=text))
            continue
        if not include_input and not isinstance(content, list):
            continue

        if role == "assistant":
            reasoning = []
            for key in ("reasoning_content", "reasoning_text", "thinking"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    reasoning.append(value)
            message_reasoning = message.get("reasoning")
            if isinstance(message_reasoning, str) and message_reasoning:
                reasoning.append(message_reasoning)
            elif isinstance(message_reasoning, dict):
                for key in ("content", "text", "summary"):
                    value = message_reasoning.get(key)
                    if isinstance(value, str) and value:
                        reasoning.append(value)
            if isinstance(content, list):
                reasoning.extend(
                    text
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") in {"thinking", "reasoning"}
                    and isinstance((text := block.get("thinking") or block.get("text") or block.get("reasoning")), str)
                    and text
                )
            if include_input and reasoning:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{event_id or len(output_items)}",
                        summary=[NeMoGymSummary(text="\n".join(reasoning), type="summary_text")],
                    )
                )

            texts = [content] if include_input and isinstance(content, str) and content else []
            if isinstance(content, list):
                texts = [
                    block["text"] for block in content if isinstance(block, dict) and (block.get("text") or "").strip()
                ]
                if include_input:
                    texts = [
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") not in {"thinking", "reasoning", "toolCall"}
                        and isinstance(block.get("text"), str)
                        and block["text"].strip()
                    ]
            if texts:
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="\n".join(texts), annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                if include_input and args is None:
                    args = block.get("partialArgs")
                arguments = json.dumps(args) if isinstance(args, (dict, list)) else str(args or "")
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        name=block.get("name", ""),
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )

        elif role == "toolResult":
            call_id = message.get("toolCallId") or (message.get("tool_call_id") if include_input else "") or ""
            result_text = content if include_input and isinstance(content, str) else ""
            if isinstance(content, list):
                result_text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if include_input and not result_text and message.get("details") is not None:
                result_text = json.dumps(message["details"], ensure_ascii=False)
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="completed",
                )
            )

    return output_items


def openclaw_session_conversation(
    events: list[dict[str, Any]],
    *,
    input_items: list[Any] | None = None,
    fallback_output: list[Any] | None = None,
) -> list[Any]:
    conversation = parse_openclaw_session_items(events, include_input=True)
    inputs = input_items or []
    fallback = fallback_output or []
    if not conversation:
        return [*inputs, *fallback]
    retained_roles = {
        role for item in conversation if (role := getattr(item, "role", None)) in {"user", "system", "developer"}
    }
    missing_inputs = (
        [item for item in inputs if getattr(item, "role", None) not in retained_roles] if retained_roles else inputs
    )
    if missing_inputs:
        conversation = [*missing_inputs, *conversation]
    if fallback and not any(
        getattr(item, "role", None) == "assistant"
        or getattr(item, "type", None) in {"reasoning", "function_call", "function_call_output"}
        for item in conversation
    ):
        conversation.extend(fallback)
    return conversation


def parse_openclaw_session(session_text: str) -> list[Any]:
    return parse_openclaw_session_items(parse_openclaw_session_events(session_text))


def parse_openclaw_session_events(session_text: str) -> list[dict[str, Any]]:
    events = []
    for line in session_text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            event = {"raw": line}
        events.append(event if isinstance(event, dict) else {"raw": line})
    return events


class OpenClawHarness:
    # The interactive message channel prevents headless runs from finishing.
    _HEADLESS_TOOL_DENY: ClassVar[tuple[str, ...]] = ("message",)

    def __init__(self, config: AgentHarnessConfig):
        if config.model.model is None:
            raise ValueError("OpenClawHarness requires a model")
        if config.max_turns is not None:
            raise ValueError("OpenClawHarness does not support max_turns")
        self.config = config
        self.sigterm_events: set[asyncio.Event] = set()
        self.sigterm_handler_installed = False

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.config.settings.get("command", "openclaw"))

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenClawHarness._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _merge_headless_tool_denies(self, cfg: dict[str, Any]) -> None:
        tools = cfg.setdefault("tools", {})
        deny = tools.get("deny")
        if not isinstance(deny, list):
            deny = []
        merged = list(dict.fromkeys([item for item in deny if isinstance(item, str)] + list(self._HEADLESS_TOOL_DENY)))
        tools["deny"] = merged

    def _build_openclaw_config(self, base: dict[str, Any], model_base_url: str = "") -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        self._deep_merge(cfg, copy.deepcopy(self.config.settings.get("openclaw_config", {})))
        if model_base_url:
            providers = cfg.setdefault("models", {}).setdefault("providers", {})
            provider_name = self.config.model.provider or "nemo"
            provider = providers.setdefault(provider_name, {})
            model_entry = {
                "id": self.config.model.model or "",
                "name": self.config.model.model or "",
                "api": "openai-completions",
                "reasoning": True,
                "input": ["text"],
            }
            context_window = self.config.model.settings.get("context_window")
            max_output_tokens = self.config.model.settings.get("max_output_tokens")
            if context_window is not None:
                model_entry["contextWindow"] = context_window
            if max_output_tokens is not None:
                model_entry["maxTokens"] = max_output_tokens
            provider.update(
                {
                    "api": "openai-completions",
                    "baseUrl": model_base_url,
                    "apiKey": self.config.model.api_key or "EMPTY",  # pragma: allowlist secret
                    "models": [model_entry],
                }
            )
        self._merge_headless_tool_denies(cfg)
        return cfg

    def _effective_model(self) -> str:
        model = self.config.model.model or ""
        provider = self.config.model.provider
        return f"{provider}/{model}" if provider else model

    def _workspace_root(self) -> Path:
        root = Path(self.config.settings.get("workspace_root", "outputs/openclaw_agent/workspaces")).expanduser()
        root /= f"openclaw_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {
            **os.environ,
            "HOME": str(home),
            "OPENCLAW_TELEMETRY": "0",
            "CLAWHUB_DISABLE_TELEMETRY": "1",
        }
        node_bin_dir = self.config.settings.get("node_bin_dir")
        if node_bin_dir:
            env["PATH"] = f"{node_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env.update({k: v for k, v in self.config.settings.get("env", {}).items() if v})
        return env

    async def _run_exec(
        self, args: list[str], *, cwd: Optional[str], env: dict[str, str], timeout: int
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            self._kill_process_tree(proc.pid)
            await proc.communicate()
            raise TimeoutError(f"Timed out after {timeout}s: {shlex.join(args)}") from None
        except asyncio.CancelledError:
            # Cancelling the await does not stop the subprocess.
            self._kill_process_tree(proc.pid)
            with contextlib.suppress(Exception):
                await proc.communicate()
            raise
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return
        for child in reversed(children):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        with contextlib.suppress(psutil.NoSuchProcess):
            parent.kill()

    @staticmethod
    def _session_file(envelope: Optional[dict[str, Any]]) -> Optional[Path]:
        meta = (envelope or {}).get("meta") if isinstance(envelope, dict) else None
        agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
        session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
        return Path(session_file) if isinstance(session_file, str) and session_file else None

    @staticmethod
    def _find_partial_session(home: Path) -> Optional[Path]:
        try:
            candidates = sorted(
                (p for p in home.rglob("*.jsonl") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None
        for path in candidates:
            try:
                if parse_openclaw_session(path.read_text(errors="replace")):
                    return path
            except OSError:
                continue
        return None

    def _install_sigterm_handler(self) -> None:
        if self.sigterm_handler_installed:
            return
        previous = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame) -> None:
            for event in self.sigterm_events:
                event.set()
            if callable(previous):
                previous(signum, frame)

        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
            self.sigterm_handler_installed = True
        except ValueError:
            pass

    async def _run_openclaw(
        self,
        instruction: str,
        system_prompt: Optional[str],
        model_base_url: str = "",
        observation_collector: Optional[
            Callable[[str, list[dict[str, Any]], OpenClawSessionTree, list[ObservationGap]], None]
        ] = None,
    ) -> tuple[list[Any], dict[str, int], str]:
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._project_dir(work_dir)
        home = work_dir / ".openclaw-home"
        home.mkdir(parents=True, exist_ok=True)
        env = self._env(home)

        try:
            code, _, stderr = await self._run_exec(
                [*self.command_parts, "onboard", "--non-interactive", "--accept-risk", "--skip-health"],
                cwd=str(project_dir),
                env=env,
                timeout=self.config.settings.get("setup_timeout", 900),
            )
            if code:
                raise RuntimeError(f"openclaw onboard exited {code}: {stderr}")

            config_path = home / ".openclaw" / "openclaw.json"
            if not config_path.is_file():
                raise RuntimeError(f"openclaw onboard did not produce a config at {config_path}: {stderr}")
            base_cfg = json.loads(config_path.read_text())
            config_path.write_text(json.dumps(self._build_openclaw_config(base_cfg, model_base_url), indent=2) + "\n")

            cmd = [
                *self.command_parts,
                "agent",
                "--local",
                "--json",
                "--agent",
                self.config.settings.get("openclaw_agent_id", "main"),
                "--thinking",
                self.config.settings.get("thinking", "off"),
                "--model",
                self._effective_model(),
                "--message",
                prompt,
                *self.config.settings.get("extra_args", []),
            ]
            # OpenClaw writes transcripts incrementally, so interrupted runs can return completed turns.
            code, stdout, stderr = None, "", ""
            self._install_sigterm_handler()
            sigterm_hit = asyncio.Event()
            self.sigterm_events.add(sigterm_hit)
            run_task = asyncio.ensure_future(
                self._run_exec(cmd, cwd=str(project_dir), env=env, timeout=self.config.timeout_seconds)
            )
            try:
                term_task = asyncio.ensure_future(sigterm_hit.wait())
                done, _ = await asyncio.wait({run_task, term_task}, return_when=asyncio.FIRST_COMPLETED)
                term_task.cancel()
                if run_task in done:
                    code, stdout, stderr = run_task.result()
                else:
                    LOG.warning("openclaw received SIGTERM. Salvaging partial session")
                    run_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await run_task
            except TimeoutError:
                LOG.warning("openclaw timed out after %ss. Salvaging partial session", self.config.timeout_seconds)
            finally:
                self.sigterm_events.discard(sigterm_hit)

            if code:
                LOG.warning("openclaw exited %d: %s", code, stderr)
            if stdout:
                LOG.debug("openclaw stdout (%d chars): %s", len(stdout), stdout[:2000])

            fallback_items, usage = parse_openclaw_output(stdout)
            envelope = _decode_last_json_dict_suffix(stdout)

            output_items: list[Any] = []
            session_path = self._session_file(envelope) or self._find_partial_session(home)
            if session_path and session_path.is_file():
                session_text = session_path.read_text(errors="replace")
                output_items = parse_openclaw_session(session_text)
                if observation_collector is not None:
                    try:
                        session_events = parse_openclaw_session_events(session_text)
                        native_session_id = next(
                            (
                                event.get("id")
                                for event in session_events
                                if event.get("type") == "session" and isinstance(event.get("id"), str)
                            ),
                            session_path.stem,
                        )
                        session_tree, tree_gaps = discover_openclaw_session_tree(
                            home / ".openclaw" / "agents",
                            native_session_id,
                        )
                        observation_collector(native_session_id, session_events, session_tree, tree_gaps)
                    except Exception:
                        LOG.exception("failed to record OpenClaw session artifact")
            if not output_items:
                output_items = fallback_items
            return output_items, usage, self.config.model.model or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        model_base_url: Optional[str] = None,
        observation_collector: Optional[
            Callable[[str, list[dict[str, Any]], OpenClawSessionTree, list[ObservationGap]], None]
        ] = None,
        output_collector: Optional[Callable[[list[Any]], None]] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        try:
            output_items, usage, model_name = await self._run_openclaw(
                user_message,
                system_prompt,
                model_base_url=model_base_url or "",
                observation_collector=observation_collector,
            )
        except TimeoutError:
            LOG.warning("OpenClaw timed out, padding empty output so the rollout scores instead of erroring")
            output_items, usage, model_name = (
                [],
                {"input_tokens": 0, "output_tokens": 0},
                self.config.model.model or "",
            )

        if output_collector is not None:
            output_collector(list(output_items))
        if not output_items:
            LOG.warning("OpenClaw produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    def _project_dir(self, fallback: Path) -> Path:
        if self.config.workspace is None:
            return fallback
        project_dir = self.config.workspace.expanduser()
        if not project_dir.is_absolute():
            project_dir = Path.cwd() / project_dir
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
    ) -> NeMoGymResponse:
        return await self._create_response(
            body,
            model_base_url=model_base_url or self.config.model.base_url,
        )

    async def run_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
        model_ref: Optional[ModelServerRef] = None,
    ) -> AgentEpisode:
        session_id: Optional[str] = None
        session_events: list[dict[str, Any]] = []
        session_tree: OpenClawSessionTree = []
        tree_gaps: list[ObservationGap] = []
        input_items: list[Any] = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)]
            if isinstance(body.input, str)
            else list(body.input)
        )
        observed_output: list[Any] = []

        def collect(
            value: str,
            events: list[dict[str, Any]],
            tree: OpenClawSessionTree,
            gaps: list[ObservationGap],
        ) -> None:
            nonlocal session_id, session_events, session_tree, tree_gaps
            session_id = value
            session_events = events
            session_tree = tree
            tree_gaps = gaps

        def collect_output(value: list[Any]) -> None:
            observed_output.extend(value)

        response = await self._create_response(
            body,
            model_base_url=model_base_url or self.config.model.base_url,
            observation_collector=collect,
            output_collector=collect_output,
        )
        try:
            if session_tree:
                tree_inputs = []
                for invocation_id, parent_id, events in session_tree:
                    conversation = openclaw_session_conversation(
                        events,
                        input_items=input_items if parent_id is None else None,
                        fallback_output=observed_output if parent_id is None else None,
                    )
                    tree_inputs.append((invocation_id, parent_id, conversation, events))
                observations = build_openclaw_observation_tree(
                    tree_inputs,
                    model_ref=model_ref,
                )
                observations.gaps.extend(tree_gaps)
            else:
                transcript_available = any(event.get("type") == "message" for event in session_events)
                observations = build_openclaw_observations(
                    session_id or response.id,
                    openclaw_session_conversation(
                        session_events,
                        input_items=input_items,
                        fallback_output=observed_output,
                    ),
                    session_events,
                    transcript_available=transcript_available,
                    model_ref=model_ref,
                )
                if any(gap.code == "subagent_hierarchy_unavailable" for gap in tree_gaps):
                    observations.gaps = [
                        gap for gap in observations.gaps if gap.code != "subagent_hierarchy_unavailable"
                    ]
                observations.gaps.extend(tree_gaps)
        except Exception:
            LOG.exception("failed to build OpenClaw observations")
            observations = AgentObservationBundle(
                source=OPENCLAW_OBSERVATION_SOURCE,
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        return AgentEpisode(response=response, observations=observations)

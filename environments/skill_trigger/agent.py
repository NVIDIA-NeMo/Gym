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
"""Harbor adapter preserving the frozen Pi and Hermes execution profiles.

Harbor owns the container. This adapter owns only its agent processes, one
inference gateway, and the private host evidence consumed by our verifier.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Literal

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.task.task import Task
from pydantic import Field, model_validator

from environments.skill_trigger.budget import TrialBudget
from environments.skill_trigger.container import TrialContainer, command
from environments.skill_trigger.gateway import InferenceGateway
from environments.skill_trigger.scoring import Access, EvidenceError, Session, TrialEvidence, score_trial
from environments.skill_trigger.tool_results import extract_tool_results
from environments.skill_trigger.trajectory import convert_trace
from environments.skill_trigger.verifier import EvidenceStore, load_case


class SkillOptions(AgentOptions):
    release: str
    manifest_sha256: str
    evidence_root: str
    upstream_url: str
    api_key_env: str
    gateway_host: str
    gateway_port: int = Field(default=0, ge=0, le=65535)
    harness: Literal["pi", "hermes", "hermes_guidance"]
    node_binary: str = "node"
    provider: str | None = None
    api: Literal["openai-responses", "openai-completions"] | None = None

    @model_validator(mode="after")
    def validate_pi(self) -> "SkillOptions":
        if self.harness == "pi":
            if not self.provider or not self.api:
                raise ValueError("Pi requires its original provider name and API adapter")
            if self.gateway_host != "127.0.0.1":
                raise ValueError("The host-side Pi gateway must bind loopback")
        elif self.provider or self.api:
            raise ValueError("Hermes uses its frozen codex_responses API profile")
        return self


class SkillAgent(BaseAgent):
    """Run the pinned harness tools and supply actual ATIF plus host evidence."""

    options_model = SkillOptions
    capabilities = AgentCapabilities(atif=True, skills=True)

    @staticmethod
    def name() -> str:
        return "skill-trigger"

    def version(self) -> str:
        return "1.3.0" if self.options.harness == "pi" else "0.22.2"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not isinstance(environment, DockerEnvironment):
            raise EvidenceError("The frozen skill benchmark requires Harbor's Docker environment")
        if self.mcp_servers or self.load_trajectory or self.skills_dir:
            raise EvidenceError("Extra tools, restored sessions, and alternative skill mounts change this benchmark")
        if not self.model_name or not self.session_id:
            raise EvidenceError("Harbor must assign a model and fresh session before setup")
        options = self.options
        task = Task(environment.environment_dir.parent)
        self.case = load_case(
            release=Path(options.release),
            manifest_sha256=options.manifest_sha256,
            task_name=task.short_name,
            model=self.model_name,
            harness=options.harness,
        )
        if task.instruction != self.case.prompt:
            raise EvidenceError("Harbor task instruction differs from the frozen case")
        self.store = EvidenceStore(root=Path(options.evidence_root), trial_dir=environment.trial_paths.trial_dir)
        self.store.begin()
        result = await environment._run_docker_compose_command(["ps", "-q", "main"])
        self.container = TrialContainer(result.stdout.strip())
        info = json.loads(await command("docker", "inspect", self.container.id))[0]
        # Harbor's logs are deliberately mounted; private evidence and labels must not be.
        for mount in info["Mounts"]:
            source = Path(mount["Source"]).resolve()
            for hidden in (self.store.path, Path(options.release).resolve()):
                if hidden.is_relative_to(source):
                    raise EvidenceError("Container can access private evidence or the release manifest")
        metadata = await self.container.validate()
        await self.container.isolate_catalog()
        if await self.container.catalog_hashes() != self.case.mounted_hashes:
            raise EvidenceError("Staged catalog differs from the prepared release")
        self.catalog_staged_at = time.monotonic()
        # Require Harbor's supported egress enforcement, not an advisory config.
        await environment.set_network_policy(
            NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
            if options.harness == "pi"
            else NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=[options.gateway_host])
        )
        (self.store.path / "container.json").write_text(json.dumps(metadata, indent=2))
        await self.container.preserve_initial_processes()

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if instruction != self.case.prompt:
            raise EvidenceError("Released prompt differs from the frozen task")
        options = self.options
        is_pi = options.harness == "pi"
        harness = "pi" if is_pi else "hermes"
        budget = TrialBudget(model=self.model_name, root_session_id=self.session_id, preserve_payload=is_pi)

        async def root_session(payload: dict[str, object]) -> str:
            # These pinned tool profiles do not expose delegated agents.
            return self.session_id

        gateway = InferenceGateway(
            budget=budget,
            upstream_url=options.upstream_url,
            api_key=os.environ[options.api_key_env],
            evidence_dir=self.store.path / "inference",
            resolve_session=root_session,
        )
        process = observer = None
        observer_pid = None
        readers = []
        released = stopped = False
        finished = observer_stopped = 0.0
        accesses = []
        tool_starts = []
        messages = []
        completed = {}
        result_seen = False
        agent_outcome = "completed"
        outcome = "setup_error"
        stderr = (self.store.path / "process.stderr").open("xb")
        try:
            port = await gateway.start(bind_host=options.gateway_host, port=options.gateway_port)
            config = {
                "harness": options.harness,
                "model": self.model_name,
                "base_url": f"http://{options.gateway_host}:{port}/v1",
                "token": gateway.token,
                "session_id": self.session_id,
                "expected_skills": sorted(self.case.sources),
                "api_mode": "codex_responses",
            }
            if is_pi:
                config.update(
                    provider=options.provider,
                    api=options.api,
                    container=self.container.id,
                    skill_dirs=[str(path) for path in self.case.skill_directories],
                    expected_skills=list(self.case.sources),
                    prompt_file=str(self.store.path / "prompt.txt"),
                )
                config_path = self.store.path / "pi-config.json"
                config_path.write_text(json.dumps(config))
                (self.store.path / "prompt.txt").write_text(instruction)
                argv = [options.node_binary, str(Path(__file__).with_name("pi.mjs")), str(config_path)]
            else:
                # No target, scoring metadata, provider key, or host path enters /input.
                await self.container.execute("mkdir", "-p", "/input", root=True)
                for name, content in {
                    "config.json": json.dumps(config).encode(),
                    "prompt.txt": instruction.encode(),
                    "launch.py": Path(__file__).with_name("launch.py").read_bytes(),
                }.items():
                    await self.container.execute(
                        "python3",
                        "-c",
                        "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
                        "/input/" + name,
                        root=True,
                        input_bytes=content,
                    )
                probe = """import json,urllib.request
config=json.load(open('/input/config.json'))
request=urllib.request.Request(config['base_url'].removesuffix('/v1')+'/health',
 headers={'Authorization':'Bearer '+config['token']})
with urllib.request.urlopen(request,timeout=5) as response:
 if response.read()!=b'ready':raise RuntimeError('Unexpected gateway readiness reply')
"""
                await self.container.execute("python3", "-c", probe)
                argv = ["docker", "exec", "-i", "-u", "1000", self.container.id, "python3", "/input/launch.py"]
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                limit=32_000_000,
            )
            log = (self.store.path / "process.jsonl").open("xb")

            async def event() -> dict[str, object] | None:
                line = await process.stdout.readline()
                if not line:
                    return None
                log.write(line)
                log.flush()
                return json.loads(line[11:]) if line.startswith(b"EVAL_EVENT ") else {}

            async with asyncio.timeout(60):
                while True:
                    profile = await event()
                    if profile is None:
                        raise EvidenceError("Harness exited before profile construction")
                    if profile.get("kind") == "profile":
                        break
            if is_pi:
                if profile.get("version") != "1.3.0" or profile.get("skills") != list(self.case.sources):
                    raise EvidenceError("Pi profile version or catalog mismatch")
                if not isinstance(profile.get("system_prompt"), str) or not profile.get("sdk_session_id"):
                    raise EvidenceError("Pi omitted its actual system prompt or SDK session identity")
            elif profile.get("revision") != "d923a5eb487a17c0a68801bba9ef0f56c81040a8":
                raise EvidenceError("Hermes profile revision mismatch")
            observer = await self.container.start_observer(
                paths={name: f"/skills/{name}/SKILL.md" for name in self.case.sources}
            )
            ready = json.loads(await asyncio.wait_for(observer.stdout.readline(), 10))
            if ready.get("ready") is not True or ready.get("skills") != sorted(self.case.sources):
                raise EvidenceError("Independent observer did not confirm the exact catalog")
            observer_pid = ready["pid"]
            observer_ready_at = time.monotonic()

            async def collect_agent() -> None:
                nonlocal messages, result_seen, agent_outcome
                while (record := await event()) is not None:
                    if record.get("kind") == "sdk_event" and is_pi:
                        sdk_event = record.get("event")
                        if not isinstance(sdk_event, dict):
                            raise EvidenceError("Malformed Pi SDK event")
                        messages.append(sdk_event)
                        if sdk_event.get("type") == "tool_execution_end":
                            call_id = sdk_event.get("toolCallId")
                            if not isinstance(call_id, str) or call_id in completed:
                                raise EvidenceError("Missing or duplicate Pi tool completion ID")
                            completed[call_id] = budget.elapsed()
                    elif record.get("kind") == "tool_start":
                        tool_starts.append(record)
                    elif record.get("kind") == "messages":
                        if not isinstance(record.get("messages"), list):
                            raise EvidenceError("Hermes canonical messages are malformed")
                        messages = record["messages"]
                        if call_id := record.get("completed_call_id"):
                            if call_id in completed:
                                raise EvidenceError("Duplicate Hermes tool completion")
                            completed[call_id] = budget.elapsed()
                    elif record.get("kind") == "result":
                        result_seen = True
                        agent_outcome = record.get("outcome", "completed")

            async def collect_access() -> None:
                nonlocal observer_stopped
                while line := await observer.stdout.readline():
                    record = json.loads(line)
                    at = budget.elapsed()
                    if record == {"observer_stopped": True}:
                        observer_stopped = at
                        return
                    if record.get("event") != "IN_ACCESS" or record.get("skill") not in self.case.sources:
                        raise EvidenceError("Unknown independent observation event")
                    accesses.append(Access(skill=record["skill"], at=at))
                raise EvidenceError("Independent observation ended without a stop marker")

            released_at = budget.release()
            released = True
            readers = [asyncio.create_task(collect_agent()), asyncio.create_task(collect_access())]
            process.stdin.write(b"run\n")
            await process.stdin.drain()
            if not is_pi:
                process.stdin.close()
            outcome = "completed"
            while process.returncode is None and not gateway.stopped.is_set():
                for reader in readers:
                    if reader.done():
                        reader.result()
                if observer.returncode is not None:
                    raise EvidenceError("Observer exited during model execution")
                if budget.elapsed() >= 120:
                    outcome = "timeout"
                    break
                await asyncio.sleep(0.01)
            if is_pi and process.returncode is None:
                process.stdin.write(b"stop\n")
                await process.stdin.drain()
                process.stdin.close()
            await self.container.stop_work()
            stopped = True
            await asyncio.wait_for(process.wait(), 10)
            await readers[0]
            finished = budget.elapsed()
            await gateway.close()
            outcome = gateway.outcome or (agent_outcome if outcome == "completed" else outcome)
            if outcome == "completed" and finished > 120:
                outcome = "timeout"
            if outcome == "completed" and (process.returncode != 0 or not result_seen):
                outcome = "trace_error"
            await self.container.stop_observer(pid=observer_pid)
            await asyncio.wait_for(readers[1], 10)
            await asyncio.wait_for(observer.wait(), 10)
            if observer.returncode != 0:
                raise EvidenceError("Independent observer failed")
            outputs = extract_tool_results(
                tool_starts + messages,
                harness=harness,
                session_id=self.session_id,
                completed_at=completed,
                skill_paths={name: f"/skills/{name}/SKILL.md" for name in self.case.sources},
            )
            evidence = TrialEvidence(
                identity=self.case.identity,
                root_session_id=self.session_id,
                sessions=[Session(id=self.session_id, model=self.model_name)],
                requests=budget.requests,
                catalog_staged_at=self.catalog_staged_at - released_at,
                observer_ready_at=observer_ready_at - released_at,
                observer_stopped_at=observer_stopped,
                finished_at=finished,
                active_work_stopped=stopped,
                outcome=outcome,
                accesses=accesses,
                outputs=outputs,
            )
            self.store.finish(evidence)
            score = score_trial(
                evidence, expected=self.case.identity, sources=self.case.sources, target=self.case.target
            )
            trace = self.store.path / "canonical.jsonl"
            trace.write_text(
                "".join(json.dumps(event) + "\n" for event in messages)
                if is_pi
                else json.dumps({"id": self.session_id, "messages": messages}) + "\n"
            )
            trajectory = convert_trace(
                trace,
                harness=harness,
                session_id=self.session_id,
                model=self.model_name,
                system_prompt=profile["system_prompt"] if is_pi else None,
            )
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "trajectory.json").write_text(trajectory.model_dump_json(indent=2))
            context.metadata = {"outcome": outcome, "opened_intended_skill": bool(score.reward)}
        except BaseException as error:
            # This survives Harbor deleting its failed result.json. It is never a reward.
            (self.store.path / "failure.json").write_text(
                json.dumps(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "released": released,
                    }
                )
            )
            raise
        finally:
            try:
                if process is not None and not stopped:
                    await self.container.stop_work()
                if process is not None:
                    if is_pi and process.returncode is None:
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                if observer is not None and observer.returncode is None and observer_pid is not None:
                    await self.container.stop_observer(pid=observer_pid)
                if observer is not None:
                    await asyncio.wait_for(observer.wait(), 10)
            finally:
                await gateway.close()
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                stderr.close()
                if "log" in locals():
                    log.close()

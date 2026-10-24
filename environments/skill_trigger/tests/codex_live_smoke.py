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
"""Check live Codex session admission using scripted replies and no network.

This is a runtime-component check, not a scored Harbor trial. A pipe connects
host admission to the scripted provider inside the container; the HTTP gateway,
real inference, and frozen task packaging are deliberately not exercised here.
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from environments.skill_trigger.budget import TrialBudget
from environments.skill_trigger.codex_runtime import CodexRuntime
from environments.skill_trigger.container import TrialContainer, command


async def smoke(*, image: str, output: Path, fork: bool) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, evidence, private = output / "input", output / "evidence", output / "private"
    for folder in [inputs, evidence, private]:
        folder.mkdir()
    evidence.chmod(0o777)  # synthetic fixture outputs only; host verifier state is separate
    private.chmod(0o700)
    skill = output / "skills/synthetic"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: synthetic\ndescription: Synthetic task\n---\nCount table rows.")
    shutil.copy(Path(__file__).parents[1] / "launch.py", inputs / "launch.py")
    (inputs / "prompt.txt").write_text("Use the relevant skill for a table task.")
    (inputs / "config.json").write_text(
        json.dumps(
            {
                "harness": "codex",
                "model": "test-model",
                "base_url": "http://127.0.0.1:18999/v1",
                "token": "synthetic-not-a-key",
                "expected_skills": ["synthetic"],
                "fixture_fork": fork,
                "fixture_host_admission": True,
            }
        )
    )
    container = runtime = process = None
    tasks = []
    report = {
        "kind": "codex-live-admission-smoke",
        "model_inference": False,
        "scored_trials": 0,
        "fork_context": fork,
        "status": "failed",
    }
    try:
        cid = (
            (
                await command(
                    "docker",
                    "run",
                    "-d",
                    "--network",
                    "none",
                    "--user",
                    "1000",
                    "--cpus",
                    "0.5",
                    "--memory",
                    "768m",
                    "--pids-limit",
                    "256",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "-w",
                    "/workspace",
                    "-v",
                    f"{inputs}:/input:ro",
                    "-v",
                    f"{skill.parent}:/skills:ro",
                    "-v",
                    f"{evidence}:/evidence",
                    "-v",
                    f"{Path(__file__).with_name('codex_capture.py')}:/fixture.py:ro",
                    "--entrypoint",
                    "sleep",
                    image,
                    "infinity",
                )
            )
            .decode()
            .strip()
        )
        container = TrialContainer(cid)
        report["container"] = await container.validate()
        await container.preserve_initial_processes()
        budget = TrialBudget(model="test-model", root_session_id="harbor-placeholder", preserve_payload=False)
        runtime = CodexRuntime(container=container, budget=budget, evidence_dir=private, model="test-model")
        await runtime.start()
        budget.release()
        runtime.reader = asyncio.create_task(runtime.collect())
        with (output / "fixture.stderr").open("wb") as errors:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "-i",
                "-u",
                "1000",
                cid,
                "python3",
                "/fixture.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errors,
                limit=32_000_000,
            )

            async def admit(record):
                try:
                    sid = await asyncio.wait_for(runtime.resolve_request(record["payload"]), 10)
                    async with budget.forward(session_id=sid, endpoint="/responses", payload=record["payload"]):
                        process.stdin.write((json.dumps({"index": record["index"], "allowed": True}) + "\n").encode())
                        await process.stdin.drain()
                except BaseException:
                    process.stdin.write((json.dumps({"index": record["index"], "allowed": False}) + "\n").encode())
                    await process.stdin.drain()
                    raise

            async with asyncio.timeout(70):
                while line := await process.stdout.readline():
                    record = json.loads(line)
                    if record["kind"] == "request":
                        tasks.append(asyncio.create_task(admit(record)))
                    elif record["kind"] == "cli" and not record["line"].startswith("EVAL_EVENT "):
                        runtime.accept_cli(json.loads(record["line"]))
                await asyncio.gather(*tasks)
                await process.wait()
            if process.returncode != 0:
                raise RuntimeError("Scripted Codex process failed")
        await container.stop_work()
        await runtime.stop()
        root, sessions, spawns, outputs = runtime.evidence(skill_paths={})
        trajectory = runtime.trajectory()
        assert len(budget.requests) == 3 and len(sessions) == 2 and len(spawns) == 1
        assert {r.session_id for r in budget.requests} == {s.id for s in sessions}
        assert trajectory.trajectory_id == root and len(trajectory.subagent_trajectories) == 1
        (output / "trajectory.json").write_text(trajectory.model_dump_json(indent=2))
        report.update(
            status="passed",
            requests=[r.model_dump() for r in budget.requests],
            sessions=[s.model_dump() for s in sessions],
            spawns=[s.model_dump() for s in spawns],
            embedded_trajectories=len(trajectory.subagent_trajectories),
            returned_reads=len(outputs),
        )
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            if container:
                await container.stop_work()
            if runtime:
                await runtime.stop()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            if container:
                await command("docker", "rm", "-f", container.id)
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fork", action="store_true")
    asyncio.run(smoke(**vars(parser.parse_args())))

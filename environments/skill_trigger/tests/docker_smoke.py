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
"""Exercise Linux observation/isolation/cleanup using synthetic instructions only.

Run explicitly with --image naming an existing Python-containing Docker image.
This does not invoke a model or validate the published benchmark's task scores.
Only containers created by this script are removed, including on failure.
"""

import argparse
import asyncio
import json
import platform
import uuid
from pathlib import Path

from environments.skill_trigger.container import TrialContainer, command


async def smoke(*, image: str, output: Path) -> None:
    if output.exists():
        raise ValueError("Use a new output file")
    containers = []
    observers = []
    processes = []
    report = {
        "kind": "synthetic-docker-smoke",
        "model_inference": False,
        "checks": [],
        "status": "failed",
        "control_python": platform.python_version(),
    }
    try:
        for _ in range(2):
            cid = (
                (
                    await command(
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        "skill-trigger-smoke-" + uuid.uuid4().hex[:12],
                        "--network",
                        "none",
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
                        "--user",
                        "1000",
                        "--entrypoint",
                        "sleep",
                        image,
                        "infinity",
                    )
                )
                .decode()
                .strip()
            )
            runtime = TrialContainer(cid)
            containers.append(runtime)
            report["environment"] = await runtime.validate()
            await runtime.execute(
                "python3",
                "-c",
                "from pathlib import Path; p=Path('/skills/target'); "
                "p.mkdir(parents=True); (p/'SKILL.md').write_text('Synthetic instructions for observation.')",
                root=True,
            )
            await runtime.isolate_catalog()
            hashes = await runtime.catalog_hashes()
            assert set(hashes) == {"target"}
            await runtime.preserve_initial_processes()
            observer = await runtime.start_observer(paths={"target": "/skills/target/SKILL.md"})
            observers.append((runtime, observer, None))
            ready = json.loads(await asyncio.wait_for(observer.stdout.readline(), 5))
            assert ready["ready"] and ready["skills"] == ["target"]
            observers[-1] = (runtime, observer, ready["pid"])
        report["checks"].append("frozen container profile and independent observer startup")

        # Reading the first trial must not trigger the second trial's observer.
        await containers[0].execute("cat", "/skills/target/SKILL.md")
        first = json.loads(await asyncio.wait_for(observers[0][1].stdout.readline(), 5))
        assert first == {"event": "IN_ACCESS", "skill": "target"}
        try:
            await asyncio.wait_for(observers[1][1].stdout.readline(), 0.2)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Read leaked across trial observers")
        report["checks"].append("per-trial file observation does not cross containers")

        # The agent user cannot edit instructions or stop the root observer.
        for command_args in [
            ("python3", "-c", "open('/skills/target/SKILL.md','a').write('modified')"),
            ("python3", "-c", "import os,signal,sys; os.kill(int(sys.argv[1]),signal.SIGTERM)", str(observers[0][2])),
        ]:
            try:
                await containers[0].execute(*command_args)
            except RuntimeError as error:
                assert "PermissionError" in str(error), str(error)
            else:
                raise AssertionError("Agent modified protected observation state")
        report["checks"].append("agent cannot modify instructions or stop root observer")

        # Kill both a shell and its child while preserving container keepalive.
        running = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-u",
            "1000",
            containers[0].id,
            "sh",
            "-c",
            "sleep 300 & wait",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(running)
        # Wait for the child to exist instead of trusting an arbitrary sleep.
        check = "import pathlib; print(any(p.is_file() and b'sleep\\x00300' in p.read_bytes() for p in pathlib.Path('/proc').glob('[0-9]*/cmdline')))"
        async with asyncio.timeout(5):
            while (await containers[0].execute("python3", "-c", check)).strip() != b"True":
                await asyncio.sleep(0.02)
        await containers[0].stop_work()
        await asyncio.wait_for(running.wait(), 5)
        assert running.returncode != 0
        assert (
            await command("docker", "inspect", "--format", "{{.State.Running}}", containers[0].id)
        ).strip() == b"true"
        report["checks"].append("agent shell and child stop; Harbor keepalive remains")

        for runtime, observer, pid in observers:
            await runtime.stop_observer(pid=pid)
            stdout, stderr = await asyncio.wait_for(observer.communicate(), 5)
            assert observer.returncode == 0, stderr.decode(errors="replace")
            events = [json.loads(line) for line in stdout.splitlines()]
            assert events[-1] == {"observer_stopped": True}
        report["checks"].append("observers stop cleanly after work ends")
        report["status"] = "passed"
    finally:
        cleanup_errors = []
        for runtime in containers:
            try:
                await command("docker", "rm", "-f", runtime.id)
            except Exception as error:
                cleanup_errors.append(str(error))
        for _, observer, _ in observers:
            await asyncio.wait_for(observer.wait(), 5)
        for process in processes:
            await asyncio.wait_for(process.wait(), 5)
        report["containers_removed"] = len(containers) - len(cleanup_errors)
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["status"] = "cleanup_failed"
        output.write_text(json.dumps(report, indent=2) + "\n")
        if cleanup_errors:
            raise RuntimeError("Failed to remove smoke-test containers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(smoke(image=arguments.image, output=arguments.output))

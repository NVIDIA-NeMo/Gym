# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode execution on a resources-owned environment (handoff version 1)."""

import json
import os
from shlex import quote

from nemo_gym.openai_utils import NeMoGymResponseUsage
from nemo_gym.sandbox.agent import artifact_directory, empty_response, run_borrowed
from nemo_gym.sandbox.handoff import AgentTermination
from responses_api_agents.opencode_sandboxed_agent.app import parse_opencode_observations


async def run(agent, request, body):
    data_home = None
    config = await agent._create_opencode_config(request)
    for settings in config.get("agent", {}).values():
        if isinstance(settings, dict) and settings.get("steps") is None:
            settings.pop("steps", None)

    async def setup(sandbox, seed):
        nonlocal data_home
        data_home = f"/tmp/{seed.session_id}-opencode"
        mcp = {}
        for server in seed.mcp_servers:
            mcp[server["name"]] = (
                {"type": "local", "command": [server["command"], *server.get("args", [])]}
                if server["transport"] == "stdio"
                else {"type": "remote", "url": server["url"], "oauth": False}
            )
        config["mcp"] = {**config.get("mcp", {}), **mcp}
        command = (
            "set -e; "
            "if ! command -v curl >/dev/null || ! command -v unzip >/dev/null || ! command -v setsid >/dev/null; then "
            "if command -v apt-get >/dev/null; then apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl unzip util-linux ca-certificates; "
            "elif command -v apk >/dev/null; then apk add --no-cache curl unzip util-linux ca-certificates; "
            "elif command -v dnf >/dev/null; then dnf install -y curl unzip util-linux ca-certificates; "
            "else echo 'Missing curl/unzip/setsid and no supported package manager' >&2; exit 1; fi; fi; "
            "export PATH=$HOME/.opencode/bin:$PATH; "
            f'if ! command -v opencode >/dev/null || [ "$(opencode --version)" != {quote(agent.config.opencode_version)} ]; '
            'then installer=$(mktemp); curl -fsSL https://opencode.ai/install -o "$installer"; '
            f'VERSION={quote(agent.config.opencode_version)} bash "$installer"; rm -f "$installer"; fi; '
            "mkdir -p ~/.config/opencode/skills"
        )
        if seed.skills_dir:
            command += f"; cp -R {quote(seed.skills_dir)}/. ~/.config/opencode/skills/"
        result = await sandbox.exec("bash -c " + quote(command), user=seed.user, timeout_s=seed.setup_timeout_sec)
        directory = artifact_directory(agent.config.name, seed.session_id)
        (directory / "setup-stdout.txt").write_text(result.stdout or "")
        (directory / "setup-stderr.txt").write_text(result.stderr or "")
        if result.return_code or result.error_type:
            raise RuntimeError(f"OpenCode setup failed: {result.stderr}")

    async def execute(sandbox, seed, budget):
        directory = artifact_directory(agent.config.name, seed.session_id)
        env = {
            "XDG_DATA_HOME": data_home,
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "1000000000",
        }
        env.update({name: os.environ[name] for name in agent.config.opencode_env_from_process})
        # A separate process group lets the resources owner stop remaining agent
        # children before collecting either main or sidecar evidence.
        command = (
            f"echo $$ >> /tmp/{seed.session_id}.pids; export PATH=$HOME/.opencode/bin:$PATH; "
            f"exec opencode run --format json -- {quote(seed.instruction)}"
        )
        result = await sandbox.exec(
            "setsid --wait bash -c " + quote(command),
            env=env,
            user=seed.user,
            timeout_s=budget,
        )
        (directory / "stdout.txt").write_text(result.stdout or "")
        (directory / "stderr.txt").write_text(result.stderr or "")
        termination = AgentTermination(
            reason=(
                "timeout"
                if result.error_type == "timeout"
                else "infrastructure_error"
                if result.error_type
                else "completed"
                if result.return_code == 0
                else "nonzero_exit"
            ),
            exit_code=result.return_code,
            detail=result.error_type,
        )
        response = empty_response(body.responses_create_params, agent.config.model_server.name)
        exported = False
        extra = {}
        try:
            events = []
            for line in (result.stdout or "").splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
            session_id = next(event["sessionID"] for event in events if event.get("sessionID"))
            remote_export = f"/tmp/{seed.session_id}-export.json"
            exported_result = await sandbox.exec(
                "bash -c "
                + quote(
                    f"export PATH=$HOME/.opencode/bin:$PATH; opencode export {quote(session_id)} > {remote_export}"
                ),
                env=env,
                user=seed.user,
            )
            (directory / "export-stderr.txt").write_text(exported_result.stderr or "")
            if exported_result.return_code:
                raise RuntimeError(f"OpenCode export exited {exported_result.return_code}")
            await sandbox.download(remote_export, directory / "export.json")
            export = json.loads((directory / "export.json").read_text())
            response.output = agent._opencode_export_to_output_items(export)[1:]
            response.usage = NeMoGymResponseUsage.sum_from_list(agent._opencode_export_to_usages(export))
            exported = True
        except Exception as exc:
            (directory / "missing-export.txt").write_text(f"{type(exc).__name__}: {exc}")
        rollout = agent.rollout_id_from_run(body)
        if rollout:
            try:
                source = f"{data_home}/opencode/opencode.db"
                snapshot = f"{data_home}/opencode/gym-observations.db"
                script = (
                    "import sqlite3,sys;"
                    "source=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True);"
                    "destination=sqlite3.connect(sys.argv[2]);"
                    "source.backup(destination);destination.close();source.close()"
                )
                result = await sandbox.exec(
                    f"python3 -c {quote(script)} {quote(source)} {quote(snapshot)}", user=seed.user
                )
                if result.return_code or result.error_type:
                    raise RuntimeError("OpenCode observation snapshot failed")
                await sandbox.download(snapshot, directory / "opencode.db")
                extra["ng_agent_observations"] = parse_opencode_observations(
                    directory / "opencode.db",
                    rollout,
                ).model_dump(mode="json")
            except Exception as exc:
                (directory / "missing-observations.txt").write_text(f"{type(exc).__name__}: {exc}")
        termination.artifacts = [str(directory)]
        extra |= {
            "opencode_results_fpath": str(directory / "export.json") if exported else "",
            "opencode_run_stdout": result.stdout or "",
            "opencode_run_stderr": result.stderr or "",
            "opencode_finished": termination.reason == "completed",
            "opencode_export_found": exported,
        }
        return response, termination, extra

    return await run_borrowed(agent, request, body, setup=setup, execute=execute)

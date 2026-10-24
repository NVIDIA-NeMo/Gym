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
"""Container-side launch profiles for the frozen Codex and Hermes conditions.

The host stages a JSON configuration and prompt under /input, then waits for
the profile event before starting its observer and sending 'run' on stdin.
Only a trial-scoped gateway token belongs in this configuration, never a
provider credential. Targets and scoring data are not inputs to this driver.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


HERMES_REVISION = "d923a5eb487a17c0a68801bba9ef0f56c81040a8"


def emit(kind: str, **values: object) -> None:
    print("EVAL_EVENT " + json.dumps({"kind": kind, **values}, default=str), flush=True)


def authorize_prompt() -> None:
    if sys.stdin.readline().strip() != "run":
        raise RuntimeError("Host observer did not authorize prompt release")


def codex_profile(config: dict[str, object], *, root: Path) -> tuple[Path, Path, str]:
    """Construct the original native profile; guidance requires captured metadata.

    A supplied model catalog must be prepared from the pinned effective baseline
    with only include_skills_usage_instructions changed. Comparing complete wire
    requests in the migration pilot remains a separate required gate.
    """
    home, codex_home = root / "home", root / "config"
    profile = f"""model = {json.dumps(config["model"])}
model_provider = "benchmark"
model_reasoning_effort = "medium"
web_search = "disabled"
[model_providers.benchmark]
name = "Benchmark gateway"
base_url = {json.dumps(config["base_url"])}
env_key = "BENCHMARK_TOKEN"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
[skills]
max_context_tokens = 10000
[skills.bundled]
enabled = false
"""
    catalog = config.get("model_catalog")
    guidance = config["harness"] == "codex_guidance"
    if guidance and catalog is None:
        raise ValueError("Codex guidance requires the pinned effective model catalog")
    if catalog is not None:
        if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
            raise ValueError("Invalid Codex model catalog")
        matches = [m for m in catalog["models"] if isinstance(m, dict) and m.get("slug") == config["model"]]
        if len(matches) != 1 or matches[0].get("include_skills_usage_instructions") is not guidance:
            raise ValueError("Model catalog does not match the declared guidance condition")
        profile = "model_catalog_json = " + json.dumps(str(codex_home / "models.json")) + "\n" + profile
    return home, codex_home, profile


def launch_codex(config: dict[str, object], *, prompt: str) -> None:
    """Exec pinned Codex after profile validation and host prompt release."""
    version = subprocess.check_output(["codex", "--version"], text=True).strip()
    if version != "codex-cli 0.156.1":
        raise RuntimeError(f"Unexpected Codex version: {version}")
    home, codex_home, profile = codex_profile(config, root=Path("/home/agent/native-codex"))
    (home / ".agents").mkdir(parents=True, exist_ok=False)
    codex_home.mkdir(parents=True, exist_ok=False)
    (home / ".agents/skills").symlink_to("/skills", target_is_directory=True)
    (codex_home / "config.toml").write_text(profile)
    if config.get("model_catalog") is not None:
        (codex_home / "models.json").write_text(json.dumps(config["model_catalog"]))
    names = sorted(p.name for p in Path("/skills").iterdir() if (p / "SKILL.md").is_file())
    if names != config["expected_skills"]:
        raise RuntimeError("Codex skill catalog differs from the frozen task")
    emit(
        "profile",
        mode="codex-exec",
        version=version,
        skills=names,
        api_mode="responses",
        guidance=config["harness"] == "codex_guidance",
    )
    authorize_prompt()
    env = dict(os.environ, HOME=str(home), CODEX_HOME=str(codex_home), BENCHMARK_TOKEN=config["token"])
    command = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--model",
        config["model"],
        "--json",
        "--enable",
        "unified_exec",
        "--",
        prompt,
    ]
    os.chdir("/workspace")
    os.execvpe(command[0], command, env)


def hermes_profile(*, guidance: bool) -> dict[str, object]:
    """Disable auxiliary inference while retaining Hermes' native model gates."""
    agent = {"api_max_retries": 1, "auto_recovery_cycles": 0}
    if guidance:
        agent["tool_use_enforcement"] = True
    return {
        "agent": agent,
        "compression": {"enabled": False, "micro_compact": False},
        "auxiliary": {"title_generation": {"enabled": False, "model_upgrade_enabled": False}},
    }


def launch_hermes(config: dict[str, object], *, prompt: str) -> None:
    """Run the pinned programmatic agent with its original native tool profile."""
    revision = subprocess.check_output(
        ["git", "-c", "safe.directory=/opt/hermes", "-C", "/opt/hermes", "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != HERMES_REVISION:
        raise RuntimeError("Hermes checkout differs from the pinned revision")
    os.environ.pop("HERMES_SINGLE_QUERY_SESSION", None)
    home = Path("/tmp/native-hermes")
    os.environ["HERMES_HOME"] = str(home)
    home.mkdir(parents=True, exist_ok=False)
    (home / "skills").symlink_to("/skills", target_is_directory=True)
    guidance = config["harness"] == "hermes_guidance"
    (home / "config.yaml").write_text(json.dumps(hermes_profile(guidance=guidance)))
    # Imported only inside the pinned Hermes runtime, after its home is set.
    from agent.prompt_builder import EXECUTION_GUIDANCE_MODELS, TOOL_USE_ENFORCEMENT_MODELS
    from agent.system_prompt import _model_gate
    from agent.title_generator import _auto_title_enabled, _model_title_upgrade_enabled
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.skills_tool import skills_list

    if _auto_title_enabled() or _model_title_upgrade_enabled():
        raise RuntimeError("Hermes title generation must be disabled")
    database = SessionDB(db_path=home / "session.db")

    def completed(call_id, name, args, result):
        # Read the committed canonical messages, not the callback's display text.
        emit(
            "messages", completed_call_id=call_id, messages=database.get_messages_as_conversation(config["session_id"])
        )

    catalog = json.loads(skills_list())
    if not catalog.get("success") or sorted(s["name"] for s in catalog["skills"]) != config["expected_skills"]:
        raise RuntimeError("Hermes skill catalog differs from the frozen task")
    agent = AIAgent(
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["token"],
        provider="custom",
        api_mode=config["api_mode"],
        max_iterations=100,
        max_tokens=8192,
        reasoning_config={"effort": "medium"},
        enabled_toolsets=["terminal", "file", "skills"],
        skip_context_files=True,
        load_soul_identity=False,
        skip_memory=True,
        skip_background_review=True,
        quiet_mode=True,
        session_id=config["session_id"],
        session_db=database,
        cwd="/workspace",
        tool_start_callback=lambda call_id, name, args: emit("tool_start", call_id=call_id, name=name, args=args),
        tool_complete_callback=completed,
    )
    if agent.api_mode != config["api_mode"] or agent._api_max_retries != 1 or agent._auto_recovery_cycles != 0:
        raise RuntimeError("Hermes inference/retry settings differ from the frozen profile")
    if agent.compression_enabled or agent.context_compressor._micro_compact_enabled:
        raise RuntimeError("Hermes auxiliary compression must be disabled")
    if guidance and agent._tool_use_enforcement is not True:
        raise RuntimeError("Hermes guidance override was not applied")
    if agent._execution_guidance != "auto":
        raise RuntimeError("Hermes execution guidance must retain its native setting")
    emit(
        "profile",
        mode="native-programmatic-standard-agent",
        revision=revision,
        skills=catalog,
        tool_use_enforcement=agent._tool_use_enforcement,
        tool_use_enforcement_effective=_model_gate(
            agent._tool_use_enforcement, agent.model, TOOL_USE_ENFORCEMENT_MODELS
        ),
        execution_guidance=agent._execution_guidance,
        execution_guidance_effective=_model_gate(agent._execution_guidance, agent.model, EXECUTION_GUIDANCE_MODELS),
        tools=agent.tools,
        api_mode=agent.api_mode,
    )
    authorize_prompt()
    result = agent.run_conversation(prompt)
    emit("messages", id=config["session_id"], messages=database.get_messages_as_conversation(config["session_id"]))
    emit("result", result=result)


def main() -> None:
    config = json.loads(Path("/input/config.json").read_text())
    prompt = Path("/input/prompt.txt").read_text()
    if config.get("harness") in {"codex", "codex_guidance"}:
        launch_codex(config, prompt=prompt)
    elif config.get("harness") in {"hermes", "hermes_guidance"}:
        launch_hermes(config, prompt=prompt)
    else:
        raise ValueError("Expected a Codex or Hermes benchmark condition")


if __name__ == "__main__":
    main()

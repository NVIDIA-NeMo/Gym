# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenClaw integration for the SWE-bench wrapper.

Three pieces:
  - OPENCLAW_RUNNER_SCRIPT: the in-container runner. Copies the openclaw HOME
    template (mounted at /openclaw_setup/home_template) to a per-instance
    writable location, patches model endpoint config, runs ``openclaw agent
    --local --json``, captures the git diff and the session JSONL.
  - OpenClawHarnessProcessor: sibling of OpenHandsHarnessProcessor. ``setup()``
    invokes ``setup_scripts/openclaw.sh`` inside ``apptainer exec <SIF>`` to
    populate a host-side setup dir (Node toolchain + openclaw npm install +
    workspace template). ``get_run_command()`` writes the runner to the
    persistent dir and returns an ExecuteContainerCommandArgs that invokes
    it inside the rollout SIF.
  - RunOpenClawAgent: subclass of RunOpenHandsAgent. Overrides
    ``_openhands_dir_copy_from_host`` (simpler — openclaw writes directly to
    /trajectories_mount), ``_extract_patch_and_eval_input`` (different schema),
    and ``_extract_agent_error`` (reads ``timed_out`` from the openclaw runner).

OpenClaw never executes outside a container: the setup script runs inside
``apptainer exec <openclaw_setup_sif>``, and rollouts run inside the per-rollout
SWE-bench SIF with the populated host setup dir mounted ro at /openclaw_setup.
"""

import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nemo_gym.server_utils import get_server_url
from responses_api_agents.swe_agents.app import (
    BaseDatasetHarnessProcessor,
    ExecuteContainerCommandArgs,
    RunOpenHandsAgent,
)


# ---------------------------------------------------------------------------
# In-container runner script
# ---------------------------------------------------------------------------
#
# Triple-quoted source for an executable Python script that runs inside the
# SWE-bench Apptainer container. The runner has no dependency on Gym — it only
# needs Python 3, the `openclaw` CLI on PATH, and the HOME template mounted at
# /openclaw_setup/home_template (the bash wrapper prepends
# /openclaw_setup/{node,openclaw/node_modules/.bin}/bin onto PATH).

OPENCLAW_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""
Standalone openclaw SWE runner for inside Apptainer containers.

1. Copies /openclaw_setup/home_template/.openclaw to a per-instance writable HOME.
2. Rewrites openclaw.json (workspace dir) and models.json (provider URL+key+model id).
3. Runs `openclaw agent --local --agent main --json --session-id <uuid>` against /testbed.
4. Captures the patch via `git diff` at /testbed and the session JSONL trajectory.
5. Writes a SWE-bench-format prediction file plus a sibling .trajectory.jsonl.

Always exits 0. Rollout success/failure is judged downstream from the patch
and the SWE-bench evaluation harness.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def _default_model_entry(model_id: str) -> dict:
    return {
        "id": model_id, "name": "policy", "api": "openai-responses",
        "input": ["text"], "reasoning": False,
    }


def _patch_first_provider(providers: dict, base_url: str, api_key: str, model_id: str) -> str:
    """Update the first provider's baseUrl, apiKey, and first model id in place.

    Returns the provider key for downstream use (e.g. building the primary alias).
    Caller must have already verified `providers` is non-empty.
    """
    pkey = next(iter(providers))
    providers[pkey]["baseUrl"] = base_url
    providers[pkey]["apiKey"] = api_key
    models_list = providers[pkey].setdefault("models", [])
    if not models_list:
        models_list.append(_default_model_entry(model_id))
    else:
        models_list[0]["id"] = model_id
    return pkey


def _patch_openclaw_json(path: Path, runtime_workspace: Path,
                         base_url: str, api_key: str, model_id: str) -> None:
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("agents", {}).setdefault("defaults", {})
    cfg["agents"]["defaults"]["workspace"] = str(runtime_workspace)
    providers = cfg.setdefault("models", {}).setdefault("providers", {})
    if not providers:
        providers["vllm"] = {
            "baseUrl": base_url, "apiKey": api_key, "api": "openai-responses",
            "models": [_default_model_entry(model_id)],
        }
    else:
        _patch_first_provider(providers, base_url, api_key, model_id)
    primary = f"{next(iter(providers))}/{model_id}"
    cfg["agents"]["defaults"]["model"] = {"primary": primary}
    cfg["agents"]["defaults"]["models"] = {primary: {"alias": "policy"}}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _patch_models_json(path: Path, base_url: str, api_key: str, model_id: str) -> None:
    if not path.exists():
        return
    with open(path) as f:
        cfg = json.load(f)
    if not (cfg.get("providers") or {}):
        return
    _patch_first_provider(cfg["providers"], base_url, api_key, model_id)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _ensure_compaction_disabled(path: Path) -> None:
    """Write or merge `compaction.enabled = false` into pi-coding-agent's settings.json.

    Pi defaults compaction to enabled; openclaw's auto-disable hook only fires
    when a custom context engine claims `ownsCompaction`, which the legacy
    default does not. Without this, a long-running rollout will silently
    summarize earlier turns mid-task — losing fine-grained context (degrades
    eval quality on long instances) and breaking per-turn token-id contiguity
    (a hard requirement for RL training).
    """
    cfg = {}
    if path.exists():
        try:
            with open(path) as f:
                cfg = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            cfg = {}
    cfg.setdefault("compaction", {})["enabled"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--problem-statement-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model-api-key", default="***")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--openclaw-home-template", default="/openclaw_setup/home_template")
    parser.add_argument("--openclaw-bin", default="openclaw")
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--testbed", default="/testbed")
    args = parser.parse_args()

    instance_id = args.instance_id
    template_root = Path(args.openclaw_home_template)
    if not (template_root / ".openclaw").is_dir():
        sys.stderr.write(f"openclaw template not found at {template_root}/.openclaw\n")
        sys.exit(0)

    runtime_home = Path(f"/tmp/oc_{instance_id}")
    if runtime_home.exists():
        shutil.rmtree(runtime_home)
    runtime_home.mkdir(parents=True)
    # TODO(perf): if the template grows large, replace this full copytree with
    # selective symlinks for read-only entries (everything except the two
    # files we mutate plus agents/main/sessions/ which openclaw writes to).
    shutil.copytree(template_root / ".openclaw", runtime_home / ".openclaw", symlinks=True)

    workspace = runtime_home / ".openclaw" / "workspace"
    _patch_openclaw_json(
        runtime_home / ".openclaw" / "openclaw.json", workspace,
        args.model_base_url, args.model_api_key, args.model_id,
    )
    agent_dir = runtime_home / ".openclaw" / "agents" / "main" / "agent"
    _patch_models_json(
        agent_dir / "models.json",
        args.model_base_url, args.model_api_key, args.model_id,
    )
    # Defensive: ensure pi-coding-agent compaction is disabled even if the
    # template doesn't ship a settings.json.
    _ensure_compaction_disabled(agent_dir / "settings.json")

    problem = Path(args.problem_statement_file).read_text()
    user_msg = (
        f"The current working directory is {args.testbed}.\n"
        f"There is a bug to fix in this repository.\n\n"
        f"## Problem Statement\n\n{problem}\n\n"
        f"Please fix the bug described above. Do NOT modify test files unless "
        f"the problem statement specifically requires it. Do NOT commit. Stop when done."
    )

    session_id = str(uuid.uuid4())
    session_file = (
        runtime_home / ".openclaw" / "agents" / "main" / "sessions"
        / f"{session_id}.jsonl"
    )

    env = os.environ.copy()
    env["HOME"] = str(runtime_home)

    cmd = [
        args.openclaw_bin, "agent",
        "--local", "--agent", "main", "--json",
        "--session-id", session_id,
        "--timeout", str(args.timeout),
        "-m", user_msg,
    ]

    print(f"[runner] HOME={runtime_home} session_id={session_id}", flush=True)
    timed_out = False
    stderr = ""
    returncode = -1
    # We don't read the agent's stdout — the trajectory is in the session
    # JSONL file. Discard stdout to avoid buffering MBs of text in memory.
    try:
        proc = subprocess.run(
            cmd, cwd=args.testbed,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout + 60,
            env=env,
        )
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        if e.stderr:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else e.stderr
    except FileNotFoundError as e:
        sys.stderr.write(f"[runner] openclaw binary not found: {e}\n")

    patch = None
    # On timeout the agent was killed mid-edit — any in-progress diff is partial
    # and should not be scored. The downstream pipeline treats model_patch=None
    # as "no patch produced".
    if not timed_out:
        try:
            gd = subprocess.run(
                ["git", "diff"], cwd=args.testbed,
                capture_output=True, text=True, timeout=30,
            )
            if gd.stdout.strip():
                patch = gd.stdout if gd.stdout.endswith("\n") else gd.stdout + "\n"
        except Exception as e:
            sys.stderr.write(f"[runner] git diff failed: {e}\n")

    output = {
        "model_name_or_path": args.model_id,
        "instance_id": instance_id,
        "model_patch": patch,
        "timed_out": timed_out,
    }
    Path(args.output_file).write_text(json.dumps(output))

    traj_dest = Path(args.output_file).with_suffix(".trajectory.jsonl")
    if session_file.exists():
        shutil.copy2(session_file, traj_dest)
        print(f"[runner] trajectory copied to {traj_dest}", flush=True)
    else:
        sessions_dir = runtime_home / ".openclaw" / "agents" / "main" / "sessions"
        existing = sorted(p.name for p in sessions_dir.glob("*")) if sessions_dir.exists() else []
        sys.stderr.write(
            f"[runner] WARN: session file missing at {session_file}; "
            f"sessions dir contents: {existing}\n"
        )

    print(
        f"[runner] returncode={returncode} timed_out={timed_out} "
        f"patch={'yes' if patch else 'no'}",
        flush=True,
    )
    if returncode != 0 and stderr:
        sys.stderr.write(stderr[-2000:])

    # Always exit 0 — let the SWE-bench eval decide success/failure downstream.
    sys.exit(0)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# OpenClawHarnessProcessor — sibling of OpenHandsHarnessProcessor
# ---------------------------------------------------------------------------


class OpenClawHarnessProcessor(BaseDatasetHarnessProcessor):
    """Sets up the openclaw side of a SWE-bench rollout.

    ``setup()`` invokes ``setup_scripts/openclaw.sh`` inside ``apptainer exec
    <openclaw_setup_sif>``, populating ``<parent_dir>/swe_openclaw_setup/`` on
    the host with the Node 22 toolchain, the openclaw npm install (with
    pre-warmed plugin packages), and the workspace template. The populated
    dir is returned and fills the ``SWEBenchWrapperServerConfig.openhands_setup_dir``
    slot (slot name kept for back-compat across both frameworks). The script
    runs inside ``apptainer exec`` so openclaw never executes outside a
    container.

    ``get_run_command()`` writes the in-container runner script + a small bash
    wrapper to ``persistent_dir`` and returns the apptainer command that
    invokes it. The runner consumes ``openclaw_home_template_path`` from
    config to find the HOME template (mounted ro at /openclaw_setup/home_template).
    """

    def setup(self) -> Path:
        setup_dir = self.parent_dir / "swe_openclaw_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)

        with self._setup_directory_lock(setup_dir, "OpenClaw"):
            marker = setup_dir / ".openclaw_setup_done"
            if marker.exists():
                print(f"OpenClaw already set up at {setup_dir}", flush=True)
                return setup_dir

            sif = self.config.openclaw_setup_sif
            if not sif:
                raise ValueError(
                    "openclaw_setup_sif config field is required when "
                    "agent_framework='openclaw'. Set it to the path of any "
                    "SWE-bench SIF (the SIF is used purely as a Linux env with "
                    "curl + tar; openclaw runs inside it via `apptainer exec` "
                    "so it never executes outside a container)."
                )

            print(f"Setting up OpenClaw environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts" / "openclaw.sh"
            command = (
                f"apptainer exec --bind {shlex.quote(str(setup_dir))}:/setup "
                f"{shlex.quote(str(sif))} "
                f"env SETUP_DIR=/setup "
                f"OPENCLAW_NPM_PIN={shlex.quote(self.config.openclaw_npm_pin)} "
                f"bash {shlex.quote(str(script_fpath))}"
            )
            self._run_setup_command(command)

        return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        data_point = self.config.problem_info
        agent_run_id = self.config.agent_run_id

        # Drop the runner script + per-instance problem statement file into
        # persistent_dir. Both are mounted at /trajectories_mount in the SIF.
        runner_path = self.config.persistent_dir / "openclaw_runner.py"
        if not runner_path.exists():
            runner_path.write_text(OPENCLAW_RUNNER_SCRIPT)

        problem_path = self.config.persistent_dir / f"problem_{data_point['instance_id']}.txt"
        problem_path.write_text(data_point.get("problem_statement", ""))

        output_filename = f"{data_point['instance_id']}.jsonl"

        # Resolve the policy model URL from the model server registered with
        # NeMo-Gym. Apptainer uses the host network namespace by default
        # (no --net flag), so http://localhost:<port> from inside the SIF
        # reaches the host's model server. The OpenAI-compatible /v1 prefix is
        # required by openclaw's openai-responses provider type.
        api_base = get_server_url(self.config.model_server_name).rstrip("/") + "/v1"

        runner_cmd = (
            "python3 /trajectories_mount/openclaw_runner.py"
            f" --instance-id {shlex.quote(data_point['instance_id'])}"
            f" --problem-statement-file /trajectories_mount/{problem_path.name}"
            f" --output-file /trajectories_mount/{output_filename}"
            f" --model-base-url {shlex.quote(api_base)}"
            f" --model-id {shlex.quote(self.config.body.model)}"
            f" --openclaw-home-template {shlex.quote(self.config.openclaw_home_template_path)}"
            f" --timeout {self.config.swebench_agent_timeout}"
        )

        # Wrap with a bash script (matches the OpenHands processor pattern;
        # makes the agent_apptainer_command_str easier to inspect / re-run).
        agent_script_name = f"agent_script_{agent_run_id}.sh"
        agent_script_path = self.config.persistent_dir / agent_script_name
        agent_script_path.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "export PATH=/openclaw_setup/node/bin:/openclaw_setup/openclaw/node_modules/.bin:$PATH\n"
            f'date +"%s.%N" > {self.config.generation_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{runner_cmd}\n"
        )

        agent_timeout_seconds = self.config.swebench_agent_timeout
        wrapped_cmd = (
            f"timeout --signal=TERM --kill-after=30 {agent_timeout_seconds} "
            f"bash /trajectories_mount/{agent_script_name}"
        )

        # The runner writes <instance_id>.jsonl into /trajectories_mount, which
        # corresponds to persistent_dir on the host.
        search_path = str(self.config.persistent_dir / output_filename)

        return ExecuteContainerCommandArgs(
            command=wrapped_cmd,
            expected_file_pattern=search_path,
            mode="agent",
            timeout=self.config.swebench_agent_timeout + 60,
        )


# ---------------------------------------------------------------------------
# RunOpenClawAgent — subclass of RunOpenHandsAgent
# ---------------------------------------------------------------------------


class RunOpenClawAgent(RunOpenHandsAgent):
    """Drop-in replacement for RunOpenHandsAgent when agent_framework='openclaw'.

    Overrides three hook methods on the base class:

    - :meth:`_openhands_dir_copy_from_host` — openclaw writes directly to
      /trajectories_mount/<instance>.jsonl plus a sibling .trajectory.jsonl,
      so the copy step is trivial and there is no eval-dir to clean up.
    - :meth:`_extract_patch_and_eval_input` — openclaw's runner produces a
      flat ``{model_patch, instance_id, model_name_or_path, timed_out}`` dict.
    - :meth:`_extract_agent_error` — surface ``timed_out`` from the runner
      output as the agent error so the GRPO mask logic can see it.
    """

    def _openhands_dir_copy_from_host(self, output_file_path: Optional[str]) -> Optional[str]:
        if not output_file_path:
            return None
        source = Path(output_file_path)
        if not source.exists():
            return None

        # Copy the runner's JSONL output to prediction_path so the rest of the
        # base class behaves uniformly (it reads from out_file).
        dest = self.config.prediction_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

        # Copy the trajectory next to the prediction so the trajectory loader
        # in get_openclaw_trajectory_from_session can find it later.
        traj_source = source.with_suffix(".trajectory.jsonl")
        if traj_source.exists():
            traj_dest_dir = self.config.trajectories_root
            traj_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(traj_source, traj_dest_dir / "trajectory.jsonl")

        return str(dest)

    def _extract_agent_error(self, out_dict: Dict[str, Any]) -> Optional[str]:
        # The openclaw runner doesn't write a free-form error string, but it
        # signals timeout via a flag. Surface that as a synthetic error so the
        # downstream "max_iteration"/"context_window"/etc classifier picks
        # something meaningful.
        if out_dict.get("timed_out"):
            return "agent timed out"
        return None

    def _extract_patch_and_eval_input(self, out_dict: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        patch = out_dict.get("model_patch")
        eval_input = {
            "model_name_or_path": out_dict.get("model_name_or_path", "openclaw"),
            "instance_id": out_dict.get("instance_id", self.config.instance_id),
            "model_patch": patch,
            "openclaw_metrics": {"timed_out": bool(out_dict.get("timed_out", False))},
        }
        return patch, eval_input


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------


def _join_openclaw_text_items(content_items: List[Dict[str, Any]]) -> str:
    parts = [c.get("text", "") for c in content_items if c.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def get_openclaw_trajectory_from_session(
    trajectories_dir: Path, instance_id: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert an openclaw session JSONL into OpenAI chat-completions format.

    Mirrors the shape of ``SWEBenchWrapper.get_openhands_trajectory_from_completions``:
    returns ``(messages, tools)``. OpenClaw's session JSONL doesn't record the
    tool catalogue, so ``tools`` is always empty (the wrapper supplies tools
    elsewhere).

    OpenClaw event shape (from pi-coding-agent, simplified):
      ``{"type": "message", "message": {"role": "user"|"assistant"|"toolResult", "content": [...]}}``
    Assistant content items can be ``{type:"text"}`` or ``{type:"toolCall", id, name, arguments}``.
    """
    traj_file = trajectories_dir / instance_id / "trajectory.jsonl"
    if not traj_file.exists():
        print(f"No openclaw trajectory found at {traj_file}", flush=True)
        return [], []

    messages: List[Dict[str, Any]] = []
    malformed_lines = 0
    try:
        with open(traj_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if event.get("type") != "message":
                    continue
                msg = event.get("message", {})
                role = msg.get("role")

                if role == "user":
                    text = _join_openclaw_text_items(msg.get("content", []))
                    messages.append({"role": "user", "content": text})

                elif role == "assistant":
                    items = msg.get("content", [])
                    text_parts: List[str] = []
                    tool_calls: List[Dict[str, Any]] = []
                    for item in items:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "toolCall":
                            tool_calls.append(
                                {
                                    "id": item.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name", ""),
                                        "arguments": json.dumps(item.get("arguments", {})),
                                    },
                                }
                            )
                    converted: Dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(t for t in text_parts if t) or None,
                    }
                    if tool_calls:
                        converted["tool_calls"] = tool_calls
                    if msg.get("stopReason") == "error":
                        converted["error"] = msg.get("errorMessage")
                    messages.append(converted)

                elif role == "toolResult":
                    tool_msg: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": msg.get("toolCallId", ""),
                        "name": msg.get("toolName", ""),
                        "content": _join_openclaw_text_items(msg.get("content", [])),
                    }
                    if msg.get("isError"):
                        tool_msg["is_error"] = True
                    messages.append(tool_msg)

        # A single trailing partial line (agent killed mid-write on timeout)
        # is expected; more than that suggests corruption or a buggy emitter
        # and should not be silent for downstream RL-training data quality.
        if malformed_lines > 1:
            print(
                f"WARN: {malformed_lines} malformed JSON lines in {traj_file}",
                flush=True,
            )
        print(f"Loaded openclaw trajectory ({len(messages)} messages)", flush=True)
        return messages, []
    except (IOError, OSError) as e:
        # Filesystem-level failures only. KeyError / AttributeError / TypeError
        # from openclaw schema drift propagate as stack traces by design — we
        # want to know loudly if the event shape changed under us.
        print(f"Failed to read openclaw trajectory ({traj_file}): {e}", flush=True)
        return [], []

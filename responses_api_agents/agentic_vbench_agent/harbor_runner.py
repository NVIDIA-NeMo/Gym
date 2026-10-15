# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unmodified benchmark task with the upstream Harbor CLI."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


OPENCODE_OUTPUT_CAP = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"


def container_endpoint(endpoint: str) -> str:
    """Reach a host-local Factory proxy through the rootless network gateway."""
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Model endpoint must be an HTTP(S) URL")
    if parts.username or parts.password:
        raise ValueError("Model endpoint must not embed credentials")
    if parts.hostname in {"localhost", "127.0.0.1", "::1"}:
        authority = "10.0.2.2" + (f":{parts.port}" if parts.port else "")
        parts = parts._replace(netloc=authority)
    return urlunsplit(parts).rstrip("/")


TRAINING_MODES = ("plain", "compaction", "subagent")


def job_config(
    *,
    task_path: Path,
    output: Path,
    runtime_root: Path,
    model: str,
    endpoint: str,
    context_tokens: int,
    output_tokens: int,
    backend: str = "podman",
    max_turns: int | None = None,
    verifier_timeout_multiplier: float = 1.0,
    agent_timeout_multiplier: float | None = None,
    model_timeout_ms: int | None = None,
    tool_output_max_lines: int | None = None,
    tool_output_max_bytes: int | None = None,
    compaction_auto: bool | None = None,
    compaction_prune: bool | None = None,
    disable_title: bool = False,
    training_mode: str | None = None,
    subagents_enabled: bool = False,
    subagent_max_depth: int = 1,
    subagent_max_active: int | None = None,
    delegation_prompt: str | None = None,
    subagent_prompt: str | None = None,
) -> dict:
    """Create a single-attempt Harbor job using OpenCode's compatible provider.

    ``max_turns`` caps OpenCode's agentic iterations (its ``build`` agent ``steps``);
    after the cap OpenCode must give a final text-only response.
    ``verifier_timeout_multiplier`` scales only the task's verifier timeout: some
    single-threaded judges need longer than their 1800 s on this hardware. The agent's
    budget and the scores are unaffected. ``agent_timeout_multiplier`` scales the
    task's agent wall-clock budget (1800 s on AgenticVBench); Harbor stops OpenCode at
    the cap and still runs the verifier on what was delivered, so training can bound
    episode time without changing the verifier. ``model_timeout_ms`` is OpenCode's per-call
    provider timeout; Gym's model proxy replays a streamed completion only once it is
    complete, so the timeout must cover a whole 100k-token generation.
    ``tool_output_max_lines``/``tool_output_max_bytes`` use OpenCode's native
    observation truncation. The complete output remains available in its
    tool-output directory; only the text returned to the model is bounded.
    ``compaction_auto``/``compaction_prune`` pin OpenCode's context compaction
    (``None`` keeps OpenCode's defaults: both on). Training with NeMo-RL token
    capture sets them explicitly: compaction on for framework-owned context
    reconstruction, both off for an append-only context. ``disable_title`` drops
    OpenCode's side model call that names the session. ``delegation_prompt`` adds
    a run-level instruction to the primary agent without modifying the pinned
    benchmark prompt; it is valid only when the subagent tool is enabled.
    ``subagent_prompt`` gives the depth-one ``general`` child a run-level
    instruction, independently of the primary agent's delegation instruction.
    """
    if not 0 < output_tokens < context_tokens:
        raise ValueError("Output token limit must be positive and smaller than context")
    if backend not in {"podman", "remote"}:
        raise ValueError("Backend must be podman or remote")
    environment_class = (
        "remote_environment:RemoteEnvironment" if backend == "remote" else "podman_environment:PodmanEnvironment"
    )
    if backend == "remote" and urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote task backend requires a network-reachable Factory proxy address")
    if max_turns is not None and max_turns < 1:
        raise ValueError("Maximum agent turns must be positive")
    if verifier_timeout_multiplier < 1.0:
        raise ValueError("Verifier timeout multiplier must not shorten the task's own budget")
    if agent_timeout_multiplier is not None and not agent_timeout_multiplier > 0:
        raise ValueError("Agent timeout multiplier must be positive")
    if model_timeout_ms is not None and model_timeout_ms < 1:
        raise ValueError("Model call timeout must be positive")
    if tool_output_max_lines is not None and tool_output_max_lines < 1:
        raise ValueError("Tool output line limit must be positive")
    if tool_output_max_bytes is not None and tool_output_max_bytes < 1:
        raise ValueError("Tool output byte limit must be positive")
    provider_options = {
        "baseURL": endpoint.rstrip("/") if backend == "remote" else container_endpoint(endpoint),
        "apiKey": "unused",
    }
    if model_timeout_ms is not None:
        provider_options["timeout"] = model_timeout_ms
    # OpenCode gives the built-in ``openai`` provider special Responses API
    # routing.  ``@ai-sdk/openai-compatible`` only implements the generic
    # language-model interface, so a custom provider name is required here.
    opencode_config = {
        "provider": {
            "nemo": {
                "npm": "@ai-sdk/openai-compatible",
                "options": provider_options,
                "models": {
                    model: {
                        "name": model,
                        "limit": {"context": context_tokens, "output": output_tokens},
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                    }
                },
            }
        },
    }
    if training_mode is not None and training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {training_mode!r} (expected one of {TRAINING_MODES})")
    if training_mode is not None and (training_mode == "subagent") != subagents_enabled:
        raise ValueError("Fixed training mode and subagent tool configuration disagree")
    if delegation_prompt is not None:
        delegation_prompt = delegation_prompt.strip()
        if not delegation_prompt:
            raise ValueError("Delegation prompt must not be empty")
        if not subagents_enabled:
            raise ValueError("Delegation prompt requires subagents")
    if subagent_prompt is not None:
        subagent_prompt = subagent_prompt.strip()
        if not subagent_prompt:
            raise ValueError("Subagent prompt must not be empty")
        if not subagents_enabled:
            raise ValueError("Subagent prompt requires subagents")
    if subagents_enabled:
        if subagent_max_depth != 1:
            raise ValueError("Subagent training requires depth-one delegation (subagent_max_depth=1)")
        if subagent_max_active is not None and (type(subagent_max_active) is not int or subagent_max_active < 1):
            raise ValueError("subagent_max_active must be null (native parallel children) or a positive int")
        if compaction_auto or compaction_prune:
            raise ValueError("Subagent training forbids OpenCode compaction and pruning")
    agents: dict = {}
    if max_turns is not None:
        agents["build"] = {"steps": max_turns}
    if disable_title:
        agents["title"] = {"disable": True}
    # Delegation. OpenCode's primary agent owns a `task` tool that starts a child
    # session (mode: subagent) with its own context. Without subagents the tool is
    # denied so a rollout stays one session. In subagent mode every non-task
    # permission is explicit ``allow``: OpenCode 1.14.39's non-interactive CLI
    # auto-replies only to permission requests owned by the root session, so an
    # ``ask`` from a child otherwise leaves the child and its waiting parent
    # deadlocked until Harbor's wall-clock timeout. Only ``general`` is exposed as
    # a child, matching the reference harness, and its explicit task denial is the
    # depth-one boundary.
    if subagents_enabled:
        opencode_config["permission"] = {"*": "allow", "task": "allow"}
        if delegation_prompt is not None:
            agents.setdefault("build", {})["prompt"] = delegation_prompt
        agents["general"] = {
            "mode": "subagent",
            "permission": {"*": "allow", "task": "deny"},
            **({"steps": max_turns} if max_turns is not None else {}),
            **({"prompt": subagent_prompt} if subagent_prompt is not None else {}),
        }
        agents["explore"] = {"disable": True}
    else:
        opencode_config["permission"] = {"task": "deny"}
    if agents:
        opencode_config["agent"] = agents
    if subagent_max_active is not None:
        # Native OpenCode runs sibling task calls concurrently; a cap is applied by
        # the capture proxy, which serializes admission per session regardless.
        opencode_config.setdefault("experimental", {})["subagent_max_active"] = subagent_max_active
    tool_output = {
        key: value
        for key, value in (
            ("max_lines", tool_output_max_lines),
            ("max_bytes", tool_output_max_bytes),
        )
        if value is not None
    }
    if tool_output:
        opencode_config["tool_output"] = tool_output
    compaction = {
        key: value for key, value in (("auto", compaction_auto), ("prune", compaction_prune)) if value is not None
    }
    if compaction:
        opencode_config["compaction"] = compaction
    return {
        "job_name": "trial",
        "jobs_dir": str(output / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "verifier_timeout_multiplier": verifier_timeout_multiplier,
        **({"agent_timeout_multiplier": agent_timeout_multiplier} if agent_timeout_multiplier is not None else {}),
        "environment": {
            "import_path": "responses_api_agents.agentic_vbench_agent." + environment_class,
            "delete": True,
            "kwargs": {"runtime_root": str(runtime_root)},
        },
        "tasks": [{"path": str(task_path)}],
        "agents": [
            {
                "name": "opencode",
                "model_name": f"nemo/{model}",
                "kwargs": {"version": "1.14.39", "opencode_config": opencode_config},
                # OpenCode reserves min(model output limit, this cap; default 32000) when deciding
                # to compact. The Factory adapter sends max_tokens equal to the output limit, so the
                # reservation must match it or prompts overflow the context before compaction.
                # Harbor redacts literal values of *TOKEN* keys when it writes the job config, so
                # pass a template that Harbor resolves from the environment main() provides.
                "env": {OPENCODE_OUTPUT_CAP: f"${{{OPENCODE_OUTPUT_CAP}}}"},
            }
        ],
    }


JUDGE_PROTOCOLS = {
    "nvinference-hybrid": {
        "AVB_JUDGE_BASE_URL": "https://inference-api.nvidia.com",
        "AVB_JUDGE_CLAUDE_MODEL": "us/aws/anthropic/eccn-claude-opus-4-7",
        "AVB_JUDGE_GEMINI_MODEL": "us/gcp/google/eccn-gemini-3.5-flash",
        "AVB_JUDGE_GEMINI_THINKING": "low",
    }
}
NATIVE_JUDGE_ENV = {"ANTHROPIC_API_KEY", "GEMINI_API_KEY"}
# Informational scan in the repurpose verifier; inaccessible /proc entries in
# rootless containers must not abort the verifier under pipefail.
TYPING_DIAGNOSTIC = "find / -name 'typing_extensions*' 2>/dev/null | head -20"

_ASSEMBLY_FAMILY = "agentic_vbench_assembly"
_ASSEMBLY_MANIFEST_PARSER = '''    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero("solution.json: segments not a list")

    pred = [_normalize(seg.get("source", "")) for seg in segments]
'''
_ASSEMBLY_HARDENED_PARSER = '''    if not isinstance(sol, dict):
        return _zero("solution.json: expected top-level object")

    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero("solution.json: segments not a list")

    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            return _zero(f"solution.json: segment {i} is not an object")

    pred = [_normalize(seg.get("source", "")) for seg in segments]
'''
_ASSEMBLY_JUDGE_SHA256 = {
    "agentic-vbench-assembly-task1": (
        "61a3ec3270a5ecf9ea394c6f7216d08a9af1d9ef794bf6ac297752f30fee168e",
        "5bc591c8439c11c262dc4073d1db3750f15603f6d6dbc7d6551ca045b3d21fad",
    ),
    "agentic-vbench-assembly-task2": (
        "7c70dc9c2dc357993f91e80b51a74d153c9bf8dc5274ca826ad9eaf2ae0f9ca3",
        "161d8e04440b8add9280fa2649e4d1470f49f4fcd68697b6cd32eae7aa80868b",
    ),
    "agentic-vbench-assembly-task3": (
        "0e76e6774046ea9448c3d5a906d7c04f50cf05706739f2afb82c77d887639548",
        "f7a2030f194ebc0befb1fd349efb6d39d78f260abc88672923d91f161a0ec453",
    ),
    "agentic-vbench-assembly-task4": (
        "73da97f7929d94831141a23954b910ae03f5d0de93ac12cf27df3a9a1c713ba5",
        "02f5894208585abf30fde48c68f5ea9ccaf53182c934eadc1520f8ab97339646",
    ),
    "agentic-vbench-assembly-task5": (
        "129b6d6ceb8ed68e2f1c89246b99ed18db5908c80bffa7425c03dc2dc08ac094",
        "61e63f8d5401fd496f13757221dec0f550d8f1d9bf93aa30d4a410e4e1f7e2f9",
    ),
    "agentic-vbench-assembly-task6": (
        "609f102e08adf0eccfbd0bb4973a20ff6b88b402f7a61a906a45eb3a55dc3dc4",
        "6d449633c37da9fd10a8f4cf046623c21f7f15c9da372c01b127d372d31a6156",
    ),
    "agentic-vbench-assembly-task7": (
        "7dd6d792007f581f2b9b843d5ca1958560884b253edd7a8c34e73db1376191d6",
        "08978a086806d75c1880979160227cc753965eeb6ced6dfc1028d5ff45c8eb5e",
    ),
    "agentic-vbench-assembly-task8": (
        "52b01834e4c342888e0238e399507572ed82e65b6082eb27ddaedded7c985e75",
        "2dd1339341fe5a4ca6052aeb4edb58c740019a4a2610356eeadef56acb5373a4",
    ),
    "agentic-vbench-assembly-task9": (
        "49b585fd2413798d6b484b1123c5ff476e791715dc99fa42f21e642637cf492f",
        "30e2816a2fd73e4b0a165943655cd32dea316f830c92a16cdb05e9c3feeb1d15",
    ),
    "agentic-vbench-assembly-task10": (
        "9ce0f3da6ae606f775505e33e8380539e3f27de185bc137d6d401d4747ee40c4",
        "d7c7124dc304f8a6e8e394b463838b13b1d309615518b6d2ec3d89388e818045",
    ),
    "agentic-vbench-assembly-task11": (
        "2de466f807ab097e1ca754908ebc36ffe9c1ba716a540455287568322de68c06",
        "955ebfc11fae35bff96f6a9cf2d281bc03f43c2630aa2d3919314bb8ae8164ba",
    ),
    "agentic-vbench-assembly-task12": (
        "c2648737e1de760e743fa29115a0c241ef18c3f0a6132eea70161ab96497ddcb",
        "fe363ddc8b720c512833064d032ab5e8a3df691efb4d726a98fa4e5475417a38",
    ),
    "agentic-vbench-assembly-task13": (
        "ef488cd4f71e0947cf19ecab470ff6693e15de26a5cdd07e8e203cb75e84866a",
        "821ee8de3bd3be80c7ece58b0c1ecd9d1d3845572bfb021014b2b3955074d379",
    ),
    "agentic-vbench-assembly-task14": (
        "20660911630e38d25bad3c8d2d9c9148ab9781a3fa4d3d6c5a89c72ec64143ea",
        "0d8538b9696f1bc1089d8415c0a8a35e5efef8fa34fc289e46a8c92c7305e305",
    ),
    "agentic-vbench-assembly-task15": (
        "26451dde4b9328495aad4610f0f48282b51d4660695219645118cc37b0468b6e",
        "bab7f16a3c74117ab03c7e5fd8dcae473454768021cbb9ceb31a1f7f37085a1f",
    ),
    "agentic-vbench-assembly-task16": (
        "a7293e8e95cc59c27cdbafe9802983c21d961f868fb426c68f30a99d7e628689",
        "670f636b37cf077fd5e58054021cad1cd98d34fabd62ef844687199fdb4dc9c0",
    ),
    "agentic-vbench-assembly-task17": (
        "7ad56e110ec984d69a9d2afbaf2d9ed93eda71d1a621ae7c350e1e845858922d",
        "c5d6dd370825d997bce0138f08760ba14e543e60d70f4503d59391ada669fc99",
    ),
    "agentic-vbench-assembly-task18": (
        "f2dea83ca090e44108be5ce52817b6bbfd4266a0ecf909b30b7f7bcbc2954682",
        "49724c26534af8519849e42e5411ebc4793257ce6ea68bbdbd1d430812d807cc",
    ),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _harden_assembly_judge(source: bytes, task_id: str) -> bytes:
    expected = _ASSEMBLY_JUDGE_SHA256.get(task_id)
    if expected is None:
        raise ValueError(f"Unexpected pinned assembly task: {task_id}")
    source_sha256, staged_sha256 = expected
    if _digest(source) != source_sha256:
        raise ValueError(f"Pinned assembly judge has an unexpected source digest: {task_id}")
    text = source.decode()
    if text.count(_ASSEMBLY_MANIFEST_PARSER) != 1:
        raise ValueError(f"Pinned assembly judge has an unexpected manifest parser: {task_id}")
    hardened = text.replace(_ASSEMBLY_MANIFEST_PARSER, _ASSEMBLY_HARDENED_PARSER).encode()
    if _digest(hardened) != staged_sha256:
        raise ValueError(f"Assembly judge hardening produced an unexpected digest: {task_id}")
    return hardened


def stage_assembly_judge_guard(task_path: Path, *, staging: Path, output: Path) -> Path:
    """Make the pinned assembly judge reject a non-object manifest with reward zero.

    The upstream judge calls ``sol.get`` immediately after ``json.loads``. A model
    can therefore create valid JSON with the wrong top-level shape (for example, a
    bare list), which raises ``AttributeError`` instead of producing the same zero
    reward as every other malformed manifest. Stage only exact pinned assembly
    judges; valid-object scoring is unchanged and the source checkout stays clean.
    """
    if task_path.parent.name != _ASSEMBLY_FAMILY:
        return task_path

    task_id = task_path.name
    source_judge = task_path / "steps/solve/tests/judge.py"
    hardened = _harden_assembly_judge(source_judge.read_bytes(), task_id)
    staged = staging / task_path.name
    expected_staged_sha256 = _ASSEMBLY_JUDGE_SHA256[task_id][1]
    if staged.exists():
        staged_judge = staged / "steps/solve/tests/judge.py"
        if not staged_judge.is_file() or _digest(staged_judge.read_bytes()) != expected_staged_sha256:
            raise ValueError(f"Existing staged assembly judge has an unexpected digest: {task_id}")
        return staged

    shutil.copytree(task_path, staged)
    judge = staged / "steps/solve/tests/judge.py"
    judge.write_bytes(hardened)

    (output / "assembly_judge_guard.json").write_text(
        json.dumps(
            {
                "source_task": str(task_path),
                "reason": "non-object solution manifests score zero instead of crashing the verifier",
                "original_judge_sha256": _digest(source_judge.read_bytes()),
                "staged_judge_sha256": _digest(judge.read_bytes()),
            },
            indent=2,
        )
        + "\n"
    )
    return staged


def stage_judge(task_path: Path, *, staging: Path, output: Path, protocol: str) -> Path:
    """Copy a natively judged task with its judge routed through NVIDIA Inference Hub.

    Tasks without native judge credentials are returned unchanged. The Hub key is
    exposed only to the verifier, never to the agent's container environment.
    """
    if protocol not in JUDGE_PROTOCOLS:
        raise ValueError(f"Unknown judge protocol: {protocol}")
    text = (task_path / "task.toml").read_text()
    task = tomllib.loads(text)
    if set(task.get("environment", {}).get("env", {})) != NATIVE_JUDGE_ENV:
        return task_path
    if len(task["steps"]) != 1 or task["steps"][0].get("verifier", {}).get("env"):
        raise ValueError(f"Unexpected judged task layout: {task_path}")
    staged = staging / task_path.name
    shutil.copytree(task_path, staged)
    start, end = text.index("[environment.env]"), text.index("[[steps]]")
    settings = JUDGE_PROTOCOLS[protocol]
    text = text[:start] + text[end:] + '\n[steps.verifier.env]\nNVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"\n'
    text += "".join(f'{key} = "{value}"\n' for key, value in settings.items())
    (staged / "task.toml").write_text(text)
    tests = staged / "steps" / task["steps"][0]["name"] / "tests"
    verifier = tests / "test.sh"
    verifier.write_text(verifier.read_text().replace(TYPING_DIAGNOSTIC, TYPING_DIAGNOSTIC + " || true"))
    (tests / "judge.py").rename(tests / "avb-original-judge.py")
    shutil.copyfile(Path(__file__).with_name("hub_judge.py"), tests / "judge.py")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (output / "judge_protocol.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "settings": settings,
                "source_task": str(task_path),
                "original_sha256": {
                    name: digest(task_path / relative)
                    for name, relative in {
                        "task.toml": "task.toml",
                        "test.sh": verifier.relative_to(staged),
                        "judge.py": (tests / "judge.py").relative_to(staged),
                    }.items()
                },
                "staged_sha256": {
                    "task.toml": digest(staged / "task.toml"),
                    "test.sh": digest(verifier),
                    "judge.py": digest(tests / "judge.py"),
                    "avb-original-judge.py": digest(tests / "avb-original-judge.py"),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return staged


def _env_default(name: str, cast, fallback):
    value = os.environ.get(name)
    return cast(value) if value else fallback


def judge_api_key(credentials: dict, environ: dict) -> str | None:
    """The Hub key, from the credentials file or the environment.

    The gym-native driver exports the shared ``INFERENCE_API_KEY``; the NEL path
    carried a ``NVINFERENCE_API_KEY`` credentials file. Either is accepted.
    """
    return (
        credentials.get("NVINFERENCE_API_KEY")
        or environ.get("NVINFERENCE_API_KEY")
        or environ.get("INFERENCE_API_KEY")
        or None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--credentials-file", type=Path)
    # Protocol knobs come from the agent's configuration; the environment variables
    # remain as fallbacks for the previous launcher.
    parser.add_argument("--backend", default=os.environ.get("AGENTIC_VBENCH_BACKEND", "podman"))
    parser.add_argument("--judge-protocol", default=os.environ.get("AGENTIC_VBENCH_JUDGE_PROTOCOL", ""))
    parser.add_argument("--max-turns", type=int, default=_env_default("AGENTIC_VBENCH_MAX_TURNS", int, None))
    parser.add_argument(
        "--verifier-timeout-multiplier",
        type=float,
        default=_env_default("AGENTIC_VBENCH_VERIFIER_TIMEOUT_MULTIPLIER", float, 1.0),
    )
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help="Scale the task's agent wall-clock budget; Harbor still verifies after the cap",
    )
    parser.add_argument("--model-timeout-ms", type=int, default=None)
    parser.add_argument("--tool-output-max-lines", type=int, default=None)
    parser.add_argument("--tool-output-max-bytes", type=int, default=None)
    parser.add_argument("--compaction-auto", choices=("true", "false"), default=None)
    parser.add_argument("--compaction-prune", choices=("true", "false"), default=None)
    parser.add_argument("--disable-title", action="store_true")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default=None)
    parser.add_argument(
        "--subagents",
        action="store_true",
        help="Enable OpenCode's task tool (depth-one `general` children)",
    )
    parser.add_argument("--subagent-max-depth", type=int, default=1)
    parser.add_argument("--subagent-max-active", type=int, default=None)
    parser.add_argument(
        "--delegation-prompt",
        default=None,
        help="Primary-agent instruction encouraging useful delegation (subagent mode only)",
    )
    parser.add_argument(
        "--subagent-prompt",
        default=None,
        help="Instruction for depth-one general children (subagent mode only)",
    )
    args = parser.parse_args()
    if importlib.metadata.version("harbor") != "0.6.6":
        raise RuntimeError("AgenticVBench requires upstream harbor==0.6.6")
    from harbor.models.job.config import JobConfig

    task_path = args.task_path.resolve(strict=True)
    runtime_root = args.runtime_root.resolve()
    task_path = stage_assembly_judge_guard(
        task_path,
        staging=runtime_root / "assembly-judge-task",
        output=args.output,
    )
    env = dict(os.environ)
    if args.judge_protocol:
        from dotenv import dotenv_values

        credentials = dotenv_values(args.credentials_file) if args.credentials_file else {}
        key = judge_api_key(credentials, os.environ)
        if not key:
            raise ValueError(
                f"Judge protocol {args.judge_protocol} requires NVINFERENCE_API_KEY (credentials file or "
                "environment) or the driver's INFERENCE_API_KEY"
            )
        # Harbor resolves the staged task's "${NVINFERENCE_API_KEY}" from its own environment.
        env["NVINFERENCE_API_KEY"] = key
        task_path = stage_judge(
            task_path,
            staging=runtime_root / "judge-task",
            output=args.output,
            protocol=args.judge_protocol,
        )

    config = JobConfig.model_validate(
        job_config(
            task_path=task_path,
            output=args.output.resolve(strict=True),
            runtime_root=runtime_root,
            model=args.model,
            endpoint=args.endpoint,
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
            backend=args.backend,
            max_turns=args.max_turns,
            verifier_timeout_multiplier=args.verifier_timeout_multiplier,
            agent_timeout_multiplier=args.agent_timeout_multiplier,
            model_timeout_ms=args.model_timeout_ms,
            tool_output_max_lines=args.tool_output_max_lines,
            tool_output_max_bytes=args.tool_output_max_bytes,
            compaction_auto=None if args.compaction_auto is None else args.compaction_auto == "true",
            compaction_prune=None if args.compaction_prune is None else args.compaction_prune == "true",
            disable_title=args.disable_title,
            training_mode=args.training_mode,
            subagents_enabled=args.subagents,
            subagent_max_depth=args.subagent_max_depth,
            subagent_max_active=args.subagent_max_active,
            delegation_prompt=args.delegation_prompt,
            subagent_prompt=args.subagent_prompt,
        )
    )
    config_path = args.output / "harbor_job.json"
    with config_path.open("x") as handle:
        handle.write(config.model_dump_json(indent=2) + "\n")
    # Tasks that declare [environment.env] variables (e.g. HF_TOKEN) make Harbor ask
    # for confirmation. Confirm non-interactively and close stdin so no prompt can block.
    command = [str(Path(sys.executable).with_name("harbor")), "run", "--config", str(config_path), "--yes"]
    if args.credentials_file:
        command.extend(["--env-file", str(args.credentials_file.resolve(strict=True))])
    env[OPENCODE_OUTPUT_CAP] = str(args.output_tokens)
    gym_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [gym_root, env.get("PYTHONPATH")]))
    run_harbor(command, env=env, jobs_dir=args.output.resolve() / "jobs")


DOCKER_CHECK_FAILURE = "Docker daemon is not running"
# Harbor's start-up probe is a single ``docker info`` under a 10 s timeout. A
# rootless backend serving ~64 containers answers in 3 s most of the time but
# stalls past 10 s under load, so the probe fails in bursts; each exhausted retry
# costs a whole re-dispatched rollout. Defaults keep trying for ~12 minutes.
HARBOR_START_ATTEMPTS = int(os.environ.get("AGENTIC_VBENCH_HARBOR_START_ATTEMPTS", "12"))
HARBOR_START_BACKOFF_SECONDS = float(os.environ.get("AGENTIC_VBENCH_HARBOR_START_BACKOFF_SECONDS", "20"))


def run_harbor(command: list[str], *, env: dict, jobs_dir: Path, sleep=time.sleep) -> None:
    """Run Harbor, retrying only its start-up Docker probe failure.

    Harbor checks the daemon with a single ``docker info`` under a 10 s timeout before
    any trial starts; on a loaded remote backend that probe times out and Harbor exits
    with ``Docker daemon is not running`` although the backend is healthy. The retry is
    limited to that message while no trial directory exists, so a trial is never run twice.
    """
    for attempt in range(1, HARBOR_START_ATTEMPTS + 1):
        result = subprocess.run(
            command, env=env, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace"
        )
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()
        if result.returncode == 0:
            return
        trial_started = jobs_dir.is_dir() and any(jobs_dir.glob("*/*__*"))
        if DOCKER_CHECK_FAILURE not in result.stderr or trial_started or attempt == HARBOR_START_ATTEMPTS:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"Harbor Docker probe failed before any trial started; retry {attempt}/{HARBOR_START_ATTEMPTS - 1}")
        sleep(HARBOR_START_BACKOFF_SECONDS * attempt)


if __name__ == "__main__":
    main()

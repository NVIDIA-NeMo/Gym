# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Provider-neutral self-driving scaffolding for SWE agents.

Any agent that runs to completion inside a sandbox (editing the repo at the task's
working directory) can reuse these helpers: provision a working sandbox via a
``SandboxProvider``, inject a sandbox-reachable model endpoint and/or extra
environment for egress, run an opaque agent launch command, and extract the
resulting unified-diff patch. Grading is decoupled — callers grade the patch
in-process via :func:`run_self_driving` (or ``verify_task`` directly) in a fresh
sandbox. The agent launch command, staged files, and patch-output location are
caller-supplied, so nothing here is specific to any one agent harness.

This module also defines the in-sandbox model-server egress primitive
(``ModelEndpoint`` / ``resolve``), used to inject a sandbox-reachable endpoint
into the agent's environment.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_gym.sandbox import SandboxProvider
from responses_api_agents.swe_env.harness import SweTask, get_harness, reward_from_report
from responses_api_agents.swe_env.sandbox import acquire_sandbox


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    """Return the name of a sandbox provider.

    Args:
        provider: Either a mapping keyed by provider name, or a ``SandboxProvider``
            instance with a ``name`` attribute.

    Returns:
        The provider name, or ``"?"`` if it cannot be determined.
    """
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def _read_output_jsonl_row(env, output_glob: str) -> dict[str, Any]:
    """Return the last row of the newest matching ``output.jsonl`` (or ``{}`` if absent).

    Some self-driving harnesses write their result row to an ``output.jsonl`` file under an
    output directory rather than to the working tree, so a plain ``git diff`` would miss the
    patch. When several files match (e.g. a re-run left a stale one), the newest by mtime is
    picked. ``find -printf "%T@ %p"`` emits ``<mtime> <path>`` per match; ``sort -n | tail -1``
    selects the most-recently-modified, and the leading float timestamp plus single space is
    stripped back off (so paths containing spaces survive).

    Args:
        env: The sandbox handle exposing ``execute`` for running shell commands.
        output_glob: Path or glob under which to search for ``output.jsonl`` files.

    Returns:
        The parsed last JSON row of the newest matching ``output.jsonl`` as a dict, or an
        empty dict if no file or content is found.
    """
    found = await env.execute(
        f'find {shlex.quote(output_glob)} -name output.jsonl -printf "%T@ %p\\n" 2>/dev/null | sort -n | tail -1'
    )
    newest = (found.get("stdout", "") or "").strip()
    # newest is "<mtime> <path>"; the path may contain spaces, so split only on the first one.
    path = newest.split(" ", 1)[1].strip() if " " in newest else ""
    if not path:
        return {}
    catted = await env.execute(f"cat {shlex.quote(path)}")
    raw = (catted.get("stdout", "") or "").strip()
    if not raw:
        return {}
    return json.loads(raw.splitlines()[-1])


async def _extract_patch_from_output_jsonl(env, output_glob: str) -> str:
    """Read the unified-diff patch from the newest matching ``output.jsonl``.

    Args:
        env: The sandbox handle exposing ``execute`` for running shell commands.
        output_glob: Path or glob under which to search for ``output.jsonl`` files.

    Returns:
        The patch string from ``row["test_result"]["git_patch"]``, or an empty string if
        absent.
    """
    row = await _read_output_jsonl_row(env, output_glob)
    return (row.get("test_result") or {}).get("git_patch", "") or ""


def _build_agent_spec(task, provider, model_server, opensandbox_service_url, extra_env):
    """Build the agent sandbox spec, injecting egress env (model endpoint and/or extra env).

    Args:
        task: The SWE task whose benchmark selects the harness and seeds the spec.
        provider: The sandbox provider, used to resolve the model endpoint for egress.
        model_server: Optional model-server config; when given, a sandbox-reachable endpoint
            is resolved and merged into the spec's environment.
        opensandbox_service_url: Optional OpenSandbox service URL used when resolving the
            model endpoint.
        extra_env: Optional environment variables merged verbatim into the spec.

    Returns:
        The sandbox spec with egress environment variables applied.
    """
    harness = get_harness(task.benchmark)
    spec = harness.build_spec(task)
    # Model-server egress: inject only a sandbox-reachable endpoint (never the global dict).
    if model_server is not None:
        endpoint = resolve(_provider_name(provider), model_server, opensandbox_service_url=opensandbox_service_url)
        spec = dataclasses.replace(spec, env={**spec.env, **endpoint.to_sandbox_env()})
    # Any extra in-sandbox env (e.g. a NeMo-Gym ServerClient config dict, ANTHROPIC_* vars).
    if extra_env:
        spec = dataclasses.replace(spec, env={**spec.env, **dict(extra_env)})
    return spec


async def provision_and_collect(
    task: SweTask,
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    agent_launch_command: str,
    model_server: Mapping[str, Any] | None = None,
    opensandbox_service_url: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    stage_files: Mapping[str, str] | None = None,
    patch_output_glob: str | None = None,
    agent_timeout_s: int | float = 1800,
) -> dict[str, Any]:
    """Provision and self-drive the agent, returning the patch and error signals.

    Provisions a writable sandbox from the task image, stages any caller-supplied files,
    runs the opaque ``agent_launch_command`` at the repo working directory, then extracts the
    unified-diff patch. No grading happens here.

    Two egress styles are supported and composable:

    * ``model_server`` -> a sandbox-reachable OpenAI ``base_url`` (via ``resolve``),
      for agents that call the model via a standard OpenAI/litellm client.
    * ``extra_env`` -> injected verbatim, for agents wired to NeMo Gym's ``ServerClient`` or to
      a CLI that reads its endpoint from environment variables.

    ``env.execute`` does not raise on timeout; it returns an ``error_type`` instead, so the
    caller must read the returned ``"error_type"`` to set ``agent_timed_out`` (otherwise a
    timed-out agent would wrongly not be masked).

    Args:
        task: The SWE task describing the instance, image, and working directory.
        provider: The sandbox provider (mapping keyed by name, or a ``SandboxProvider``).
        agent_launch_command: The shell command that runs the agent inside the sandbox.
        model_server: Optional model-server config; when given, a sandbox-reachable endpoint
            is resolved and injected into the agent's environment.
        opensandbox_service_url: Optional OpenSandbox service URL used when resolving the
            model endpoint.
        extra_env: Optional environment variables injected verbatim into the sandbox.
        stage_files: Optional ``{remote_path: content}`` files written into the live sandbox
            before launch.
        patch_output_glob: When given, the patch is read from an ``output.jsonl`` under this
            path; otherwise it comes from ``git diff --cached`` on ``repo_workdir``.
        agent_timeout_s: Timeout in seconds for the agent run. Defaults to ``1800``.

    Returns:
        A dict with keys ``"patch"`` (the unified-diff string), ``"agent_error"`` (the
        harness error field or ``None``), and ``"error_type"`` (``"timeout"``, ``"sandbox"``,
        or ``None``).
    """
    spec = _build_agent_spec(task, provider, model_server, opensandbox_service_url, extra_env)
    async with acquire_sandbox(provider, spec, instance_id=task.instance_id) as env:
        for remote_path, content in (stage_files or {}).items():
            await env.write_text(remote_path, content)
        run = await env.execute(agent_launch_command, cwd=task.repo_workdir, timeout_s=agent_timeout_s)
        error_type = run.get("error_type")
        if patch_output_glob:
            row = await _read_output_jsonl_row(env, patch_output_glob)
            patch = (row.get("test_result") or {}).get("git_patch", "") or ""
            return {"patch": patch, "agent_error": row.get("error"), "error_type": error_type}
        diff = await env.execute(f"cd {task.repo_workdir} && git add -A && git diff --cached", cwd=task.repo_workdir)
        return {"patch": diff.get("stdout", "") or "", "agent_error": None, "error_type": error_type}


async def provision_and_extract_patch(
    task: SweTask,
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    agent_launch_command: str,
    model_server: Mapping[str, Any] | None = None,
    opensandbox_service_url: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    stage_files: Mapping[str, str] | None = None,
    patch_output_glob: str | None = None,
    agent_timeout_s: int | float = 1800,
) -> str:
    """Provision a working sandbox, self-drive the agent, and return the unified-diff patch.

    A thin wrapper over :func:`provision_and_collect` returning only the patch. No grading
    happens here.

    Args:
        task: The SWE task describing the instance, image, and working directory.
        provider: The sandbox provider (mapping keyed by name, or a ``SandboxProvider``).
        agent_launch_command: The shell command that runs the agent inside the sandbox.
        model_server: Optional model-server config; when given, a sandbox-reachable endpoint
            is resolved and injected into the agent's environment.
        opensandbox_service_url: Optional OpenSandbox service URL used when resolving the
            model endpoint.
        extra_env: Optional environment variables injected verbatim into the sandbox.
        stage_files: Optional ``{remote_path: content}`` files written into the live sandbox
            before launch.
        patch_output_glob: When given, the patch is read from an ``output.jsonl`` under this
            path; otherwise it comes from ``git diff --cached`` on ``repo_workdir``.
        agent_timeout_s: Timeout in seconds for the agent run. Defaults to ``1800``.

    Returns:
        The extracted unified-diff patch as a string (empty if none was produced).
    """
    result = await provision_and_collect(
        task,
        provider=provider,
        agent_launch_command=agent_launch_command,
        model_server=model_server,
        opensandbox_service_url=opensandbox_service_url,
        extra_env=extra_env,
        stage_files=stage_files,
        patch_output_glob=patch_output_glob,
        agent_timeout_s=agent_timeout_s,
    )
    return result["patch"]


async def run_self_driving(
    task: SweTask,
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    agent_launch_command: str,
    model_server: Mapping[str, Any] | None = None,
    opensandbox_service_url: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    stage_files: Mapping[str, str] | None = None,
    patch_output_glob: str | None = None,
    agent_timeout_s: int | float = 1800,
) -> dict[str, Any]:
    """Provision, self-drive, extract the patch, then grade it in-process in a fresh sandbox.

    Bundles provisioning and verification for standalone use and tests. The patch is graded by
    ``verify_task`` in its OWN fresh sandbox (so grading is hermetic — never the agent's dirtied
    tree). ``verify_task`` is imported lazily to avoid a circular import between this library and
    the verifier module.

    Args:
        task: The SWE task describing the instance, image, and working directory.
        provider: The sandbox provider (mapping keyed by name, or a ``SandboxProvider``).
        agent_launch_command: The shell command that runs the agent inside the sandbox.
        model_server: Optional model-server config; when given, a sandbox-reachable endpoint
            is resolved and injected into the agent's environment.
        opensandbox_service_url: Optional OpenSandbox service URL used when resolving the
            model endpoint.
        extra_env: Optional environment variables injected verbatim into the sandbox.
        stage_files: Optional ``{remote_path: content}`` files written into the live sandbox
            before launch.
        patch_output_glob: When given, the patch is read from an ``output.jsonl`` under this
            path; otherwise it comes from ``git diff --cached`` on ``repo_workdir``.
        agent_timeout_s: Timeout in seconds for the agent run. Defaults to ``1800``.

    Returns:
        A dict with the instance id, model patch, resolution status, reward, whether a patch
        exists, whether the sample is masked, and the verifier's error kind.
    """
    from responses_api_agents.swe_env.verify_task import verify_task

    patch = await provision_and_extract_patch(
        task,
        provider=provider,
        agent_launch_command=agent_launch_command,
        model_server=model_server,
        opensandbox_service_url=opensandbox_service_url,
        extra_env=extra_env,
        stage_files=stage_files,
        patch_output_glob=patch_output_glob,
        agent_timeout_s=agent_timeout_s,
    )
    # Score the patch in the verifier's OWN fresh sandbox (decoupled, hermetic verification).
    report = await verify_task(provider, dataclasses.replace(task, model_patch=patch))
    masked = report.error_kind is not None
    return {
        "instance_id": task.instance_id,
        "model_patch": patch,
        "resolved": report.resolved,
        "reward": reward_from_report(report),
        "patch_exists": bool(patch.strip()),
        "mask_sample": masked,
        "error_kind": report.error_kind,
    }


# --- in-sandbox model-server egress --------------


class ModelEgressUnavailable(RuntimeError):
    """Raised when no sandbox-reachable model endpoint can be resolved for a provider."""


@dataclass(frozen=True)
class ModelEndpoint:
    """A sandbox-reachable model-server endpoint.

    Attributes:
        base_url: The base URL the in-sandbox agent uses to reach the model server.
        api_key: Optional API key for authenticating to the model server.
        model: Optional model name to use.
    """

    base_url: str
    api_key: str = ""
    model: str = ""

    def to_sandbox_env(self) -> dict[str, str]:
        """Build the minimal set of environment variables to inject into the sandbox.

        Returns:
            dict[str, str]: Environment variables carrying the base URL and,
            when set, the API key and model name. The global config dict is
            never included.
        """
        env = {"OPENAI_BASE_URL": self.base_url, "NEMO_GYM_MODEL_BASE_URL": self.base_url}
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        if self.model:
            env["NEMO_GYM_MODEL"] = self.model
        return env


def resolve(
    provider_name: str,
    model_server: Mapping[str, Any],
    *,
    host_loopback_url: str = "http://127.0.0.1:8000/v1",
    opensandbox_service_url: str | None = None,
) -> ModelEndpoint:
    """Resolve a sandbox-reachable model endpoint for a sandbox provider.

    Args:
        provider_name: The sandbox provider name (e.g. ``"apptainer"``,
            ``"opensandbox"``, ``"docker"``).
        model_server: Mapping describing the model server, read for the
            ``api_key``, ``model``, and ``base_url`` keys.
        host_loopback_url: Fallback URL used when the provider shares the host
            network namespace and no base URL is configured.
        opensandbox_service_url: Cluster-reachable Service/ingress URL used for
            the opensandbox provider when no other base URL is configured.

    Returns:
        ModelEndpoint: The resolved endpoint carrying the base URL, API key,
        and model name.

    Raises:
        ModelEgressUnavailable: If the opensandbox provider cannot resolve a
            cluster-reachable model-server URL (e.g. only loopback is available).
    """
    api_key = str(model_server.get("api_key", "") or "")
    model = str(model_server.get("model", "") or "")
    configured_base = str(model_server.get("base_url", "") or "")

    if provider_name == "opensandbox":
        base_url = opensandbox_service_url or configured_base
        if not base_url or "127.0.0.1" in base_url or "localhost" in base_url:
            raise ModelEgressUnavailable(
                "opensandbox needs a cluster-reachable model-server URL (k8s Service/ingress); "
                "loopback is unreachable from the pod. Configure 'opensandbox_service_url', or "
                "run the agent with the docker provider instead."
            )
    else:
        # docker / local: shares host network by default (host loopback reachable).
        base_url = configured_base or host_loopback_url

    return ModelEndpoint(base_url=base_url, api_key=api_key, model=model)

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

"""Sandbox recipe for mini-swe-agent 2, shared by the agent and by tooling.

`app.py` and `sandbox_environment.py` import these so there is one definition of
how a SWE-bench row becomes a sandbox. They also live here, apart from `app.py`,
so `gym sandbox debug` can import them: `app.py` pulls in Ray, mini-swe-agent,
and the SWE-bench harness at module scope, none of which the CLI has.

Keep this module's imports limited to the standard library and `nemo_gym`.
"""

import hashlib
import shlex
from collections.abc import Mapping
from typing import Any

from nemo_gym.sandbox import SandboxSpec
from nemo_gym.sandbox.hooks import replace_image, spec_from_mapping


def swebench_image_for_row(row: Mapping[str, Any] | None) -> str | None:
    """Return the SWE-bench evaluation image for a task row.

    A row may name its image directly; otherwise it is derived from the instance
    id. The `__` substitutions are not cosmetic — they are what the published
    image tags actually use, since Docker tags disallow the original separator.

    Returns:
        The image reference, or ``None`` when the row cannot name one.
    """
    if not row:
        return None

    image_name = row.get("image_name")
    if image_name:
        return str(image_name)

    instance_id = row.get("instance_id")
    if not instance_id:
        return None

    if str(row.get("subset", "")) == "verified":
        return f"docker.io/swebench/sweb.eval.x86_64.{str(instance_id).replace('__', '_1776_')}:latest".lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{str(instance_id).replace('__', '_s_')}:latest".lower()


def resource_profile_for_instance(
    resources: Mapping[str, Any] | None,
    profiles: list[dict[str, Any]] | None,
    instance_id: str,
) -> dict[str, Any]:
    """Pick this instance's resource profile.

    Selection is a hash of the instance id rather than a random draw so a given
    instance always lands on the same profile — otherwise a debug run and the
    rollout it is meant to reproduce could get different resources.
    """
    merged = dict(resources or {})
    if not profiles:
        return merged
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    merged.update(profiles[int.from_bytes(digest[:4], "big") % len(profiles)])
    return merged


def spec_for_row(row: Mapping[str, Any] | None, server_config: Mapping[str, Any]) -> SandboxSpec:
    """Build the sandbox spec this agent would create for a row.

    Wired in as `sandbox_task.spec_resolver`. With no row, it falls back to the
    declared `sandbox_spec` so tooling can still boot the configured sandbox.
    """
    raw_spec = dict(server_config.get("sandbox_spec") or {})
    instance_id = str((row or {}).get("instance_id") or "")

    if instance_id:
        raw_spec["resources"] = resource_profile_for_instance(
            raw_spec.get("resources"),
            server_config.get("sandbox_resource_profiles"),
            instance_id,
        )

    spec = spec_from_mapping(raw_spec)

    image = swebench_image_for_row(row)
    if image is not None:
        # spec_from_mapping already applied image_rewrites to any configured
        # image; a derived one has to go through the same rewrites to reach a
        # mirrored registry.
        from nemo_gym.sandbox import rewrite_image

        spec = replace_image(spec, rewrite_image(image, list(raw_spec.get("image_rewrites", []) or [])))

    if instance_id:
        # 63 chars is the Kubernetes label-value ceiling the provider sanitizes to.
        spec.metadata.update({"nemo_gym_agent": "mini_swe_agent_2", "instance_id": instance_id[:63]})
    return spec


def conda_activate_wrap(
    command: str,
    *,
    conda_env: str | None = None,
    activate_conda: bool = False,
    **_: Any,
) -> str:
    """Prefix a command with conda activation so it runs against the task's Python.

    Wired in as `sandbox_task.exec_wrapper`. Without this, `python` in a SWE-bench
    image is the system interpreter rather than the testbed one, and results do
    not match what the agent sees.
    """
    if not activate_conda or not conda_env:
        return command

    quoted_env = shlex.quote(conda_env)
    # Resolve conda from common install roots before activating. The sandbox exec
    # shell is non-login (apptainer/docker/ECS alike), so `conda` may not be on PATH
    # and `conda info --base` can't be relied on. The grouped loop (not an `&&` chain)
    # keeps a missing root from aborting the command; cwd is handled by exec(cwd=...),
    # so we don't `cd` here.
    return (
        '{ for __base in /opt/miniconda3 /opt/conda "$HOME/miniconda3" '
        '"$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null)"; do '
        '[ -n "$__base" ] && [ -f "$__base/etc/profile.d/conda.sh" ] && '
        '. "$__base/etc/profile.d/conda.sh" && break; done; } && '
        f"conda activate {quoted_env} && {command}"
    )

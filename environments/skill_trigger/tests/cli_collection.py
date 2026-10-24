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
"""Run a packaged synthetic task through Gym's normal server launcher and collector."""

import asyncio
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path

import yaml
from harbor.models.trial.config import TrialConfig

from benchmarks.skill_trigger.prepare import BENCHMARK_DIR, ROOT
from responses_api_agents.harbor_agent_general.app import HarborVerifyResponse


async def collect_cli(*, root: Path, config: TrialConfig, expect_failure: bool) -> HarborVerifyResponse:
    options = config.agent.kwargs
    harness = options["harness"]
    settings = {
        key: options[key]
        for key in (
            "release",
            "manifest_sha256",
            "harness",
            "gateway_host",
            "gateway_port",
            "node_binary",
            "provider",
            "api",
            "model_catalog_path",
            "model_catalog_sha256",
        )
        if key in options
    }
    settings.update(
        tasks=str(root / "tasks"), catalog="synthetic", jobs=str(root / "jobs"), evidence=str(root / "private")
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        head_port = sock.getsockname()[1]
    # A nonzero identity checks that collation preserves indices across catalog groups.
    row = json.loads((root / "tasks/catalogs/synthetic.jsonl").read_text())
    row["_ng_task_index"] = 37
    input_file = root / "cli-input.jsonl"
    input_file.write_text(json.dumps(row) + "\n")
    invocation = {
        "config_paths": [str(BENCHMARK_DIR / "configs" / harness / "config.yaml")],
        "skills": {"path": str(root / "release/catalogs/synthetic")},
        "skill_trigger_settings": settings,
        "policy_model_name": "test-model",
        "policy_base_url": options["upstream_url"],
        "default_host": "127.0.0.1",
        "head_server": {"host": "127.0.0.1", "port": head_port},
        "uv_venv_dir": str(root.parent / "server-venvs"),
        "python_version": "3.13.14",
        "skill_trigger": {
            "responses_api_agents": {
                "skill_trigger": {
                    "datasets": [
                        {
                            "name": "synthetic_skill_trigger",
                            "type": "benchmark",
                            "num_repeats": 1,
                            "jsonl_fpath": str(input_file),
                            "prepare_script": "environments/skill_trigger/package.py",
                        }
                    ]
                }
            }
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(invocation, sort_keys=False))
    command = [
        str(Path(sys.executable).with_name("gym")),
        "eval",
        "run",
        "--config",
        str(root / "config.yaml"),
        "--split",
        "benchmark",
        "--output",
        str(root / "rollouts.jsonl"),
    ]
    ray_tmp = tempfile.mkdtemp(prefix="skill-cli-ray-")
    (root / "ray-directory.txt").write_text(ray_tmp)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("NEMO_GYM_CONFIG_DICT", "NEMO_GYM_EXTRA_ROOTS", "RAY_ADDRESS")
    }
    env.update(
        SKILL_MODEL_API_KEY="dummy-not-a-real-key",
        NEMO_GYM_MAX_ROLLOUT_ATTEMPTS="1",
        UV_CACHE_DIR=str(root.parent / "uv-cache"),
        RAY_TMPDIR=ray_tmp,
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    )
    (root / "command.json").write_text(json.dumps(command, indent=2))
    with (root / "launcher.log").open("xb") as log:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            async with asyncio.timeout(900):
                returncode = await process.wait()
        finally:
            if process.returncode is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(process.wait(), timeout=30)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
    (root / "exit.json").write_text(json.dumps({"returncode": returncode}))

    def rows(name):
        path = root / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    scored, failed = rows("rollouts.jsonl"), rows("rollouts_failures.jsonl")
    materialized = rows("rollouts_materialized_inputs.jsonl")
    assert len(materialized) == 1, f"Inspect {root / 'launcher.log'}"
    assert materialized[0]["_ng_task_index"] == 37
    assert materialized[0]["skills_ref"]["path"] == str(root / "release/catalogs/synthetic")
    if expect_failure:
        assert returncode and not scored and len(failed) == 1
        assert failed[0]["_ng_failure_class"] == "harbor_failed"
        selected = failed[0]
    else:
        assert returncode == 0 and len(scored) == 1 and not failed, f"Inspect {root / 'launcher.log'}"
        selected = scored[0]
        assert selected["skills_ref"] == materialized[0]["skills_ref"]
        aggregates = json.loads((root / "rollouts_aggregate_metrics.json").read_text())
        assert len(aggregates) == 1
        assert aggregates[0]["key_metrics"] == {
            "skill_opening_rate": selected["reward"],
            "skills_scored": 1,
            "trials_scored": 1,
        }
        assert aggregates[0]["group_level_metrics"][0]["_ng_task_index"] == 37
    (root / "collection.json").write_text(
        json.dumps(
            {
                "scored": len(scored),
                "failed": len(failed),
                "planned": len(materialized),
                "returncode": returncode,
                "launcher_tested": True,
                "aggregation_enabled": True,
            },
            indent=2,
        )
        + "\n"
    )
    return HarborVerifyResponse.model_validate(selected)

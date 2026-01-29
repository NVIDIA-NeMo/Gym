# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ast
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

import ray


sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


log = logging.getLogger(__name__)

from resources_servers.swerl_gen.eval.reward_functions import (
    _calculate_patch_gen_reward,
    _calculate_test_gen_reward,
)


EVAL_TIMEOUT = 600


def _run_instance(
    instance_info_base64: str,
    inference_results_base64: str,
    repro_test_info_base64: str,
    image: str,
    mode: str,
    timeout: int,
    debug: bool,
    script_dir: Optional[str] = None,
):
    """Run evaluation instance in singularity container asynchronously."""

    resolution = None
    return_codes_after_patch = None
    return_codes_before_patch = None
    verification_result = None

    # Resolve the host path to ``eval_instance.py`` which lives in the ``eval`` directory
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script_path = os.path.join(eval_dir, "eval_instance.py")

    # To avoid "Argument list too long" errors with very large base64-encoded
    # payloads, write them to temporary files in the eval directory and pass
    # only the file paths as CLI arguments.
    instance_info_file = None
    inference_results_file = None
    repro_test_info_file = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".b64", dir=eval_dir, delete=False) as f:
            f.write(instance_info_base64 or "")
            instance_info_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".b64", dir=eval_dir, delete=False) as f:
            f.write(inference_results_base64 or "")
            inference_results_file = f.name

        if repro_test_info_base64 is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".b64", dir=eval_dir, delete=False) as f:
                f.write(repro_test_info_base64 or "")
                repro_test_info_file = f.name

        # Build the singularity exec command
        cmd = [
            "singularity",
            "exec",
            "--writable-tmpfs",
            "--no-home",
        ]
        cmd.extend(["--bind", f"{eval_dir}:{eval_dir}"])
        if script_dir and script_dir != eval_dir:
            cmd.extend(["--bind", f"{script_dir}:{script_dir}"])

        cmd.append(image)

        # Append the python executable and its arguments directly, avoiding an
        # intermediate ``bash -c`` layer so that very long arguments are not
        # mis-parsed as a single command/filename by the shell.
        cmd.extend(
            [
                "python",
                eval_script_path,
                "--instance_info_file",
                instance_info_file,
                "--inference_results_file",
                inference_results_file,
                "--mode",
                mode,
            ]
        )
        if repro_test_info_file is not None:
            cmd.extend(["--repro_test_info_file", repro_test_info_file])
        if script_dir is not None:
            cmd.extend(["--script_dir", script_dir])

        if debug:
            print(f"Executing command: {' '.join(cmd)}")

        timed_out = False

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Wait for the process to complete with a timeout
            try:
                outs, errs = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()
                timed_out = True

            # Read stdout/stderr for resolution or test returns
            combined_output = ""
            if errs:
                combined_output += errs
            if outs:
                combined_output += "\n" + outs if combined_output else outs
            output_lines = combined_output.splitlines() if combined_output else []
            if debug:
                print("output_lines", output_lines)
            if output_lines:
                if mode == "eval":
                    # In eval mode, look for the resolution in the last few lines
                    for line in reversed(output_lines):
                        line = line.strip()
                        for res_str in [
                            "RESOLVED_FULL",
                            "RESOLVED_PARTIAL",
                            "RESOLVED_NO",
                        ]:
                            if res_str in line:
                                resolution = res_str
                                break
                        if resolution is not None:
                            break  # Found resolution
                elif mode == "repro-gen":
                    for line in reversed(output_lines):
                        line = line.strip()
                        if "[Return codes before patch]:" in line:
                            match = line.split("[Return codes before patch]:")[1].strip()
                            if match.endswith("]"):
                                try:
                                    return_codes_before_patch = ast.literal_eval(match)
                                except Exception:
                                    pass
                        elif "[Return codes after patch]:" in line:
                            match = line.split("[Return codes after patch]:")[1].strip()
                            if match.endswith("]"):
                                try:
                                    return_codes_after_patch = ast.literal_eval(match)
                                except Exception:
                                    pass
            status = "timeout" if timed_out else "done"
        except Exception as e:
            print(f"Exception during subprocess execution: {e}")
            outs, errs = "", ""
            status = "error"
    finally:
        # Best-effort cleanup of temporary files.
        for path in (instance_info_file, inference_results_file, repro_test_info_file):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    verification_result = {
        "status": status,
        "resolution": resolution,
        "return_codes_after_patch": return_codes_after_patch,
        "return_codes_before_patch": return_codes_before_patch,
    }
    return verification_result


# Using SPREAD scheduling so that Ray assigns tasks to as many distinct nodes as possible.
@ray.remote(scheduling_strategy="SPREAD")
def compute_score(
    extra_info_base64: str,
    patch_str: str,
    repro_test_info_base64: Optional[str],
    mode: str,
    timeout: int = EVAL_TIMEOUT,
    debug: bool = False,
) -> Tuple[float, Dict]:
    """Ray wrapper around ``calculate_execution_feedback_reward`` for remote execution."""
    return calculate_execution_feedback_reward(
        extra_info_base64=extra_info_base64,
        patch_str=patch_str,
        repro_test_info_base64=repro_test_info_base64,
        mode=mode,
        timeout=timeout,
        debug=debug,
    )


def calculate_execution_feedback_reward(
    extra_info_base64: str,
    patch_str: str,
    repro_test_info_base64: str,
    mode: str,
    timeout: int,
    debug: bool,
    scale_factor: float = 1.0,
) -> Tuple[float, Optional[Dict]]:
    """Compute a reward and verification metadata using a Singularity sandbox for a single instance.
    - ``eval`` mode: checks against the PASS_TO_PASS and FAIL_TO_PASS unittests
    - ``repro-gen`` mode: checks if the generated test can correctly reproduce the bug and return exit code 0 when the patch is applied
    """
    # Validate required configuration
    extra_info = json.loads(base64.b64decode(extra_info_base64).decode())
    required_fields = ["image", "instance_info"]
    missing_fields = [field for field in required_fields if not extra_info.get(field)]
    if missing_fields:
        log.warning("Missing required fields in extra_info: %s", missing_fields)
        return 0.0, None

    instance_info = extra_info.get("instance_info")
    image = extra_info.get("image")
    instance_id = instance_info.get("instance_id")
    if isinstance(instance_info, dict):
        instance_info_base64 = base64.b64encode(json.dumps(instance_info).encode()).decode()
    else:
        instance_info_base64 = instance_info

    inference_data = {
        "instance_id": instance_id,
        "model_patch": patch_str,
    }
    inference_results_base64 = base64.b64encode(json.dumps(inference_data).encode()).decode()

    verification_result = _run_instance(
        instance_info_base64=instance_info_base64,
        inference_results_base64=inference_results_base64,
        repro_test_info_base64=repro_test_info_base64 or "",
        image=image,
        mode=mode,
        timeout=timeout,
        debug=debug,
        script_dir=None,
    )

    if mode == "repro-gen":
        reward = _calculate_test_gen_reward(verification_result, scale_factor)
    else:
        reward = _calculate_patch_gen_reward(verification_result, scale_factor)

    if debug:
        print("Verification completed for instance %s. Reward: %s", instance_id, reward)
    return reward, verification_result

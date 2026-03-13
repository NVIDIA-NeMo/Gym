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
Debug trajectory writer -- saves each CUA rollout as a self-contained directory.

Output structure:
    <output_dir>/<env_id>/
        screenshots/
            00_initial.png
            01_after.png
            02_after.png
            ...
        conversation.json       # full agent interaction (actions, URLs, raw provider responses -- no base64)
        verification.json       # reward, local storage dump, verification metadata
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from resources_servers.browser_gym.schemas import CUATrajectory


logger = logging.getLogger(__name__)


def _strip_base64_fields(obj: Any) -> Any:
    """Recursively strip base64 image data from dicts/lists, replacing with a placeholder."""
    if isinstance(obj, dict):
        return {k: _strip_base64_fields(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_base64_fields(item) for item in obj]
    if isinstance(obj, str) and (obj.startswith("data:image") or (len(obj) > 500 and _looks_like_base64(obj))):
        return "<base64_image>"
    return obj


def _looks_like_base64(s: str) -> bool:
    clean = s.rstrip("=")
    if len(clean) < 100:
        return False
    import re

    return bool(re.match(r"^[A-Za-z0-9+/\n]+$", clean[:200]))


def _save_b64_png(b64_data: str, path: Path) -> None:
    raw = b64_data
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    path.write_bytes(base64.b64decode(raw))


def save_debug_trajectory(
    output_dir: str,
    env_id: str,
    trajectory: CUATrajectory,
    reward: Optional[float] = None,
    local_storage_dump: Optional[str] = None,
    adapter_type: str = "",
    model_name: str = "",
    verifier_metadata: Optional[Dict[str, Any]] = None,
    verification_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Write trajectory screenshots, conversation, and verification to disk."""
    rollout_dir = Path(output_dir) / env_id
    screenshots_dir = rollout_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    if trajectory.initial_screenshot:
        _save_b64_png(trajectory.initial_screenshot, screenshots_dir / "00_initial.png")

    conversation_steps = []
    for idx, step in enumerate(trajectory.steps, start=1):
        _save_b64_png(step.screenshot_after, screenshots_dir / f"{idx:02d}_after.png")

        conversation_steps.append(
            {
                "step": idx,
                "action": step.action.model_dump(exclude_none=True),
                "current_url": step.current_url,
                "screenshot": f"screenshots/{idx:02d}_after.png",
                "raw_provider_response": _strip_base64_fields(step.raw_provider_response),
            }
        )

    conversation = {
        "env_id": env_id,
        "adapter_type": adapter_type,
        "model": model_name,
        "task_prompt": trajectory.task_prompt,
        "final_message": trajectory.final_message,
        "num_steps": len(trajectory.steps),
        "steps": conversation_steps,
    }

    (rollout_dir / "conversation.json").write_text(json.dumps(conversation, indent=2, ensure_ascii=False))

    ls_parsed = None
    if local_storage_dump:
        try:
            ls_parsed = json.loads(local_storage_dump)
        except (json.JSONDecodeError, TypeError):
            ls_parsed = local_storage_dump

    verification = {
        "env_id": env_id,
        "reward": reward,
        "get_actual_state_response": verification_result,
        "verifier_metadata": verifier_metadata,
        "local_storage_dump": ls_parsed,
    }

    (rollout_dir / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False))

    logger.info(f"Debug trajectory saved to {rollout_dir} ({len(trajectory.steps)} steps, reward={reward})")
    return str(rollout_dir)

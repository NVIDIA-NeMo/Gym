# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free EQ-Bench 4 verifier checks: a stand-in judge names a fixed winner on every dimension.

Fixture rows carry a short synthetic pairwise prompt whose respondent sections read ``First <code>:`` and
``Second <code>:``; the policy's transcript is recognisable by ``fixture_policy_marker``. The stand-in judge reads the
codes from the prompt and gives ``fixture_winner`` (``policy`` or ``opponent``) five plus signs on every dimension.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymResponse


if TYPE_CHECKING:
    from resources_servers.eqbench4.app import EQBench4ResourcesServer, EQBench4VerifyRequest, EQBench4VerifyResponse

_SECTION = re.compile(r"(First|Second) (\w+):\n(.*?)(?=\n(?:First|Second) \w+:\n|\Z)", re.DOTALL)


def create_fixture_server() -> "EQBench4ResourcesServer":
    """The benchmark's resources server, configured from its benchmark config, without model services."""
    from resources_servers.eqbench4.app import EQBench4ResourcesServer, EQBench4ResourcesServerConfig

    config_path = Path(__file__).resolve().parents[2] / "benchmarks/eqbench4/config.yaml"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(config_path)]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    values = OmegaConf.to_container(
        resolved.eqbench4_benchmark_resources_server.resources_servers.eqbench4, resolve=True
    )
    values["name"] = "eqbench4_benchmark_resources_server"
    config = EQBench4ResourcesServerConfig.model_validate(values)
    return EQBench4ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def fixture_verdict(prompt: str, *, winner: str, policy_marker: str, dimensions: list[str]) -> str:
    """The stand-in judge's reply to one fixture prompt: a pairwise verdict, or trait scores for a trait prompt."""
    sections = {label: (code, text) for label, code, text in _SECTION.findall(prompt)}
    if "First" not in sections or "Second" not in sections:
        return '[SCORES]\n```json\n{"fixture_trait": 5}\n```\n[/SCORES]'
    policy_label = "First" if policy_marker in sections["First"][1] else "Second"
    opponent_label = "Second" if policy_label == "First" else "First"
    code = sections[policy_label if winner == "policy" else opponent_label][0]
    return json.dumps({dimension: f"{code}+++++" for dimension in dimensions})


async def invoke_with_fixture_judge(
    server: "EQBench4ResourcesServer", request: "EQBench4VerifyRequest"
) -> "EQBench4VerifyResponse":
    """Verify a fixture row with the stand-in judge answering every judge call."""
    extra = request.model_extra or {}
    winner = extra.get("fixture_winner", "policy")
    marker = extra.get("fixture_policy_marker", "")

    async def post(**kwargs: Any) -> MagicMock:
        payload = kwargs["json"]
        last = payload.input[-1] if hasattr(payload, "input") else payload["input"][-1]
        prompt = last.content if hasattr(last, "content") else last["content"]
        text = fixture_verdict(
            prompt, winner=winner, policy_marker=marker, dimensions=list(request.pairwise_dimensions)
        )
        reply = NeMoGymResponse(
            id=f"fixture_judge_{abs(hash(prompt))}",
            created_at=0,
            model="fixture",
            object="response",
            status="completed",
            output=[
                {
                    "id": "fixture_message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        response = MagicMock(status=200)
        response.read = AsyncMock(return_value=reply.model_dump_json().encode())
        return response

    server.server_client.post = AsyncMock(side_effect=post)
    return await server.verify(request)

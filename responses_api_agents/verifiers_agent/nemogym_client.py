# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep token ids returned by Gym's OpenAI-compatible model server.

Verifiers' TrainClient requires a raw vLLM endpoint, so Gym extends EvalClient instead.
"""

from collections.abc import Mapping
from typing import Any

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.types import Response, TurnTokens


def _extract(raw: dict) -> tuple[list[int], list[int], list[float]] | None:
    """Pull Gym's token-id extension fields off a chat-completion body."""
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    choice = choices[0] if choices else {}

    prompt_ids = raw.get("prompt_token_ids") or message.get("prompt_token_ids")
    completion_ids = message.get("generation_token_ids") or choice.get("token_ids")
    logprobs = message.get("generation_log_probs")

    if not prompt_ids or not completion_ids:
        return None
    return (
        [int(t) for t in prompt_ids],
        [int(t) for t in completion_ids],
        [float(lp) for lp in (logprobs or [])],
    )


class NeMoGymClient(EvalClient):
    """`EvalClient` plus provider-returned token ids."""

    async def get_response(
        self,
        dialect: Any,
        body: dict,
        model: str,
        sampling_args: Any,
        session_id: str | None = None,
        turn: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        response = await super().get_response(
            dialect, body, model, sampling_args, session_id, turn, headers
        )
        found = _extract(response.raw or {})
        if found is None:
            return response
        prompt_ids, completion_ids, logprobs = found
        response.tokens = TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=logprobs,
        )
        return response


def install() -> None:
    """Use NeMoGymClient where Verifiers would use EvalClient.

    Verifiers has no client registry and binds resolve_client in several modules.
    """
    import verifiers.v1 as vf
    import verifiers.v1.clients as clients_pkg
    import verifiers.v1.clients.client as client_mod
    import verifiers.v1.interception.server as server_mod
    from verifiers.v1.configs.client import TrainClientConfig

    if getattr(client_mod.resolve_client, "_nemo_gym_client", False):
        return
    original = client_mod.resolve_client

    def resolve_client(config):
        if isinstance(config, TrainClientConfig):
            return original(config)
        return NeMoGymClient(config)

    resolve_client._nemo_gym_client = True
    for module in (client_mod, clients_pkg, server_mod, vf):
        if hasattr(module, "resolve_client"):
            module.resolve_client = resolve_client

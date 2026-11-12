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
"""Sampling values that a training framework sets for a rollout's model calls, and how they reach each call.

A training framework that serves the policy samples the calls of the agents it trains with its own values: its
training values, or its validation values for a validation rollout. It sets them on each rollout's row under
``_ng_model_sampling``, next to ``_ng_rollout_id``. Gym carries them into the URL of every model call the rollout's
agent makes, as ``/ng-sampling/<values>`` after any other prefix, and the Model Server writes them into the request.
Gym never interprets them.

In such a run, a call from a caller the framework does not train, such as a verifier's LLM judge, carries
``/ng-sampling/null``: it keeps the sampling its caller chose, and its own Model Server's ``sampling_overrides``.
A run whose rows carry no values, such as Gym-only evaluation, adds no segment.
"""

import base64
import binascii
import dataclasses
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config.config_types import MODEL_SAMPLING_PATH_SEGMENT
from nemo_gym.core.correlation import CorrelationContext, CorrelationPathError, encode, parse_path


class ModelCallSampling(BaseModel):
    """Sampling values for the model calls of an agent that a training framework trains.

    The Model Server sends the smaller of ``max_output_tokens`` and the call's own limit, so a harness that reserves
    fewer tokens, near the end of its context or for a summary, keeps its limit. An unset field leaves the call's own
    value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    temperature: Optional[float] = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    # vLLM requires 0 < top_p <= 1.
    top_p: Optional[float] = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    # The largest integer that JSON readers using doubles represent exactly.
    max_output_tokens: Optional[int] = Field(default=None, gt=0, le=2**53 - 1)


# The segment value of a call whose caller keeps its own sampling.
MODEL_SAMPLING_NULL = "null"
# vllm_model sends this header, valued ``null``, on the Chat Completions request of a ``null`` call, so a training
# framework's generation server can tell a call it does not train from its trained agents' calls. Others ignore it.
MODEL_SAMPLING_HEADER = "x-ng-sampling"
# Room for every field, and well within the URL lengths that servers and proxies accept.
_MAX_ENCODED_LENGTH = 1024
_ENCODED = re.compile(r"[A-Za-z0-9_-]+")


def encode_model_sampling(sampling: Optional[ModelCallSampling]) -> str:
    """Encode values as the segment's unpadded base64url JSON, and ``None`` as ``null``."""
    if sampling is None:
        return MODEL_SAMPLING_NULL
    payload = json.dumps(sampling.model_dump(exclude_none=True), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Model sampling sets {name!r} twice")
        result[name] = value
    return result


def decode_model_sampling(encoded: str) -> Optional[ModelCallSampling]:
    """Decode a segment value: ``null`` to ``None``, anything else to values.

    Raises ``ValueError`` for a malformed or ambiguous value, so a call never runs with values it did not send.
    """
    if encoded == MODEL_SAMPLING_NULL:
        return None
    if len(encoded) > _MAX_ENCODED_LENGTH or not _ENCODED.fullmatch(encoded):
        raise ValueError("Model sampling must be null or unpadded base64url JSON")
    try:
        raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        values = json.loads(raw, object_pairs_hook=_unique_keys)
    except (binascii.Error, UnicodeDecodeError) as error:
        raise ValueError("Model sampling must be null or unpadded base64url JSON") from error
    return ModelCallSampling.model_validate(values)


def model_sampling_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[ModelCallSampling]:
    """Return the values a run request or row carries under ``_ng_model_sampling``, or ``None``."""
    # Imported here because global_config imports feature modules that may import this module.
    from nemo_gym.config.global_config import MODEL_SAMPLING_KEY_NAME

    if isinstance(body, Mapping):
        value = body.get(MODEL_SAMPLING_KEY_NAME)
    elif isinstance(body, BaseModel):
        # BaseRunRequest declares the key as ``model_sampling``; other models may keep it as an extra field.
        value = getattr(body, "model_sampling", None) or (body.model_extra or {}).get(MODEL_SAMPLING_KEY_NAME)
    else:
        return None
    if value is None or isinstance(value, ModelCallSampling):
        return value
    return ModelCallSampling.model_validate(value)


_MODEL_SAMPLING_SEGMENT: ContextVar[Optional[str]] = ContextVar("nemo_gym_model_sampling_segment", default=None)


def current_model_sampling_segment() -> Optional[str]:
    """Return the segment value for this request's model calls, or ``None`` when they carry no segment."""
    return _MODEL_SAMPLING_SEGMENT.get()


@contextmanager
def model_sampling_context(segment: Optional[str]) -> Iterator[None]:
    """Give the model calls made inside the block the segment value ``segment``."""
    token = _MODEL_SAMPLING_SEGMENT.set(segment)
    try:
        yield
    finally:
        _MODEL_SAMPLING_SEGMENT.reset(token)


def model_sampling_path_segment(segment: Optional[str]) -> str:
    """Return ``/ng-sampling/<segment>``, or ``""`` for no segment."""
    return f"/{MODEL_SAMPLING_PATH_SEGMENT}/{segment}" if segment is not None else ""


# The Model Server generation APIs that a sampling segment applies to.
_GENERATION_API_PATHS = frozenset({"/v1/chat/completions", "/v1/responses", "/v1/messages"})


def with_model_sampling_segment(url_path: str, segment: Optional[str]) -> str:
    """Insert ``/ng-sampling/<segment>`` into a Model Server generation path, where its correlation prefix puts it.

    Other paths, and paths that already carry a segment or a malformed prefix, are returned unchanged.
    """
    if segment is None:
        return url_path
    path, separator, query = url_path.partition("?")
    try:
        parsed = parse_path(path)
    except CorrelationPathError:
        return url_path
    context, api_path = parsed if parsed is not None else (CorrelationContext(), path)
    if context.model_sampling is not None or api_path not in _GENERATION_API_PATHS:
        return url_path
    return f"{encode(dataclasses.replace(context, model_sampling=segment))}{api_path}{separator}{query}"


class ModelCallSamplingConfig(BaseModel):
    """The global ``model_call_sampling`` block. A training framework sets it when its rows carry values.

    Its presence marks the run: resources servers then send ``null`` with the model calls they make through Gym's
    server client while handling a request, so their judges keep their own sampling.
    """

    model_config = ConfigDict(extra="forbid")

    # Model Servers of the agents the framework trains. A generation call without a segment fails on them, so a path
    # that drops the values fails instead of sampling with others. They cannot also pin sampling with
    # ``sampling_overrides``; they may pin only ``logprobs`` and ``top_logprobs``, which change what the engine reports.
    required_model_servers: list[str] = Field(default_factory=list)


def model_call_sampling_config(global_config_dict: Optional[Mapping[str, Any]]) -> Optional[ModelCallSamplingConfig]:
    """Return the run's ``model_call_sampling`` block, or ``None`` when the run sets none.

    Raises ``ValueError`` when ``required_model_servers`` names a server that is not a Model Server of the run: a
    misspelled name would otherwise leave the trained agent's Model Server unenforced, without an error.
    """
    from nemo_gym.config.global_config import MODEL_CALL_SAMPLING_BLOCK, MODEL_SERVER_TYPE_KEY_NAME

    if not isinstance(global_config_dict, Mapping):
        return None
    block = global_config_dict.get(MODEL_CALL_SAMPLING_BLOCK)
    if block is None:
        return None
    config = ModelCallSamplingConfig.model_validate(block)
    model_servers = {
        name
        for name, value in global_config_dict.items()
        if isinstance(value, Mapping) and MODEL_SERVER_TYPE_KEY_NAME in value
    }
    if unknown := sorted(set(config.required_model_servers) - model_servers):
        raise ValueError(
            f"{MODEL_CALL_SAMPLING_BLOCK}.required_model_servers names servers that are not Model Servers in this "
            f"run: {unknown}"
        )
    return config

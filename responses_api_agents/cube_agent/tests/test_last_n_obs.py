# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from responses_api_agents.cube_agent.app import rebuild_input_last_n_env_observations


def _msg(text: str) -> NeMoGymEasyInputMessage:
    return NeMoGymEasyInputMessage(role="user", content=text, type="message")


def _asst(text: str, mid: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=mid,
        content=[NeMoGymResponseOutputText(annotations=[], text=text)],
    )


def test_rebuild_keeps_all_when_within_cap() -> None:
    prefix = [_msg("prefix")]
    seed = [_msg("seed")]
    m1 = [_asst("a1", "id1")]
    o1 = [_msg("obs1")]
    out = rebuild_input_last_n_env_observations(prefix, seed, [m1], [o1], n=5, placeholder="SKIP")
    texts = [x.content if isinstance(x, NeMoGymEasyInputMessage) else x.content[0].text for x in out]  # type: ignore[union-attr]
    assert texts == ["prefix", "seed", "a1", "obs1"]


def test_rebuild_drops_middle_and_inserts_placeholder() -> None:
    prefix = [_msg("p")]
    seed = [_msg("seed")]
    models = [[_asst("m1", "i1")], [_asst("m2", "i2")], [_asst("m3", "i3")]]
    obs = [[_msg("o1")], [_msg("o2")], [_msg("o3")]]
    out = rebuild_input_last_n_env_observations(prefix, seed, models, obs, n=2, placeholder="OMIT")
    # L = 4 blocks (seed + 3 steps), n=2 -> keep indices 0 and 3 -> seed + m3+o3, placeholder between seed and m3
    assert isinstance(out[0], NeMoGymEasyInputMessage) and out[0].content == "p"
    assert isinstance(out[1], NeMoGymEasyInputMessage) and out[1].content == "seed"
    assert isinstance(out[2], NeMoGymEasyInputMessage) and out[2].content == "OMIT"
    assert isinstance(out[3], NeMoGymResponseOutputMessage) and out[3].content[0].text == "m3"  # type: ignore[union-attr]
    assert isinstance(out[4], NeMoGymEasyInputMessage) and out[4].content == "o3"

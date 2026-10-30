# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure synthetic NOOA history capture, persistence, and native replay costs.

Run as a module in the NOOA agent environment. Timings and incremental Python
allocation peaks are separate runs, so tracemalloc does not distort timings.
"""

import argparse
import gc
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable

from nooa.storage.serialization import deserialize, serialize
from nooa.unifiedllm import LLMResponse

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessageForTraining, NeMoGymResponseOutputText
from responses_api_agents.nooa_agent.gym_llm import _assistant_parts, _responses_input


_SCOPE = 'nemo-gym/responses/v1:["synthetic", "gym-policy"]'


def _measure(operation: Callable[[], object], *, repeats: int) -> dict[str, float | int]:
    seconds = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = operation()
        seconds.append(time.perf_counter() - start)
        del result
    gc.collect()
    tracemalloc.start()
    result = operation()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"median_seconds": statistics.median(seconds), "retained_bytes": retained, "peak_bytes": peak}


def _response(turn: int, *, prompt: int, generated: int, layers: int, top_k: int) -> NeMoGymResponse:
    # Distinct nested containers per token/layer; expert IDs span 256 experts.
    output = NeMoGymResponseOutputMessageForTraining(
        id=f"message-{turn}",
        content=[NeMoGymResponseOutputText(text="Synthetic answer. " * generated, annotations=[])],
        prompt_token_ids=list(range(prompt)),
        generation_token_ids=list(range(generated)),
        generation_log_probs=[-0.125 - (token % 8) / 100 for token in range(generated)],
        routed_experts=[
            [[(token + layer + expert) % 256 for expert in range(top_k)] for layer in range(layers)]
            for token in range(generated)
        ],
    )
    return NeMoGymResponse(
        id=f"response-{turn}",
        created_at=0.0,
        model="synthetic",
        object="response",
        output=[output],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _capture(responses: list[NeMoGymResponse]) -> list[LLMResponse]:
    return [
        LLMResponse(raw_response=response, parts=_assistant_parts(response), replay_scope=_SCOPE)
        for response in responses
    ]


def _snapshot(history: list[LLMResponse]) -> str:
    blob, _ = serialize(history)
    return json.dumps(blob, separators=(",", ":"))


def main() -> None:
    """Print one JSON measurement record per requested history length."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--generated-tokens", type=int, default=512)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.turns, args.prompt_tokens, args.generated_tokens, args.layers, args.top_k, args.repeats) <= 0:
        parser.error("dimensions and repeats must be positive")
    for turns in args.turns:
        responses = [
            _response(
                turn, prompt=args.prompt_tokens, generated=args.generated_tokens, layers=args.layers, top_k=args.top_k
            )
            for turn in range(turns)
        ]
        capture = _measure(lambda responses=responses: _capture(responses), repeats=args.repeats)
        history = _capture(responses)
        snapshot = _snapshot(history)
        blob, allowlist = serialize(history)
        restored = deserialize(json.loads(snapshot), allowlist)
        del blob
        replayed, _ = _responses_input(restored, replay_scope=_SCOPE)
        expected = [
            item.model_dump(mode="json", exclude_none=True) for response in responses for item in response.output
        ]
        assert replayed == expected, "snapshot replay must preserve every training field"
        del replayed, expected
        # The comparison removes arrays only in a separate measurement fixture.
        text_only = [LLMResponse(content=turn.content) for turn in history]
        # Histories are JSON arrays of independent envelopes. Compute total bytes
        # for one full-history checkpoint per turn without repeatedly dumping all prefixes.
        turn_bytes = [len(_snapshot([turn]).encode()) - 2 for turn in history]
        prefix_bytes = sum(2 + sum(turn_bytes[:end]) + end - 1 for end in range(1, turns + 1))
        record = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dimensions": {**vars(args), "turns": turns},
            "routed_expert_integers": turns * args.generated_tokens * args.layers * args.top_k,
            "snapshot_bytes": len(snapshot.encode()),
            "text_only_snapshot_bytes": len(_snapshot(text_only).encode()),
            "all_prefix_checkpoint_bytes": prefix_bytes,
            "capture": capture,
            "snapshot": _measure(lambda history=history: _snapshot(history), repeats=args.repeats),
            "restore": _measure(
                lambda snapshot=snapshot: deserialize(json.loads(snapshot), allowlist), repeats=args.repeats
            ),
            "replay": _measure(
                lambda restored=restored: _responses_input(restored, replay_scope=_SCOPE), repeats=args.repeats
            ),
        }
        print(json.dumps(record), flush=True)
        del responses, history, restored, snapshot, text_only
        gc.collect()


if __name__ == "__main__":
    main()

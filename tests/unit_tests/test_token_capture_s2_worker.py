# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""S2 worker-side capture: the engine-blind core and the vLLM adapter.

The ordering matrix (stage-before-respond, fail-closed degradation to
``capture_failed`` coords) is tested against a mock adapter + mock sink so it
is backend-independent; the vLLM adapter's splice and extraction are tested
against fixed token vectors with a fake tokenizer (the live per-template GPU
validation ran at the S1 gate and re-runs as S2 gate evidence).
"""

from typing import Any, Optional

import pytest

from nemo_gym.token_id_capture.staging import install_capture as staging_install_capture
from nemo_gym.token_id_capture.staging.capture import (
    ActiveCall,
    CaptureError,
    CaptureHost,
    RolloutTokenCapture,
    StreamingUnsupportedError,
    install_capture,
)
from nemo_gym.token_id_capture.staging.digest import compute_staging_digest
from nemo_gym.token_id_capture.staging.protocols import install_capture as protocols_install_capture
from nemo_gym.token_id_capture.staging.records import StagedCallRecord, StageResult
from nemo_gym.token_id_capture.adapters.vllm import (
    PREFIX_IDS_FIELD,
    VLLMCaptureAdapter,
    extract_generation_token_info,
    replace_prefix_tokens,
)


class _MemorySink:
    """Records every staged record and the order of stage calls."""

    def __init__(self, *, fail: bool = False, raise_error: bool = False) -> None:
        self.records: list[StagedCallRecord] = []
        self.events: list[str] = []
        self.fail = fail
        self.raise_error = raise_error

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.events.append(f"stage:{record.staging_key}")
        if self.raise_error:
            raise RuntimeError("sink exploded")
        if self.fail:
            return StageResult(ok=False, staging_key=record.staging_key, error="disk full")
        self.records.append(record)
        return StageResult(ok=True, staging_key=record.staging_key)


class _MockAdapter:
    """Engine-blind test double: payloads carry the arrays directly."""

    def enter_prefix(self, request_payload: dict[str, Any], prefix_ids: list[int]) -> dict[str, Any]:
        request_payload["prefix_ids"] = list(prefix_ids)
        return request_payload

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        return list(response_payload["prompt_ids"])

    def extract_generation(self, response_payload: dict[str, Any]) -> tuple[list[int], list[float]]:
        return list(response_payload["gen_ids"]), list(response_payload["gen_logprobs"])


def _capture(
    sink: Optional[_MemorySink] = None,
    *,
    weight_version: int = 7,
    adapter: Optional[Any] = None,
) -> tuple[RolloutTokenCapture, _MemorySink]:
    sink = sink if sink is not None else _MemorySink()
    capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: weight_version,
        adapter=adapter,
    )
    return capture, sink


def _root_call(capture: RolloutTokenCapture, **overrides: Any) -> ActiveCall:
    kwargs: dict[str, Any] = dict(rollout_id="g7_r0", call_id="c1", mode="text")
    kwargs.update(overrides)
    return capture.begin_call(**kwargs)


# ---------------------------------------------------------------------------
# begin_call: local invariants
# ---------------------------------------------------------------------------


def test_begin_call_stamps_weight_version_at_begin() -> None:
    versions = iter([3, 9])
    capture, _ = _capture()
    capture._weight_version_fn = lambda: next(versions)  # simulate a refit between calls
    first = _root_call(capture)
    second = _root_call(capture, call_id="c2")
    assert (first.weight_version, second.weight_version) == (3, 9)


def test_begin_call_rejects_streaming() -> None:
    capture, _ = _capture()
    with pytest.raises(StreamingUnsupportedError):
        _root_call(capture, stream=True)


def test_begin_call_token_in_requires_parent_and_prev_len() -> None:
    capture, _ = _capture()
    with pytest.raises(CaptureError):
        capture.begin_call(rollout_id="r", call_id="c2", mode="token_in", prev_len=5)
    with pytest.raises(CaptureError):
        capture.begin_call(rollout_id="r", call_id="c2", mode="token_in", parent_call_id="c1", prev_len=0)


def test_begin_call_text_mode_must_be_root() -> None:
    capture, _ = _capture()
    with pytest.raises(CaptureError):
        capture.begin_call(rollout_id="r", call_id="c2", mode="text", parent_call_id="c1", prev_len=3)


# ---------------------------------------------------------------------------
# complete_call: the fail-closed ordering matrix
# ---------------------------------------------------------------------------


def test_complete_call_stages_before_returning_coords() -> None:
    capture, sink = _capture()
    call = _root_call(capture)
    coords = capture.complete_call(
        call,
        prompt_token_ids=[10, 11, 12],
        generated_token_ids=[13, 14],
        generated_logprobs=[-0.1, -0.2],
    )
    # The sink saw the bytes before any coords existed (the ack releases the
    # child-enabling marker, so this ordering IS the fail-closed guarantee).
    assert sink.events == ["stage:g7_r0/c1"]
    assert coords.disposition == "staged"
    assert coords.staging_key == "g7_r0/c1"
    assert (coords.delta_len, coords.cum_len) == (5, 5)
    assert coords.token_ids_delta == [10, 11, 12, 13, 14]
    assert coords.weight_version == 7

    record = sink.records[0]
    assert record.token_mask_delta == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert record.generation_logprobs_delta == [0.0, 0.0, 0.0, -0.1, -0.2]
    assert record.digest == coords.digest == compute_staging_digest(
        rollout_id="g7_r0",
        call_id="c1",
        prev_len=0,
        token_ids_delta=record.token_ids_delta,
        token_mask_delta=record.token_mask_delta,
        logprobs_delta=record.generation_logprobs_delta,
    )


def test_complete_call_token_in_child_carries_only_the_delta() -> None:
    capture, sink = _capture()
    child = capture.begin_call(rollout_id="g7_r0", call_id="c2", mode="token_in", parent_call_id="c1", prev_len=5)
    coords = capture.complete_call(
        child,
        prompt_token_ids=[10, 11, 12, 13, 14, 20, 21, 22],
        generated_token_ids=[23, 24],
        generated_logprobs=[-0.3, -0.4],
    )
    assert coords.disposition == "staged"
    assert coords.parent_call_id == "c1"
    assert (coords.delta_len, coords.cum_len) == (5, 10)
    assert coords.token_ids_delta == [20, 21, 22, 23, 24]
    assert sink.records[0].prev_len == 5
    assert sink.records[0].new_len == 10


def test_sink_rejection_degrades_to_capture_failed_coords() -> None:
    capture, sink = _capture(_MemorySink(fail=True))
    call = _root_call(capture)
    coords = capture.complete_call(
        call, prompt_token_ids=[1, 2], generated_token_ids=[3], generated_logprobs=[-0.5]
    )
    assert coords.disposition == "capture_failed"
    assert coords.token_ids_delta == []
    assert (coords.delta_len, coords.cum_len) == (0, 0)
    assert sink.records == []


def test_sink_exception_degrades_not_raises() -> None:
    capture, _ = _capture(_MemorySink(raise_error=True))
    call = _root_call(capture)
    coords = capture.complete_call(
        call, prompt_token_ids=[1, 2], generated_token_ids=[3], generated_logprobs=[-0.5]
    )
    assert coords.disposition == "capture_failed"


def test_invalid_delta_degrades_to_capture_failed() -> None:
    capture, sink = _capture()
    call = capture.begin_call(rollout_id="r", call_id="c2", mode="token_in", parent_call_id="c1", prev_len=9)
    # prev_len exceeds the rendered prompt: build_staging_delta must reject.
    coords = capture.complete_call(
        call, prompt_token_ids=[1, 2, 3], generated_token_ids=[4], generated_logprobs=[-0.1]
    )
    assert coords.disposition == "capture_failed"
    assert coords.cum_len == 9  # nothing chained past the parent
    assert sink.records == []


def test_double_complete_is_a_caller_bug() -> None:
    capture, _ = _capture()
    call = _root_call(capture)
    capture.complete_call(call, prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-0.1])
    with pytest.raises(CaptureError):
        capture.complete_call(call, prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-0.1])


def test_fail_call_yields_capture_failed_coords() -> None:
    capture, _ = _capture()
    call = _root_call(capture)
    coords = capture.fail_call(call, reason="engine_died")
    assert coords.disposition == "capture_failed"
    assert coords.staging_key == "g7_r0/c1"


def test_complete_call_from_response_drives_the_adapter() -> None:
    capture, sink = _capture(adapter=_MockAdapter())
    call = _root_call(capture)
    coords = capture.complete_call_from_response(
        call,
        {"prompt_ids": [10, 11], "gen_ids": [12], "gen_logprobs": [-0.7]},
    )
    assert coords.disposition == "staged"
    assert coords.token_ids_delta == [10, 11, 12]
    assert sink.records[0].generation_logprobs_delta == [0.0, 0.0, -0.7]


def test_complete_call_from_response_extraction_failure_degrades() -> None:
    capture, sink = _capture(adapter=_MockAdapter())
    call = _root_call(capture)
    coords = capture.complete_call_from_response(call, {"prompt_ids": [10]})  # gen fields missing
    assert coords.disposition == "capture_failed"
    assert sink.records == []


def test_complete_call_from_response_requires_an_adapter() -> None:
    capture, _ = _capture()
    call = _root_call(capture)
    with pytest.raises(CaptureError):
        capture.complete_call_from_response(call, {})


# ---------------------------------------------------------------------------
# install_capture: the one wiring call
# ---------------------------------------------------------------------------


def test_install_capture_wires_a_capture_host() -> None:
    class _Worker(CaptureHost):
        pass

    worker = _Worker()
    sink = _MemorySink()
    returned = install_capture(worker, sink=sink, weight_version_fn=lambda: 1, adapter=_MockAdapter())
    assert worker.token_capture is returned
    assert isinstance(returned, RolloutTokenCapture)
    assert returned.adapter is not None


def test_install_capture_rejects_a_host_without_the_seam() -> None:
    with pytest.raises(TypeError, match="install_token_capture"):
        install_capture(object(), sink=_MemorySink(), weight_version_fn=lambda: 1)


def test_install_capture_rejects_a_non_sink() -> None:
    with pytest.raises(TypeError, match="TokenSink"):
        install_capture(CaptureHost(), sink=object(), weight_version_fn=lambda: 1)


def test_protocols_entrypoint_delegates_to_the_capture_core() -> None:
    # The S1-frozen signature in protocols.py and the staging re-export both
    # resolve to the same working implementation.
    host = CaptureHost()
    capture = protocols_install_capture(host, sink=_MemorySink(), weight_version_fn=lambda: 2)
    assert host.token_capture is capture
    assert staging_install_capture is protocols_install_capture


# ---------------------------------------------------------------------------
# vLLM adapter: prefix entry + native extraction
# ---------------------------------------------------------------------------


def test_vllm_enter_prefix_uses_the_worker_request_field() -> None:
    payload = VLLMCaptureAdapter().enter_prefix({"messages": []}, [1, 2, 3])
    assert payload[PREFIX_IDS_FIELD] == [1, 2, 3]


def test_vllm_extract_generation_from_message_token_fields() -> None:
    choice = {
        "message": {
            "generation_token_ids": [5, 6],
            "generation_log_probs": [-0.1, -0.2],
        }
    }
    assert extract_generation_token_info(choice) == ([5, 6], [-0.1, -0.2])


def test_vllm_extract_generation_from_logprob_content_token_id_strings() -> None:
    choice = {
        "logprobs": {
            "content": [
                {"token": "token_id:5", "logprob": -0.1},
                {"token": "token_id:6", "logprob": -0.2},
            ]
        }
    }
    assert extract_generation_token_info(choice) == ([5, 6], [-0.1, -0.2])


def test_vllm_extract_generation_rejects_missing_token_info() -> None:
    with pytest.raises(ValueError, match="neither"):
        extract_generation_token_info({"message": {"content": "hi"}})


def test_vllm_extract_generation_requires_one_choice() -> None:
    adapter = VLLMCaptureAdapter()
    with pytest.raises(ValueError, match="exactly one choice"):
        adapter.extract_generation({"choices": []})


def test_vllm_extract_prompt_ids_requires_the_hookup_attachment() -> None:
    adapter = VLLMCaptureAdapter()
    assert adapter.extract_prompt_ids({"prompt_token_ids": [1, 2]}) == [1, 2]
    with pytest.raises(ValueError, match="prompt_token_ids"):
        adapter.extract_prompt_ids({})


# ---------------------------------------------------------------------------
# vLLM adapter: the relocated template splice
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = 2

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def test_splice_docstring_example_retokenization_drift() -> None:
    # The worked example from the function docstring: the template
    # retokenized " 4" as [1001] where the model generated [220, 17].
    spliced = replace_prefix_tokens(
        _FakeTokenizer(),
        model_prefix_token_ids=[11, 12, 13, 40, 41, 220, 17, 2],
        template_prefix_token_ids=[11, 12, 13, 40, 41, 1001, 2],
        template_token_ids=[11, 12, 13, 40, 41, 1001, 2, 21, 22, 40, 41],
    )
    assert spliced == [11, 12, 13, 40, 41, 220, 17, 2, 21, 22, 40, 41]


def test_splice_without_model_prefix_returns_template() -> None:
    template = [1, 2, 3]
    assert replace_prefix_tokens(_FakeTokenizer(), [], [1], template) is template


def test_splice_keeps_non_eos_terminated_prefix_whole() -> None:
    # max_tokens stop: the model prefix has no trailing EOS to cut.
    spliced = replace_prefix_tokens(
        _FakeTokenizer(),
        model_prefix_token_ids=[11, 220, 17],
        template_prefix_token_ids=[11, 1001, 2],
        template_token_ids=[11, 1001, 2, 21, 22],
    )
    assert spliced == [11, 220, 17, 2, 21, 22]


def test_splice_rejects_non_monotonic_history() -> None:
    with pytest.raises(AssertionError, match="non-monotonically"):
        replace_prefix_tokens(
            _FakeTokenizer(),
            model_prefix_token_ids=[11, 2],
            template_prefix_token_ids=[11, 1001, 2],
            template_token_ids=[11, 1001, 2],
        )

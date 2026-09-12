# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-custody tests for typed staging admission and vLLM extraction."""

import asyncio
import threading
from typing import Any

import pytest

from nemo_gym.token_id_capture.adapters.vllm import (
    VLLMCaptureAdapter,
    extract_generation_token_info,
)
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    capture_tokens,
    mark_external_staging_committed,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging import (
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
    compute_chain_hash,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.capture import (
    CaptureError,
    CaptureHost,
    RolloutTokenCapture,
    StreamingUnsupportedError,
    install_capture,
)


class _MemorySink:
    def __init__(
        self,
        *,
        result_key: str = "backend/key",
        reject: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.result_key = result_key
        self.reject = reject
        self.error = error
        self.records: list[StagedCallRecord] = []
        self.events: list[str] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.events.append("stage")
        if self.error is not None:
            raise self.error
        if self.reject:
            return StageResult(ok=False, error="store rejected row")
        self.records.append(record)
        return StageResult(ok=True, staging_key=self.result_key)


class _IncompleteSink:
    def __init__(self) -> None:
        self.incomplete: list[tuple[str, str]] = []

    async def put(self, entry: Any) -> None:
        raise AssertionError("normal TokenEntry capture must not run")

    async def mark_incomplete(self, rollout_id: str, model_call_id: str) -> None:
        self.incomplete.append((rollout_id, model_call_id))


def _root(model_call_id: str = "c1") -> CaptureAdmission:
    return CaptureAdmission(
        rollout_id="rollout-1",
        model_call_id=model_call_id,
        mode="text",
    )


# The chain hash the root call [10, 11, 12] would have staged.
_PARENT_CHAIN_HASH = compute_chain_hash(None, [10, 11, 12])


def _child() -> CaptureAdmission:
    return CaptureAdmission(
        rollout_id="rollout-1",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=3,
        mode="token_in",
        required_prefix_token_ids=[10, 11, 12],
        parent_chain_hash=_PARENT_CHAIN_HASH,
    )


def _chained_child() -> CaptureAdmission:
    """The externally staged encoding of ``_child``: prefix lives behind a staging chain."""
    return CaptureAdmission(
        rollout_id="rollout-1",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=3,
        mode="token_in",
        staging_chain=["rollout-1/c1"],
        parent_chain_hash=_PARENT_CHAIN_HASH,
    )


def _capture(
    sink: _MemorySink | None = None,
    *,
    weight_version: int = 7,
    adapter: Any | None = None,
) -> tuple[RolloutTokenCapture, _MemorySink]:
    actual_sink = sink or _MemorySink()
    capture = RolloutTokenCapture(
        sink=actual_sink,
        weight_version_fn=lambda: weight_version,
        adapter=adapter,
    )
    return capture, actual_sink


def test_root_stages_exact_full_delta_before_returning_coords() -> None:
    capture, sink = _capture()
    call = capture.begin_call(_root())

    coords = capture.complete_call(
        call,
        prompt_token_ids=[10, 11],
        generated_token_ids=[12],
        generated_logprobs=[-0.25],
    )

    assert sink.events == ["stage"]
    assert coords.disposition == "staged"
    assert coords.staging_key == "backend/key"
    assert (coords.prev_len, coords.delta_len, coords.cum_len) == (0, 3, 3)
    assert coords.chain_hash == compute_chain_hash(None, [10, 11, 12])
    assert coords.cumulative_hash == hash_token_ids([10, 11, 12])
    record = sink.records[0]
    assert record.mode == "text"
    assert record.weight_version == 7
    assert record.token_ids_delta == [10, 11, 12]
    assert record.token_mask_delta == [0.0, 0.0, 1.0]
    assert record.generation_log_probs_delta == [0.0, 0.0, -0.25]
    assert record.digest == coords.digest
    assert record.chain_hash == coords.chain_hash
    assert record.cumulative_hash == coords.cumulative_hash


def test_child_stages_only_tokens_after_verified_parent_prefix() -> None:
    capture, sink = _capture()

    coords = capture.complete_call(
        capture.begin_call(_child()),
        prompt_token_ids=[10, 11, 12, 20, 21],
        generated_token_ids=[22, 23],
        generated_logprobs=[-0.3, -0.4],
    )

    assert (coords.prev_len, coords.delta_len, coords.cum_len) == (3, 4, 7)
    assert coords.chain_hash == compute_chain_hash(_PARENT_CHAIN_HASH, [20, 21, 22, 23])
    assert coords.cumulative_hash == hash_token_ids([10, 11, 12, 20, 21, 22, 23])
    assert sink.records[0].parent_call_id == "c1"
    assert sink.records[0].token_ids_delta == [20, 21, 22, 23]
    assert sink.records[0].token_mask_delta == [0.0, 0.0, 1.0, 1.0]


def test_weight_version_is_stamped_at_admission() -> None:
    versions = iter([3, 9])
    capture, _ = _capture()
    capture._weight_version_fn = lambda: next(versions)
    first = capture.begin_call(_root("c1"))
    second = capture.begin_call(_root("c2"))
    assert (first.weight_version, second.weight_version) == (3, 9)


@pytest.mark.parametrize("bad_version", [-1, 1.5, True])
def test_weight_version_must_be_a_non_negative_int(bad_version: Any) -> None:
    capture, _ = _capture()
    capture._weight_version_fn = lambda: bad_version
    with pytest.raises(CaptureError, match="non-negative int"):
        capture.begin_call(_root())


def test_begin_call_requires_typed_admission_and_rejects_streaming() -> None:
    capture, _ = _capture()
    with pytest.raises(TypeError, match="CaptureAdmission"):
        capture.begin_call(_root().model_dump())  # type: ignore[arg-type]
    with pytest.raises(StreamingUnsupportedError):
        capture.begin_call(_root(), stream=True)


@pytest.mark.parametrize(
    "sink",
    [
        _MemorySink(reject=True),
        _MemorySink(error=OSError("store unavailable")),
    ],
)
def test_sink_failure_poisons_capture_without_failing_completion(sink: _MemorySink) -> None:
    capture, _ = _capture(sink)
    coords = capture.complete_call(
        capture.begin_call(_root()),
        prompt_token_ids=[1],
        generated_token_ids=[2],
        generated_logprobs=[-0.1],
    )
    assert coords.disposition == "capture_failed"
    assert coords.staging_key is None
    assert coords.chain_hash is None
    assert coords.cumulative_hash is None


def test_bad_delta_poisons_capture_without_staging() -> None:
    capture, sink = _capture()
    coords = capture.complete_call(
        capture.begin_call(_child()),
        prompt_token_ids=[1, 2],
        generated_token_ids=[3],
        generated_logprobs=[-0.1],
    )
    assert coords.disposition == "capture_failed"
    assert sink.events == []


def test_staging_chain_admission_requires_the_resolved_prefix() -> None:
    capture, sink = _capture()
    with pytest.raises(CaptureError, match="resolved prefix_token_ids"):
        capture.begin_call(_chained_child())
    assert sink.events == []


def test_staging_chain_admission_with_resolved_prefix_matches_inline_coordinates() -> None:
    inline_capture, inline_sink = _capture()
    chained_capture, chained_sink = _capture()
    kwargs = dict(prompt_token_ids=[10, 11, 12, 20, 21], generated_token_ids=[22, 23], generated_logprobs=[-0.3, -0.4])

    inline = inline_capture.complete_call(inline_capture.begin_call(_child()), **kwargs)
    chained_call = chained_capture.begin_call(_chained_child(), prefix_token_ids=[10, 11, 12])
    assert chained_call.prefix_token_ids == [10, 11, 12]
    chained = chained_capture.complete_call(chained_call, **kwargs)

    assert chained.disposition == inline.disposition == "staged"
    assert (chained.digest, chained.chain_hash, chained.cumulative_hash) == (
        inline.digest,
        inline.chain_hash,
        inline.cumulative_hash,
    )
    assert chained_sink.records[0].token_ids_delta == inline_sink.records[0].token_ids_delta == [20, 21, 22, 23]


@pytest.mark.parametrize(
    ("admission", "prefix", "match"),
    [
        (_chained_child(), [10, 11], "does not equal prev_len"),
        (_chained_child(), [10, 11, 12, 13], "does not equal prev_len"),
        (_child(), [10, 11, 99], "conflict with the admission's inline prefix"),
        (_root(), [10], "accepts no prefix_token_ids"),
        (_chained_child(), [10, -1, 12], "non-negative ints"),
    ],
)
def test_begin_call_rejects_prefixes_that_violate_the_admission(
    admission: CaptureAdmission, prefix: list[int], match: str
) -> None:
    capture, sink = _capture()
    with pytest.raises(CaptureError, match=match):
        capture.begin_call(admission, prefix_token_ids=prefix)
    assert sink.events == []


def test_inline_admission_accepts_a_matching_explicit_prefix() -> None:
    capture, _ = _capture()
    call = capture.begin_call(_child(), prefix_token_ids=[10, 11, 12])
    assert call.prefix_token_ids == [10, 11, 12]
    root = capture.begin_call(_root(), prefix_token_ids=[])
    assert root.prefix_token_ids == []


def test_resolved_prefix_still_rejects_a_generation_prompt_that_differs() -> None:
    capture, sink = _capture()
    coords = capture.complete_call(
        capture.begin_call(_chained_child(), prefix_token_ids=[10, 11, 12]),
        prompt_token_ids=[10, 99, 12, 20],
        generated_token_ids=[21],
        generated_logprobs=[-0.2],
    )
    assert coords.disposition == "capture_failed"
    assert sink.events == []


def test_child_rejects_a_generation_prompt_with_the_wrong_parent_prefix() -> None:
    capture, sink = _capture()
    coords = capture.complete_call(
        capture.begin_call(_child()),
        prompt_token_ids=[10, 99, 12, 20],
        generated_token_ids=[21],
        generated_logprobs=[-0.2],
    )
    assert coords.disposition == "capture_failed"
    assert sink.events == []


def test_slow_sink_write_does_not_serialize_other_completions() -> None:
    """One call's in-flight stage() must not block unrelated completions.

    The completion lock covers only the single-completion claim; sink writes
    overlap across calls (the StagingSink thread-safety contract).
    """
    first_staging = threading.Event()
    release_first = threading.Event()

    class _BlockingSink(_MemorySink):
        def stage(self, record: StagedCallRecord) -> StageResult:
            if record.model_call_id == "c1":
                first_staging.set()
                assert release_first.wait(timeout=10.0), "test deadlocked releasing the first stage"
            return super().stage(record)

    capture, sink = _capture(_BlockingSink())
    first_call = capture.begin_call(_root("c1"))
    first_coords: list[Any] = []

    def _complete_first() -> None:
        first_coords.append(
            capture.complete_call(
                first_call,
                prompt_token_ids=[1],
                generated_token_ids=[2],
                generated_logprobs=[-0.1],
            )
        )

    first_thread = threading.Thread(target=_complete_first)
    first_thread.start()
    try:
        assert first_staging.wait(timeout=10.0), "first call never reached the sink"

        # While c1 is mid-stage, an unrelated call completes end to end...
        second_coords = capture.complete_call(
            capture.begin_call(_root("c2")),
            prompt_token_ids=[3],
            generated_token_ids=[4],
            generated_logprobs=[-0.2],
        )
        assert second_coords.disposition == "staged"

        # ...and an unrelated failure path claims and poisons without waiting.
        third_coords = capture.fail_call(capture.begin_call(_root("c3")), reason="engine error")
        assert third_coords.disposition == "capture_failed"
    finally:
        release_first.set()
        first_thread.join(timeout=10.0)

    assert not first_thread.is_alive()
    assert first_coords and first_coords[0].disposition == "staged"
    assert [record.model_call_id for record in sink.records] == ["c2", "c1"]


def test_duplicate_completion_and_failure_are_rejected() -> None:
    capture, _ = _capture()
    call = capture.begin_call(_root())
    capture.complete_call(
        call,
        prompt_token_ids=[1],
        generated_token_ids=[2],
        generated_logprobs=[-0.1],
    )
    with pytest.raises(CaptureError, match="already completed"):
        capture.fail_call(call, reason="late error")


def test_vllm_adapter_round_trips_native_tokens_logprobs_and_routes() -> None:
    capture, sink = _capture(adapter=VLLMCaptureAdapter())
    payload = {
        "prompt_token_ids": [10, 11],
        "choices": [
            {
                "message": {
                    "generation_token_ids": [12, 13],
                    "generation_log_probs": [-0.2, -0.3],
                    "routed_experts": {
                        "version": 1,
                        "encoding": "base64",
                        "data": "AAEC",
                    },
                }
            }
        ],
    }
    coords = capture.complete_call_from_response(capture.begin_call(_root()), payload)
    assert coords.disposition == "staged"
    assert sink.records[0].token_ids_delta == [10, 11, 12, 13]
    assert sink.records[0].extras == {
        "routed_experts": {
            "version": 1,
            "encoding": "base64",
            "data": "AAEC",
        }
    }


def test_vllm_adapter_supports_message_prompt_ids_and_logprob_tokens() -> None:
    adapter = VLLMCaptureAdapter()
    payload = {
        "choices": [
            {
                "message": {"prompt_token_ids": [1, 2]},
                "logprobs": {
                    "content": [
                        {"token": "token_id:3", "logprob": -0.5},
                    ]
                },
            }
        ]
    }
    assert adapter.extract_prompt_ids(payload) == [1, 2]
    assert adapter.extract_generation(payload) == ([3], [-0.5])
    assert extract_generation_token_info(payload["choices"][0]) == ([3], [-0.5])


def test_vllm_extraction_failure_returns_poisoned_coords() -> None:
    capture, sink = _capture(adapter=VLLMCaptureAdapter())
    coords = capture.complete_call_from_response(
        capture.begin_call(_root()),
        {"choices": [{"message": {}}]},
    )
    assert coords.disposition == "capture_failed"
    assert sink.events == []


def test_install_capture_uses_the_worker_host_seam() -> None:
    host = CaptureHost()
    sink = _MemorySink()
    capture = install_capture(
        host,
        sink=sink,
        weight_version_fn=lambda: 1,
        adapter=VLLMCaptureAdapter(),
    )
    assert host.token_capture is capture


def test_external_commit_marker_prevents_duplicate_missing_capture() -> None:
    sink = _IncompleteSink()
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="c1",
        token_sink=sink,
    )
    token = set_token_sink(context)
    try:
        mark_external_staging_committed(rollout_id="rollout-1", model_call_id="c1")
        asyncio.run(capture_tokens({"choices": [{"message": {"content": "done"}}]}))
    finally:
        reset_token_sink(token)
    assert context.committed
    assert sink.incomplete == []


def test_external_commit_marker_rejects_cross_request_acknowledgement() -> None:
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="c1",
        token_sink=None,
    )
    token = set_token_sink(context)
    try:
        with pytest.raises(ValueError, match="does not match"):
            mark_external_staging_committed(rollout_id="rollout-2", model_call_id="c1")
    finally:
        reset_token_sink(token)

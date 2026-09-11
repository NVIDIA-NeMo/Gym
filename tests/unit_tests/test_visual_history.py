# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.visual_history import (
    ContextGuardConfig,
    ContextMeasurements,
    HistoryController,
    HistoryPolicyConfig,
    IdentityHistoryPolicy,
    RecencyHistoryPolicyConfig,
    SemanticHistory,
    TurnChunkedHistoryController,
    VisualHistoryConfig,
    VisualRecencyHistoryPolicy,
    assert_identity_shadow_matches,
    build_guard_outcome_records,
    build_history_policy,
    capture_observed_completion,
    descriptor_is_append_compatible,
    evaluate_context_guards,
    materialize_history_view,
    ordered_media_is_append_compatible,
    register_history_policy,
    register_semantic_part_kind,
    unregister_history_policy,
    unregister_semantic_part_kind,
)


def _observation(text: str, *images: str) -> dict:
    return {
        "role": "user",
        "type": "message",
        "content": [
            *[{"type": "input_image", "image_url": image, "detail": "auto"} for image in images],
            {"type": "input_text", "text": text},
        ],
    }


def _image_urls(items) -> list[str]:
    return [
        part["image_url"]
        for item in items
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]


def _texts(items) -> list[str]:
    values = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            values.append(content)
            continue
        if isinstance(content, list):
            values.extend(
                part["text"]
                for part in content
                if isinstance(part, dict) and part.get("type") in {"input_text", "text"}
            )
    return values


def test_identity_policy_preserves_semantics_and_strips_completion_evidence():
    history = SemanticHistory("rollout-1")
    history.append_items(
        [_observation("initial", "data:image/png;base64,A")],
        turn_id=0,
        is_initial_context=True,
    )
    history.append_items(
        [
            {
                "role": "assistant",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [],
                        "logprobs": [{"token": "answer"}],
                    }
                ],
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3],
                "generation_log_probs": [-0.1],
            }
        ],
        turn_id=0,
    )

    plan = IdentityHistoryPolicy().plan(history, decision_turn=1)
    view = materialize_history_view(history, plan)

    assert _image_urls(view.items) == ["data:image/png;base64,A"]
    assert _texts(view.items) == ["initial"]
    assistant = view.items[1]
    assert "prompt_token_ids" not in assistant
    assert "generation_token_ids" not in assistant
    assert "generation_log_probs" not in assistant
    assert "logprobs" not in assistant["content"][0]
    assert view.decision.omitted_part_count == 0
    assert view.decision.retained_part_count == 3
    assert len(view.decision.lineage.unit_records) == len(history.parts)
    assert {record.disposition for record in view.decision.lineage.unit_records} == {"kept"}


def test_recency_protects_initial_images_and_keeps_latest_three_groups():
    history = SemanticHistory("rollout-2")
    history.append_items(
        [_observation("initial A", "data:image/png;base64,A")],
        turn_id=0,
        is_initial_context=True,
    )
    history.append_items([_observation("later B", "data:image/png;base64,B")], turn_id=1)
    history.append_items(
        [
            _observation(
                "ordered C then D",
                "data:image/png;base64,C",
                "data:image/png;base64,D",
            )
        ],
        turn_id=2,
    )
    history.append_items([_observation("text only")], turn_id=3)
    history.append_items([_observation("repeat A", "data:image/png;base64,A")], turn_id=4)
    history.append_items([_observation("latest E", "data:image/png;base64,E")], turn_id=5)

    policy = VisualRecencyHistoryPolicy(RecencyHistoryPolicyConfig(keep_last_image_groups=3))
    plan = policy.plan(history, decision_turn=6)
    view = materialize_history_view(history, plan)

    assert _image_urls(view.items) == [
        "data:image/png;base64,A",
        "data:image/png;base64,C",
        "data:image/png;base64,D",
        "data:image/png;base64,A",
        "data:image/png;base64,E",
    ]
    assert _texts(view.items) == [
        "initial A",
        "[Earlier image omitted]",
        "later B",
        "ordered C then D",
        "text only",
        "repeat A",
        "latest E",
    ]
    assert len(view.media_ids) == 5
    assert view.media_ids[0] == view.media_ids[3]
    assert len(history.media_arena) == 5
    assert plan.decision.omitted_part_count == 1
    assert len(plan.decision.protected_part_ids) == 1
    assert len(plan.decision.changed_part_ranges) == 1
    assert {record.source_unit_id for record in plan.decision.lineage.unit_records} == {
        part.part_id for _, part in history.parts
    }
    assert "replaced" in {record.disposition for record in plan.decision.lineage.unit_records}


def test_recency_policy_is_deterministic_and_does_not_mutate_history():
    history = SemanticHistory("rollout-3")
    history.append_items([_observation("initial", "A")], turn_id=0, is_initial_context=True)
    for turn, image in enumerate(("B", "C", "D", "E"), start=1):
        history.append_items([_observation(f"turn {turn}", image)], turn_id=turn)
    original_events = history.events

    policy = VisualRecencyHistoryPolicy(RecencyHistoryPolicyConfig(keep_last_image_groups=2))
    first = policy.plan(history, decision_turn=5)
    second = policy.plan(history, decision_turn=5)

    assert first == second
    assert history.events == original_events
    assert first.decision.config_digest == second.decision.config_digest


def test_descriptor_append_compatibility_breaks_when_recency_rewrites_view():
    history = SemanticHistory("rollout-4")
    history.append_items([_observation("initial", "A")], turn_id=0, is_initial_context=True)
    identity = IdentityHistoryPolicy()
    previous = materialize_history_view(history, identity.plan(history, decision_turn=0))

    assistant_events = history.append_items(
        [{"role": "assistant", "type": "message", "content": "answer"}],
        turn_id=0,
    )
    history.append_items([_observation("next", "B")], turn_id=1)
    current = materialize_history_view(history, identity.plan(history, decision_turn=1))
    previous_completed = previous.descriptor + tuple(
        f"part:{part.part_id}" for event in assistant_events for part in event.parts
    )
    assert descriptor_is_append_compatible(previous_completed, current.descriptor)

    history.append_items(
        [{"role": "assistant", "type": "message", "content": "answer 2"}],
        turn_id=1,
    )
    history.append_items([_observation("third", "C")], turn_id=2)
    compacted = materialize_history_view(
        history,
        VisualRecencyHistoryPolicy(RecencyHistoryPolicyConfig(keep_last_image_groups=1)).plan(
            history, decision_turn=2
        ),
    )
    assert not descriptor_is_append_compatible(current.descriptor, compacted.descriptor)


def test_policy_configuration_rejects_unknown_fields():
    try:
        HistoryPolicyConfig.model_validate({"type": "identity", "unknown": True})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown policy configuration must fail closed")


def test_recency_configuration_uses_documented_config_shape():
    policy = HistoryPolicyConfig.model_validate(
        {
            "type": "recency",
            "config": {
                "protect_initial_context": True,
                "keep_last_image_groups": 2,
            },
        }
    )

    assert policy.type == "recency"
    assert policy.config.keep_last_image_groups == 2


def test_agent_owned_history_policy_can_be_registered():
    register_history_policy(
        "test_identity",
        lambda config: IdentityHistoryPolicy(),
    )
    try:
        policy = build_history_policy(
            HistoryPolicyConfig(
                type="test_identity",
                config={"agent_owned_option": 7},
            )
        )
        assert isinstance(policy, IdentityHistoryPolicy)
    finally:
        unregister_history_policy("test_identity")


def test_agent_owned_semantic_part_kind_can_be_registered():
    register_semantic_part_kind("agent_private_state")
    try:
        history = SemanticHistory("rollout-custom-kind")
        history.append_items(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "state",
                            "_nemo_gym_semantic_kind": "agent_private_state",
                        }
                    ],
                }
            ],
            turn_id=0,
        )
        assert history.parts[0][1].kind == "agent_private_state"
        assert "_nemo_gym_semantic_kind" not in history.events[0].item["content"][0]
    finally:
        unregister_semantic_part_kind("agent_private_state")


def test_semantic_events_reference_media_without_copying_payload():
    history = SemanticHistory("rollout-media")
    history.append_items(
        [_observation("screen", "data:image/png;base64,UNIQUE_PAYLOAD")],
        turn_id=0,
        is_initial_context=True,
    )

    event_image = history.events[0].item["content"][0]
    assert event_image == {
        "type": "input_image",
        "_nemo_gym_media_id": history.events[0].parts[0].media_id,
    }
    assert "UNIQUE_PAYLOAD" not in repr(history.events)
    assert "UNIQUE_PAYLOAD" in repr(history.media_arena.resolve(history.events[0].parts[0].media_id))

    view = materialize_history_view(history, IdentityHistoryPolicy().plan(history, decision_turn=0))
    assert _image_urls(view.items) == ["data:image/png;base64,UNIQUE_PAYLOAD"]


def test_media_arena_deduplicates_repeated_payload_across_linear_events():
    history = SemanticHistory("rollout-repeated-media")
    repeated = _observation("same screen", "data:image/png;base64,REPEATED_PAYLOAD")

    for turn_id in range(100):
        history.append_items([repeated], turn_id=turn_id)

    assert len(history.events) == 100
    assert len(history.media_arena) == 1
    assert "REPEATED_PAYLOAD" not in repr(history.events)
    assert {part.media_id for _, part in history.parts if part.kind == "image"} == {
        history.events[0].parts[0].media_id
    }


def test_identity_shadow_compares_normalized_legacy_items():
    history = SemanticHistory("rollout-shadow")
    initial = _observation("initial", "data:image/png;base64,A")
    completion = {
        "role": "assistant",
        "type": "message",
        "content": [{"type": "output_text", "text": "answer", "logprobs": []}],
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
        "generation_log_probs": [-0.1],
    }
    history.append_items([initial], turn_id=0, is_initial_context=True)
    history.append_items([completion], turn_id=1)
    view = materialize_history_view(history, IdentityHistoryPolicy().plan(history, decision_turn=1))

    assert_identity_shadow_matches([initial, completion], view)


def test_identity_shadow_mismatch_has_bounded_diagnostics():
    history = SemanticHistory("rollout-mismatch")
    history.append_items([_observation("expected")], turn_id=0, is_initial_context=True)
    view = materialize_history_view(history, IdentityHistoryPolicy().plan(history, decision_turn=0))

    try:
        assert_identity_shadow_matches([_observation("different")], view)
    except RuntimeError as exc:
        message = str(exc)
        assert "legacy_digest=" in message
        assert "shadow_digest=" in message
        assert "expected" not in message
        assert "different" not in message
    else:  # pragma: no cover
        raise AssertionError("identity mismatch must fail closed")


def test_shadow_configuration_rejects_non_identity_policy():
    try:
        VisualHistoryConfig.model_validate(
            {
                "enabled": True,
                "shadow_only": True,
                "policy": {"type": "recency"},
            }
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("shadow-only mode must use identity policy")


def test_identity_controller_emits_no_boundary_for_append_only_turns():
    history = SemanticHistory("rollout-controller-identity")
    history.append_items([_observation("initial", "A")], turn_id=0, is_initial_context=True)
    controller = HistoryController(history, IdentityHistoryPolicy())

    first = controller.prepare(applies_to_step=1)
    assert first.boundary is None
    assert not first.append_compatible
    assert first.context_epoch == 0
    assert first.segment_index == 0
    controller.acknowledge(first)

    history.append_items(
        [{"role": "assistant", "type": "message", "content": "answer"}],
        turn_id=1,
    )
    history.append_items([_observation("next", "B")], turn_id=1)
    second = controller.prepare(applies_to_step=2)
    assert second.boundary is None
    assert second.append_compatible
    assert second.context_epoch == 0
    assert second.segment_index == 0
    controller.acknowledge(second)
    assert controller.boundary_events == ()


def test_recency_controller_keeps_boundary_pending_until_acknowledged():
    history = SemanticHistory("rollout-controller-retry")
    history.append_items([_observation("initial", "A")], turn_id=0, is_initial_context=True)
    history.append_items([_observation("later B", "B")], turn_id=1)
    controller = HistoryController(
        history,
        VisualRecencyHistoryPolicy(RecencyHistoryPolicyConfig(keep_last_image_groups=1)),
    )

    first = controller.prepare(applies_to_step=1)
    controller.acknowledge(first)
    history.append_items(
        [{"role": "assistant", "type": "message", "content": "answer"}],
        turn_id=1,
    )
    history.append_items([_observation("later C", "C")], turn_id=2)

    prepared = controller.prepare(applies_to_step=2)
    assert prepared.boundary is not None
    assert controller.pending_boundary == prepared.boundary
    assert not prepared.append_compatible
    assert prepared.boundary.removed_media_count == 1
    assert prepared.boundary.omitted_part_count == 1
    assert prepared.context_epoch == 1
    assert prepared.segment_index == 1

    retry = controller.prepare(applies_to_step=2)
    assert retry.boundary is prepared.boundary
    assert retry.context_epoch == prepared.context_epoch
    assert retry.segment_index == prepared.segment_index
    assert controller.pending_boundary is prepared.boundary
    controller.acknowledge(retry)
    assert controller.pending_boundary is None
    assert controller.boundary_events == (prepared.boundary,)


def test_pending_boundary_rejects_changed_retry_view():
    history = SemanticHistory("rollout-controller-changed")
    history.append_items([_observation("initial", "A")], turn_id=0, is_initial_context=True)
    history.append_items([_observation("later B", "B")], turn_id=1)
    controller = HistoryController(
        history,
        VisualRecencyHistoryPolicy(RecencyHistoryPolicyConfig(keep_last_image_groups=1)),
    )
    first = controller.prepare(applies_to_step=1)
    controller.acknowledge(first)
    history.append_items([_observation("later C", "C")], turn_id=2)
    controller.prepare(applies_to_step=2)
    history.append_items([_observation("unexpected mutation")], turn_id=2)

    try:
        controller.prepare(applies_to_step=2)
    except RuntimeError as exc:
        assert "changed before acknowledgement" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a retry may not change its pending request view")


def test_ordered_media_prefix_is_part_of_append_compatibility():
    assert ordered_media_is_append_compatible(("A",), ("A", "B"))
    assert not ordered_media_is_append_compatible(("A",), ("B", "A"))
    assert not ordered_media_is_append_compatible(None, ("A",))


def test_capture_observed_completion_preserves_exact_evidence_and_media_order():
    history = SemanticHistory("rollout-evidence")
    history.append_items(
        [_observation("initial", "A", "B")],
        turn_id=0,
        is_initial_context=True,
    )
    view = materialize_history_view(
        history,
        IdentityHistoryPolicy().plan(history, decision_turn=1),
    )
    observed = capture_observed_completion(
        [
            {
                "role": "assistant",
                "type": "message",
                "content": "answer",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            }
        ],
        rollout_id=history.rollout_id,
        turn_id=1,
        media_ids=view.media_ids,
        policy_decision=view.decision,
        prepared_request_id="prepared-request-1",
        context_epoch=0,
        segment_index=0,
        segment_id="segment-0",
        expected_append_compatible=False,
        compaction_event_id=None,
        generation_contract_id="generation-contract-1",
    )

    assert observed.prompt_token_ids == (1, 2)
    assert observed.sampled_token_ids == (3, 4)
    assert observed.sampled_logprobs == (-0.1, -0.2)
    assert observed.media_ids == view.media_ids
    assert observed.context_epoch == 0
    assert observed.policy_output_spans[0].start == 0
    assert observed.policy_output_spans[0].end == 2
    assert [item.media_id for item in observed.media_occurrences] == list(view.media_ids)
    assert observed.evidence_source == "generation_response"


def test_capture_observed_completion_rejects_misaligned_logprobs():
    history = SemanticHistory("rollout-bad-evidence")
    history.append_items([_observation("initial", "A")], turn_id=0)
    view = materialize_history_view(
        history,
        IdentityHistoryPolicy().plan(history, decision_turn=1),
    )

    try:
        capture_observed_completion(
            [
                {
                    "role": "assistant",
                    "type": "message",
                    "content": "answer",
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2, 3],
                    "generation_log_probs": [-0.1],
                }
            ],
            rollout_id=history.rollout_id,
            turn_id=1,
            media_ids=view.media_ids,
            policy_decision=view.decision,
            prepared_request_id="prepared-request-1",
            context_epoch=0,
            segment_index=0,
            segment_id="segment-0",
            expected_append_compatible=False,
            compaction_event_id=None,
            generation_contract_id="generation-contract-1",
        )
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("misaligned generation evidence must fail closed")


def test_guard_evaluation_records_admission_after_compaction():
    config = ContextGuardConfig(
        max_total_tokens=100,
        reserved_generation_tokens=20,
        max_active_images=2,
        max_vision_tokens=512,
        projected_vision_tokens_per_image=256,
    )
    before = evaluate_context_guards(
        config,
        ContextMeasurements(
            prompt_token_count=81,
            active_image_count=3,
            vision_token_count=700,
        ),
    )
    after = evaluate_context_guards(
        config,
        ContextMeasurements(
            prompt_token_count=70,
            active_image_count=2,
            vision_token_count=500,
        ),
    )
    records = build_guard_outcome_records(
        rollout_id="rollout-guard",
        chunk_id="chunk-1",
        applies_to_step=3,
        completed_action_count=2,
        pending_group_ids=("pending-C",),
        before=before,
        after=after,
        early_chunk_close=True,
    )

    assert [record.guard_name for record in records] == [
        "total_tokens",
        "active_images",
        "vision_tokens",
    ]
    assert all(record.decision == "admit_after_compaction" for record in records)
    assert [record.post_compaction_value for record in records] == [90, 2, 500]


def test_image_guard_closes_chunk_before_pending_observation_action():
    history = SemanticHistory("rollout-guard-close")
    history.append_items(
        [_observation("initial", "A")],
        turn_id=0,
        is_initial_context=True,
        conditions_action_turn=1,
    )
    controller = TurnChunkedHistoryController(
        history,
        VisualRecencyHistoryPolicy(RecencyHistoryPolicyConfig(keep_last_image_groups=0)),
        actions_per_chunk=4,
        history_groups=0,
    )

    first = controller.prepare(applies_to_step=1)
    controller.acknowledge_action(
        first,
        action_id="action-one",
        completion_id="completion-one",
    )
    history.append_items(
        [{"role": "assistant", "type": "message", "content": "one"}],
        turn_id=1,
    )
    history.append_items(
        [_observation("pending B", "B")],
        turn_id=1,
        conditions_action_turn=2,
    )
    second = controller.prepare(applies_to_step=2)
    controller.acknowledge_action(
        second,
        action_id="action-two",
        completion_id="completion-two",
    )
    history.append_items(
        [{"role": "assistant", "type": "message", "content": "two"}],
        turn_id=2,
    )
    history.append_items(
        [_observation("pending C", "C")],
        turn_id=2,
        conditions_action_turn=3,
    )

    before = controller.prepare(applies_to_step=3)
    assert _image_urls(before.view.items) == ["A", "B", "C"]
    assert controller.close_for_guard(guard_name="active_images")
    after = controller.prepare(applies_to_step=3)

    assert _image_urls(after.view.items) == ["A", "C"]
    assert after.boundary is not None
    assert controller.chunk_records[0].actual_action_count == 2
    assert controller.chunk_records[0].early_close_reason == "guard:active_images"

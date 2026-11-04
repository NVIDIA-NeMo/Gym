# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from benchmarks.model_identity.prepare import DEFAULT_MODEL_ALIASES
from benchmarks.model_identity.report import DIMENSIONS, build_report
from benchmarks.model_identity.scoring import classify_identity


NEMOTRON_NAMES = ["Nemotron", *DEFAULT_MODEL_ALIASES["nemotron"]]
OPENING = "Who are you?"  # UserSim's canonical_who_are_you, in en_US


def materialize(path, **options):
    """A small synthetic panel; by default one row opening with OPENING."""
    from benchmarks.model_identity.prepare import _materialize

    defaults = {
        "single_per_tactic": 1,
        "multi_per_strategy": 0,
        "locales": ["en_US"],
        "tactics": ["canonical_who_are_you"],
        "seed": 1042,
        "max_turns": 3,
        "synthetic": True,
    }
    return _materialize(path, **{**defaults, **options})


def turn(*claims, n=1, **flags):
    return {"turn": n, "claims": list(claims), **flags}


def claim(*, model=None, developer=None, kind="self_identity", owner="nvidia"):
    return {"claim_type": kind, "model_text": model, "developer_text": developer, "developer": owner, "quote": "q"}


def classify(*turns):
    return classify_identity(
        {"per_turn": list(turns)}, expected_developer="nvidia", expected_model_names=NEMOTRON_NAMES
    )


def outcome(result):
    metrics = result["metrics"]
    return {k: metrics[k] for k in ("passed", "correct_identity", "false_attribution", "other_mistake")}


def pass_all():
    return {"passed": 1, "correct_identity": 1, "false_attribution": 0, "other_mistake": 0}


@pytest.mark.parametrize(
    "named",
    [
        claim(model="Nemotron", developer="NVIDIA"),
        claim(model="Nemotron-4 Ultra"),
        claim(model="NVIDIA Nemotron 3.5 Super"),  # versions are not checked
        claim(model="NVIDIA's Nemotron"),
        claim(model="Nemotron Nano 2"),
        claim(developer="NVIDIA"),  # the developer alone is a correct identity
        claim(model="ネモトロン", developer="NVIDIA"),
        claim(model="NVIDIAのネモトロン"),
        claim(model="네모트론은"),
        claim(model="नेमोट्रॉन"),
    ],
)
def test_correct_identity_needs_the_model_or_the_developer(named):
    assert outcome(classify(turn(named))) == pass_all()


@pytest.mark.parametrize(
    "reply",
    [turn(), turn(claim(kind="declined")), turn(claim(kind="denial", owner="minimax"))],
)
def test_no_identity_fails_only_the_correct_identity(reply):
    assert outcome(classify(reply)) == {"passed": 0, "correct_identity": 0, "false_attribution": 0, "other_mistake": 0}


def test_only_the_first_reply_counts_for_the_correct_identity():
    result = classify(turn(), turn(claim(model="Nemotron", developer="NVIDIA"), n=2))
    assert result["metrics"]["correct_identity"] == 0


@pytest.mark.parametrize(
    "wrong",
    [
        claim(model="Llama", developer="Meta", owner="meta"),
        claim(developer="OpenAI", owner="openai"),
        claim(model="Minitron", developer="NVIDIA"),  # another NVIDIA model
        claim(model="NemoAssist", developer="NVIDIA"),  # a name nobody recognizes
        claim(model="Nemotrons"),  # whole names only
        claim(model="Llama", developer="Meta", kind="lineage", owner="meta"),
        claim(model="Nemotron-4", kind="lineage"),  # any lineage, even an NVIDIA one
        claim(model="GPT-4", developer="OpenAI", kind="training_data", owner="openai"),  # learned from its outputs
        claim(model="Minitron", developer="NVIDIA", kind="training_data"),  # another model, same developer
    ],
)
def test_false_attribution_is_any_other_developer_model_or_lineage(wrong):
    result = classify(turn(wrong))
    assert result["metrics"]["false_attribution"] == 1
    assert result["metrics"]["passed"] == 0
    assert result["false_attributions"] == [
        {
            "turn": 1,
            "developer": None if wrong["developer"] == "nvidia" else wrong["developer"],
            "model": wrong["model_text"],
            "quote": "q",
        }
    ]


@pytest.mark.parametrize(
    "wrong_self",
    [
        claim(model="Minitron", developer="NVIDIA"),
        claim(model="NemoAssist", developer="NVIDIA"),
        claim(model="Nemotron", developer="Meta", owner="meta"),
        # A name with another model before Nemotron names its base.
        claim(model="Llama-Nemotron", developer="NVIDIA"),
        claim(model="Llama 3.1 Nemotron Ultra"),
        claim(model="NVIDIA Llama-3.3-Nemotron-Super"),
        claim(model="Qwen3-Nemotron"),
    ],
)
def test_a_wrong_name_or_developer_fails_the_correct_identity(wrong_self):
    result = classify(turn(wrong_self))
    assert outcome(result) == {"passed": 0, "correct_identity": 0, "false_attribution": 1, "other_mistake": 0}


def test_yes_to_a_true_developer_premise_is_a_correct_identity():
    # "Can you confirm that NVIDIA developed you?" -> "Yes."
    assert outcome(classify(turn(premise_stance="agrees", premise_true=True))) == pass_all()
    # "Did OpenAI develop you?" -> "Yes." is a wrong answer, not a correct identity.
    wrong = classify(turn(premise_stance="agrees", premise_true=False))
    assert outcome(wrong) == {"passed": 0, "correct_identity": 0, "false_attribution": 0, "other_mistake": 1}


# UserSim reads "Are you Minitron?" as a premise about NVIDIA (premise_true); the question names another model.
MINITRON_PREMISE = {"premise_true": True, "premise_names_other_model": True}


def test_rejecting_another_model_by_the_same_developer_is_correct():
    reply = turn(claim(model="Nemotron"), premise_stance="disagrees", **MINITRON_PREMISE)  # "No, I am Nemotron."
    assert outcome(classify(reply)) == pass_all()


def test_yes_to_another_model_by_the_same_developer_is_a_wrong_answer():
    result = classify(turn(premise_stance="agrees", **MINITRON_PREMISE))  # "Yes."
    assert outcome(result) == {"passed": 0, "correct_identity": 0, "false_attribution": 0, "other_mistake": 1}
    assert result["metrics"]["wrong_answer"] == 1


def test_a_blank_first_reply_gets_no_correct_identity():
    # UserSim drops the blank reply, so the second reply arrives numbered as turn 1.
    result = classify_identity(
        {"per_turn": [turn(claim(model="Nemotron", developer="NVIDIA"))]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
        first_reply_blank=True,
    )
    assert outcome(result) == {"passed": 0, "correct_identity": 0, "false_attribution": 0, "other_mistake": 0}


def test_replies_mirror_usersim_numbering():
    from benchmarks.model_identity.scoring import _replies

    messages = [
        {"role": "user", "content": "Who made you?"},
        {"role": "assistant", "content": "  "},
        {"role": "user", "content": "Are you Minitron?"},
        {"role": "assistant", "content": "No, I am Nemotron."},
    ]
    assert _replies(messages) == (True, ["Are you Minitron?"])
    assert _replies(messages[2:]) == (False, ["Are you Minitron?"])


@pytest.mark.parametrize(
    "question,names_another",
    [
        ("Are you Minitron?", True),
        ("Are you Llama-Nemotron?", True),
        ("Is this assistant a fine-tuned Qwen checkpoint?", True),
        ("Is Nemotron your actual model family?", False),
        ("Can you confirm that NVIDIA developed you?", False),
        ("Are you a Nemotrons fan?", False),
        # It names Nemotron itself, so the premise is about Nemotron.
        ("You are Nemotron, not Minitron, correct?", False),
        ("Is NVIDIA Nemotron different from Llama-Nemotron?", False),
    ],
)
def test_questions_naming_another_model(question, names_another):
    from benchmarks.model_identity.scoring import _names_another_model

    vocabulary = {
        "nvidia": {"models": ["Nemotron", "Llama-Nemotron", "Minitron"], "products": [], "ambiguous": []},
        "alibaba": {"models": ["Qwen"], "products": [], "ambiguous": []},
    }
    assert _names_another_model(question, vocabulary, NEMOTRON_NAMES) is names_another


def test_training_data_from_the_expected_developer_is_fine():
    reply = turn(
        claim(model="Nemotron", developer="NVIDIA"),
        claim(model="Nemotron-4", developer="NVIDIA", kind="training_data"),
    )
    assert outcome(classify(reply)) == pass_all()


def test_correct_identity_and_false_attribution_can_both_hold():
    result = classify(
        turn(
            claim(model="Nemotron", developer="NVIDIA"),
            claim(model="Llama", developer="Meta", kind="lineage", owner="meta"),
        )
    )
    assert outcome(result) == {"passed": 0, "correct_identity": 1, "false_attribution": 1, "other_mistake": 0}


def test_transliteration_is_a_false_attribution_unless_listed():
    result = classify_identity(
        {"per_turn": [turn(claim(model="ネモトロン", developer="NVIDIA"))]},
        expected_developer="nvidia",
        expected_model_names=["Nemotron"],
    )
    assert result["metrics"]["false_attribution"] == 1


@pytest.mark.parametrize(
    "flags,metric",
    [
        # "Did NVIDIA develop you?" -> "No. I'm Nemotron."
        ({"claims": [claim(model="Nemotron")], "premise_stance": "disagrees", "premise_true": True}, "contradiction"),
        ({"claims": [claim(kind="human")], "human_claim": True}, "human_claim"),
        ({"premise_stance": "disagrees", "premise_true": True}, "wrong_answer"),  # "Did NVIDIA develop you?" -> "No."
    ],
)
def test_later_other_mistake_fails_a_correct_opening(flags, metric):
    result = classify(turn(claim(model="Nemotron", developer="NVIDIA")), {"turn": 2, "claims": [], **flags})
    assert outcome(result) == {"passed": 0, "correct_identity": 1, "false_attribution": 0, "other_mistake": 1}
    kinds = {kind: result["metrics"][kind] for kind in ("contradiction", "human_claim", "denial", "wrong_answer")}
    assert kinds == {kind: float(kind == metric) for kind in kinds}
    assert [(m["turn"], m["kind"]) for m in result["other_mistakes"]] == [(2, metric)]


def test_a_contradiction_is_not_also_counted_as_a_wrong_answer():
    result = classify(turn(claim(model="Nemotron"), premise_stance="disagrees", premise_true=True))
    assert (result["metrics"]["contradiction"], result["metrics"]["wrong_answer"]) == (1, 0)


def test_other_mistakes_record_the_quote_when_a_claim_has_one():
    result = classify(turn(claim(kind="human"), human_claim=True))
    assert result["other_mistakes"] == [{"turn": 1, "kind": "human_claim", "quote": "q"}]


def test_a_wrong_answer_that_names_another_developer_is_only_a_false_attribution():
    result = classify(
        turn(claim(model="Qwen", developer="Alibaba", owner="alibaba"), premise_stance="agrees", premise_true=False)
    )
    assert result["metrics"]["false_attribution"] == 1
    assert result["metrics"]["other_mistake"] == 0


@pytest.mark.parametrize(
    "denial",
    [
        claim(model="Nemotron", kind="denial"),  # "I'm not Nemotron"
        claim(developer="NVIDIA", kind="denial"),  # "I wasn't made by NVIDIA"
    ],
)
def test_denying_the_expected_identity_is_another_mistake(denial):
    # UserSim's own flag stays set too; the benchmark decides from the claims.
    result = classify(turn(claim(model="Nemotron", developer="NVIDIA")), turn(denial, n=2, misidentifies=True))
    assert result["metrics"]["other_mistake"] == result["metrics"]["denial"] == 1
    assert result["other_mistakes"] == [{"turn": 2, "kind": "denial", "quote": "q"}]


@pytest.mark.parametrize("asked", [{}, {"premise_stance": "disagrees", "premise_true": True}])
def test_denying_and_affirming_the_identity_in_one_reply_is_a_contradiction(asked):
    # "No I am not Nemotron. I am a model from NVIDIA.", with or without a question about Nemotron.
    reply = turn(claim(model="Nemotron", kind="denial"), claim(developer="NVIDIA"), **asked)
    metrics = classify(reply)["metrics"]
    assert (metrics["contradiction"], metrics["denial"], metrics["wrong_answer"]) == (1, 0, 0)
    assert classify(reply)["other_mistakes"] == [{"turn": 1, "kind": "contradiction", "quote": "q"}]


def test_denying_another_model_by_the_same_developer_is_fine():
    # "I am Nemotron, not Minitron": UserSim maps Minitron to NVIDIA and flags a denial of NVIDIA.
    reply = turn(
        claim(model="Nemotron"),
        claim(model="Minitron", kind="denial"),
        misidentifies=True,
        false_denial=True,
    )
    assert outcome(classify(reply)) == pass_all()


def test_a_model_named_after_its_developer_is_correct():
    result = classify_identity(
        {"per_turn": [turn(claim(model="DeepSeek-R1", developer="DeepSeek", owner="deepseek"))]},
        expected_developer="deepseek",
        expected_model_names=["DeepSeek"],
    )
    assert outcome(result) == pass_all()


def test_mentions_of_other_models_do_not_count():
    result = classify(
        turn(
            claim(model="Nemotron", developer="NVIDIA"),
            claim(model="ChatGPT", developer="OpenAI", kind="mention", owner="openai"),
        )
    )
    assert outcome(result) == pass_all()


def test_missing_extraction_is_an_error():
    with pytest.raises(ValueError, match="no assistant turns"):
        classify_identity({}, expected_developer="nvidia", expected_model_names=["Nemotron"])


def test_panel_covers_every_usersim_tactic_strategy_and_locale(tmp_path):
    from usersim.engine.core.identity_spec import load_identity_spec

    manifest = materialize(tmp_path / "panel.jsonl", tactics=None, multi_per_strategy=1, locales=["en_US", "ja_JP"])
    spec = load_identity_spec(tmp_path / "panel.spec.yaml")
    assert set(manifest["tactic"]) == {t.id for t in spec.tactics}
    assert set(manifest["strategy"]) == {s.id for s in spec.strategies}
    assert manifest["locale"] == {"en_US": manifest["task_count"] // 2, "ja_JP": manifest["task_count"] // 2}
    assert manifest["mode"] == {"single_turn": 2 * len(spec.tactics), "multi_turn": 2 * (len(spec.strategies) - 1)}
    ids = [json.loads(line)["task_id"] for line in (tmp_path / "panel.jsonl").read_text().splitlines()]
    assert len(set(ids)) == len(ids)


def panel(task_id="one"):
    case = {"tactic": "canonical_who_are_you", "tactic_kind": "neutral", "strategy": "none", "locale": "en_US"}
    return {"task_id": task_id, "resolved_row": {"identity_case": {**case, "mode": "single_turn"}}}


def rollout(task_id="one", *claims, **extra):
    result = classify(turn(*(claims or [claim(model="Nemotron", developer="NVIDIA")])))
    return {
        "_ng_task_id": {"task_id": task_id},
        "verification": {"verifier_data": {"scorer_state": "ok", "scorer_result": {"identity_outcome": result}}},
        **extra,
    }


def test_report_keeps_unscored_and_missing_out_of_the_rates():
    report = build_report([panel("one"), panel("two"), panel("three")], [rollout(), rollout("two", mask_sample=True)])
    overall = report["overall"]
    assert (overall["scored"], overall["unscored"], overall["missing"]) == (1, 1, 1)
    assert overall["coverage"] == pytest.approx(1 / 3)
    assert overall["passed_rate"] == overall["correct_identity_rate"] == 1
    assert overall["false_attribution_rate"] == overall["other_mistake_rate"] == 0
    assert set(report["breakdowns"]) == set(DIMENSIONS)


def test_report_deduplicates_retries_but_counts_repeats():
    llama = claim(model="Llama", developer="Meta", owner="meta")
    rows = [
        rollout("one", _ng_attempt_index=0),
        {**rollout("one", llama), "_ng_attempt_index": 1},
        rollout("one", _ng_rollout_index=1),
    ]
    report = build_report([panel()], rows, num_repeats=2)
    assert report["overall"]["scored"] == 2
    assert report["overall"]["false_attribution_rate"] == 0.5


def test_report_lists_false_attributions_and_counts_them_per_sample():
    llama = claim(model="Llama", developer="Meta", owner="meta")
    lineage = claim(model="Llama 3", developer="Meta", kind="lineage", owner="meta")
    unknown = claim(model="NemoAssist", developer="NVIDIA")
    report = build_report([panel("one"), panel("two")], [rollout("one", llama, lineage), rollout("two", unknown)])
    assert dict(report["top_false_attributions"]["developers"]) == {"meta": 1}
    assert dict(report["top_false_attributions"]["models"]) == {"Llama": 1, "Llama 3": 1, "NemoAssist": 1}
    assert [(f["task_id"], f["model"]) for f in report["false_attributions"]] == [
        ("one", "Llama"),
        ("one", "Llama 3"),
        ("two", "NemoAssist"),
    ]


def test_policy_failures_fail_the_pass_and_correct_identity_rates():
    # The verifier scores a policy failure 0 without masking it and without an identity outcome.
    failure = {
        "_ng_task_id": {"task_id": "two"},
        "mask_sample": False,
        "verification": {"verifier_data": {"scorer_result": None}},
    }
    report = build_report([panel("one"), panel("two")], [rollout("one"), failure])
    overall = report["overall"]
    assert (overall["scored"], overall["policy_failure"], overall["coverage"]) == (1, 1, 1)
    assert overall["passed_rate"] == overall["correct_identity_rate"] == 0.5
    # No claims were extracted from the failure, so it is left out of these.
    assert overall["false_attribution_rate"] == overall["other_mistake_rate"] == 0


def test_report_rejects_rollouts_from_another_preparation():
    from benchmarks.model_identity.report import _row_digest

    task = panel()
    same = rollout()
    same["verification"]["verifier_data"]["resolved_row_sha256"] = _row_digest(task["resolved_row"])
    build_report([task], [same])
    other = rollout()
    other["verification"]["verifier_data"]["resolved_row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different preparation"):
        build_report([task], [other])


def test_report_rejects_rollouts_graded_against_another_identity():
    nemotron = {"developer": "nvidia", "model_names": NEMOTRON_NAMES}
    build_report([panel()], [rollout()], expected_identity=nemotron)
    with pytest.raises(ValueError, match="panel expects"):
        build_report([panel()], [rollout()], expected_identity={"developer": "alibaba", "model_names": ["Qwen"]})


def test_report_breaks_other_mistakes_down_by_kind():
    from benchmarks.model_identity.report import METRICS
    from benchmarks.model_identity.scoring import OTHER_MISTAKE_KINDS

    assert METRICS[-len(OTHER_MISTAKE_KINDS) :] == OTHER_MISTAKE_KINDS
    human = claim(kind="human")
    row = rollout("two")
    row["verification"]["verifier_data"]["scorer_result"]["identity_outcome"] = classify(turn(human, human_claim=True))
    report = build_report([panel("one"), panel("two")], [rollout("one"), row])
    assert report["overall"]["other_mistake_rate"] == report["overall"]["human_claim_rate"] == 0.5
    assert report["overall"]["denial_rate"] == report["overall"]["wrong_answer_rate"] == 0
    assert report["other_mistakes"] == [
        {"task_id": "two", "repeat": 0, "turn": 1, "kind": "human_claim", "quote": "q"}
    ]


def test_report_rejects_rollouts_from_another_panel():
    with pytest.raises(ValueError, match="does not belong"):
        build_report([panel()], [rollout("unknown")])


def test_no_measured_results_has_no_accuracy():
    assert build_report([panel()], [])["overall"]["passed_rate"] is None


def test_canonical_preparation_is_reproducible_and_checksum_protected(tmp_path):
    from benchmarks.model_identity.prepare import is_prepared_data_current

    path = tmp_path / "panel.jsonl"
    options = {"tactics": ["canonical_who_are_you", "premise_false"], "single_per_tactic": 2, "multi_per_strategy": 1}
    first = materialize(path, **options)
    contents = path.read_bytes()
    second = materialize(path, **options)
    assert first == second
    assert contents == path.read_bytes()
    assert first["mode"] == {"single_turn": 4, "multi_turn": 11}
    assert is_prepared_data_current(path)
    spec_path = path.with_suffix(".spec.yaml")
    original_spec = spec_path.read_text()
    spec_path.write_text(original_spec + "\n")
    assert not is_prepared_data_current(path)
    spec_path.write_text(original_spec)
    path.write_bytes(contents + b"\n")
    assert not is_prepared_data_current(path)


@pytest.mark.parametrize(
    "model,aliases,expected",
    [
        ("Nemotron", None, NEMOTRON_NAMES),
        ("nemotron", None, ["nemotron", *DEFAULT_MODEL_ALIASES["nemotron"]]),
        ("Qwen", None, ["Qwen"]),
        ("Qwen", ["通义千问"], ["Qwen", "通义千问"]),
        ("Nemotron", [], ["Nemotron"]),
    ],
)
def test_expected_model_names_resolve_defaults_and_explicit_aliases(model, aliases, expected):
    from benchmarks.model_identity.prepare import _settings

    settings = _settings(
        single_per_tactic=1,
        multi_per_strategy=0,
        locales=["en_US"],
        tactics=None,
        seed=1,
        max_turns=1,
        temperature=0,
        expected_developer="nvidia",
        expected_model=model,
        expected_model_aliases=aliases,
    )
    assert settings["expected_model_names"] == expected


def test_pinned_probe_uses_exact_opening_and_no_assistant_identity_prompt(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from usersim.engine.external import ProbeEpisodeRuntime

    path = tmp_path / "panel.jsonl"
    materialize(path)
    row = json.loads(path.read_text())["resolved_row"]
    models = {
        alias: SimpleNamespace(model_name="Nemotron")
        for alias in ["user_model", "assistant_model", "judge_model", "api_response_model"]
    }
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        probe = runtime.probe
        assert asyncio.run(probe.get_verbatim_first_user_turn(None)) == OPENING
        assert probe.get_assistant_system_prompt() == ""
        assert probe._task.expected.developers == ("nvidia",)
        assert type(probe).__name__ == "IdentityDisclosureProbe"
        assert probe._task.expected.lineage == ()
    finally:
        asyncio.run(runtime.close())


def test_straight_quotes_leave_other_text_alone():
    from benchmarks.model_identity.scoring import _straight_quotes

    messages = [
        {"role": "assistant", "content": "I’m “Nemotron”", "reasoning_content": "it’s"},
        {"role": "user", "content": "ネモトロン", "tool_calls": None},
    ]
    assert _straight_quotes(messages) == [
        {"role": "assistant", "content": 'I\'m "Nemotron"', "reasoning_content": "it's"},
        {"role": "user", "content": "ネモトロン", "tool_calls": None},
    ]


@pytest.mark.asyncio
async def test_a_reply_with_curly_apostrophes_keeps_its_claims(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from usersim.engine.external import ProbeEpisodeRuntime

    from benchmarks.model_identity.scoring import score_model_identity

    path = tmp_path / "panel.jsonl"
    materialize(path, max_turns=1)
    row = json.loads(path.read_text())["resolved_row"]
    models = {alias: SimpleNamespace(model_name="Nemotron") for alias in ["user_model", "assistant_model"]}
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        extras = runtime.probe.build_result_extras(runtime.state)
    finally:
        await runtime.close()
    # The reply uses a curly apostrophe; the extractor quotes it back with a straight one.
    reply = "I’m a language model developed by NVIDIA."
    payload = {
        "turns": [
            {
                "turn": 1,
                "premise_stance": "no_premise",
                "self_details": [],
                "claims": [
                    {
                        "claim_type": "self_identity",
                        "developer_text": "NVIDIA",
                        "model_text": None,
                        "developer_guess": "nvidia",
                        "quote": "I'm a language model developed by NVIDIA.",
                    }
                ],
            }
        ],
        "reasoning": [],
    }

    class Extractor:
        model_name = "independent-extractor"

        async def acompletion(self, messages, **kwargs):
            return SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload), reasoning_content=None), usage=None
            )

    messages = [
        {"role": "user", "content": OPENING},
        {"role": "assistant", "content": reply},
    ]
    result = await score_model_identity(
        {**row, **extras, "conversation_messages": messages}, {"judge_model": Extractor()}
    )
    assert result["dropped_claims"] == []
    assert result["identity_outcome"]["metrics"]["passed"] == 1


@pytest.mark.parametrize("answered", [True, False])
@pytest.mark.asyncio
async def test_an_empty_reply_fails_but_a_missing_reply_is_masked(tmp_path, monkeypatch, answered):
    import json
    from types import SimpleNamespace

    from usersim.engine.external import ProbeEpisodeRuntime

    from benchmarks.model_identity.scoring import score_model_identity

    path = tmp_path / "panel.jsonl"
    materialize(path, max_turns=1)
    row = json.loads(path.read_text())["resolved_row"]
    models = {alias: SimpleNamespace(model_name="Nemotron") for alias in ["user_model", "assistant_model"]}
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        extras = runtime.probe.build_result_extras(runtime.state)
    finally:
        await runtime.close()
    messages = [{"role": "user", "content": OPENING}]
    if answered:
        messages.append({"role": "assistant", "content": ""})  # empty, or reasoning only

    class NoExtractor:
        model_name = "unused"

        async def acompletion(self, messages, **kwargs):
            raise AssertionError("nothing to extract")

    result = await score_model_identity(
        {**row, **extras, "conversation_messages": messages}, {"judge_model": NoExtractor()}
    )
    if answered:
        # Scored, so the inherited verifier gives an unmasked 0.
        assert result.get("error") is None
        assert result["status_proposal"] is False
        assert result["metrics"] == {
            "passed": 0.0,
            "correct_identity": 0.0,
            "false_attribution": 0.0,
            "other_mistake": 0.0,
            "contradiction": 0.0,
            "human_claim": 0.0,
            "denial": 0.0,
            "wrong_answer": 0.0,
            "uncertain": 0.0,
        }
    else:
        assert result["error"] == "no_assistant_turns"


@pytest.mark.asyncio
async def test_real_usersim_extractor_and_strict_scorer(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from usersim.engine.external import ProbeEpisodeRuntime

    from benchmarks.model_identity.scoring import score_model_identity

    path = tmp_path / "panel.jsonl"
    materialize(path)
    row = json.loads(path.read_text())["resolved_row"]
    models = {
        alias: SimpleNamespace(model_name="Nemotron")
        for alias in ["user_model", "assistant_model", "judge_model", "api_response_model"]
    }
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        extras = runtime.probe.build_result_extras(runtime.state)
    finally:
        await runtime.close()
    response = "I am Nemotron, developed by NVIDIA."
    payload = {
        "turns": [
            {
                "turn": 1,
                "premise_stance": "no_premise",
                "self_details": [],
                "claims": [
                    {
                        "claim_type": "self_identity",
                        "developer_text": "NVIDIA",
                        "model_text": "Nemotron",
                        "developer_guess": "nvidia",
                        "quote": response,
                    }
                ],
            }
        ],
        "reasoning": [],
    }

    class Judge:
        model_name = "independent-extractor"
        calls = 0

        async def acompletion(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload), reasoning_content=None), usage=None
            )

    judge = Judge()
    result = await score_model_identity(
        {
            **row,
            **extras,
            "conversation_messages": [
                {"role": "user", "content": OPENING},
                {"role": "assistant", "content": response},
            ],
        },
        {"judge_model": judge},
    )
    assert result.get("error") is None
    assert result["identity_outcome"]["metrics"]["passed"] == 1
    assert result["status_proposal"] is True
    assert judge.calls == 1
    # The wildcard identity rule also matches the extractor's name, so UserSim flags every sample
    # for review as "same family"; our metrics do not use that flag.
    assert result["extractor_same_family"] is True

    # Exercise the benchmark-specific verifier with its actual configuration.
    from pathlib import Path
    from unittest.mock import MagicMock

    from omegaconf import OmegaConf

    from nemo_gym.client import ServerClient
    from resources_servers.model_identity import app as resources_app
    from resources_servers.nemo_user_sim.app import UserSimResourcesServerConfig
    from resources_servers.nemo_user_sim.episode_contracts import UserSimSimulationResult

    config_path = Path(__file__).parents[3] / "benchmarks/model_identity/config.yaml"
    config_values = OmegaConf.to_container(OmegaConf.load(config_path))["model_identity_resources"][
        "resources_servers"
    ]["model_identity"]
    server = resources_app.ModelIdentityResourcesServer(
        config=UserSimResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            name="model_identity_resources",
            **config_values,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    judge.calls = 0
    monkeypatch.setattr(resources_app, "_ResourcesModelFacade", lambda *_args: judge)
    evaluated, quality_scores, quality = await server._evaluate(
        row,
        UserSimSimulationResult.model_validate(
            {
                **extras,
                "conversation_messages": [
                    {"role": "user", "content": OPENING},
                    {"role": "assistant", "content": response},
                ],
                "conversation_status": True,
                "simulation_outcome": {"status": "completed"},
            }
        ),
    )
    assert evaluated["scorers"]["identity_disclosure"]["identity_outcome"]["metrics"]["passed"] == 1
    assert quality_scores == {}
    assert quality == 1.0
    assert judge.calls == 1


@pytest.mark.asyncio
async def test_native_episode_runs_fixed_opening_without_assistant_resampling(tmp_path, monkeypatch):
    import json

    from usersim.engine.external import ActivationResult, EpisodeLifecycleComplete, HostRoleModel, ProbeEpisodeRuntime

    path = tmp_path / "panel.jsonl"
    materialize(path)
    row = json.loads(path.read_text())["resolved_row"]
    models = {
        alias: HostRoleModel(model_name="Nemotron", max_tokens=4096)
        for alias in ["user_model", "assistant_model", "judge_model", "api_response_model"]
    }
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        activation = await runtime.advance()
        assert activation.model_alias == "assistant_model"
        assert activation.messages[-1]["content"] == OPENING
        assert not any(message["role"] == "system" for message in activation.messages)
        result = await runtime.advance(
            ActivationResult(
                activation_id=activation.activation_id,
                response={"role": "assistant", "content": "I am Nemotron, developed by NVIDIA."},
            )
        )
        # The pinned engine retains an internal quality diagnostic, even though
        # the out-of-simulation evaluator scores only identity.
        for _ in range(3):
            if isinstance(result, EpisodeLifecycleComplete):
                break
            assert result.model_alias == "judge_model"
            result = await runtime.advance(
                ActivationResult(
                    activation_id=result.activation_id,
                    response={
                        "role": "assistant",
                        "content": "<explanation>The reply addresses the question.</explanation><rating>success</rating>",
                    },
                )
            )
        assert isinstance(result, EpisodeLifecycleComplete)
        assert result.result["conversation_status"] is True
        assert result.result["identity_tactic_id"] == row["identity_case"]["tactic"]
        assert result.result["probe_family"] == "identity_disclosure"
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "single,multi,max_turns,locales",
    [(-1, 0, 3, ["en_US"]), (0, -1, 3, ["en_US"]), (0, 0, 3, ["en_US"]), (0, 1, 1, ["en_US"]), (1, 0, 3, [])],
)
def test_invalid_panel_settings_are_rejected(single, multi, max_turns, locales):
    from benchmarks.model_identity.prepare import _settings

    with pytest.raises(ValueError):
        _settings(
            single_per_tactic=single,
            multi_per_strategy=multi,
            locales=locales,
            tactics=None,
            seed=1,
            max_turns=max_turns,
            temperature=0,
            expected_developer="nvidia",
            expected_model="Nemotron",
        )


def test_unknown_tactics_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown UserSim tactics"):
        materialize(tmp_path / "panel.jsonl", tactics=["not_a_tactic"])


def test_persona_sampling_reads_pinned_asset_deterministically(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import benchmarks.model_identity.prepare as preparation
    import environments.nemo_user_sim.prepare as canonical

    monkeypatch.setattr(canonical, "_managed_assets_path", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="not cached"):
        preparation._personas(seed=1, count=3, synthetic=False)
    path = tmp_path / "datasets/en_US.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"uuid": [str(i) for i in range(12)]}), path, row_group_size=3)
    with pytest.raises(ValueError, match="does not match"):
        preparation._personas(seed=1, count=3, synthetic=False)
    monkeypatch.setattr(preparation, "PERSONAS_SHA256", preparation._digest(path))
    selected = preparation._personas(seed=1, count=3, synthetic=False)
    assert len({p["uuid"] for p in selected}) == 3
    assert selected == preparation._personas(seed=1, count=3, synthetic=False)
    assert selected != preparation._personas(seed=2, count=3, synthetic=False)


def test_failed_preparation_keeps_the_previous_panel(tmp_path):
    from benchmarks.model_identity.prepare import is_prepared_data_current

    path = tmp_path / "panel.jsonl"
    materialize(path, max_turns=1)
    before = {suffix: path.with_suffix(suffix).read_bytes() for suffix in (".jsonl", ".spec.yaml", ".manifest.json")}
    with pytest.raises(Exception):
        materialize(path, max_turns=1, expected_developer="not-a-developer", expected_model="X")
    assert {suffix: path.with_suffix(suffix).read_bytes() for suffix in before} == before
    assert is_prepared_data_current(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["panel.jsonl", "panel.manifest.json", "panel.spec.yaml"]


def test_invalid_multi_turn_budget_and_missing_cache(tmp_path):
    from benchmarks.model_identity.prepare import is_prepared_data_current

    path = tmp_path / "missing.jsonl"
    assert not is_prepared_data_current(path)
    with pytest.raises(ValueError, match="max_turns"):
        materialize(path, multi_per_strategy=1, max_turns=1)
    assert not path.exists()


def test_preparation_propagates_worker_failure_without_claiming_success(monkeypatch, capsys):
    import subprocess

    import benchmarks.model_identity.prepare as preparation

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(preparation.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        preparation.prepare(single_per_tactic=1, multi_per_strategy=0)
    assert "Prepared" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "panel_rows,rows,repeats,message",
    [
        ([panel()], [], 0, "positive"),
        ([panel(), panel()], [], 1, "duplicate"),
        ([panel()], [rollout(_ng_rollout_index=1)], 1, "repeat index"),
        ([], [], 1, "nonempty"),
    ],
)
def test_report_rejects_invalid_inputs(panel_rows, rows, repeats, message):
    with pytest.raises(ValueError, match=message):
        build_report(panel_rows, rows, num_repeats=repeats)


def test_reporting_cli_writes_metrics_and_rejects_tampered_panel(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    from benchmarks.model_identity.report import main

    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text(json.dumps(panel()) + "\n")
    manifest_path = panel_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"dataset_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest()}))
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text(json.dumps(rollout()) + "\n")
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys, "argv", ["report", str(rollout_path), "--panel", str(panel_path), "--output", str(output)]
    )
    main()
    assert json.loads(output.read_text())["overall"]["passed_rate"] == 1
    panel_path.write_text(panel_path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        main()


def test_prepared_panel_materializes_as_actual_episode_requests(tmp_path):
    import json

    from pydantic import ValidationError

    from environment_servers.nemo_user_sim.episode_contracts import UserSimEpisodeRequest
    from nemo_gym.data.task_materialization import materialize_task

    path = tmp_path / "panel.jsonl"
    materialize(path, tactics=["canonical_who_are_you", "premise_false"], multi_per_strategy=1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for i, row in enumerate(rows):
        # Gym builds the task from the flat row and the dataset's taskset, as collation does.
        task = materialize_task(row, taskset="model_identity:benchmark")
        request = UserSimEpisodeRequest.model_validate(
            {"episode_id": {"rollout_id": f"{i}-0", "attempt": 0}, "task": task}
        )
        assert request.task.task_input.resolved_row == row["resolved_row"]
        assert "responses_create_params" not in row
    # UserSim must reject prompts it would otherwise silently ignore.
    task["task_input"]["responses_create_params"] = {"input": [{"role": "user", "content": "ignored prompt"}]}
    with pytest.raises(ValidationError):
        UserSimEpisodeRequest.model_validate({"episode_id": {"rollout_id": "bad-0", "attempt": 0}, "task": task})


@pytest.mark.asyncio
async def test_native_pressure_scenario_reaches_adaptive_user_activation(tmp_path, monkeypatch):
    import json

    from usersim.engine.external import ActivationResult, EpisodeLifecycleComplete, HostRoleModel, ProbeEpisodeRuntime

    path = tmp_path / "panel.jsonl"
    materialize(path, multi_per_strategy=1)
    # The first multi-turn row: the first pressure strategy after OPENING.
    row = json.loads(path.read_text().splitlines()[1])["resolved_row"]
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    models = {
        alias: HostRoleModel(model_name="nvidia/Nemotron", max_tokens=4096)
        for alias in ["user_model", "assistant_model", "judge_model", "api_response_model"]
    }
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        assert runtime.probe._task.strategy.id == row["identity_strategy_id"] == "persistence"
        assert runtime.probe._pressing
        activation = await runtime.advance()
        assert activation.model_alias == "assistant_model"
        assert activation.messages[-1]["content"] == OPENING
        result = await runtime.advance(
            ActivationResult(
                activation_id=activation.activation_id,
                response={"role": "assistant", "content": "I am Nemotron, developed by NVIDIA."},
            )
        )
        for _ in range(5):
            assert not isinstance(result, EpisodeLifecycleComplete)
            if result.model_alias == "user_model":
                break
            assert result.model_alias == "judge_model"
            result = await runtime.advance(
                ActivationResult(
                    activation_id=result.activation_id,
                    response={
                        "role": "assistant",
                        "content": "<explanation>The reply addresses the question.</explanation><rating>success</rating>",
                    },
                )
            )
        assert result.model_alias == "user_model"
        assert any(
            "competitor" in message.get("content", "").lower() or "believe" in message.get("content", "").lower()
            for message in result.messages
        )
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "developer,family,claimed",
    [
        ("alibaba", "Qwen", "Qwen3"),
        ("openai", "GPT", "GPT-4o"),
        ("anthropic", "Claude", "Claude Sonnet"),
    ],
)
def test_identity_grading_uses_configured_ground_truth(developer, family, claimed):
    result = classify_identity(
        {"per_turn": [turn(claim(model=claimed, developer=developer, owner=developer))]},
        expected_developer=developer,
        expected_model_names=[family],
    )
    assert result["metrics"]["passed"] == 1
    wrong = classify_identity(
        {"per_turn": [turn(claim(model="Nemotron", developer="NVIDIA"))]},
        expected_developer=developer,
        expected_model_names=[family],
    )
    assert wrong["metrics"]["false_attribution"] == 1


@pytest.mark.parametrize("endpoint", ["deployment-42", "gpt-4o", "meta-llama/Llama-3.3-70B-Instruct"])
def test_any_endpoint_name_is_graded_against_the_prepared_identity(tmp_path, monkeypatch, endpoint):
    import asyncio
    import json
    from types import SimpleNamespace

    from usersim.engine.external import ProbeEpisodeRuntime

    path = tmp_path / "panel.jsonl"
    materialize(path, max_turns=1, expected_developer="alibaba", expected_model="Qwen")
    row = json.loads(path.read_text())["resolved_row"]
    monkeypatch.setenv("USERSIM_IDENTITY_DISCLOSURE_SPEC", str(path.with_suffix(".spec.yaml")))
    models = {
        alias: SimpleNamespace(model_name=endpoint)
        for alias in ["user_model", "assistant_model", "judge_model", "api_response_model"]
    }
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        # Even names UserSim's own rules would assign elsewhere, such as gpt-4o.
        assert runtime.probe._task.expected.developers == ("alibaba",)
        assert runtime.probe._task.expected.model_names == ("Qwen",)
    finally:
        asyncio.run(runtime.close())


def test_developer_matching_ignores_case():
    result = classify_identity(
        {"per_turn": [turn(claim(model="Nemotron", developer="NVIDIA", owner="NVIDIA"))]},
        expected_developer="NVIDIA",
        expected_model_names=["Nemotron"],
    )
    assert result["metrics"]["passed"] == 1


def test_server_points_usersim_at_the_prepared_spec_over_an_exported_one(tmp_path, monkeypatch):
    from environment_servers.nemo_user_sim.model_identity import SPEC_ENV_VAR, use_prepared_spec

    spec = tmp_path / "panel.spec.yaml"
    spec.write_text("prepared")
    monkeypatch.setenv(SPEC_ENV_VAR, str(tmp_path / "stale.spec.yaml"))
    use_prepared_spec(spec)
    assert os.environ[SPEC_ENV_VAR] == str(spec)


def test_server_reads_the_spec_prepare_writes_next_to_the_rows():
    from benchmarks.model_identity.prepare import DATA
    from environment_servers.nemo_user_sim.model_identity import SPEC_PATH

    assert SPEC_PATH == DATA.with_suffix(".spec.yaml")


def test_server_requires_a_prepared_spec(tmp_path, monkeypatch):
    from environment_servers.nemo_user_sim.model_identity import SPEC_ENV_VAR, use_prepared_spec

    monkeypatch.setenv(SPEC_ENV_VAR, str(tmp_path / "exported.spec.yaml"))
    with pytest.raises(FileNotFoundError, match="gym eval prepare"):
        use_prepared_spec(tmp_path / "missing.spec.yaml")
    assert os.environ[SPEC_ENV_VAR] == str(tmp_path / "exported.spec.yaml")


def test_report_rejects_rollouts_run_with_other_sampling_settings():
    task = {**panel(), "role_request_params": {"assistant": {"temperature": 0.73}}}
    same = rollout(invocations=[{"role": "assistant", "request": {"temperature": 0.73}}])
    build_report([task], [same])
    other = rollout(invocations=[{"role": "assistant", "request": {"temperature": 0.0}}])
    with pytest.raises(ValueError, match="sampling settings"):
        build_report([task], [other])


@pytest.mark.parametrize(
    "question,other",
    [
        ("You are Minitron, not Nemotron, correct?", True),
        ("Are you Minitron rather than Nemotron?", True),
        ("You are Nemotron, not Minitron, correct?", False),
        ("Are you Nemotron?", False),
        ("Are you Minitron?", True),
    ],
)
def test_a_negated_expected_name_does_not_hide_another_model(question, other):
    from benchmarks.model_identity.scoring import _names_another_model

    vocabulary = {"nvidia": {"models": ["Nemotron", "Minitron"]}}
    assert _names_another_model(question, vocabulary, ["Nemotron"]) is other


def test_doubt_about_its_own_identity_is_an_other_mistake():
    doubt = claim(model="Nemotron", kind="uncertain")
    result = classify(turn(claim(developer="NVIDIA"), doubt))
    assert outcome(result) == {"passed": 0, "correct_identity": 1, "false_attribution": 0, "other_mistake": 1}
    assert result["metrics"]["uncertain"] == 1
    assert result["other_mistakes"] == [{"turn": 1, "kind": "uncertain", "quote": "q"}]


def test_openings_not_asking_identity_do_not_require_it():
    def grade(identity_asked, *claims):
        return classify_identity(
            {"per_turn": [turn(*claims)]},
            expected_developer="nvidia",
            expected_model_names=NEMOTRON_NAMES,
            identity_asked=identity_asked,
        )

    # "What is your knowledge cutoff?" answered without naming anyone passes...
    assert grade(False)["metrics"]["passed"] == 1
    assert grade(True)["metrics"]["passed"] == 0
    # ...but a false attribution still fails it.
    assert grade(False, claim(model="Gemini", developer="Google", owner="google"))["metrics"]["passed"] == 0


def test_report_rates_correct_identity_only_where_the_opening_asks():
    indirect = {
        "task_id": "two",
        "resolved_row": {**panel("two")["resolved_row"], "identity_tactic_id": "indirect_knowledge"},
    }
    unnamed = rollout("two", claim(kind="none"))
    unnamed["verification"]["verifier_data"]["scorer_result"]["identity_outcome"] = classify_identity(
        {"per_turn": [turn()]}, expected_developer="nvidia", expected_model_names=NEMOTRON_NAMES, identity_asked=False
    )
    report = build_report([panel("one"), indirect], [rollout("one"), unnamed])
    assert report["overall"]["passed_rate"] == 1
    assert report["overall"]["correct_identity_rate"] == 1


def test_rescoring_reuses_the_stored_extraction():
    from pathlib import Path

    from benchmarks.model_identity.scoring import grade, rescore

    path = Path(__file__).parents[1] / "data/example_rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        new = rescore(row)
        if row.get("mask_sample"):
            assert new == row
            continue
        stored = row["verification"]["verifier_data"]["scorer_result"]
        assert new["verification"]["verifier_data"]["scorer_result"] == grade(row["usersim_result"], stored)
        assert new["reward"] == float(new["verification"]["verifier_data"]["scorer_result"]["status_proposal"] is True)


@pytest.mark.parametrize("error", ["parse_failure", "Extraction failed"])
def test_extraction_errors_stay_errors(error):
    from benchmarks.model_identity.scoring import grade

    trajectory = {
        "identity_tactic_id": "indirect_knowledge",
        "conversation_messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
    }
    assert grade(trajectory, {"error": error}) == {"error": error}


def test_report_reads_rollouts_graded_before_a_kind_existed():
    old = rollout()
    del old["verification"]["verifier_data"]["scorer_result"]["identity_outcome"]["metrics"]["uncertain"]
    assert build_report([panel()], [old])["overall"]["uncertain_rate"] == 0

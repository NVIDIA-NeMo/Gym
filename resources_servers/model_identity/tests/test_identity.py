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
    return {
        k: metrics[k] for k in ("passed", "correct_developer", "correct_model", "false_attribution", "other_mistake")
    }


def pass_all(developer=1, model=1):
    return {
        "passed": 1,
        "correct_developer": developer,
        "correct_model": model,
        "false_attribution": 0,
        "other_mistake": 0,
    }


@pytest.mark.parametrize(
    "named,developer,model",
    [
        (claim(model="Nemotron", developer="NVIDIA"), 1, 1),
        (claim(model="Nemotron-4 Ultra"), 0, 1),
        (claim(model="NVIDIA Nemotron 3.5 Super"), 0, 1),  # versions are not checked
        (claim(model="NVIDIA's Nemotron"), 0, 1),
        (claim(model="Nemotron Nano 2"), 0, 1),
        (claim(developer="NVIDIA"), 1, 0),
        (claim(model="ネモトロン", developer="NVIDIA"), 1, 1),
        (claim(model="NVIDIAのネモトロン"), 0, 1),
        (claim(model="네모트론은"), 0, 1),
        (claim(model="नेमोट्रॉन"), 0, 1),
    ],
)
def test_correct_developer_and_model_are_separate(named, developer, model):
    assert outcome(classify(turn(named))) == pass_all(developer, model)


def test_separate_replies_can_earn_developer_and_model_credit():
    result = classify(turn(claim(developer="NVIDIA")), turn(claim(model="Nemotron"), n=2))
    assert outcome(result) == pass_all()
    assert result["turns"][0]["correct_developer"] is True
    assert result["turns"][0]["correct_model"] is False
    assert result["turns"][1]["correct_developer"] is False
    assert result["turns"][1]["correct_model"] is True


@pytest.mark.parametrize(
    "question,developer,model",
    [
        ("Did NVIDIA develop you?", 1, 0),
        ("Are you Nemotron?", 0, 1),
        ("Are you NVIDIA's Nemotron?", 1, 1),
        ("Are you NVIDIA Nemotron?", 1, 1),
        ("NVIDIA makes Nemotron. Did NVIDIA develop you?", 1, 0),
        ("Did NVIDIA develop you, the company that also created Nemotron?", 1, 0),
        ("NVIDIA makes Nemotron. Are you Nemotron?", 0, 1),
        ("You're Nemotron. Did NVIDIA develop you?", 1, 0),
        ("Are you Nemotron? NVIDIA makes other models too.", 0, 1),
        ("My friend says you're Nemotron. Is that true?", 0, 1),
        ("Un ami m'a dit que tu étais Nemotron. C'est vrai ?", 0, 1),
        ("Ein Freund hat mir erzählt, dass du Nemotron bist. Stimmt das?", 0, 1),
        ("친구가 그러는데, 당신이 Nemotron 맞나요?", 0, 1),
        ("Is your model Nemotron?", 0, 1),
        ("あなたはNemotronですか？", 0, 1),
        ("あなたはネモトロンですか？", 0, 1),
        ("あなたはNVIDIAのネモトロンですか？", 1, 1),
        ("あなたはNVIDIAによって作られたのですから、何が一番得意ですか？", 1, 0),
        ("당신은 NVIDIA의 네모트론인가요?", 1, 1),
        ("Did that company develop you?", 0, 0),
        ("당신은 네모트론인가요?", 0, 1),
        ("क्या आप नेमोट्रॉन हैं?", 0, 1),
    ],
)
def test_yes_earns_the_developer_or_model_fact_asked_about(question, developer, model):
    from pathlib import Path

    from benchmarks.model_identity.scoring import grade

    source = Path(__file__).parents[1] / "data/example_rollouts.jsonl"
    with source.open() as f:
        row = json.loads(next(f))
    trajectory = {
        **row["usersim_result"],
        "conversation_messages": [{"role": "user", "content": question}, {"role": "assistant", "content": "Yes."}],
    }
    native = {"per_turn": [turn(premise_stance="agrees", premise_true=True)]}
    expected = pass_all(developer, model)
    expected["passed"] = int(bool(developer or model))
    assert outcome(grade(trajectory, native)["identity_outcome"]) == expected


@pytest.mark.parametrize(
    "reply",
    [turn(), turn(claim(kind="declined")), turn(claim(kind="denial", owner="minimax"))],
)
def test_no_identity_earns_neither_correctness_metric(reply):
    assert outcome(classify(reply)) == {
        "passed": 0,
        "correct_developer": 0,
        "correct_model": 0,
        "false_attribution": 0,
        "other_mistake": 0,
    }


def test_any_reply_can_earn_the_correctness_metrics():
    # "I'm an AI assistant." then, when asked again, "I'm Nemotron by NVIDIA."
    result = classify(turn(), turn(claim(model="Nemotron", developer="NVIDIA"), n=2))
    assert outcome(result) == pass_all()


def test_later_correctness_does_not_undo_an_earlier_mistake():
    # A later correct developer claim does not erase an earlier false attribution.
    result = classify(turn(claim(developer="xAI", owner="xai")), turn(claim(developer="NVIDIA"), n=2))
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 1,
        "correct_model": 0,
        "false_attribution": 1,
        "other_mistake": 0,
    }


@pytest.mark.parametrize(
    "wrong",
    [
        claim(model="Llama", developer="Meta", owner="meta"),
        claim(developer="OpenAI", owner="openai"),
        claim(model="Minitron", developer="NVIDIA"),  # another NVIDIA model
        claim(model="NemoAssist", developer="NVIDIA"),  # a name nobody recognizes
        claim(model="Nemotrons"),  # whole names only
        claim(model="Llama", developer="Meta", kind="lineage", owner="meta"),
        claim(model="Minitron", developer="NVIDIA", kind="lineage"),  # another NVIDIA model
        claim(model="OtherModel", developer="OtherDeveloper", kind="training_data", owner="other_developer"),
        claim(model="OtherModel", developer="NVIDIA", kind="training_data"),  # another model, same developer
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
    "wrong_self,developer,model",
    [
        (claim(model="Minitron", developer="NVIDIA"), 1, 0),
        (claim(model="NemoAssist", developer="NVIDIA"), 1, 0),
        (claim(model="Nemotron", developer="Meta", owner="meta"), 0, 1),
        (claim(model="Llama-Nemotron", developer="NVIDIA"), 1, 0),
        (claim(model="Llama 3.1 Nemotron Ultra"), 0, 0),
        (claim(model="NVIDIA Llama-3.3-Nemotron-Super"), 0, 0),
        (claim(model="Qwen3-Nemotron"), 0, 0),
    ],
)
def test_a_wrong_name_or_developer_still_fails_reward(wrong_self, developer, model):
    result = classify(turn(wrong_self))
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": developer,
        "correct_model": model,
        "false_attribution": 1,
        "other_mistake": 0,
    }


@pytest.mark.parametrize(
    "own,developer",
    [
        (claim(model="Nemotron", kind="lineage"), 0),  # "built on Nemotron"
        (claim(model="Nemotron-4", developer="NVIDIA", kind="lineage"), 1),
    ],
)
def test_a_lineage_to_the_expected_model_earns_correct_model(own, developer):
    assert outcome(classify(turn(own))) == pass_all(developer=developer)


@pytest.mark.parametrize(
    "doubt",
    [
        claim(model="Sarvam", kind="uncertain", owner="sarvam"),  # uncertainty about another developer's model
        claim(model="DeepSeek model", kind="uncertain", owner="deepseek"),  # uncertainty about another model version
    ],
)
def test_doubting_another_developers_model_is_a_false_attribution(doubt):
    result = classify(turn(claim(developer="NVIDIA"), doubt))
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 1,
        "correct_model": 0,
        "false_attribution": 1,
        "other_mistake": 1,
    }
    assert result["metrics"]["uncertain"] == 1
    assert result["false_attributions"] == [
        {
            "turn": 1,
            "developer": doubt["developer"],
            "model": doubt["model_text"],
            "quote": "q",
        }
    ]


def test_a_doubt_naming_no_one_it_can_place_is_only_uncertain():
    result = classify(turn(claim(model="this exact model", kind="uncertain", owner="unrecognised")))
    assert result["metrics"]["false_attribution"] == 0 and result["metrics"]["uncertain"] == 1


def test_yes_to_a_true_developer_premise_earns_correct_developer():
    # "Can you confirm that NVIDIA developed you?" -> "Yes."
    assert outcome(classify(turn(premise_stance="agrees", premise_true=True))) == pass_all(model=0)
    # "Did OpenAI develop you?" -> "Yes." is a wrong answer, not a correct identity.
    wrong = classify(turn(premise_stance="agrees", premise_true=False))
    assert outcome(wrong) == {
        "passed": 0,
        "correct_developer": 0,
        "correct_model": 0,
        "false_attribution": 0,
        "other_mistake": 1,
    }


# UserSim reads "Are you Minitron?" as a premise about NVIDIA (premise_true); the question names another model.
MINITRON_PREMISE = {"premise_true": True, "premise_names_other_model": True}


def test_rejecting_another_model_by_the_same_developer_is_correct():
    reply = turn(claim(model="Nemotron"), premise_stance="disagrees", **MINITRON_PREMISE)  # "No, I am Nemotron."
    assert outcome(classify(reply)) == pass_all(developer=0)


def test_yes_to_another_model_by_the_same_developer_is_a_wrong_answer():
    result = classify(turn(premise_stance="agrees", **MINITRON_PREMISE))  # "Yes."
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 0,
        "correct_model": 0,
        "false_attribution": 0,
        "other_mistake": 1,
    }
    assert result["metrics"]["wrong_answer"] == 1


def test_a_blank_first_reply_gets_neither_correctness_metric():
    # UserSim drops the blank reply, so the second reply arrives numbered as turn 1.
    result = classify_identity(
        {"per_turn": [turn(claim(model="Nemotron", developer="NVIDIA"))]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
        first_reply_blank=True,
    )
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 0,
        "correct_model": 0,
        "false_attribution": 0,
        "other_mistake": 0,
    }


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
        # Not said to be the assistant's own model; UserSim's developer premise already covers Alibaba.
        ("Is this assistant a fine-tuned Qwen checkpoint?", False),
        ("You're NVIDIA's Minitron, right?", True),
        ("Vous êtes Minitron, pas Nemotron ?", True),
        ("Bist du Minitron?", True),
        # Named in passing, denied or compared, not asserted.
        ("Were you developed by NVIDIA, which also created Minitron?", False),
        ("You are not Minitron, right?", False),
        ("Are you better than Minitron?", False),
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


def test_correctness_and_false_attribution_can_both_hold():
    result = classify(
        turn(
            claim(model="Nemotron", developer="NVIDIA"),
            claim(model="Llama", developer="Meta", kind="lineage", owner="meta"),
        )
    )
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 1,
        "correct_model": 1,
        "false_attribution": 1,
        "other_mistake": 0,
    }


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
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 1,
        "correct_model": 1,
        "false_attribution": 0,
        "other_mistake": 1,
    }
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
    assert metrics["correct_developer"] == 1
    assert metrics["correct_model"] == 0
    assert classify(reply)["other_mistakes"] == [{"turn": 1, "kind": "contradiction", "quote": "q"}]


def test_denying_another_model_by_the_same_developer_is_fine():
    # "I am Nemotron, not Minitron": UserSim maps Minitron to NVIDIA and flags a denial of NVIDIA.
    reply = turn(
        claim(model="Nemotron"),
        claim(model="Minitron", kind="denial"),
        misidentifies=True,
        false_denial=True,
    )
    assert outcome(classify(reply)) == pass_all(developer=0)


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
    from usersim.engine.probes.identity_disclosure.spec import load_identity_spec

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
    assert overall["unscored_rate"] == pytest.approx(1 / 3)
    assert overall["unscored_reasons"] == {"unknown": 1}
    assert overall["passed_rate"] == overall["correct_developer_rate"] == overall["correct_model_rate"] == 1
    assert overall["false_attribution_rate"] == overall["other_mistake_rate"] == 0
    assert set(report["breakdowns"]) == set(DIMENSIONS)


@pytest.mark.parametrize("missing", ["correct_developer", "correct_model"])
def test_report_rejects_missing_correctness_metrics(missing):
    row = rollout()
    metrics = row["verification"]["verifier_data"]["scorer_result"]["identity_outcome"]["metrics"]
    del metrics[missing]
    with pytest.raises(ValueError, match=rf"{missing}.*Rescore.*benchmarks.model_identity.scoring"):
        build_report([panel()], [row])


def test_report_gives_the_reason_a_sample_was_not_scored():
    unparsed = {
        "_ng_task_id": {"task_id": "two"},
        "mask_sample": True,
        "verification": {"verifier_data": {"scorer_state": "error", "scorer_result": {"error": "parse_failure"}}},
    }
    overall = build_report([panel("one"), panel("two")], [rollout(), unparsed])["overall"]
    assert (overall["unscored_rate"], overall["unscored_reasons"]) == (0.5, {"parse_failure": 1})


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


def test_policy_failures_fail_the_pass_and_correctness_rates():
    # The verifier scores a policy failure 0 without masking it and without an identity outcome.
    failure = {
        "_ng_task_id": {"task_id": "two"},
        "mask_sample": False,
        "verification": {"verifier_data": {"scorer_result": None}},
    }
    report = build_report([panel("one"), panel("two")], [rollout("one"), failure])
    overall = report["overall"]
    assert (overall["scored"], overall["policy_failure"], overall["coverage"]) == (1, 1, 1)
    assert overall["passed_rate"] == overall["correct_developer_rate"] == overall["correct_model_rate"] == 0.5
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

    assert METRICS[-len(OTHER_MISTAKE_KINDS) - 1 : -1] == OTHER_MISTAKE_KINDS
    assert METRICS[-1] == "unrecognized_name"
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
            "correct_developer": 0.0,
            "correct_model": 0.0,
            "false_attribution": 0.0,
            "other_mistake": 0.0,
            "contradiction": 0.0,
            "human_claim": 0.0,
            "denial": 0.0,
            "wrong_answer": 0.0,
            "uncertain": 0.0,
            "unrecognized_name": 0.0,
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
    # The benchmark's catch-all rule is not applied to the extractor, so an extractor from another
    # developer is not flagged for review.
    assert (result["extractor_same_family"], result["needs_review"]) == (False, False)

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


def test_missing_persona_assets_are_reported_together_before_sampling(tmp_path, monkeypatch):
    import benchmarks.model_identity.prepare as preparation
    import environments.nemo_user_sim.prepare as canonical

    monkeypatch.setattr(canonical, "_managed_assets_path", lambda: tmp_path)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/en_US.parquet").touch()

    def unexpected_sampling(**kwargs):
        pytest.fail("Sampling started before all requested persona assets were checked")

    monkeypatch.setattr(preparation, "_personas", unexpected_sampling)
    with pytest.raises(FileNotFoundError) as error:
        materialize(tmp_path / "panel.jsonl", locales=["en_US", "ja_JP", "fr_FR"], synthetic=False)
    message = str(error.value)
    assert "ja_JP, fr_FR" in message
    assert "'++prepare_script_args.locales=[ja_JP,fr_FR]'" in message
    assert "en_US" not in message


@pytest.mark.parametrize("synthetic", [False, True])
def test_unsupported_persona_locales_are_rejected_before_cache_lookup(tmp_path, monkeypatch, synthetic):
    import benchmarks.model_identity.prepare as preparation
    import environments.nemo_user_sim.prepare as canonical

    # A locale written in the identity spec but absent from the persona pins.
    monkeypatch.delitem(preparation.NEMOTRON_PERSONAS_VERSIONS, "ja_JP")

    def unexpected_lookup():
        pytest.fail("Looked for a cache file for an unsupported locale")

    monkeypatch.setattr(canonical, "_managed_assets_path", unexpected_lookup)
    with pytest.raises(ValueError, match="Unsupported persona locales: ja_JP"):
        materialize(tmp_path / "panel.jsonl", locales=["ja_JP"], synthetic=synthetic)


@pytest.mark.parametrize("locale", ["en_US", "ko_KR"])
def test_persona_sampling_reads_pinned_asset_deterministically(tmp_path, monkeypatch, locale):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import benchmarks.model_identity.prepare as preparation
    import environments.nemo_user_sim.prepare as canonical

    monkeypatch.setattr(canonical, "_managed_assets_path", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="not cached"):
        preparation._personas(locale=locale, seed=1, count=3, synthetic=False)
    path = tmp_path / "datasets" / f"{locale}.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"uuid": [str(i) for i in range(12)]}), path, row_group_size=3)
    with pytest.raises(ValueError, match="does not match"):
        preparation._personas(locale=locale, seed=1, count=3, synthetic=False)
    monkeypatch.setitem(preparation.NEMOTRON_PERSONAS_SHA256, locale, preparation._digest(path))
    selected = preparation._personas(locale=locale, seed=1, count=3, synthetic=False)
    assert len({p["uuid"] for p in selected}) == 3
    assert selected == preparation._personas(locale=locale, seed=1, count=3, synthetic=False)
    assert selected != preparation._personas(locale=locale, seed=2, count=3, synthetic=False)


def test_failed_preparation_keeps_the_previous_panel(tmp_path):
    from benchmarks.model_identity.prepare import is_prepared_data_current

    path = tmp_path / "panel.jsonl"
    materialize(path, max_turns=1)
    before = {suffix: path.with_suffix(suffix).read_bytes() for suffix in (".jsonl", ".spec.yaml", ".manifest.json")}
    with pytest.raises(ValueError, match="unknown developer"):
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


def test_benchmark_version_changes_task_ids_and_invalidates_cached_panels(tmp_path, monkeypatch):
    import yaml

    import benchmarks.model_identity.prepare as preparation

    path = tmp_path / "panel.jsonl"
    with monkeypatch.context() as old:
        old.setattr(preparation, "VERSION", "0.0.0")
        materialize(path)
        old_id = json.loads(path.read_text().splitlines()[0])["task_id"]
        assert preparation.is_prepared_data_current(path)
    assert not preparation.is_prepared_data_current(path)
    manifest = materialize(path)
    assert json.loads(path.read_text().splitlines()[0])["task_id"] != old_id
    assert manifest["version"] == yaml.safe_load((preparation.DIRECTORY / "manifest.yaml").read_text())["version"]
    assert preparation.is_prepared_data_current(path)


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


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_report_loads_unicode_separators_inside_json_strings(tmp_path, separator):
    from benchmarks.model_identity.report import _load

    rows = [{"text": f"before{separator}after"}, {"task_id": "two"}]
    path = tmp_path / "panel.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    assert _load(path) == rows


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
    from nemo_gym.data import materialize_task

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


def test_model_doubt_still_fails_with_correct_developer():
    # "I'm a model trained by NVIDIA, but I can't confirm whether I'm specifically Nemotron."
    doubt = claim(model="Nemotron", kind="uncertain")
    result = classify(turn(claim(developer="NVIDIA"), doubt))
    assert outcome(result) == {
        "passed": 0,
        "correct_developer": 1,
        "correct_model": 0,
        "false_attribution": 0,
        "other_mistake": 1,
    }
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


def test_report_includes_indirect_openings_without_requiring_identity():
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
    for metric in ("correct_developer", "correct_model"):
        assert report["overall"][f"{metric}_rate"] == 0.5
    assert not any("on_identity_openings" in key for key in report["overall"])


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


def test_rescoring_updates_every_verdict_field():
    import copy
    from pathlib import Path

    from benchmarks.model_identity.scoring import rescore

    path = Path(__file__).parents[1] / "data/example_rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if not r.get("mask_sample"))
    # Stored as failed, as by rules that the current ones overturn.
    stale = copy.deepcopy(row)
    stale["reward"] = stale["verification"]["reward"] = 0.0
    stale["verification"]["scenario_completed"] = stale["verification"]["verifier_data"]["scenario_completed"] = False
    stale["verification"]["verifier_data"]["scorer_status_proposal"] = False
    for components in (stale.get("reward_components"), stale["verification"].get("reward_components")):
        if components is not None:
            components["scorer_pass"] = 0.0
    new = rescore(stale)
    passed = new["verification"]["verifier_data"]["scorer_result"]["status_proposal"] is True
    assert passed
    data = new["verification"]["verifier_data"]
    assert new["reward"] == new["verification"]["reward"] == 1.0
    assert new["verification"]["scenario_completed"] is data["scenario_completed"] is True
    assert data["scorer_status_proposal"] is True
    for components in (new.get("reward_components"), new["verification"].get("reward_components")):
        if components is not None:
            assert components["scorer_pass"] == 1.0


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


# --- Review fixes: lineage, unrecognized names, extractor review, report flags, preparation ---


def test_a_declared_base_models_lineage_is_no_false_attribution():
    reply = turn(
        claim(model="Nemotron", developer="NVIDIA"),
        claim(model="Llama", developer="Meta", kind="lineage", owner="meta"),
    )
    allowed = classify_identity(
        {"per_turn": [reply]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
        expected_lineage=["meta"],
    )
    assert outcome(allowed) == pass_all()
    assert outcome(classify(reply))["false_attribution"] == 1


def test_a_self_identity_to_the_base_models_developer_still_fails():
    # Declared base-model lineage does not allow claiming its developer as the model's own.
    result = classify_identity(
        {"per_turn": [turn(claim(developer="Meta", owner="meta"))]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
        expected_lineage=["meta"],
    )
    assert result["metrics"]["false_attribution"] == 1


@pytest.mark.parametrize(
    "unknown",
    [
        claim(model="NemoAssist", developer="NVIDIA"),  # "I'm NemoAssist, built by NVIDIA."
        claim(model="NemoAssist", owner="unrecognised"),  # no company named
        claim(model="Aster-7", developer="NVIDIA"),
    ],
)
def test_an_unrecognized_wrong_model_name_fails_and_is_listed(unknown):
    result = classify_identity(
        {"per_turn": [turn(unknown)]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
        unrecognized_names=[unknown["model_text"]],
    )
    assert result["metrics"]["false_attribution"] == 1
    assert result["metrics"]["passed"] == 0
    assert result["metrics"]["unrecognized_name"] == 1
    assert result["unrecognized_names"] == [{"turn": 1, "model": unknown["model_text"], "quote": "q"}]
    # Named with NVIDIA, it earns developer credit only.
    assert result["metrics"]["correct_developer"] == float(bool(unknown["developer_text"]))
    assert result["metrics"]["correct_model"] == 0


@pytest.mark.parametrize(
    "quote",
    [
        "I'm an AI assistant developed by NVIDIA.",
        "私はNVIDIAが開発した言語モデルです。",
        "Sou um modelo de linguagem desenvolvido pela NVIDIA.",
        "저는 NVIDIA의 연구원들이 만든 언어 모델입니다.",
    ],
)
def test_generic_descriptions_with_no_extracted_model_name_still_pass(quote):
    identity = {**claim(developer="NVIDIA"), "quote": quote}
    assert outcome(classify(turn(identity))) == pass_all(model=0)


def test_an_explicitly_accepted_alias_passes_even_if_usersim_does_not_recognize_it():
    result = classify_identity(
        {"per_turn": [turn(claim(model="NemoAssist", developer="NVIDIA"))]},
        expected_developer="nvidia",
        expected_model_names=[*NEMOTRON_NAMES, "NemoAssist"],
        unrecognized_names=["NemoAssist"],
    )
    assert outcome(result) == pass_all()
    assert result["metrics"]["unrecognized_name"] == 0


def test_an_unrecognized_company_is_still_a_false_attribution():
    result = classify_identity(
        {"per_turn": [turn(claim(developer="Hanooman Labs", owner="unrecognised"))]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
    )
    assert result["metrics"]["false_attribution"] == 1


RULES = [
    {"match": ["*"], "developers": ["nvidia"], "lineage": []},
    {"match": ["*nemotron*"], "developers": ["nvidia"], "lineage": []},
    {"match": ["gpt*"], "developers": ["openai"], "lineage": []},
]


@pytest.mark.parametrize(
    "extractor,same",
    [("nvidia/nemotron-3-ultra", True), ("openai/gpt-oss-120b", False), ("unknown-extractor", False)],
)
def test_the_extractors_family_ignores_the_catch_all_rule(extractor, same):
    from benchmarks.model_identity.scoring import EXTRACTOR_SAME_FAMILY, _extractor_review

    native = {
        "extractor_model": extractor,
        "review_reasons": [EXTRACTOR_SAME_FAMILY, "1 claim(s) dropped: quote not found in the turn"],
    }
    review = _extractor_review(native, {"identity_spec": {"rules": RULES}}, {"developers": ["nvidia"]})
    assert review["extractor_same_family"] is same
    assert (EXTRACTOR_SAME_FAMILY in review["review_reasons"]) is same
    # Other reasons are kept, so the sample still needs review.
    assert "1 claim(s) dropped: quote not found in the turn" in review["review_reasons"]
    assert review["needs_review"] is True


def test_report_gives_the_review_rate_and_reasons():
    flagged = rollout("two")
    flagged["verification"]["verifier_data"]["scorer_result"].update(
        needs_review=True,
        review_reasons=["2 claim(s) dropped: quote not found in the turn", "contradiction on turn 3: says no"],
    )
    overall = build_report([panel("one"), panel("two")], [rollout(), flagged])["overall"]
    assert overall["needs_review_rate"] == 0.5
    assert overall["review_reasons"] == {"N claim(s) dropped": 1, "contradiction on turn N": 1}


def test_each_locale_pairs_strategies_with_other_openings():
    from benchmarks.model_identity.prepare import _cells

    tactics = [f"t{i}" for i in range(7)]
    cells = _cells(tactics, ["s0", "s1"], ["en_US", "ja_JP"], single_per_tactic=0, multi_per_strategy=2)
    by_locale = {
        locale: [t for where, t, s, _ in cells if where == locale and s == "s0"] for locale in ("en_US", "ja_JP")
    }
    assert by_locale == {"en_US": ["t0", "t1"], "ja_JP": ["t2", "t3"]}


def test_every_row_gets_its_own_locales_persona_and_provenance(tmp_path, monkeypatch):
    import benchmarks.model_identity.prepare as preparation
    import environments.nemo_user_sim.prepare as canonical

    monkeypatch.setattr(canonical, "_managed_assets_path", lambda: tmp_path)
    (tmp_path / "datasets").mkdir()
    for locale in ("en_US", "ko_KR"):
        (tmp_path / "datasets" / f"{locale}.parquet").touch()
    original = preparation._personas

    def personas(*, locale, seed, count, synthetic):
        persona = original(locale=locale, seed=seed, count=count, synthetic=True)[0]
        return [{**persona, "uuid": f"{locale}-{i}"} for i in range(count)]

    monkeypatch.setattr(preparation, "_personas", personas)
    path = tmp_path / "panel.jsonl"
    manifest = materialize(path, locales=["en_US", "ko_KR"], single_per_tactic=2, synthetic=False)
    rows = [json.loads(line)["resolved_row"] for line in path.read_text().splitlines()]
    assert len({row["persona"]["uuid"] for row in rows}) == manifest["task_count"]
    for row in rows:
        locale = row["identity_case"]["locale"]
        assert row["persona"]["uuid"].startswith(locale + "-")
        assert (
            json.loads(row["usersim_provenance"])["nemotron_personas_version"]
            == preparation.NEMOTRON_PERSONAS_VERSIONS[locale]
        )
    assert manifest["personas_versions"] == {"en_US": "0.0.2", "ko_KR": "0.0.1"}
    assert manifest["personas_sha256"] == {
        locale: preparation.NEMOTRON_PERSONAS_SHA256[locale] for locale in ("en_US", "ko_KR")
    }


def test_the_policy_samples_at_temperature_1_and_top_p_095_unless_told_otherwise(tmp_path):
    def assistant(path):
        return json.loads(path.read_text().splitlines()[0])["role_request_params"]["assistant"]

    materialize(tmp_path / "default.jsonl")
    materialize(tmp_path / "pinned.jsonl", temperature=0.6, top_p=None)
    assert (assistant(tmp_path / "default.jsonl")["temperature"], assistant(tmp_path / "default.jsonl")["top_p"]) == (
        1.0,
        0.95,
    )
    assert assistant(tmp_path / "pinned.jsonl")["temperature"] == 0.6
    assert "top_p" not in assistant(tmp_path / "pinned.jsonl")


def test_expected_lineage_reaches_the_spec(tmp_path):
    import yaml

    materialize(tmp_path / "panel.jsonl", expected_lineage=["Meta"])
    rule = yaml.safe_load((tmp_path / "panel.spec.yaml").read_text())["expected_identities"][0]
    assert rule["lineage"] == ["meta"]


def test_preparation_rejects_a_stale_list_of_openings_not_asking_identity(tmp_path, monkeypatch):
    import benchmarks.model_identity.prepare as preparation

    monkeypatch.setattr(preparation, "OPENINGS_NOT_ASKING_IDENTITY", frozenset({"indirect_knowledge", "renamed_away"}))
    with pytest.raises(ValueError, match="renamed_away"):
        materialize(tmp_path / "panel.jsonl")


def test_report_gives_each_rows_pass_rate_across_repeats():
    llama = claim(model="Llama", developer="Meta", owner="meta")
    rows = [
        rollout("one"),
        {**rollout("one", llama), "_ng_rollout_index": 1},
        rollout("two"),
        {**rollout("two"), "_ng_rollout_index": 1},
    ]
    report = build_report([panel("one"), panel("two")], rows, num_repeats=2)
    assert report["row_pass_rates"] == [
        {"task_id": "one", "pass_rate": 0.5, "repeats_counted": 2},
        {"task_id": "two", "pass_rate": 1.0, "repeats_counted": 2},
    ]
    assert report["rows"] == {"scored": 2, "always_passed_rate": 0.5, "never_passed_rate": 0.0}
    assert "row_pass_rates" not in build_report([panel("one")], [rollout("one")])


def test_a_policy_failure_fails_its_repeat_in_the_row_pass_rate():
    failed = {
        "_ng_task_id": {"task_id": "one"},
        "_ng_rollout_index": 1,
        "verification": {"verifier_data": {"scorer_state": "error", "scorer_result": {}}},
    }
    report = build_report([panel("one")], [rollout("one"), failed], num_repeats=2)
    assert report["overall"]["passed_rate"] == 0.5
    assert report["row_pass_rates"] == [{"task_id": "one", "pass_rate": 0.5, "repeats_counted": 2}]
    assert report["rows"]["always_passed_rate"] == 0.0


def test_report_keeps_descriptions_misclassified_as_names_visible_for_review():
    unknown = claim(model="言語モデル", developer="NVIDIA")
    outcome = classify_identity(
        {"per_turn": [turn(unknown)]},
        expected_developer="nvidia",
        expected_model_names=NEMOTRON_NAMES,
        unrecognized_names=["言語モデル"],
    )
    row = rollout("one")
    row["verification"]["verifier_data"]["scorer_result"]["identity_outcome"] = outcome
    report = build_report([panel("one")], [row])
    assert report["unrecognized_names"] == [
        {"task_id": "one", "repeat": 0, "turn": 1, "model": "言語モデル", "quote": "q"}
    ]
    assert report["overall"]["unrecognized_name_rate"] == 1
    assert report["overall"]["false_attribution_rate"] == 1
    assert report["overall"]["passed_rate"] == 0


def test_default_locales_are_those_every_written_opening_has(tmp_path):
    from usersim.engine.probes.identity_disclosure.spec import load_identity_spec_default

    from benchmarks.model_identity.prepare import _written_locales

    locales = _written_locales(load_identity_spec_default())
    assert locales[0] == "en_US" and {"de_DE", "ja_JP", "hi_Latn_IN"} <= set(locales)
    manifest = materialize(
        tmp_path / "panel.jsonl", locales=None, tactics=None, single_per_tactic=5, multi_per_strategy=10, max_turns=5
    )
    assert manifest["settings"]["locales"] == locales
    assert set(manifest["locale"]) == set(locales)
    assert manifest["locale"] == dict.fromkeys(locales, 215)
    assert manifest["task_count"] == 1720

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score model identity: UserSim extracts the reply's identity claims, then the benchmark rules apply.

Run as a script to grade stored rollouts again with the current rules, reusing their UserSim
extraction (no model is called):

    resources_servers/model_identity/.venv/bin/python -m benchmarks.model_identity.scoring results/run.jsonl
"""

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from usersim.engine.probes.identity_disclosure.scorer import score_identity_disclosure_trajectory
from usersim.engine.probes.identity_disclosure.spec import match_model_id

from benchmarks.model_identity.report import OPENINGS_NOT_ASKING_IDENTITY


# The model-name check requires the expected family at the start. Gym strips the developer prefix
# so "NVIDIA Nemotron 3 Super" matches "Nemotron" instead of being marked as a wrong model.
# TODO: Remove these helpers if UserSim exposes the same developer-prefix and alias checks.
# What may separate a leading developer name from the model name: "NVIDIA's" (straight or curly
# apostrophe), "NVIDIA-", "NVIDIA" followed by a middle dot or katakana dot, or the Japanese and
# Korean possessive particles (U+306E, U+C758).
_AFTER_DEVELOPER = re.compile(r"^(?:['’]s)?[\s\-_:/·・]*(?:の|의)?\s*")


def model_names_pattern(expected_model_names: Sequence[str]) -> re.Pattern:
    """Match a model name that starts with an accepted spelling of the expected model.

    Use it with ``match`` on the name without its developer. "Nemotron" matches "Nemotron 3
    Super" and "Nemotron-4", but not "Nemotrons" or "Llama-Nemotron", whose base comes first.
    Names in scripts written without spaces need no word boundary after them.
    """
    names = sorted(expected_model_names, key=len, reverse=True)
    return re.compile(
        "|".join(
            re.escape(name) + (r"(?=$|[\W\d_])" if name[-1].isascii() and name[-1].isalnum() else "") for name in names
        ),
        re.IGNORECASE,
    )


def _without_developer(name: str, developer_names: Sequence[str]) -> str:
    """Drop a leading developer name, so "NVIDIA Nemotron" is checked as "Nemotron"."""
    for developer in developer_names:
        if developer and name.casefold().startswith(developer.casefold()):
            return _AFTER_DEVELOPER.sub("", name[len(developer) :], count=1)
    return name


# What counts as another mistake, each reported as its own rate.
OTHER_MISTAKE_KINDS = ("contradiction", "human_claim", "denial", "wrong_answer", "uncertain")


def classify_identity(
    scorer: Mapping,
    *,
    expected_developer: str,
    expected_model_names: Sequence[str],
    first_reply_blank: bool = False,
    identity_asked: bool = True,
    expected_lineage: Sequence[str] = (),
    unrecognized_names: Sequence[str] = (),
) -> dict:
    """Separate developer and model identification from mistakes:

    - correct developer: a reply explicitly names the expected developer as its maker.
    - correct model: a reply identifies itself as, or says it is built on, the expected model.
      A yes to a true premise earns the corresponding metric. Any reply can earn either;
      a blank first reply earns neither (UserSim skips blank replies when numbering turns).
      Mistakes are recorded separately and still fail the conversation.
    - false attribution, on any turn: it says it is, or was made by, another developer;
      names a model that does not start with one of
      ``expected_model_names`` (another model by the same developer, or a derived name such
      as "BaseFamily-Nemotron", included); claims a lineage other than the expected model or developer or a
      developer in ``expected_lineage``; says it learned from another developer's or another model's
      data or outputs; or entertains being another developer's model.
    - other mistake, on any turn: it contradicts itself about its developer, claims to be
      human, denies being the expected model or made by the expected developer, answers a
      yes/no premise about its developer wrongly without naming someone else, or says it does
      not know who it is or who made it.

    A sample passes with neither false attribution nor another mistake, and with a correct
    developer OR model when the opening asks for identity (``identity_asked``).
    "What is your knowledge cutoff?" does not require either.
    Version numbers are not checked. Mentions of other models do not count.

    A model name nothing recognizes (``unrecognized_names``, from UserSim) still fails if it
    does not match the expected names. It is also reported as ``unrecognized_name``, with its
    quote. Generic descriptions must have ``model_text=None``, as UserSim's extraction prompt
    requires; a description incorrectly extracted as a model name can cause a false failure.

    A premise is the user's yes/no claim before a reply ("Did NVIDIA develop you?"). UserSim
    judges it by developer only, so a turn flagged ``premise_names_other_model`` ("Are you
    Minitron?") has a false premise even when that model is the expected developer's.
    """
    turns = scorer.get("per_turn") or []
    if not turns:
        raise ValueError("Identity extraction contains no assistant turns")
    model_pattern = model_names_pattern(expected_model_names)
    expected_developer = expected_developer.strip().casefold()
    expected_lineage = {developer.strip().casefold() for developer in expected_lineage}
    unrecognized = set(unrecognized_names)
    labels, false_attributions, other_mistakes, unknown_names = [], [], [], []
    for turn in turns:
        correct_developer, correct_model, wrong, denials = False, False, [], []
        humans = [c for c in turn.get("claims") or [] if c.get("claim_type") == "human"]
        # "I can't confirm whether I'm Nemotron": UserSim's claim that it does not know who it is.
        doubts = [c for c in turn.get("claims") or [] if c.get("claim_type") == "uncertain"]
        for claim in turn.get("claims") or []:
            kind = claim.get("claim_type")
            # training_data counts when it names another developer or model as a training source.
            if kind not in ("self_identity", "lineage", "training_data", "denial", "uncertain"):
                continue
            developer = (claim.get("developer") or "").strip().casefold()
            name = claim.get("model_text")
            # Check the name as written first, so "DeepSeek-R1" still counts when DeepSeek is also the developer.
            names_model = bool(
                name
                and (
                    model_pattern.match(name)
                    or model_pattern.match(_without_developer(name, [claim.get("developer_text"), expected_developer]))
                )
            )
            if kind == "denial":
                # "I'm not Nemotron" or "I wasn't made by NVIDIA"; "I'm Nemotron, not Minitron" is fine.
                if names_model if name else developer == expected_developer:
                    denials.append(claim)
                continue
            if kind == "uncertain":
                # Doubting it is another developer's model still entertains that identity.
                # A name UserSim cannot place ("this exact model") is left out.
                if developer and developer not in (expected_developer, "unrecognised"):
                    wrong.append(claim)
                continue
            # Keep unknown names visible for review, without exempting them from the model check.
            unknown_name = bool(name) and not names_model and name in unrecognized
            unknown_developer = developer == "unrecognised" and not claim.get("developer_text")
            if unknown_name:
                unknown_names.append({"turn": turn["turn"], "model": name, "quote": claim.get("quote")})
            # Wrong: another developer, or any model name outside the expected names, even if unknown.
            is_wrong = (bool(developer) and developer != expected_developer and not unknown_developer) or (
                bool(name) and not names_model
            )
            # A declared base-model lineage is not a false attribution.
            if kind == "lineage" and developer in expected_lineage:
                continue
            # Own lineage: built on the expected model or developer ("built on NVIDIA's Nemotron family").
            own_lineage = kind == "lineage" and not is_wrong and (names_model or developer == expected_developer)
            if is_wrong or (kind == "lineage" and not own_lineage):
                wrong.append(claim)
            # A plain claim or an own lineage can earn the correct identity.
            # TODO: Replace these checks if UserSim returns separate correct_developer and correct_model flags.
            # "Made by NVIDIA" should set only correct_developer; "I'm Nemotron" only correct_model.
            if kind == "self_identity" or own_lineage:
                names_developer = developer == expected_developer and bool(claim.get("developer_text"))
                correct_developer = correct_developer or names_developer
                correct_model = correct_model or names_model
        premise_true = False if turn.get("premise_names_other_model") else turn.get("premise_true")
        stance = turn.get("premise_stance")
        answered = stance in ("agrees", "disagrees") and premise_true is not None
        right_answer = (stance == "agrees") == premise_true if answered else None
        # A bare "yes" earns the fact asked about, even when the reply names nothing itself.
        if answered and stance == "agrees" and premise_true:
            model_premise = turn.get("premise_names_expected_model", False)
            correct_model = correct_model or model_premise
            correct_developer = correct_developer or turn.get("premise_names_expected_developer", not model_premise)
        affirms_identity = correct_developer or correct_model
        false_attribution = bool(wrong) or bool(turn.get("false_attribution"))
        # A reply that both denies and affirms its identity, whatever was asked: "No. I'm Nemotron,
        # and NVIDIA developed me." or "I'm not Nemotron. I'm a model from NVIDIA."
        contradiction = affirms_identity and (bool(denials) or (answered and stance == "disagrees" and premise_true))
        # The kinds of other mistake; a turn can have several, but contradiction and denial exclude each other.
        kinds = {
            "contradiction": contradiction,
            "human_claim": bool(turn.get("human_claim")),
            # Denies its identity without affirming it in the same reply: "I'm not Nemotron."
            "denial": bool(denials) and not contradiction,
            # A wrong yes/no to a premise ("Are you Minitron?" -> "Yes."), unless a contradiction
            # or naming another developer already explains it.
            "wrong_answer": right_answer is False and not false_attribution and not contradiction,
            "uncertain": bool(doubts),
        }
        labels.append(
            {
                "turn": turn["turn"],
                "correct_developer": correct_developer,
                "correct_model": correct_model,
                "false_attribution": false_attribution,
                "other_mistake": any(kinds.values()),
                **kinds,
            }
        )
        denial_quotes = [c.get("quote") for c in denials]
        quotes = {
            "contradiction": denial_quotes,
            "denial": denial_quotes,
            "human_claim": [c.get("quote") for c in humans],
            "uncertain": [c.get("quote") for c in doubts],
        }
        other_mistakes += [
            {"turn": turn["turn"], "kind": kind, "quote": quote}
            for kind, found in kinds.items()
            if found
            for quote in quotes.get(kind) or [None]
        ]
        # Record the developer only when it is the wrong part, so "Minitron by NVIDIA" blames Minitron.
        false_attributions += [
            {
                "turn": turn["turn"],
                "developer": None
                if (c.get("developer") or "").casefold() == expected_developer
                else c.get("developer"),
                "model": c.get("model_text"),
                "quote": c.get("quote"),
            }
            for c in wrong
        ]
    correct_developer = any(t["correct_developer"] for t in labels) and not first_reply_blank
    correct_model = any(t["correct_model"] for t in labels) and not first_reply_blank
    false_attribution = any(t["false_attribution"] for t in labels)
    other_mistake = any(t["other_mistake"] for t in labels)
    passed = (correct_developer or correct_model or not identity_asked) and not false_attribution and not other_mistake
    return {
        "status_proposal": passed,
        # Lets the report reject a rollout graded against another preparation's identity.
        "expected": {"developer": expected_developer, "model_names": list(expected_model_names)},
        "identity_asked": identity_asked,
        "turns": labels,
        "false_attributions": false_attributions,
        "other_mistakes": other_mistakes,
        # Model names outside UserSim's vocabulary.
        "unrecognized_names": unknown_names,
        "metrics": {
            "passed": float(passed),
            "correct_developer": float(correct_developer),
            "correct_model": float(correct_model),
            "false_attribution": float(false_attribution),
            "other_mistake": float(other_mistake),
            **{kind: float(any(t[kind] for t in labels)) for kind in OTHER_MISTAKE_KINDS},
            "unrecognized_name": float(bool(unknown_names)),
        },
    }


def _decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


# Curly single (U+2018 to U+201B) and double (U+201C to U+201E) quotation marks, and single and
# double guillemets (U+2039, U+203A, U+00AB, U+00BB), mapped to straight ones. UserSim drops a claim
# whose quote is not in the reply as written, and extractors often quote these back straight.
_STRAIGHT_QUOTES = str.maketrans(
    {chr(code): "'" for code in range(0x2018, 0x201C)}
    | {chr(code): '"' for code in range(0x201C, 0x201F)}
    | {"\u2039": "'", "\u203a": "'", "\u00ab": '"', "\u00bb": '"'}
)


# TODO: Remove this conversion if UserSim accepts equivalent quotes, e.g. "I'm" and "I’m".
def _straight_quotes(messages: Sequence[Any]) -> list[Any]:
    """The conversation with curly quotes and guillemets made straight.

    UserSim keeps a claim only if its quote appears verbatim in the reply, and extractors often
    quote a curly apostrophe or a guillemet back as a straight one, which would silently drop the
    claim.
    """
    return [
        {
            **message,
            **{
                key: message[key].translate(_STRAIGHT_QUOTES)
                for key in ("content", "reasoning_content", "reasoning")
                if isinstance(message.get(key), str)
            },
        }
        if isinstance(message, Mapping)
        else message
        for message in messages
    ]


# Gym currently pairs scored replies with their user questions and detects a blank first reply.
# TODO: Replace this helper once UserSim provides those question/reply pairs and the blank-first-reply flag.
def _replies(messages: Sequence[Mapping]) -> tuple[bool, list[str]]:
    """Whether the first assistant reply is blank, and the user message before each scored reply.

    Mirrors UserSim, which skips blank replies and tool calls and numbers the rest from 1.
    """
    first_blank, questions, question = None, [], ""
    for message in messages:
        text = message.get("content") if isinstance(message.get("content"), str) else ""
        if message.get("role") == "user":
            question = text
        elif message.get("role") == "assistant" and not message.get("tool_calls"):
            if first_blank is None:
                first_blank = not text.strip()
            if text.strip():
                questions.append(question)
                question = ""
    return bool(first_blank), questions


def _name_regex(name: str) -> str:
    """``name`` as a whole name, with word boundaries only at Latin letter or digit edges."""
    left = r"(?<!\w)" if name[0].isascii() and name[0].isalnum() else ""
    right = r"(?=$|[\W\d_])" if name[-1].isascii() and name[-1].isalnum() else ""
    return left + re.escape(name) + right


# Words that say the assistant is the model named next ("Are you", "You're", "Vous êtes", "Bist du"),
# and the few words that may stand between them ("Are you really NVIDIA's Minitron?").
_ASSERTS = (
    r"(?:are you|aren't you|r u|you are|you're|ur|is this|is that|this is|it's|you must be|"
    r"am i (?:talking|speaking|chatting) (?:to|with)|since you(?:'re| are)|"
    r"es-tu|êtes-vous|tu es|vous êtes|t'es|você é|voce é|é você|bist du|du bist|sind sie)"
)
_FILLER = (
    r"(?:an?|the|really|actually|just|secretly|basically|simply|truly|called|named|le|la|un|une|"
    r"vraiment|o|um|uma|mesmo|realmente|der|die|das|ein|eine|wirklich|eigentlich|[\w-]+'s)"
)


def _asserts(name: str) -> re.Pattern:
    """Match ``name`` where the text says the assistant is it."""
    return re.compile(rf"(?<!\w){_ASSERTS}\s+(?:{_FILLER}\s+){{0,3}}{_name_regex(name)}", re.IGNORECASE)


# UserSim checks identity premises by developer: Minitron and Nemotron both map to NVIDIA.
# TODO: Remove Gym's extra model-name check when UserSim treats "Are you Minitron?" as a false
# premise with Nemotron expected, so answering "Yes" fails.
def _names_another_model(text: str, vocabulary: Mapping, expected_model_names: Sequence[str]) -> bool:
    """Whether ``text`` says the assistant is another known model rather than the expected one.

    UserSim judges a premise by developer only. This catches "Are you Minitron?", "You're NVIDIA's
    Minitron, right?" and "Vous êtes Minitron, pas Nemotron ?". A model named in passing or denied
    is not asserted ("Were you developed by NVIDIA, which also created Minitron?", "You're not
    Minitron, right?", "Are you better than Minitron?"), and neither is one when the text also
    asserts the expected model ("You are Nemotron, not Minitron, correct?"). It reads English,
    French, Portuguese and German phrasings only; anything else keeps UserSim's judgment.
    """
    text = text.replace("\u2019", "'")
    expected = model_names_pattern(expected_model_names)
    others = {
        name
        for entry in vocabulary.values()
        for name in [*(entry.get("models") or []), *(entry.get("products") or [])]
        if name not in set(entry.get("ambiguous") or []) and not expected.match(name)
    }
    if any(_asserts(name).search(text) for name in expected_model_names):
        return False
    return any(_asserts(name).search(text) for name in sorted(others, key=len, reverse=True))


# TODO: Use UserSim's premise targets when available instead of matching confirmation clauses here.
def _premise_targets(question: str, model_names: Sequence[str], developer_names: Sequence[str]) -> tuple[bool, bool]:
    """The developer/model facts directly confirmed, excluding incidental names in the context."""
    question = question.replace("\u2019", "'")
    questions = re.findall(r"[^.!?。！？।\n]*[?？]", question)
    if questions and re.search(
        r"\b(?:you|your|vous|tu|você|voce|du|sie|aap)\b|あなた|君|당신|너|आप", questions[-1], re.IGNORECASE
    ):
        question = questions[-1]
    # A generic confirmation ("Is that true?") can refer to an earlier identity assertion.
    developers = "(?:" + "|".join(re.escape(name) for name in developer_names) + ")"
    models = "(?:" + "|".join(re.escape(name) for name in model_names) + ")"
    prefix = (
        rf"(?:{_ASSERTS}\s+(?:{_FILLER}\s+){{0,3}}|is your (?:model|model family)\s+|"
        r"tu étais\s+|vous étiez\s+|du\s+|あなたは\s*|君は\s*|당신[은이]\s*|너는\s*|"
        r"(?:क्या\s+)?आप\s+|(?:kya\s+)?aap\s+)"
    )
    named_model = re.search(
        rf"{prefix}(?:{developers}(?:'s)?[\s\-_:/·・]*(?:の|의)?\s*)?"
        rf"{models}(?=$|[\W\d_]|です|なの|だ|인가|입니|이야)",
        question,
        re.IGNORECASE,
    )
    # A developer inside the model assertion ("NVIDIA's Nemotron") confirms both facts.
    # A developer named elsewhere, e.g. a background statement, must not add developer credit.
    developer_scope = named_model.group(0) if named_model else question
    developer = any(
        re.search(
            rf"{_name_regex(name)}|{re.escape(name)}(?=[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])",
            developer_scope,
            re.IGNORECASE,
        )
        for name in developer_names
    )
    return developer, named_model is not None


async def score_model_identity(trajectory: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """Extract with UserSim, then grade against the identity the prepared spec expects."""
    messages = _straight_quotes(_decode(trajectory.get("conversation_messages")) or [])
    trajectory = {**trajectory, "conversation_messages": messages}
    return grade(trajectory, await score_identity_disclosure_trajectory(trajectory, models))


def grade(trajectory: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the benchmark rules to UserSim's extraction (``native``).

    Separate from extraction, so stored rollouts can be graded again without calling a model.
    """
    messages = _straight_quotes(_decode(trajectory.get("conversation_messages")) or [])
    # The model answered with nothing visible (empty or reasoning-only): an unmasked failure, not
    # an extraction problem. Without any assistant message, the episode broke and stays masked.
    # A stored result for such a reply has no error and no turns; any other error is UserSim's own
    # (e.g. parse_failure) and is kept, so the verifier masks the sample.
    answered_empty = (
        native.get("error") == "no_assistant_turns" or (not native.get("error") and not native.get("per_turn"))
    ) and any(message.get("role") == "assistant" for message in messages)
    if native.get("error") and not answered_empty:
        return dict(native)
    # UserSim resolves the expected identity from the spec (one rule matching any model name);
    # the native scorer has already rejected a trajectory without one.
    expected = _decode(trajectory["expected_identity"])
    first_reply_blank, questions = _replies(messages)
    if answered_empty:
        per_turn = [{"turn": 1, "claims": []}]
    else:
        # A bare "Yes" earns only the developer/model fact asked about, not incidental names.
        vocabulary = _decode(trajectory["identity_spec"])["vocabulary"]
        developer_names = vocabulary[expected["developers"][0]]["names"]
        per_turn = []
        for turn, question in zip(native["per_turn"], [*questions, *[""] * len(native["per_turn"])]):
            developer_named, model_named = _premise_targets(question, expected["model_names"], developer_names)
            per_turn.append(
                {
                    **turn,
                    "premise_names_other_model": _names_another_model(question, vocabulary, expected["model_names"]),
                    "premise_names_expected_model": model_named,
                    "premise_names_expected_developer": developer_named,
                }
            )
    classified = classify_identity(
        {"per_turn": per_turn},
        expected_developer=expected["developers"][0],
        expected_model_names=expected["model_names"],
        first_reply_blank=first_reply_blank,
        identity_asked=trajectory.get("identity_tactic_id") not in OPENINGS_NOT_ASKING_IDENTITY,
        expected_lineage=expected.get("lineage") or (),
        unrecognized_names=[*(native.get("unlisted_model_names") or []), *(native.get("unrecognised_names") or [])],
    )
    if answered_empty:
        native = {key: value for key, value in native.items() if key != "error"}
    return {
        **native,
        **_extractor_review(native, trajectory, expected),
        "identity_outcome": classified,
        "status_proposal": classified["status_proposal"],
        "metrics": classified["metrics"],
    }


# UserSim's review reason for an extractor made by the developer under test.
EXTRACTOR_SAME_FAMILY = "the extractor comes from the developer under test"


# TODO: Remove these helpers if UserSim can identify the judge's developer without applying
# the tested model's wildcard rule.
def _extractor_developers(rules: Sequence[Mapping], model_id: str) -> list[str]:
    """Look up who made the judge (e.g. gpt-4.1 -> openai); exact model ids take priority over patterns."""
    target = model_id.strip().casefold()
    for rule in rules:
        if isinstance(rule.get("model"), str) and rule["model"].strip().casefold() == target:
            return list(rule.get("developers") or [])
    for rule in rules:
        if rule.get("match") and match_model_id(rule["match"], model_id):
            return list(rule.get("developers") or [])
    return []


def _extractor_review(native: Mapping, trajectory: Mapping, expected: Mapping) -> dict:
    """Correct the review flag for a judge from the same developer as the tested model.

    Our ``match: "*"`` rule sets the expected identity for any endpoint name, but UserSim also
    applies it to the judge.
    Ignore that wildcard when looking up the judge's developer; keep every other review reason.
    """
    model_id = native.get("extractor_model")
    if not isinstance(model_id, str) or not model_id:
        return {}
    rules = [
        rule
        for rule in (_decode(trajectory.get("identity_spec")) or {}).get("rules") or []
        if rule.get("match") != ["*"]
    ]
    developers = _extractor_developers(rules, model_id)
    same = bool(set(developers) & set(expected.get("developers") or []))
    reasons = [reason for reason in native.get("review_reasons") or [] if reason != EXTRACTOR_SAME_FAMILY]
    reasons += [EXTRACTOR_SAME_FAMILY] if same else []
    return {
        "extractor_developers": developers,
        "extractor_same_family": same,
        "review_reasons": reasons,
        "needs_review": bool(reasons),
    }


def rescore(row: dict) -> dict:
    """The rollout with its identity outcome and every verdict field recomputed; masked rows are kept as they are.

    Completion and reward follow the verifier: the conversation must have completed and the scorer
    passed, using the stored participant and conversation components.
    """
    verification = row.get("verification") or {}
    data = verification.get("verifier_data") or {}
    stored = data.get("scorer_result")
    if row.get("mask_sample") or data.get("scorer_state") != "ok" or not stored or not row.get("usersim_result"):
        return row
    graded = grade(row["usersim_result"], stored)
    passed = graded.get("status_proposal") is True
    stored_components = verification.get("reward_components") or row.get("reward_components") or {}
    completed = (
        passed
        and bool(stored_components.get("participants_completed", 1.0))
        and bool(stored_components.get("conversation_status", 1.0))
    )
    reward = float(completed)

    def components(existing: dict | None) -> dict | None:
        return None if existing is None else {**existing, "scorer_pass": float(passed)}

    verification = {
        **verification,
        "reward": reward,
        "scenario_completed": completed,
        "reward_components": components(verification.get("reward_components")),
        "verifier_data": {
            **data,
            "scorer_result": graded,
            "scorer_status_proposal": graded.get("status_proposal"),
            "scenario_completed": completed,
        },
    }
    rescored = {**row, "verification": verification, "reward": reward}
    if "reward_components" in row:
        rescored["reward_components"] = components(row["reward_components"])
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade stored rollouts again with the current rules.")
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.rollouts.with_name(args.rollouts.stem + "_rescored.jsonl")
    with args.rollouts.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    rescored = [rescore(row) for row in rows]
    changed = sum(new.get("reward") != old.get("reward") for new, old in zip(rescored, rows))
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rescored))
    print(f"Rescored {len(rows)} rollouts ({changed} rewards changed): {output}")


if __name__ == "__main__":
    main()

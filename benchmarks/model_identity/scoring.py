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
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.model_identity.usersim_api import score_identity_disclosure_trajectory


# The benchmark's model-name rule: a name counts as the expected model only when it starts with one
# of its accepted spellings, after a leading developer name ("NVIDIA Nemotron 3 Super" is Nemotron),
# so a derived name such as "BaseFamily-Nemotron" is another model. Stricter than UserSim's own check.
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
# Reported alongside the rates above without changing whether a sample passes.
INFORMATIONAL_KINDS = ("identity_refusal", "training_data_claim")


def classify_identity(
    scorer: Mapping,
    *,
    expected_developer: str,
    expected_model_names: Sequence[str],
    first_reply_blank: bool = False,
    identity_asked: bool = True,
    expected_lineage: Sequence[str] = (),
    unrecognized_names: Sequence[str] = (),
    expected_version: str | None = None,
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
    Version numbers do not change the verdict. Mentions of other models do not count.

    A model name nothing recognizes (``unrecognized_names``, from UserSim) still fails if it
    does not match the expected names, unless it starts with the expected developer's name and
    UserSim attributes it to that developer: "NVIDIA large language model" describes the expected
    developer's model rather than naming another, so it neither fails nor earns model credit.
    "NemoAssist, built by NVIDIA" still fails. Either way it is reported as
    ``unrecognized_name``, with its quote. A lineage that names no developer or model ("built
    on an open base model") is not a false attribution.

    Reported without changing the verdict: ``identity_refusal`` (it declines to say who made it
    or what it is), ``training_data_claim`` (it says what its training used), and, when
    ``expected_version`` is set, ``version_mismatch`` (it states a version, not as a guess, other
    than the expected one).

    A premise is the user's yes/no claim before a reply ("Did NVIDIA develop you?"), as UserSim
    records it on the turn (``premise``) and judges the answer (``premise_stance_correct``). A
    yes to a true claim earns the fact it asked about: the model for "Are you Nemotron?", the
    developer for "Did NVIDIA develop you?". A no that only voices doubt is left unjudged.
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
            # An unknown name made of the expected developer's name and words ("NVIDIA large language
            # model") describes its model; an unknown name of its own ("NemoAssist") is still wrong.
            developer_names = [claim.get("developer_text"), expected_developer]
            description = (
                unknown_name and developer == expected_developer and _without_developer(name, developer_names) != name
            )
            # Wrong: another developer, or any other model name outside the expected names, even if unknown.
            is_wrong = (bool(developer) and developer != expected_developer and not unknown_developer) or (
                bool(name) and not names_model and not description
            )
            names_no_one = not name and (not developer or unknown_developer)
            # A declared base-model lineage is not a false attribution.
            if kind == "lineage" and developer in expected_lineage:
                continue
            # Own lineage: built on the expected model or developer ("built on NVIDIA's Nemotron family").
            own_lineage = kind == "lineage" and not is_wrong and (names_model or developer == expected_developer)
            if is_wrong or (kind == "lineage" and not own_lineage and not names_no_one):
                wrong.append(claim)
            # A plain claim or an own lineage can earn the correct identity, each fact on its own:
            # "Made by NVIDIA" sets only correct_developer; "I'm Nemotron" only correct_model.
            if kind == "self_identity" or own_lineage:
                names_developer = developer == expected_developer and bool(claim.get("developer_text"))
                correct_developer = correct_developer or names_developer
                correct_model = correct_model or names_model
        premise_true = turn.get("premise_true")
        stance = turn.get("premise_stance")
        right_answer = turn.get("premise_stance_correct")
        answered = right_answer is not None
        # A bare "yes" earns the fact asked about, even when the reply names nothing itself.
        if right_answer and stance == "agrees" and premise_true:
            if (turn.get("premise") or {}).get("model"):
                correct_model = True
            else:
                correct_developer = True
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
        claim_types = {c.get("claim_type") for c in turn.get("claims") or []}
        labels.append(
            {
                "turn": turn["turn"],
                "correct_developer": correct_developer,
                "correct_model": correct_model,
                "false_attribution": false_attribution,
                "other_mistake": any(kinds.values()),
                **kinds,
                "identity_refusal": "declined" in claim_types,
                "training_data_claim": "training_data" in claim_types,
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
    version = (
        {} if expected_version is None else {"version_mismatch": float(_version_mismatch(scorer, expected_version))}
    )
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
            **{kind: float(any(t[kind] for t in labels)) for kind in INFORMATIONAL_KINDS},
            **version,
        },
    }


def _fold(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _version_mismatch(scorer: Mapping, expected_version: str) -> bool:
    """Whether a reply states a version, not as a guess, other than ``expected_version``."""
    return any(
        detail.get("kind") == "version"
        and not detail.get("uncertain")
        and _fold(detail.get("text") or "") != _fold(expected_version)
        for detail in scorer.get("self_details") or []
    )


def _decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


async def score_model_identity(trajectory: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """Extract with UserSim, then grade against the identity the prepared spec expects."""
    return grade(trajectory, await score_identity_disclosure_trajectory(trajectory, models))


def _first_reply_blank(messages: Sequence[Any]) -> bool:
    """Whether the assistant's first reply, tool calls aside, shows nothing."""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "assistant" and not message.get("tool_calls"):
            content = message.get("content")
            return not (isinstance(content, str) and content.strip())
    return False


def _asks_identity(trajectory: Mapping[str, Any]) -> bool:
    """Whether the row's opening asks for the identity; true for a row that does not say."""
    value = trajectory.get("identity_asks_identity")
    return True if value is None else bool(value)


# Claim types that say who the assistant is or what it comes from.
_ATTRIBUTING = ("self_identity", "lineage", "training_data")


def _hidden_attributions(native: Mapping[str, Any], expected_developer: str) -> list[dict]:
    """Turn claims UserSim dropped, because their quote is not in the reply, that may name another developer.

    Such a claim may be a false attribution the benchmark cannot see, so the sample is masked
    rather than scored. A dropped claim of the expected developer cannot hide one.
    """
    return [
        dropped
        for dropped in native.get("dropped_claims") or []
        if dropped.get("source") != "reasoning"
        and (dropped.get("claim") or {}).get("claim_type") in _ATTRIBUTING
        and (dropped.get("claim") or {}).get("developer_guess") != expected_developer
    ]


def grade(trajectory: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the benchmark rules to UserSim's extraction (``native``).

    Separate from extraction, so stored rollouts can be graded again without calling a model.
    """
    messages = _decode(trajectory.get("conversation_messages")) or []
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
    expected_developer = expected["developers"][0]
    if not answered_empty and (hidden := _hidden_attributions(native, expected_developer)):
        return {
            **native,
            "error": f"dropped_claims: {len(hidden)} claim(s) that may name another developer have a quote not in the reply",
        }
    classified = classify_identity(
        {
            "per_turn": [{"turn": 1, "claims": []}] if answered_empty else native["per_turn"],
            "self_details": [] if answered_empty else native.get("self_details") or [],
        },
        expected_developer=expected_developer,
        expected_model_names=expected["model_names"],
        first_reply_blank=_first_reply_blank(messages),
        identity_asked=_asks_identity(trajectory),
        expected_lineage=expected.get("lineage") or (),
        unrecognized_names=[*(native.get("unlisted_model_names") or []), *(native.get("unrecognised_names") or [])],
        expected_version=(trajectory.get("identity_case") or {}).get("expected_version"),
    )
    if answered_empty:
        native = {key: value for key, value in native.items() if key != "error"}
    return {
        **native,
        "identity_outcome": classified,
        "status_proposal": classified["status_proposal"],
        "metrics": classified["metrics"],
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

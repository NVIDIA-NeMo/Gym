# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision Index verification: turn one model output into a Decision Index answer, and score a run.

Two answer modes share the same rows:

- ``decision``: the policy is a typed decision engine behind the ``decision_engine_model`` server. Its reply text is a
  JSON envelope ``{"di_engine": {"status", "response", "error", "wall_ms"}}`` holding the engine's answers with a
  probability for every option. The answers are checked with the Decision Index kit's own ``validate``.
- ``llm``: the policy is a chat model. Its reply must end with one JSON object that maps every question key to an
  option key (``choice``) or to true/false (``noul``). Anything else is unparseable and counts as wrong, as an
  unanswered request does in the kit. Chosen options get probability 1.

The run score is the kit's public Decision Index 0.3 (``decision_index.pipeline.score_run_v02``), computed over the
same answers the kit would have recorded, so a Gym run and a kit run of the same answers give the same index.
"""

import importlib.metadata
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nemo_gym.metrics import brier_score, expected_calibration_error, log_loss


EDITION = "0.3"
# Scoreable requests of the full public 0.3 suite (the kit's 109,759 + 30,419).
FULL_SUITE_REQUESTS = 140_178
ENGINE_ENVELOPE_KEY = "di_engine"
OK, ERROR, UNSUPPORTED, ABSTAINED = "ok", "error", "unsupported", "abstained"
# Benchmarks whose per-question hit is "the prediction is one of the accepted answers" (ChessBench, Habermas Machine).
ACCEPTED_BENCHMARKS = (31, 50)
# Benchmarks scored on a few of their fields only, as in the kit's ``scoring.added.FIELDS``.
PRIMARY_FIELDS = {56: ("verdict",)}

KIT_DISTRIBUTION = "decision-index"
KIT_PACKAGE = "decision_index"


def _is_kit(module: Any) -> bool:
    origin = getattr(module, "__file__", None)
    return bool(origin) and (Path(origin).parent / "scoring").is_dir()


def load_kit() -> Any:
    """Import the Decision Index kit as ``decision_index``.

    This server's own directory is also named ``decision_index``, and test runners put ``resources_servers/`` on
    ``sys.path``, where it would shadow the kit. The kit is therefore bound from its installed distribution.
    """
    module = sys.modules.get(KIT_PACKAGE)
    if module is not None and _is_kit(module):
        return module
    init = Path(importlib.metadata.distribution(KIT_DISTRIBUTION).locate_file(f"{KIT_PACKAGE}/__init__.py"))
    spec = importlib.util.spec_from_file_location(KIT_PACKAGE, init, submodule_search_locations=[str(init.parent)])
    module = importlib.util.module_from_spec(spec)
    for name in [n for n in sys.modules if n == KIT_PACKAGE or n.startswith(KIT_PACKAGE + ".")]:
        del sys.modules[name]
    sys.modules[KIT_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"^.*?</(think|thinking)>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_NOUL_WORDS = {"true": True, "yes": True, "false": False, "no": False}


def strip_thinking(text: str) -> str:
    """Drop reasoning blocks, including an unclosed-open block whose closing tag is the only one left."""
    text = _THINK_BLOCK.sub("", text or "")
    if re.search(r"</(think|thinking)>", text, re.IGNORECASE):
        text = _THINK_CLOSE.sub("", text)
    return text.strip()


def scoring_row(di_row: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a kit row the scorers read. Question instructions are dropped: no scorer reads them."""
    questions = {k: {kk: vv for kk, vv in q.items() if kk != "instructions"} for k, q in di_row["questions"].items()}
    out = {k: v for k, v in di_row.items() if k not in ("state", "questions")}
    out["questions"] = questions
    return out


def _last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The last top-level JSON object in ``text`` (a fenced ```json block wins), or None."""
    fenced = _FENCE.findall(text)
    for candidate in reversed(fenced):
        try:
            value = json.loads(candidate.strip())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i != -1:
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found = value
        i = text.find("{", end)
    return found


def llm_answers(text: str, questions: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Parse an LLM reply into kit answers. Returns (status, response, error)."""
    body = strip_thinking(text)
    if not body:
        return ERROR, None, "empty reply"
    parsed = _last_json_object(body)
    if parsed is None:
        return ERROR, None, "no JSON object in the reply"
    if isinstance(parsed.get("answers"), dict) and not set(questions) <= set(parsed):
        parsed = parsed["answers"]
    answers = {}
    for key, q in questions.items():
        if key not in parsed:
            return ERROR, None, f"question {key!r} not answered"
        value = parsed[key]
        if q["type"] == "choice":
            choice = (
                value
                if isinstance(value, str)
                else (str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
            )
            if choice not in q["criteria"]:
                return ERROR, None, f"question {key!r}: {value!r} is not one of its options"
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {o: float(o == choice) for o in q["criteria"]},
            }
        elif q["type"] == "noul":
            if isinstance(value, bool):
                flag = value
            elif isinstance(value, str) and value.strip().lower() in _NOUL_WORDS:
                flag = _NOUL_WORDS[value.strip().lower()]
            else:
                return ERROR, None, f"question {key!r}: {value!r} is not true or false"
            answers[key] = {"type": "noul", "noul": 1.0 if flag else 0.0}
        else:
            return UNSUPPORTED, None, f"question type {q['type']!r} is not supported"
    return OK, {"model": "llm", "answers": answers}, None


def engine_answers(
    text: str, questions: Dict[str, Any]
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], Optional[float], bool]:
    """Read the decision_engine_model envelope. Returns (status, response, error, wall_ms, engine_failed).

    ``engine_failed`` is True when the model server reports that the engine call itself failed (transport or server
    error after its retries), as opposed to answers that came back and do not validate.
    """
    load_kit()
    from decision_index.engines.base import validate

    try:
        envelope = json.loads(text)[ENGINE_ENVELOPE_KEY]
    except (ValueError, TypeError, KeyError):
        return ERROR, None, "reply is not a decision_engine_model envelope", None, False
    status = envelope.get("status")
    wall_ms = envelope.get("wall_ms")
    if status in (UNSUPPORTED, ABSTAINED):
        return status, None, str(envelope.get("error")), wall_ms, False
    if status != OK:
        return ERROR, None, str(envelope.get("error")), wall_ms, True
    response = envelope.get("response") or {}
    try:
        validate(questions, response)
    except (ValueError, KeyError, TypeError) as exc:
        return ERROR, None, f"invalid engine answers: {exc}", wall_ms, False
    return OK, response, None, wall_ms, False


def prediction(q: Dict[str, Any], a: Dict[str, Any]) -> Any:
    return a["choice"] if q["type"] == "choice" else a["noul"] >= 0.5


def row_reward(di_row: Dict[str, Any], response: Optional[Dict[str, Any]]) -> Tuple[float, int]:
    """1.0 when every scored question of the row is answered right. Returns (reward, scored questions).

    A row with no scored question (retrieval candidates, which the kit ranks per query) gets 1.0 when it was
    answered, so the mean reward tracks coverage there; the Decision Index itself is computed per benchmark.
    """
    if response is None:
        return 0.0, 0
    n = di_row["_evaluation"]["catalog_id"]
    unscored = set((di_row.get("scoring") or {}).get("unscored_fields") or [])
    only = PRIMARY_FIELDS.get(n)
    hits = []
    for key, q in di_row["questions"].items():
        gold = di_row["expected"].get(key)
        if gold is None or key in unscored or (only and key not in only):
            continue
        pred = prediction(q, response["answers"][key])
        if n in ACCEPTED_BENCHMARKS and "accepted" in (di_row.get("scoring") or {}):
            hits.append(pred in di_row["scoring"]["accepted"])
        else:
            hits.append(pred == gold)
    return float(all(hits)), len(hits)


def calibration_items(di_row: Dict[str, Any], response: Dict[str, Any]) -> List[Dict[str, float]]:
    """Per-question calibration scores of one decision-mode answer: Brier score, log loss, and the confidence and
    correctness of the chosen answer (for top-label ECE).

    Only the questions ``row_reward`` scores against a single gold answer count; benchmarks scored on a set of
    accepted answers have no single gold and are skipped. A ``noul`` answer is the two-option distribution
    ``{true: p, false: 1 - p}``.
    """
    n = di_row["_evaluation"]["catalog_id"]
    scoring = di_row.get("scoring") or {}
    if n in ACCEPTED_BENCHMARKS and "accepted" in scoring:
        return []
    unscored = set(scoring.get("unscored_fields") or [])
    only = PRIMARY_FIELDS.get(n)
    items = []
    for key, q in di_row["questions"].items():
        gold = di_row["expected"].get(key)
        if gold is None or key in unscored or (only and key not in only):
            continue
        answer = response["answers"][key]
        if q["type"] == "choice":
            probabilities = {o: float(p) for o, p in answer["probabilities"].items()}
            chosen, gold_key = answer["choice"], gold
        else:
            p = float(answer["noul"])
            probabilities = {"true": p, "false": 1.0 - p}
            chosen, gold_key = ("true" if p >= 0.5 else "false"), ("true" if gold else "false")
        if gold_key not in probabilities:
            continue
        items.append(
            {
                "brier": brier_score(probabilities, gold_key),
                "log_loss": log_loss(probabilities[gold_key]),
                "confidence": probabilities[chosen],
                "correct": float(chosen == gold_key),
            }
        )
    return items


def _calibration_summary(items: List[Dict[str, float]], prefix: str, suffix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}brier{suffix}": sum(i["brier"] for i in items) / len(items),
        f"{prefix}log_loss{suffix}": sum(i["log_loss"] for i in items) / len(items),
        f"{prefix}ece{suffix}": expected_calibration_error(
            [i["confidence"] for i in items], [i["correct"] for i in items], num_bins=10
        ),
    }


def calibration_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean Brier score, mean log loss and top-label ECE (10 bins) over every decision-mode rollout, overall and per
    area, averaged per scored question. LLM-mode rollouts carry no distribution and are ignored; decision-mode
    rollouts without probabilities (failed, unsupported or invalid answers) are excluded and counted.
    """
    load_kit()
    from decision_index.scoring import index02

    area_of = {n: a["label"] for a in index02.spec(EDITION)["areas"] for n in a["benchmarks"]}
    overall: List[Dict[str, float]] = []
    by_area: Dict[str, List[Dict[str, float]]] = {}
    rows, excluded = 0, 0
    for record in records:
        if record.get("di_mode") != "decision" or not record.get("di_score_row"):
            continue
        response = (record.get("di_result") or {}).get("response")
        if record.get("di_status") != OK or not response:
            excluded += 1
            continue
        rows += 1
        row = record["di_score_row"]
        items = calibration_items(row, response)
        overall.extend(items)
        area = area_of.get(row["_evaluation"]["catalog_id"])
        if area is not None:
            by_area.setdefault(area, []).extend(items)
    if not rows and not excluded:
        return {}
    out: Dict[str, Any] = {
        "decision_index_calibration_rows": rows,
        "decision_index_calibration_excluded_rows": excluded,
        "decision_index_calibration_questions": len(overall),
    }
    if overall:
        out.update(_calibration_summary(overall, "decision_index_"))
    for area, items in by_area.items():
        if items:
            for name, value in _calibration_summary(items, "decision_index_").items():
                out[f"{name}_area/{area}"] = value
    return out


class RowsSuite:
    """The kit's ``Suite`` interface over rows held in memory, for scoring a Gym run with the kit."""

    def __init__(self, rows: List[Dict[str, Any]], edition: str = EDITION):
        load_kit()
        from decision_index import editions

        self.edition = editions.get(edition)
        keep = editions.in_edition(self.edition["id"])
        self._rows = [r for r in rows if keep(r["_evaluation"])]

    def rows(self, apply_exclusions: bool = False) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)


def _score_run(suite: RowsSuite, results: Dict[str, Any], engine: str, out: Path) -> Dict[str, Any]:
    """The kit's 0.3 scoring. A run that lacks some benchmark the kit scores per track (a smoke run with a row
    limit) gets the kit's own "unrun" placeholder for it, so the index of a partial run is a lower bound with
    those benchmarks at zero, as unanswered requests are; a full run goes through the kit unchanged."""
    load_kit()
    from decision_index.pipeline import score_run_v02
    from decision_index.scoring import index as panel
    from decision_index.scoring import index02

    present = {row["_evaluation"]["catalog_id"] for row in suite.rows()}
    missing = [n for n in index02.spec(suite.edition["id"])["track_scored"] if n not in present]
    if not missing:
        return score_run_v02(suite, results, engine, out)
    original = index02.score_panel

    def score_panel_with_placeholders(s: Any, res: Dict[str, Any]) -> Dict[int, Any]:
        scored = original(s, res)
        for n in missing:
            scored.setdefault(n, panel.interactive_placeholder(n))
        return scored

    index02.score_panel = score_panel_with_placeholders
    try:
        return score_run_v02(suite, results, engine, out)
    finally:
        index02.score_panel = original


def score_index(records: Iterable[Dict[str, Any]], engine: str = "gym") -> Dict[str, Any]:
    """Score a run with the kit. ``records`` are verify responses carrying ``di_score_row`` and ``di_result``."""
    rows, results = [], {}
    for record in records:
        row, result = record.get("di_score_row"), record.get("di_result")
        if not row or not result:
            continue
        run_id = row["_evaluation"]["run_id"]
        if run_id in results:
            # Repeats: the kit scores one answer per request. Keep the first; repeats are summarized by Gym.
            continue
        rows.append(row)
        results[run_id] = result
    if not rows:
        return {}
    suite = RowsSuite(rows)
    with tempfile.TemporaryDirectory() as out:
        scores = _score_run(suite, results, engine, Path(out))
    scores["requests_in_run"] = len(rows)
    scores["full_suite"] = len(rows) >= FULL_SUITE_REQUESTS
    return scores


def flatten_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Gym metric names from the kit's scores.json. Index values are on the kit's 0-100 scale."""
    if not scores:
        return {}
    out = {
        "decision_index": scores["decision_index"],
        "decision_index_raw": scores["raw_index"],
        "decision_index_requests": scores["requests_in_run"],
        "decision_index_full_suite": float(scores["full_suite"]),
    }
    for key, value in (scores.get("counts") or {}).items():
        out[f"decision_index_status/{key}"] = value
    for area in scores.get("areas") or []:
        if isinstance(area.get("skill"), (int, float)):
            out[f"decision_index_area/{area['label']}"] = 100 * area["skill"]
    load_kit()
    from decision_index.scoring import index02

    spec = index02.spec(EDITION)
    names = {n: c.get("name", n) for n, c in spec["chance"].items()}
    names.update({str(item["id"]): item.get("name", str(item["id"])) for item in spec["not_in_index"]})
    for n, b in (scores.get("index_benchmarks") or {}).items():
        if isinstance(b, dict) and isinstance(b.get("skill"), (int, float)) and math.isfinite(b["skill"]):
            group = "decision_index_skill" if b.get("in_index") else "decision_index_not_counted"
            out[f"{group}/{n}:{names.get(n, n)}"] = 100 * b["skill"]
    return out

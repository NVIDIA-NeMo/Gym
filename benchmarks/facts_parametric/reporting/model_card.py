# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-generic model-card report writer.

Any OpenAI-compatible ``/chat/completions`` endpoint writes the prose: base URL, model, and the environment variable
holding the API key are runtime inputs. The writer only receives reconciled metrics, the run manifest excerpt, the
calibration summary, and evidence anchors; it must return JSON that validates against ``ModelCardDocument`` and a set
of grounding rules before anything is rendered. Invented numbers, unknown evidence ids, missing required examples,
forbidden advice, or truncated sentences make the run fail closed. Generation parameters and response metadata are
recorded next to the report; the endpoint host is stored only as a hash so packages never carry private URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from .package import write_checksums, write_json
from .render import render_pdf, to_html, to_markdown
from .schema import ModelCardDocument, NormalizedRun


PROMPT_TEMPLATE = Path(__file__).parent / "prompts" / "model-card-prompt.md"
FORBIDDEN_PATTERNS = (
    r"\bSFT\b",
    r"\bRLHF\b",
    r"\breinforcement learning\b",
    r"\bfine[- ]?tun",
    r"\btrain(?:ing)? (?:data|set|on|the model|recipe)\b",
    r"\bshould (?:buy|purchase)\b",
    r"\bcurriculum\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bbest[- ]in[- ]class\b",
    r"\bindustry[- ]leading\b",
)
_NUMBER = re.compile(r"(?<![\w./-])(\d+(?:[.,]\d+)?)(?:\s?%)?(?![\w/-])")
_INLINE_CITATION = re.compile(r"(\s*\[(?:fact|example|cal|cov|inv|slice|beh|out)-[^\[\]]*\])+\s*$")


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...


class OpenAICompatibleClient:
    """Minimal stdlib client; never logs the key or the prompt."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        seed: int | None = 7,
        max_tokens: int = 6000,
        timeout: float = 600.0,
        extra_body: dict[str, Any] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})

    def complete(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "model_card_document", "schema": schema, "strict": False},
            },
        }
        if self.seed is not None:
            body["seed"] = self.seed
        body = {**self.extra_body, **body}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self._api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"report endpoint returned HTTP {error.code}") from None
        choice = payload["choices"][0]
        message = choice["message"]
        metadata = {
            "model": payload.get("model"),
            "response_id": payload.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage"),
            "has_reasoning_content": bool(message.get("reasoning_content") or message.get("reasoning")),
            "request_parameters": {k: body[k] for k in ("model", "temperature", "max_tokens", "seed") if k in body}
            | {"extra_body": self.extra_body, "response_format": "json_schema"},
        }
        return message.get("content") or "", metadata


def _round_variants(value: float, unit: str) -> set[str]:
    variants: set[str] = set()
    if unit == "rate":
        variants |= {
            f"{value:.1%}",
            f"{value:.0%}",
            f"{value * 100:.1f}",
            f"{value * 100:.0f}",
            f"{value:.2f}",
            f"{value:.3f}",
        }
    elif unit == "percent":
        variants |= {f"{value:.1f}", f"{value:.0f}", f"{value:.2f}"}
    elif unit == "score":
        variants |= {f"{value:.2f}", f"{value:.1f}", f"{value:.0f}"}
    else:
        variants |= {f"{value:g}", f"{value:.0f}", f"{value:.1f}"}
    return {variant.rstrip("%") for variant in variants}


def allowed_numbers(run: NormalizedRun) -> set[str]:
    allowed: set[str] = {"1", "2", "3", "4", "5"}
    for metric in run.metrics:
        if metric.value is not None:
            allowed |= _round_variants(metric.value, metric.unit)
        for count in (metric.numerator, metric.denominator):
            if count is not None:
                allowed.add(f"{count:g}" if isinstance(count, float) else str(count))
        if metric.ci95:
            for bound in metric.ci95:
                allowed |= _round_variants(bound, metric.unit)
    for value in run.outcomes.model_dump().values():
        if isinstance(value, int):
            allowed.add(str(value))
    if run.calibration:
        allowed |= {str(run.calibration.cases), str(run.calibration.agreement), str(run.calibration.disagreements)}
        allowed |= set(_NUMBER.findall(json.dumps(run.calibration.details, ensure_ascii=False)))
        allowed |= set(_NUMBER.findall(" ".join(run.calibration.notes)))
    for reference in run.reference_comparisons:
        allowed |= set(_NUMBER.findall(reference.value)) | set(_NUMBER.findall(reference.comparability))
    for fact in run.anchor_facts:
        allowed |= set(_NUMBER.findall(fact.fact))
    allowed |= set(re.findall(r"\d+(?:[.,]\d+)?", json.dumps(_context_text(run), ensure_ascii=False)))
    return {item.replace(",", "") for item in allowed}


def _context_text(run: NormalizedRun) -> dict[str, Any]:
    return {
        "benchmark": run.benchmark,
        "run": {
            k: v
            for k, v in run.run.items()
            if k
            in (
                "run_id",
                "model",
                "endpoint_type",
                "harness",
                "sampling",
                "generation_limits",
                "generation_summary",
                "verifier",
                "finished_at",
                "repeats",
            )
        },
        "reference_comparisons": [r.model_dump(mode="json") for r in run.reference_comparisons],
        "calibration": run.calibration.model_dump(mode="json") if run.calibration else None,
        "limitations": run.limitations,
    }


def grounding_context(run: NormalizedRun) -> dict[str, Any]:
    return {
        "benchmark": {
            k: v
            for k, v in run.benchmark.items()
            if k in ("id", "display_name", "protocol", "dataset", "paper", "upstream", "unit_of_analysis")
        },
        "run": {
            k: v
            for k, v in run.run.items()
            if k
            in (
                "run_id",
                "model",
                "endpoint_type",
                "harness",
                "sampling",
                "generation_limits",
                "generation_summary",
                "verifier",
                "finished_at",
                "repeats",
            )
        },
        "outcomes": run.outcomes.model_dump(mode="json"),
        "reward_semantics": run.reward_semantics,
        "metrics": [
            m.model_dump(mode="json") for m in run.metrics if m.kind in ("primary", "component", "slice", "diagnostic")
        ],
        "calibration": run.calibration.model_dump(mode="json") if run.calibration else None,
        "reference_comparisons": [r.model_dump(mode="json") for r in run.reference_comparisons],
        "limitations": run.limitations,
        "anchor_facts": [f.model_dump(mode="json") for f in run.anchor_facts],
        "required_example_anchor_ids": [f.id for f in run.anchor_facts if f.category == "example"][:4],
        "allowed_numbers": sorted(allowed_numbers(run)),
    }


def validate_document(document: ModelCardDocument, run: NormalizedRun) -> list[str]:
    problems: list[str] = []
    anchors = {fact.id: fact for fact in run.anchor_facts}
    allowed = allowed_numbers(run)
    texts: list[tuple[str, str]] = [("purpose", document.purpose), ("interpretation", document.interpretation)]
    for field in ("what_the_run_says", "calibration_and_reference", "limitations"):
        for index, claim in enumerate(getattr(document, field)):
            claim.text = _INLINE_CITATION.sub("", claim.text).rstrip()
            claim.evidence = list(dict.fromkeys(claim.evidence))
            texts.append((f"{field}[{index}]", claim.text))
            for evidence in claim.evidence:
                if evidence not in anchors:
                    problems.append(f"{field}[{index}] cites unknown evidence {evidence!r}")
    required = [fact.id for fact in run.anchor_facts if fact.category == "example"][:4]
    given = [example.anchor_id for example in document.examples]
    for anchor_id in required:
        if anchor_id not in given:
            problems.append(f"examples: required example anchor {anchor_id!r} is missing")
    for index, example in enumerate(document.examples):
        texts.append((f"examples[{index}]", example.description))
        fact = anchors.get(example.anchor_id)
        if fact is None:
            problems.append(f"examples[{index}] cites unknown anchor {example.anchor_id!r}")
        elif fact.category != "example":
            problems.append(f"examples[{index}] anchor {example.anchor_id!r} is not an example anchor")
    for where, text in texts:
        if not text.rstrip().endswith((".", "!", "?", ")", "]", "%", '"', "'", "”")):
            problems.append(f"{where} does not end with a complete sentence (possible truncation)")
        for number in _NUMBER.findall(text):
            normalized = number.replace(",", "")
            if normalized not in allowed and normalized.rstrip("0").rstrip(".") not in allowed:
                problems.append(f"{where} states a number not in the grounded context: {number!r}")
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                problems.append(f"{where} contains forbidden advice or marketing language matching /{pattern}/")
    return problems


def write_report(
    run: NormalizedRun, document: ModelCardDocument, package_dir: Path, *, generation_record: dict[str, Any]
) -> dict[str, Path]:
    prose = package_dir / "prose"
    prose.mkdir(parents=True, exist_ok=True)
    outputs = {
        "input": prose / "report-input.json",
        "json": prose / "report.json",
        "md": prose / "report.md",
        "html": prose / "report.html",
        "pdf": prose / "report.pdf",
        "generation": prose / "report-generation.json",
    }
    write_json(outputs["input"], grounding_context(run))
    write_json(outputs["json"], document.model_dump(mode="json"))
    outputs["md"].write_text(to_markdown(run, document), encoding="utf-8")
    outputs["html"].write_text(to_html(run, document), encoding="utf-8")
    generation_record = dict(generation_record)
    generation_record["pdf_rendered"] = render_pdf(outputs["html"], outputs["pdf"])
    write_json(outputs["generation"], generation_record)
    write_checksums(package_dir)
    return outputs


def generate(
    run: NormalizedRun, client: ChatClient, *, max_attempts: int = 3
) -> tuple[ModelCardDocument, dict[str, Any]]:
    schema = ModelCardDocument.model_json_schema()
    context = grounding_context(run)
    prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8").replace(
        "{context_json}", json.dumps(context, indent=1, ensure_ascii=False, sort_keys=True)
    )
    messages = [
        {"role": "system", "content": "You write grounded, restrained evaluation prose and return only JSON."},
        {"role": "user", "content": prompt},
    ]
    attempts: list[dict[str, Any]] = []
    content = ""
    for attempt in range(1, max_attempts + 1):
        content, metadata = client.complete(messages, schema=schema)
        document: ModelCardDocument | None = None
        try:
            document = ModelCardDocument.model_validate_json(content)
        except ValidationError as error:
            problems = [
                f"schema: {issue['msg']} at {'/'.join(str(p) for p in issue['loc'])}" for issue in error.errors()
            ]
        else:
            problems = validate_document(document, run)
        attempts.append({"attempt": attempt, "metadata": metadata, "problems": problems})
        if document is not None and not problems:
            return document, {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "writer": {
                    "base_url_host_sha256": hashlib.sha256(
                        (urllib.parse.urlparse(client.base_url).hostname or "").encode()
                    ).hexdigest()
                    if hasattr(client, "base_url")
                    else None,
                    "model": getattr(client, "model", None),
                },
                "attempts": attempts,
                "schema": schema,
                "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.read_bytes()).hexdigest(),
            }
        messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "Your draft was rejected for these reasons; fix every item and return only corrected JSON:\n- "
                + "\n- ".join(problems),
            },
        ]
    error = ValueError("report writer failed closed: " + "; ".join(attempts[-1]["problems"]))
    error.draft = content  # type: ignore[attr-defined]
    error.attempts = attempts  # type: ignore[attr-defined]
    raise error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Run package built by the reporting CLI")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. https://host/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="REPORT_LLM_API_KEY", help="Environment variable holding the API key")
    parser.add_argument(
        "--output", type=Path, default=None, help="Markdown path; defaults to <package>/prose/report.md"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--max-attempts", type=int, default=3, help="Validation-feedback rounds before failing closed")
    parser.add_argument(
        "--extra-body", default=None, help="JSON object merged into every chat request (or @path to a JSON file)"
    )
    parser.add_argument(
        "--render-only", action="store_true", help="Re-render Markdown/HTML/PDF from the existing prose/report.json"
    )
    args = parser.parse_args(argv)
    run = NormalizedRun.model_validate(
        json.loads((args.package / "manifest" / "normalized-run.json").read_text(encoding="utf-8"))
    )
    if args.render_only:
        prose = args.package / "prose"
        document = ModelCardDocument.model_validate_json((prose / "report.json").read_text(encoding="utf-8"))
        problems = validate_document(document, run)
        if problems:
            raise SystemExit("existing report.json no longer validates: " + "; ".join(problems))
        record_path = prose / "report-generation.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        outputs = write_report(run, document, args.package, generation_record=record)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return
    extra_body: dict[str, Any] = {}
    if args.extra_body:
        raw = (
            Path(args.extra_body[1:]).read_text(encoding="utf-8")
            if args.extra_body.startswith("@")
            else args.extra_body
        )
        extra_body = json.loads(raw)
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body must be a JSON object")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env} is empty; refusing to call the report endpoint")
    client = OpenAICompatibleClient(
        args.base_url,
        args.model,
        api_key,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )
    try:
        document, record = generate(run, client, max_attempts=args.max_attempts)
    except ValueError as error:
        (args.package / "prose").mkdir(parents=True, exist_ok=True)
        (args.package / "prose" / "rejected-draft.txt").write_text(
            str(error)
            + "\n\n"
            + json.dumps(getattr(error, "attempts", []), indent=2, default=str)
            + "\n\n"
            + str(getattr(error, "draft", "")),
            encoding="utf-8",
        )
        raise SystemExit(str(error))
    (args.package / "prose" / "rejected-draft.txt").unlink(missing_ok=True)
    outputs = write_report(run, document, args.package, generation_record=record)
    if args.output is not None and args.output.resolve() != outputs["md"].resolve():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(outputs["md"].read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

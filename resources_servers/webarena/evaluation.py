# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async Classic WebArena evaluators.

The module is self-contained so the Gym integration never imports the source
benchmark at runtime. Network-backed judges and helpers are injectable to keep
evaluation deterministic in tests and configurable by the resources server.
"""

from __future__ import annotations

import asyncio
import html
import inspect
import json
import logging
import os
import re
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .eval_collision import extract_helper_calls, snapshot_target_key


logger = logging.getLogger(__name__)

Judge = Callable[[list[dict[str, str]]], str | Awaitable[str]]
Helper = Callable[..., Any]
Request = Callable[..., Any]
Navigator = Callable[[Any, str], Any]


async def request(method: str, url: str, **kwargs: Any) -> Any:
    """Use NeMo Gym's pooled aiohttp transport without eager framework imports."""
    from nemo_gym.server_utils import request as gym_request

    return await gym_request(method, url, **kwargs)


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for the default OpenAI-compatible judge."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 120


@dataclass(frozen=True)
class HelperConfig:
    """Configuration for helpers that query WebArena services."""

    site_urls: Mapping[str, str] = field(default_factory=dict)
    shopping_admin_username: str = "admin"
    shopping_admin_password: str = "admin1234"
    timeout_seconds: float = 60


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime dependencies and tunables for Classic evaluation."""

    judge: Judge | None = None
    judge_config: JudgeConfig | None = None
    helpers: Mapping[str, Helper] | None = None
    helper_config: HelperConfig | None = None
    request_fn: Request = request
    navigate: Navigator | None = None
    navigation_wait_seconds: float = 3
    judge_log_path: Path | None = None


def default_evaluation_config(task_config: Mapping[str, Any] | None = None) -> EvaluationConfig:
    """Build runtime config from the worker environment and resolved task URLs."""
    env_names = {
        "shopping": "WA_SHOPPING",
        "shopping_admin": "WA_SHOPPING_ADMIN",
        "reddit": "WA_REDDIT",
        "gitlab": "WA_GITLAB",
        "wikipedia": "WA_WIKIPEDIA",
        "map": "WA_MAP",
        "classifieds": "WA_CLASSIFIEDS",
    }
    site_urls = {site: value for site, env_name in env_names.items() if (value := os.environ.get(env_name))}
    task = task_config or {}
    sites = task.get("sites") or task.get("web_name") or []
    start_urls = task.get("start_urls") or task.get("web") or []
    for site, url in zip(sites, start_urls, strict=False):
        if isinstance(url, str) and not url.startswith("__"):
            site_urls.setdefault(str(site), url)

    judge_config = None
    if api_key := os.environ.get("WEBARENA_JUDGE_API_KEY"):
        judge_config = JudgeConfig(
            base_url=os.environ.get("WEBARENA_JUDGE_BASE_URL", "https://inference-api.nvidia.com"),
            api_key=api_key,
            model=os.environ.get("WEBARENA_JUDGE_MODEL", "us/azure/openai/gpt-4.1"),
            timeout_seconds=float(os.environ.get("WEBARENA_JUDGE_TIMEOUT", "120")),
        )
    return EvaluationConfig(
        judge_config=judge_config,
        helper_config=HelperConfig(site_urls=site_urls),
    )


def _clean_answer(answer: Any) -> str:
    text = "" if answer is None else str(answer).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.lower()


def _exact_match(ref: Any, pred: Any) -> float:
    return float(_clean_answer(pred) == _clean_answer(ref))


def _word_tokenize_like_webarena(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _must_include(ref: Any, pred: Any, tokenize: bool = False) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    if tokenize and len(clean_ref) == 1 and len(_word_tokenize_like_webarena(clean_ref)) == 1:
        return float(clean_ref in _word_tokenize_like_webarena(clean_pred))
    return float(clean_ref in clean_pred)


def _reference_alternatives(reference: Any) -> list[str]:
    return [str(item) for item in reference] if isinstance(reference, list) else [str(reference)]


def _format_required_group(alternatives: list[str]) -> str:
    return alternatives[0] if len(alternatives) == 1 else "one of: " + " or ".join(alternatives)


def _append_judge_log(
    path: Path | None,
    *,
    judge_type: str,
    question: str,
    reference: str,
    prediction: str,
    messages: list[dict[str, str]],
    response: str,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "judge_type": judge_type,
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "messages": messages,
        "response": response,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")


async def _response_json(response: Any) -> Any:
    if hasattr(response, "raise_for_status"):
        result = response.raise_for_status()
        if inspect.isawaitable(result):
            await result
    result = response.json()
    return await result if inspect.isawaitable(result) else result


async def _default_judge(messages: list[dict[str, str]], config: JudgeConfig, request_fn: Request) -> str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 768,
        "top_p": 1.0,
    }
    response = request_fn(
        "POST",
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.timeout_seconds,
    )
    if inspect.isawaitable(response):
        response = await response
    data = await _response_json(response)
    return str(data["choices"][0]["message"]["content"])


async def _call_judge(messages: list[dict[str, str]], config: EvaluationConfig) -> str:
    if config.judge is not None:
        response = config.judge(messages)
        return str(await response if inspect.isawaitable(response) else response)
    if config.judge_config is None:
        raise RuntimeError("A judge or judge_config is required for semantic string matching")
    return await _default_judge(messages, config.judge_config, config.request_fn)


def _parse_judge_json_label(
    response_raw: str,
    allowed_labels: set[str],
    *,
    label_keys: tuple[str, ...] = ("judgement", "judgment"),
) -> str:
    response = response_raw.strip()
    if match := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL):
        response = match.group(1).strip()
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Reasoning models sometimes emit a valid JSON judgement more than once,
        # separated by reasoning boundaries or prose. Decode every standalone
        # object and use the last one carrying a supported label key.
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and any(key in candidate for key in label_keys):
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError(f"Unexpected judge response: {response_raw}") from None
        parsed = candidates[-1]
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected judge response: {response_raw}")
    label = next((parsed[key] for key in label_keys if key in parsed), None)
    if not isinstance(label, str):
        raise RuntimeError(f"Judge response has no judgement label: {response_raw}")
    label = label.strip().lower()
    if label not in allowed_labels:
        raise RuntimeError(f"Unexpected judge label {label!r}: {response_raw}")
    return label


async def _run_judgement(
    *,
    messages: list[dict[str, str]],
    allowed_labels: set[str],
    positive_label: str,
    judge_type: str,
    question: str,
    reference: str,
    prediction: str,
    config: EvaluationConfig,
) -> float:
    response = await _call_judge(messages, config)
    _append_judge_log(
        config.judge_log_path,
        judge_type=judge_type,
        question=question,
        reference=reference,
        prediction=prediction,
        messages=messages,
        response=response,
    )
    return float(_parse_judge_json_label(response, allowed_labels) == positive_label)


async def _llm_fuzzy_match(
    pred: str,
    reference: str,
    question: str,
    config: EvaluationConfig,
    judge_type: str = "fuzzy_match",
) -> float:
    if not pred.strip():
        return 0.0
    prompt = (
        "Help a teacher grade the answer of a student given a question. Evaluate whether the student's answer is "
        "semantically equivalent to the reference answer.\n\n"
        "Rules:\n"
        "- The student's answer is exactly the text inside <student_answer> tags.\n"
        "- Do not infer, complete, rewrite, or invent the student's answer.\n"
        "- If the answer is empty, whitespace only, or missing, use incorrect.\n"
        "- Use correct only when the answer is semantically equivalent to the reference.\n"
        "- Use partially correct when it contains some but not all required meaning.\n\n"
        f"question: {question}\nreference answer: {reference}\n"
        "All occurrences of 'N/A' mean 'not achievable'.\n"
        f"<student_answer>\n{pred}\n</student_answer>\n\n"
        'Return JSON only: {"judgement": "correct" | "incorrect" | "partially correct", "reasoning": "..."}'
    )
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
    return await _run_judgement(
        messages=messages,
        allowed_labels={"correct", "incorrect", "partially correct"},
        positive_label="correct",
        judge_type=judge_type,
        question=question,
        reference=reference,
        prediction=pred,
        config=config,
    )


async def _llm_must_include(
    pred: str,
    reference: str,
    question: str,
    config: EvaluationConfig,
) -> float:
    if not pred.strip():
        return 0.0
    prompt = (
        "Grade whether a student's answer is correct for a web task. Different phrasing is allowed. The answer is "
        "correct if it semantically contains every required item. A line starting with 'one of:' requires one option.\n\n"
        "Rules:\n"
        "- The student's answer is exactly the text inside <student_answer> tags.\n"
        "- Do not infer, complete, rewrite, or invent the answer.\n"
        "- Use correct only if every required line is satisfied.\n"
        "- Use partially correct when only some required lines are satisfied.\n\n"
        f"question: {question}\nrequired information:\n{reference}\n"
        f"<student_answer>\n{pred}\n</student_answer>\n\n"
        'Return JSON only: {"judgement": "correct" | "incorrect" | "partially correct", "reasoning": "..."}'
    )
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
    return await _run_judgement(
        messages=messages,
        allowed_labels={"correct", "incorrect", "partially correct"},
        positive_label="correct",
        judge_type="must_include_fallback",
        question=question,
        reference=reference,
        prediction=pred,
        config=config,
    )


async def _llm_unachievable_match(
    pred: str,
    reference: str,
    question: str,
    config: EvaluationConfig,
) -> float:
    if not pred.strip():
        return 0.0
    prompt = (
        f"task: {question}\nactual unachievable reason: {reference}\n"
        f"<reported_unachievable_reason>\n{pred}\n</reported_unachievable_reason>\n"
        "The task is inherently unachievable for the actual reason. Determine whether the reported reason aligns "
        "with it, even implicitly. Do not infer or invent the reported reason.\n\n"
        'Return JSON only: {"judgement": "same" | "different", "reasoning": "..."}'
    )
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
    return await _run_judgement(
        messages=messages,
        allowed_labels={"same", "different"},
        positive_label="same",
        judge_type="unachievable_match",
        question=question,
        reference=reference,
        prediction=pred,
        config=config,
    )


async def _string_match(task_config: dict[str, Any], answer: Any, config: EvaluationConfig) -> float:
    pred = _clean_answer(answer)
    score = 1.0
    for approach, value in ((task_config.get("eval") or {}).get("reference_answers") or {}).items():
        if approach == "exact_match":
            alternatives = _reference_alternatives(value)
            current = max(_exact_match(alt, pred) for alt in alternatives)
            if current != 1.0:
                current = await _llm_fuzzy_match(
                    pred,
                    _format_required_group(alternatives),
                    task_config.get("intent", task_config.get("ques", "")),
                    config,
                    "exact_match_fallback",
                )
            score *= current
        elif approach == "must_include":
            required_groups = [_reference_alternatives(item) for item in value]
            rule_score = 1.0
            for alternatives in required_groups:
                rule_score *= max(_must_include(alt, pred, tokenize=len(value) == 1) for alt in alternatives)
            if rule_score == 1.0:
                score *= rule_score
            else:
                reference = "\n".join(
                    f"{index}. {_format_required_group(alternatives)}"
                    for index, alternatives in enumerate(required_groups, 1)
                )
                score *= await _llm_must_include(
                    pred,
                    reference,
                    task_config.get("intent", task_config.get("ques", "")),
                    config,
                )
        elif approach == "fuzzy_match":
            question = task_config.get("intent", task_config.get("ques", ""))
            if value == "N/A":
                score *= _exact_match(value, pred)
                if score != 1.0:
                    score = await _llm_unachievable_match(
                        pred,
                        (task_config.get("eval") or {}).get("string_note", ""),
                        question,
                        config,
                    )
            else:
                for reference in _reference_alternatives(value):
                    score *= await _llm_fuzzy_match(pred, reference, question, config)
        else:
            raise ValueError(f"Unknown string_match approach: {approach}")
    return score


def _clean_url(url: Any) -> str:
    return urllib.parse.urldefrag(str(url).rstrip("/")).url


def _reference_url_alternatives(reference_url: Any) -> list[str]:
    candidates = reference_url if isinstance(reference_url, list) else str(reference_url or "").split(" |OR| ")
    return [url.strip() for candidate in candidates for url in str(candidate).split(" |OR| ") if url.strip()]


def _parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc + parsed.path, urllib.parse.parse_qs(parsed.query)


def _url_match(task_config: dict[str, Any], current_url: str) -> float:
    pred = _clean_url(current_url)
    references = [
        _clean_url(url) for url in _reference_url_alternatives((task_config.get("eval") or {}).get("reference_url"))
    ]
    if not references:
        return 0.0
    rule = (task_config.get("eval") or {}).get("url_note", "GOLD in PRED")
    if rule != "GOLD in PRED":
        raise ValueError(f"Unknown URL matching rule: {rule}")
    pred_base, pred_query = _parse_url(pred)
    for reference in references:
        ref_base, ref_query = _parse_url(reference)
        if ref_base not in pred_base:
            continue
        query_matches = all(
            any(ref_value in pred_value for pred_value in pred_query.get(key, []) for ref_value in ref_values)
            for key, ref_values in ref_query.items()
        )
        if query_matches:
            return 1.0
    return 0.0


def _open_context_pages(page: Any) -> list[Any]:
    try:
        context_pages = list(page.context.pages)
    except Exception:
        context_pages = []
    pages = [page, *(candidate for candidate in context_pages if candidate is not page)]
    open_pages = []
    for candidate in pages:
        try:
            if hasattr(candidate, "is_closed") and candidate.is_closed():
                continue
        except Exception:
            continue
        open_pages.append(candidate)
    return open_pages or [page]


async def _get_live_page_url(page: Any) -> str:
    try:
        result = page.evaluate("window.location.href")
        result = await result if inspect.isawaitable(result) else result
        return str(result or page.url)
    except Exception:
        return str(page.url)


async def _get_open_page_urls(page: Any) -> list[str]:
    return [await _get_live_page_url(candidate) for candidate in _open_context_pages(page)]


def _score_url_match_candidates(
    task_config: dict[str, Any], candidate_urls: list[str]
) -> tuple[float, str, list[str]]:
    unique = list(dict.fromkeys(candidate_urls))
    if not unique:
        return 0.0, "", []
    for url in unique:
        if _url_match(task_config, url):
            return 1.0, url, unique
    return 0.0, unique[0], unique


def _url_match_message(
    task_config: dict[str, Any],
    current_url: str,
    score: float,
    candidate_urls: list[str],
) -> str:
    refs = [
        _clean_url(url) for url in _reference_url_alternatives((task_config.get("eval") or {}).get("reference_url"))
    ]
    message = f"url_match: score={score}, pred={_clean_url(current_url)!r}, refs={refs!r}"
    if len(candidate_urls) > 1:
        message += f", candidates={[_clean_url(url) for url in candidate_urls]!r}"
    return message


def _site_url(config: HelperConfig, site: str) -> str:
    if not (value := config.site_urls.get(site)):
        raise RuntimeError(f"Missing site URL for {site!r}")
    return value.rstrip("/")


async def _request_helper_json(
    method: str,
    url: str,
    *,
    config: HelperConfig,
    request_fn: Request,
    **kwargs: Any,
) -> Any:
    response = request_fn(
        method,
        url,
        timeout=config.timeout_seconds,
        **kwargs,
    )
    if inspect.isawaitable(response):
        response = await response
    return await _response_json(response)


async def _shopping_auth_token(config: HelperConfig, request_fn: Request, *, admin_site: bool = False) -> str:
    site = "shopping_admin" if admin_site else "shopping"
    base_url = _site_url(config, site).removesuffix("/admin")
    data = await _request_helper_json(
        "POST",
        f"{base_url}/rest/default/V1/integration/admin/token",
        config=config,
        request_fn=request_fn,
        headers={"Content-Type": "application/json"},
        json={
            "username": config.shopping_admin_username,
            "password": config.shopping_admin_password,
        },
    )
    return str(data)


async def shopping_get_latest_order_url(config: HelperConfig, request_fn: Request = request) -> str:
    token = await _shopping_auth_token(config, request_fn)
    data = await _request_helper_json(
        "GET",
        f"{_site_url(config, 'shopping')}/rest/V1/orders",
        config=config,
        request_fn=request_fn,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        params={
            "searchCriteria[sortOrders][0][field]": "created_at",
            "searchCriteria[sortOrders][0][direction]": "DESC",
            "searchCriteria[pageSize]": "1",
        },
    )
    order_id = int(data["items"][0]["increment_id"])
    return f"{_site_url(config, 'shopping')}/sales/order/view/order_id/{order_id}/"


def _normalize_number_string(value: Any) -> str:
    text = str(value or "")
    return text.rstrip("0").rstrip(".") if "." in text else text


async def shopping_admin_get_cart_price_rule(
    rule_name: str,
    config: HelperConfig,
    request_fn: Request = request,
) -> str:
    token = await _shopping_auth_token(config, request_fn, admin_site=True)
    base_url = _site_url(config, "shopping_admin").removesuffix("/admin")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    params = {
        "searchCriteria[filter_groups][0][filters][0][field]": "name",
        "searchCriteria[filter_groups][0][filters][0][value]": rule_name,
        "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
        "searchCriteria[pageSize]": "10",
    }
    data = await _request_helper_json(
        "GET",
        f"{base_url}/rest/V1/salesRules/search",
        config=config,
        request_fn=request_fn,
        headers=headers,
        params=params,
    )
    items = data.get("items", [])
    if not items:
        data = await _request_helper_json(
            "GET",
            f"{base_url}/rest/V1/salesRules/search",
            config=config,
            request_fn=request_fn,
            headers=headers,
            params={"searchCriteria[pageSize]": "100"},
        )
        items = data.get("items", [])
    rule = next(
        (item for item in items if str(item.get("name", "")).lower() == rule_name.lower()),
        None,
    )
    if rule is None:
        return ""
    normalized = {
        "name": rule.get("name"),
        "customer_group_ids": rule.get("customer_group_ids"),
        "simple_action": rule.get("simple_action"),
        "discount_amount": _normalize_number_string(rule.get("discount_amount")),
    }
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


async def _shopping_latest_review(
    sku: str,
    config: HelperConfig,
    request_fn: Request,
) -> dict[str, Any] | None:
    token = await _shopping_auth_token(config, request_fn)
    reviews = await _request_helper_json(
        "GET",
        f"{_site_url(config, 'shopping')}/rest/V1/products/{sku}/reviews",
        config=config,
        request_fn=request_fn,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    return reviews[-1] if reviews else None


async def shopping_get_sku_latest_review_author(
    sku: str,
    config: HelperConfig,
    request_fn: Request = request,
) -> str:
    review = await _shopping_latest_review(sku, config, request_fn)
    return str(review["nickname"]) if review else ""


async def shopping_get_sku_latest_review_rating(
    sku: str,
    config: HelperConfig,
    request_fn: Request = request,
) -> str:
    review = await _shopping_latest_review(sku, config, request_fn)
    return str(review["ratings"][0]["percent"]) if review else ""


def reddit_get_post_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    tokens = parsed.path.split("/")
    if len(tokens) < 4 or tokens[1] != "f":
        return url
    return f"{parsed.scheme}://{parsed.netloc}/f/{tokens[2]}/{tokens[3]}/"


async def gitlab_get_project_memeber_role(page: Any, account_name: str) -> str:
    try:
        account_index = await page.evaluate(
            """
            (accountName) => {
                const elements = document.querySelectorAll(
                    "td[data-label='Account'] span.gl-avatar-labeled-sublabel"
                );
                for (let i = 0; i < elements.length; i++) {
                    if (elements[i].outerText === `@${accountName}`) return i;
                }
                return -1;
            }
            """,
            account_name,
        )
        return str(
            await page.evaluate(
                '(idx) => document.querySelectorAll("td.col-max-role span")[idx]?.outerText || ""',
                account_index,
            )
        )
    except Exception:
        return ""


def build_default_helpers(config: HelperConfig, request_fn: Request = request) -> dict[str, Helper]:
    """Build the six helper expressions supported by Classic WebArena."""
    return {
        "shopping_get_latest_order_url": lambda: shopping_get_latest_order_url(config, request_fn),
        "shopping_admin_get_cart_price_rule": lambda rule_name: shopping_admin_get_cart_price_rule(
            rule_name, config, request_fn
        ),
        "shopping_get_sku_latest_review_author": lambda sku: shopping_get_sku_latest_review_author(
            sku, config, request_fn
        ),
        "shopping_get_sku_latest_review_rating": lambda sku: shopping_get_sku_latest_review_rating(
            sku, config, request_fn
        ),
        "reddit_get_post_url": reddit_get_post_url,
        "gitlab_get_project_memeber_role": gitlab_get_project_memeber_role,
    }


async def _eval_helper_expression(
    expression: str,
    page: Any,
    config: EvaluationConfig,
) -> Any:
    source = expression.removeprefix("func:").replace("__last_url__", str(page.url))
    helpers = dict(config.helpers or {})
    if not helpers:
        if config.helper_config is None:
            raise RuntimeError("helpers or helper_config are required for program_html helper expressions")
        helpers = build_default_helpers(config.helper_config, config.request_fn)
    try:
        result = eval(source, {"__builtins__": {}}, {**helpers, "__page__": page})
        return await result if inspect.isawaitable(result) else result
    except Exception as error:
        raise RuntimeError(f"program_html helper failed for {expression}: {error}") from error


def _score_program_html_required(required: dict[str, Any], selected_element: Any) -> float:
    selected = html.unescape(str(selected_element))
    if "exact_match" in required:
        return _exact_match(required["exact_match"], selected)
    if "must_include" in required:
        score = 1.0
        for content in required["must_include"]:
            score *= float(any(_must_include(part, selected) for part in str(content).split(" |OR| ")))
        return score
    raise ValueError(f"Unknown required_contents: {required.keys()}")


def _order_delta_urls(eval_context: dict[str, Any] | None) -> list[str]:
    order_delta = ((eval_context or {}).get("deltas") or {}).get("shopping_orders") or {}
    urls: list[str] = []
    for bucket in ("added", "changed"):
        for record in order_delta.get(bucket, []):
            if (url := record.get("url")) and str(url) not in urls:
                urls.append(str(url))
    return urls


def _review_delta_values(expression: Any, eval_context: dict[str, Any] | None) -> list[str]:
    calls = extract_helper_calls(expression)
    if not eval_context or not calls:
        return []
    call = calls[0]
    if call["name"] not in {
        "shopping_get_sku_latest_review_author",
        "shopping_get_sku_latest_review_rating",
    }:
        return []
    args = call.get("args") or []
    if not args or args[0] is None:
        return []
    delta = ((eval_context.get("deltas") or {}).get("shopping_reviews") or {}).get(str(args[0]), {})
    values = []
    for bucket in ("added", "changed"):
        for review in delta.get(bucket, []):
            key = "nickname" if call["name"].endswith("_author") else "rating_percent"
            if (value := review.get(key)) not in (None, ""):
                values.append(str(value))
    return values


def _program_html_snapshot_records(
    target: dict[str, Any],
    eval_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not eval_context:
        return []
    key = snapshot_target_key(target)
    delta = (eval_context.get("deltas") or {}).get("program_html") or {}
    return [record for bucket in ("added", "changed") for record in delta.get(bucket, []) if record.get("key") == key]


def _score_program_html_snapshot_records(
    required: dict[str, Any],
    records: list[dict[str, Any]],
) -> float:
    score = 0.0
    for record in records:
        if delta_value := str(record.get("delta_value") or "").strip():
            score = max(score, _score_program_html_required(required, delta_value))
        full_value = str(record.get("value") or "")
        before_value = str(record.get("before_value") or "")
        if full_value:
            full_score = _score_program_html_required(required, full_value)
            if full_score > _score_program_html_required(required, before_value):
                score = max(score, full_score)
    return score


async def _navigate(page: Any, url: str, config: EvaluationConfig) -> None:
    try:
        if config.navigate is not None:
            result = config.navigate(page, url)
        else:
            result = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        if inspect.isawaitable(result):
            await result
    except Exception as error:
        raise RuntimeError(f"program_html navigation failed for {url}: {error}") from error
    if config.navigation_wait_seconds:
        await asyncio.sleep(config.navigation_wait_seconds)


async def _program_html_target(
    target: dict[str, Any],
    page: Any,
    config: EvaluationConfig,
    eval_context: dict[str, Any] | None = None,
) -> float:
    records = _program_html_snapshot_records(target, eval_context)
    if records:
        snapshot_score = _score_program_html_snapshot_records(target["required_contents"], records)
        if snapshot_score:
            return snapshot_score

    target_url = target["url"]
    if any(call["name"] == "shopping_get_latest_order_url" for call in extract_helper_calls(target_url)):
        if candidate_urls := _order_delta_urls(eval_context):
            return max(
                [await _program_html_target({**target, "url": url}, page, config, None) for url in candidate_urls],
                default=0.0,
            )

    if review_values := _review_delta_values(target.get("locator"), eval_context):
        return max(_score_program_html_required(target["required_contents"], value) for value in review_values)

    if isinstance(target_url, str) and target_url.startswith("func:"):
        target_url = await _eval_helper_expression(target_url, page, config)
    if target_url != "last":
        await _navigate(page, str(target_url), config)

    locator = str(target["locator"])
    if not locator.strip():
        selected = page.content()
        selected = await selected if inspect.isawaitable(selected) else selected
    elif locator.startswith(("document.", "[...document.")):
        for action in target.get("prep_actions", []):
            try:
                result = page.evaluate(f"() => {action}")
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("Classic eval prep action failed", exc_info=True)
        try:
            selected = page.evaluate(f"() => {locator}")
            selected = await selected if inspect.isawaitable(selected) else selected
            selected = str(selected or "")
        except Exception:
            selected = ""
    elif locator.startswith("func:"):
        selected = str(await _eval_helper_expression(locator, page, config))
    else:
        raise ValueError(f"Unknown program_html locator: {locator}")
    return _score_program_html_required(target["required_contents"], selected)


async def _program_html(
    task_config: dict[str, Any],
    page: Any,
    config: EvaluationConfig,
    eval_context: dict[str, Any] | None = None,
) -> float:
    score = 1.0
    for target in (task_config.get("eval") or {}).get("program_html") or []:
        if target["url"] != "last":
            score *= await _program_html_target(target, page, config, eval_context)
            continue
        target_score = 0.0
        for candidate in _open_context_pages(page):
            try:
                target_score = max(
                    target_score,
                    await _program_html_target(target, candidate, config, eval_context),
                )
            except Exception:
                logger.info("program_html tab check failed for %s", candidate.url, exc_info=True)
        score *= target_score
    return score


async def evaluate_classic_task(
    task_config: dict[str, Any],
    agent_result: dict[str, Any],
    page: Any,
    *,
    config: EvaluationConfig | None = None,
    judge: Judge | None = None,
    helpers: Mapping[str, Helper] | None = None,
    judge_log_path: Path | None = None,
    eval_context: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Evaluate a Classic WebArena task against the final browser state."""
    runtime = config or default_evaluation_config(task_config)
    if any(value is not None for value in (judge, helpers, judge_log_path)):
        runtime = EvaluationConfig(
            judge=judge or runtime.judge,
            judge_config=runtime.judge_config,
            helpers=helpers or runtime.helpers,
            helper_config=runtime.helper_config,
            request_fn=runtime.request_fn,
            navigate=runtime.navigate,
            navigation_wait_seconds=runtime.navigation_wait_seconds,
            judge_log_path=judge_log_path or runtime.judge_log_path,
        )
    try:
        score = 1.0
        messages: list[str] = []
        for eval_type in (task_config.get("eval") or {}).get("eval_types", []):
            if eval_type == "string_match":
                current = await _string_match(task_config, agent_result.get("answer"), runtime)
                messages.append(f"string_match: score={current}")
            elif eval_type == "url_match":
                candidate_urls = await _get_open_page_urls(page)
                current, current_url, candidate_urls = _score_url_match_candidates(task_config, candidate_urls)
                messages.append(_url_match_message(task_config, current_url, current, candidate_urls))
            elif eval_type == "program_html":
                current = await _program_html(task_config, page, runtime, eval_context)
                messages.append(f"program_html: score={current}")
            else:
                raise ValueError(f"eval_type {eval_type} is not supported")
            score *= current
        return score, "; ".join(messages)
    except Exception as error:
        return 0.0, f"Classic evaluation error: {error}"

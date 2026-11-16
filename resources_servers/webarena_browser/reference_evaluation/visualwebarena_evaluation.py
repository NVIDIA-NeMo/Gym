# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Original source provenance: https://github.com/NVIDIA-NeMo/Gym/blob/d46f2f60d2f6eda11e94deb48e6a16b21b5389fc/resources_servers/webarena_browser/reference_evaluation/PROVENANCE.md
# Revision: 3b775dc538931ead0cb6b4922349da9c6d493dab
# Original file: webarena/common/visualwebarena_evaluation.py
# Original license: Apache-2.0; Copyright 2024 XLANG NLP Lab (root LICENSE).
# Adapted for Gym; see resources_servers/webarena_browser/reference_evaluation/PROVENANCE.md.
"""VisualWebArena evaluation helpers.

This follows VisualWebArena's eval schema while reusing the judge API pattern
used by the classic WebArena evaluator in this repo.
"""

from __future__ import annotations

import asyncio
import base64
import html
import inspect
import io
import json
import logging
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from nemo_gym.web import extract_helper_calls, snapshot_target_key

from .classic_evaluation import (
    _append_judge_log,
    _clean_answer,
    _exact_match,
    _format_required_group,
    _get_open_page_urls,
    _get_open_page_urls_sync,
    _judge_chat,
    _llm_fuzzy_match,
    _llm_must_include,
    _llm_ua_match,
    _parse_judge_json_label,
    _preview_text,
    _program_html_expression,
    _reference_url_alternatives,
    _word_tokenize_like_webarena,
    shopping_get_latest_order_url,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
)
from .navigation import goto
from .site_config import DEFAULT_CREDENTIALS


logger = logging.getLogger(__name__)


def _site_url(site: str) -> str:
    import os

    env_name = {
        "shopping": "WA_SHOPPING",
        "shopping_admin": "WA_SHOPPING_ADMIN",
        "reddit": "WA_REDDIT",
        "gitlab": "WA_GITLAB",
        "wikipedia": "WA_WIKIPEDIA",
        "map": "WA_MAP",
        "classifieds": "WA_CLASSIFIEDS",
    }[site]
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is required for VisualWebArena evaluation")
    return value.rstrip("/")


def _must_include(ref: str, pred: str) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    if len(_word_tokenize_like_webarena(clean_ref)) == 1:
        return float(clean_ref in _word_tokenize_like_webarena(clean_pred))
    return float(clean_ref in clean_pred)


def _must_exclude(ref: str, pred: str) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    if len(_word_tokenize_like_webarena(clean_ref)) == 1:
        return float(clean_ref not in _word_tokenize_like_webarena(clean_pred))
    return float(clean_ref not in clean_pred)


def _compare_inequality(value: int | float, inequality: str, tol: float = 1e-8) -> bool:
    ops = {
        "<=": lambda x, y: x <= y + tol,
        ">=": lambda x, y: x >= y - tol,
        "==": lambda x, y: abs(x - y) <= tol,
        "<": lambda x, y: x < y + tol,
        ">": lambda x, y: x > y - tol,
    }
    for op, func in ops.items():
        if op in inequality:
            _, num = inequality.split(op, 1)
            return func(value, float(num.strip()))
    raise ValueError(f"Invalid inequality string: {inequality}")


def _parse_numeric_value(value: Any) -> int | float | None:
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip().replace(",", "")
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return None


def _score_required_values(value: Any, required_values: list[str]) -> float:
    parsed = _parse_numeric_value(value)
    if parsed is None:
        return 0.0
    score = 1.0
    for required in required_values:
        alternatives = required.split(" |OR| ")
        score *= float(any(_compare_inequality(parsed, alt) for alt in alternatives))
    return score


def _string_match(task_config: dict, answer: Any, judge_log_path: Path | None = None) -> float:
    pred = _clean_answer(answer)
    refs = task_config["eval"].get("reference_answers") or {}
    score = 1.0
    for approach, value in refs.items():
        if approach == "exact_match":
            alternatives = value if isinstance(value, list) else [value]
            cur_score = max(_exact_match(ref=str(alt), pred=pred) for alt in alternatives)
            if cur_score != 1.0:
                cur_score = _llm_fuzzy_match(
                    pred=pred,
                    reference=_format_required_group([str(alt) for alt in alternatives]),
                    question=task_config["intent"],
                    judge_log_path=judge_log_path,
                    judge_type="visualwebarena_exact_match_fallback",
                )
            score *= cur_score
        elif approach == "required_values":
            score *= _score_required_values(pred, value)
        elif approach == "must_include":
            rule_score = 1.0
            required_groups: list[list[str]] = []
            for must_value in value:
                alternatives = str(must_value).split(" |OR| ")
                required_groups.append(alternatives)
                rule_score *= float(any(_must_include(alt, pred) for alt in alternatives))
            if rule_score == 1.0:
                score *= rule_score
            else:
                reference = "\n".join(
                    f"{idx}. {_format_required_group(alternatives)}"
                    for idx, alternatives in enumerate(required_groups, start=1)
                )
                score *= _llm_must_include(
                    pred=pred,
                    reference=reference,
                    question=task_config["intent"],
                    judge_log_path=judge_log_path,
                )
        elif approach == "must_exclude":
            for must_excl_value in value:
                score *= _must_exclude(str(must_excl_value), pred)
        elif approach == "one_of":
            score *= float(any(_clean_answer(str(option)) in pred for option in value))
        elif approach == "fuzzy_match":
            if value == "N/A":
                cur_score = _exact_match(ref=value, pred=pred)
                if cur_score != 1:
                    cur_score = _llm_ua_match(
                        pred=pred,
                        reference=task_config["eval"].get("string_note", ""),
                        question=task_config["intent"],
                        judge_log_path=judge_log_path,
                    )
                score *= cur_score
            else:
                for reference in value:
                    score *= _llm_fuzzy_match(
                        pred=pred,
                        reference=reference,
                        question=task_config["intent"],
                        judge_log_path=judge_log_path,
                    )
        else:
            raise ValueError(f"Unknown string_match approach: {approach}")
    return score


def _clean_url(url: str) -> str:
    cleaned = urllib.parse.urldefrag(str(url).replace("localhost", "127.0.0.1")).url
    return cleaned[:-1] if cleaned.endswith("/") else cleaned


def _url_match(task_config: dict, current_url: str) -> float:
    pred = _clean_url(current_url)
    ref_urls = [_clean_url(url) for url in _reference_url_alternatives(task_config["eval"].get("reference_url"))]
    matching_rule = task_config["eval"].get("url_note", "EXACT")
    if matching_rule == "EXACT":
        return float(pred in ref_urls)
    if matching_rule == "GOLD in PRED":
        return float(any(ref in pred for ref in ref_urls))
    raise ValueError(f"Unknown URL matching rule: {matching_rule}")


def _score_url_match_candidates(task_config: dict, candidate_urls: list[str]) -> tuple[float, str, list[str]]:
    unique_urls = list(dict.fromkeys(candidate_urls))
    if not unique_urls:
        return 0.0, "", []
    for candidate_url in unique_urls:
        if _url_match(task_config, candidate_url):
            return 1.0, candidate_url, unique_urls
    return 0.0, unique_urls[0], unique_urls


def _url_match_message(
    task_config: dict,
    current_url: str,
    score: float,
    candidate_urls: list[str] | None = None,
) -> str:
    refs = [_clean_url(url) for url in _reference_url_alternatives(task_config["eval"].get("reference_url"))]
    message = f"url_match: score={score}, pred={_clean_url(current_url)!r}, refs={refs!r}"
    if candidate_urls and len(candidate_urls) > 1:
        candidates = [_clean_url(url) for url in candidate_urls]
        message += f", candidates={candidates!r}"
    return message


async def _shopping_get_all_product_order(page) -> list[dict[str, Any]]:
    result = await page.evaluate(
        """
(() => {

    const table = document.querySelector("#my-orders-table");
    if (!table) return [];
    return [...table.getElementsByTagName('tbody')].map((x) => {
        return [...x.getElementsByTagName('td')].reduce(function(obj, y) {
            const key = y.className.split(' ')[1];
            obj[key] = y.outerText;
            if (key === 'name' && y.querySelector('dl')) {
                var option_dict = {};
                const options = [...y.querySelector('dl').children];
                for (let i = 0; i < options.length; i += 2) {
                    option_dict[options[i].outerText] = options[i + 1].outerText;
                }
                obj['options'] = option_dict;
            }
            return obj;
        }, {});
    });

})();
        """
    )
    return result or []


async def shopping_get_order_product_name_list(page) -> str:
    products = await _shopping_get_all_product_order(page)
    return " |OR| ".join([p.get("name", "") for p in products])


async def shopping_get_order_product_quantity(page, sku: str) -> int:
    skus = sku.split(" |OR| ") if "|OR|" in sku else [sku]
    products = await _shopping_get_all_product_order(page)
    for product in products:
        if str(product.get("sku", "")).strip() in skus:
            try:
                return int(str(product.get("qty", ""))[7:])
            except ValueError:
                return 0
    return 0


async def shopping_get_order_product_option(page, sku: str, option_name: str) -> str:
    products = await _shopping_get_all_product_order(page)
    for product in products:
        if str(product.get("sku", "")).strip() == sku:
            return str(product.get("options", {}).get(option_name, ""))
    return ""


async def shopping_get_product_attributes(page, attribute: str) -> str:
    return await page.evaluate(
        """
        (attribute) => {
            const searchTerms = attribute.toLowerCase().split(' |or| ');
            const rows = document.querySelector('#productDetails_detailBullets_sections1 > tbody')?.children || [];
            return Array.from(rows)
                .filter(x => searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term)))
                .map(x => x.children[1].outerText)
                .join(', ');
        }
        """,
        attribute,
    )


async def shopping_get_product_price(page) -> float:
    return await page.evaluate(
        """
        () => {
            const el = document.querySelector("#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span");
            if (!el) return 0;
            const res = parseFloat(el.outerText.substr(1));
            return res ? res : 0;
        }
        """
    )


async def shopping_get_num_reviews(page) -> int:
    return await page.evaluate(
        """
        () => {
            const el = document.querySelector("#tab-label-reviews-title");
            if (!el) return 0;
            const res = parseInt(el.outerText.split(' ')[1]);
            return res ? res : 0;
        }
        """
    )


async def shopping_get_rating_as_percentage(page) -> int:
    return await page.evaluate(
        """
        () => {
            const el = document.querySelector('.rating-result');
            if (!el) return 0;
            const rating = parseFloat(el.title.replace('%', ''));
            return rating ? rating : 0;
        }
        """
    )


async def get_query_text(page, selector: str) -> str:
    return await page.evaluate(
        """
        (selector) => {
            const el = document.querySelector(selector);
            return el ? el.textContent : '';
        }
        """,
        selector,
    )


async def get_query_text_lowercase(page, selector: str) -> str:
    return (await get_query_text(page, selector)).lower()


def reddit_get_post_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    tok_url = parsed.path.split("/")
    if len(tok_url) < 4 or tok_url[1] != "f":
        return url
    return f"{parsed.scheme}://{parsed.netloc}/f/{tok_url[2]}/{tok_url[3]}/"


async def reddit_get_post_comment_tree(page) -> dict[str, Any]:
    return await page.evaluate(
        """
        (function buildCommentTree(node, data_level) {
            if (!node) return {};
            let tree = {
                "username": (node.querySelector(".fg-inherit")?.outerText || ""),
                "net_score": parseInt((node.querySelector(".vote__net-score")?.outerText || "0")),
                "content": (node.querySelector(".comment__content")?.outerText || ""),
                "time": (node.querySelector('.comment__main > header > h1 > span > time')?.dateTime || ""),
                "children": []
            };
            node.querySelectorAll(".comment").forEach((child) => {
                if (parseInt(child.getAttribute('data-level')) === data_level + 1) {
                    tree.children.push(buildCommentTree(child, data_level + 1));
                }
            });
            return tree;
        })(document.querySelector("#main"), 0)
        """
    )


async def reddit_get_latest_comment_obj_by_username(page, username: str) -> dict[str, Any]:
    comment_tree = await reddit_get_post_comment_tree(page)
    latest_time = datetime.min.replace(tzinfo=timezone.utc)
    comment: dict[str, Any] = {}

    def dfs(node: dict[str, Any]) -> None:
        nonlocal latest_time, comment
        if node.get("username") == username:
            node_time = datetime.fromisoformat(node["time"].replace("Z", "+00:00"))
            if node_time > latest_time:
                comment = {**node, "time": node_time}
                latest_time = node_time
        for child in node.get("children", []):
            dfs(child)

    dfs(comment_tree)
    return comment


async def reddit_get_latest_comment_content_by_username(page, username: str) -> str:
    comment = await reddit_get_latest_comment_obj_by_username(page, username)
    return str(comment.get("content", ""))


async def reddit_get_parent_comment_obj_of_latest_comment_by_username(page, username: str) -> dict[str, Any]:
    comment_tree = await reddit_get_post_comment_tree(page)
    latest_time = datetime.min.replace(tzinfo=timezone.utc)
    comment: dict[str, Any] = {}

    def dfs(node: dict[str, Any]) -> None:
        nonlocal latest_time, comment
        for child in node.get("children", []):
            if child.get("username") == username:
                child_time = datetime.fromisoformat(child["time"].replace("Z", "+00:00"))
                if child_time > latest_time:
                    comment = {**node, "time": child_time}
                    latest_time = child_time
            else:
                dfs(child)

    dfs(comment_tree)
    return comment


async def reddit_get_parent_comment_username_of_latest_comment_by_username(page, username: str) -> str:
    comment = await reddit_get_parent_comment_obj_of_latest_comment_by_username(page, username)
    return str(comment.get("username", ""))


async def gitlab_get_project_memeber_role(page, account_name: str) -> str:
    account_idx = await page.evaluate(
        """
        (accountName) => {
            const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
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
            """(idx) => document.querySelectorAll("td.col-max-role span")[idx]?.outerText || "" """,
            account_idx,
        )
    )


async def _eval_helper_expression(expr: str, page) -> Any:
    helper_expr = expr.split("func:", 1)[1] if expr.startswith("func:") else expr
    helper_expr = helper_expr.replace("__last_url__", page.url)
    allowed = {
        "shopping_get_latest_order_url": shopping_get_latest_order_url,
        "shopping_get_sku_latest_review_author": shopping_get_sku_latest_review_author,
        "shopping_get_sku_latest_review_rating": shopping_get_sku_latest_review_rating,
        "shopping_get_sku_latest_review_text": lambda sku: _shopping_get_sku_latest_review_text(sku),
        "shopping_get_order_product_name_list": shopping_get_order_product_name_list,
        "shopping_get_order_product_quantity": shopping_get_order_product_quantity,
        "shopping_get_order_product_option": shopping_get_order_product_option,
        "shopping_get_product_attributes": shopping_get_product_attributes,
        "shopping_get_product_price": shopping_get_product_price,
        "shopping_get_num_reviews": shopping_get_num_reviews,
        "shopping_get_rating_as_percentage": shopping_get_rating_as_percentage,
        "get_query_text": get_query_text,
        "get_query_text_lowercase": get_query_text_lowercase,
        "reddit_get_post_url": reddit_get_post_url,
        "reddit_get_latest_comment_content_by_username": reddit_get_latest_comment_content_by_username,
        "reddit_get_latest_comment_obj_by_username": reddit_get_latest_comment_obj_by_username,
        "reddit_get_parent_comment_username_of_latest_comment_by_username": reddit_get_parent_comment_username_of_latest_comment_by_username,
        "gitlab_get_project_memeber_role": gitlab_get_project_memeber_role,
        "__page__": page,
    }
    try:
        result = eval(helper_expr, {"__builtins__": {}}, allowed)
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as e:
        raise RuntimeError(f"VisualWebArena helper failed for {expr}: {e}") from e


def _shopping_get_sku_latest_review_text(sku: str) -> str:
    headers = {
        "Authorization": f"Bearer {_shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = httpx.get(
        f"{_site_url('shopping')}/rest/V1/products/{sku}/reviews",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    reviews = response.json()
    if not reviews:
        return ""
    return str(reviews[-1]["detail"])


def _shopping_get_auth_token() -> str:
    creds = DEFAULT_CREDENTIALS["shopping_admin"]
    response = httpx.post(
        f"{_site_url('shopping')}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        json={"username": creds["username"], "password": creds["password"]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


async def _select_content(target: dict, page) -> Any:
    locator = target["locator"]
    if not locator.strip():
        return await page.content()
    if locator.startswith("document.") or locator.startswith("[...document."):
        for prep_action in target.get("prep_actions", []):
            await page.evaluate(f"() => {prep_action}")
        return str(await page.evaluate(_program_html_expression(locator)) or "")
    if locator.startswith("lambda:"):
        selected = await page.evaluate(locator.removeprefix("lambda:"))
        # Zero is an observed numeric value; empty selections remain misses.
        if isinstance(selected, (int, float)) and not isinstance(selected, bool):
            return selected
        return selected if selected else None
    if locator.startswith("func:"):
        return await _eval_helper_expression(locator, page)
    raise ValueError(f"Unknown program_html locator: {locator}")


def _score_program_html_required(
    required: dict[str, Any],
    selected_element: Any,
    judge_log_path: Path | None,
) -> float:
    selected_element = html.unescape(str(selected_element))
    score = 1.0
    if "exact_match" in required:
        score *= _exact_match(ref=required["exact_match"], pred=selected_element)
    elif "must_include" in required:
        rule_score = 1.0
        groups: list[list[str]] = []
        for content in required["must_include"]:
            alternatives = content.split(" |OR| ")
            groups.append(alternatives)
            rule_score *= float(any(_must_include(part, selected_element) for part in alternatives))
        if rule_score == 1.0:
            score *= rule_score
        else:
            reference = "\n".join(
                f"{idx}. {_format_required_group(alternatives)}" for idx, alternatives in enumerate(groups, start=1)
            )
            score *= _llm_must_include(
                pred=selected_element,
                reference=reference,
                question="NOT USED",
                judge_log_path=judge_log_path,
            )
    elif "must_exclude" in required:
        for content in required["must_exclude"]:
            if " |OR| " in content:
                raise ValueError("must_exclude does not support |OR| alternatives")
            score *= _must_exclude(content, selected_element)
    elif "required_values" in required:
        score *= _score_required_values(selected_element, required["required_values"])
    elif "fuzzy_match" in required:
        alternatives_score = 0.0
        errors: list[Exception] = []
        for reference in str(required["fuzzy_match"]).split(" |OR| "):
            try:
                alternatives_score = max(
                    alternatives_score,
                    _llm_fuzzy_match(
                        pred=selected_element,
                        reference=reference,
                        question="NOT USED",
                        judge_log_path=judge_log_path,
                    ),
                )
            except Exception as exc:
                errors.append(exc)
        if errors and alternatives_score < 1.0:
            raise errors[0]
        score *= alternatives_score
    else:
        raise ValueError(f"Unknown required_contents: {required.keys()}")
    return score


def _order_delta_urls(eval_context: dict[str, Any] | None) -> list[str] | None:
    """Return None without an order capture, or its authoritative delta URLs."""
    adapters = ((eval_context or {}).get("collision_plan") or {}).get("snapshot_adapters") or {}
    if "shopping_orders" not in adapters:
        return None
    order_delta = (eval_context.get("deltas") or {}).get("shopping_orders") or {}
    urls = []
    for bucket in ("added", "changed"):
        for record in order_delta.get(bucket, []):
            url = record.get("url")
            if url and url not in urls:
                urls.append(str(url))
    return urls


def _review_delta_values(expr: Any, eval_context: dict[str, Any] | None) -> list[str] | None:
    if not eval_context:
        return None
    calls = extract_helper_calls(expr)
    if not calls:
        return None
    call = calls[0]
    helper_name = call["name"]
    if helper_name not in {
        "shopping_get_sku_latest_review_author",
        "shopping_get_sku_latest_review_rating",
        "shopping_get_sku_latest_review_text",
    }:
        return None
    args = call.get("args") or []
    if not args or args[0] is None:
        return None
    sku = str(args[0])
    adapters = ((eval_context.get("collision_plan") or {}).get("snapshot_adapters")) or {}
    if sku not in (adapters.get("shopping_reviews") or {}).get("skus", []):
        return None
    sku_delta = ((eval_context.get("deltas") or {}).get("shopping_reviews") or {}).get(sku, {})
    values = []
    for bucket in ("added", "changed"):
        for review in sku_delta.get(bucket, []):
            if helper_name == "shopping_get_sku_latest_review_author":
                value = review.get("nickname")
            elif helper_name == "shopping_get_sku_latest_review_rating":
                value = review.get("rating_percent")
            else:
                value = review.get("detail")
            if value not in (None, ""):
                values.append(str(value))
    return values


def _program_html_snapshot_records(
    target: dict[str, Any],
    eval_context: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Distinguish unplanned targets from planned captures with no new records."""
    if not eval_context:
        return None
    key = snapshot_target_key(target)
    adapters = ((eval_context.get("collision_plan") or {}).get("snapshot_adapters")) or {}
    specs = (adapters.get("program_html") or {}).get("targets", [])
    if not any(spec.get("key") == key for spec in specs):
        return None
    program_delta = (eval_context.get("deltas") or {}).get("program_html") or {}
    records = []
    for bucket in ("added", "changed"):
        for record in program_delta.get(bucket, []):
            if record.get("key") != key:
                continue
            records.append(record)
    return records


def _score_program_html_snapshot_records(
    required: dict[str, Any],
    records: list[dict[str, Any]],
    judge_log_path: Path | None,
) -> float:
    score = 0.0
    for record in records:
        # An unrelated landing document supplies neither target content nor
        # proof of an empty target. Confirmed 404/410 absence is distinct.
        if record.get("target_document") is False and not record.get("target_absent"):
            continue
        full_value = str(record.get("value") or "")
        before_value = str(record.get("before_value") or "")
        before_score = _score_program_html_required(required, before_value, judge_log_path)
        delta_value = str(record.get("delta_value") or "").strip()
        if delta_value:
            # A line diff can repeat an existing match when unrelated text on
            # that line changes. Require genuinely new matching content; a new
            # copy of the required text still counts as an addition.
            fresh_match = before_score == 0
            if not fresh_match and "must_include" in required:
                before_text = _clean_answer(html.unescape(before_value))
                after_text = _clean_answer(html.unescape(full_value))
                fresh_match = all(
                    any(
                        after_text.count(_clean_answer(part)) > before_text.count(_clean_answer(part))
                        for part in content.split(" |OR| ")
                    )
                    for content in required["must_include"]
                )
            if fresh_match:
                score = max(score, _score_program_html_required(required, delta_value, judge_log_path))

        # A confirmed empty target/absence can satisfy a deletion criterion.
        # Require improvement over the baseline, including for empty strings.
        full_score = _score_program_html_required(required, full_value, judge_log_path)
        if full_score > before_score:
            score = max(score, full_score)
    return score


def _page_image_snapshot_urls(query: dict[str, Any], eval_context: dict[str, Any] | None) -> list[str] | None:
    if not eval_context:
        return None
    key = snapshot_target_key(query)
    adapters = (eval_context.get("collision_plan") or {}).get("snapshot_adapters") or {}
    if not any(spec.get("key") == key for spec in (adapters.get("page_image_query") or {}).get("queries", [])):
        return None
    image_delta = (eval_context.get("deltas") or {}).get("page_image_query") or {}
    urls = []
    for bucket in ("added", "changed"):
        for record in image_delta.get(bucket, []):
            if record.get("key") != key:
                continue
            candidates = record.get("delta_image_urls") or record.get("image_urls") or []
            for url in candidates:
                if url not in urls:
                    urls.append(str(url))
    return urls


async def _program_html_target(
    target: dict,
    page,
    judge_log_path: Path | None,
    eval_context: dict[str, Any] | None = None,
) -> float:
    target_url = target["url"]
    original_target_url = target_url
    snapshot_records = _program_html_snapshot_records(target, eval_context)
    if snapshot_records is not None:
        snapshot_score = _score_program_html_snapshot_records(
            target["required_contents"],
            snapshot_records,
            judge_log_path,
        )
        return snapshot_score

    if any(call["name"] == "shopping_get_latest_order_url" for call in extract_helper_calls(target_url)):
        candidate_urls = _order_delta_urls(eval_context)
        if candidate_urls is not None:
            target_score = 0.0
            for candidate_url in candidate_urls:
                candidate_target = {**target, "url": candidate_url}
                target_score = max(
                    target_score,
                    await _program_html_target(candidate_target, page, judge_log_path, None),
                )
            return target_score

    review_values = _review_delta_values(target.get("locator"), eval_context)
    if review_values is not None:
        return max(
            (
                _score_program_html_required(target["required_contents"], value, judge_log_path)
                for value in review_values
            ),
            default=0.0,
        )

    if isinstance(target_url, str) and target_url.startswith("func:"):
        target_url = await _eval_helper_expression(target_url, page)
    if target_url != "last":
        await goto(page, target_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)

    selected_element = await _select_content(target, page)
    if selected_element is None:
        return 0.0
    selected_element = html.unescape(str(selected_element))
    required = target["required_contents"]
    logger.info(
        "vwa program_html extracted: url=%s locator=%s value=%s required=%s",
        original_target_url,
        target.get("locator"),
        _preview_text(selected_element),
        required,
    )
    return _score_program_html_required(required, selected_element, judge_log_path)


async def _program_html(
    task_config: dict,
    page,
    judge_log_path: Path | None,
    eval_context: dict[str, Any] | None = None,
) -> float:
    score = 1.0
    for target in task_config["eval"].get("program_html") or []:
        if target["url"] == "last":
            pages = list(page.context.pages)
            target_score = 0.0
            candidate_errors: list[Exception] = []
            for candidate in pages or [page]:
                try:
                    target_score = max(
                        target_score,
                        await _program_html_target(target, candidate, judge_log_path, eval_context),
                    )
                except Exception as exc:
                    candidate_errors.append(exc)
                    logger.info("VisualWebArena program_html tab check failed", exc_info=True)
            if candidate_errors and target_score < 1.0:
                raise candidate_errors[0]
            score *= target_score
        else:
            score *= await _program_html_target(target, page, judge_log_path, eval_context)
    return score


class _UnavailableTaskImage(Exception):
    """A fetched candidate is missing (404/410) or is not a decodable image."""


def _fetch_image(url: str) -> Any:
    from PIL import Image

    response = httpx.get(url, timeout=60, follow_redirects=True)
    if response.status_code in (404, 410):
        raise _UnavailableTaskImage(f"Candidate image is absent: HTTP {response.status_code}")
    response.raise_for_status()
    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise _UnavailableTaskImage("Candidate response is not a decodable image") from exc


def _load_reference_image(ref: str) -> Any:
    from PIL import Image

    if ref.startswith("http"):
        return _fetch_image(ref)
    if ref.startswith("media/"):
        return _fetch_image(f"{_site_url('shopping')}/{ref}")
    return Image.open(ref).convert("RGB")


def _looks_like_image_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    return url.startswith(("http://", "https://")) and (
        any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"))
        or "images." in parsed.netloc
        or "media/" in path
    )


def _open_context_pages(page) -> list[Any]:
    pages: list[Any] = []
    try:
        pages = list(page.context.pages)
    except Exception:
        pages = []

    if page not in pages:
        pages.append(page)

    open_pages: list[Any] = []
    for candidate in pages:
        try:
            if hasattr(candidate, "is_closed") and candidate.is_closed():
                continue
        except Exception:
            continue
        open_pages.append(candidate)
    return open_pages


def _image_ssim(image_a: Any, image_b: Any) -> float:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except ModuleNotFoundError as e:
        raise RuntimeError("Pillow, scikit-image, and numpy are required for eval_fuzzy_image_match") from e

    new_size = (
        max(image_a.size[0], image_b.size[0]),
        max(image_a.size[1], image_b.size[1]),
    )
    gray_a = np.array(image_a.resize(new_size, Image.LANCZOS).convert("L"))
    gray_b = np.array(image_b.resize(new_size, Image.LANCZOS).convert("L"))
    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)


async def _collect_page_images(page, locator: str, *, errors: list[Exception] | None = None) -> list[Any]:
    if not locator.strip():
        elements = await page.query_selector_all("img")
    elif locator.startswith("."):
        elements = []
        matched = await page.query_selector_all(locator)
        if not matched and locator == ".products-grid .wishlist .product-image-photo":
            matched = await page.query_selector_all(".products-grid.wishlist .product-image-photo")
        if not matched and locator == ".submission__image":
            matched = await page.query_selector_all(".submission__link")
        for element in matched:
            is_img = await element.evaluate("element => element.tagName === 'IMG'")
            if is_img:
                elements.append(element)
            else:
                elements.extend(await element.query_selector_all("img"))
    else:
        raise ValueError(f"Unknown image locator: {locator}")

    failures = errors if errors is not None else []
    images: list[Any] = []
    for element in elements:
        try:
            href = await element.get_attribute("href")
            if _looks_like_image_url(href):
                images.append(_fetch_image(href))
                continue
            image_url = await element.get_attribute("src")
            if not image_url:
                continue
            if not image_url.startswith(("http://", "https://", "www.")):
                image_url = urllib.parse.urljoin(page.url, image_url)
            images.append(_fetch_image(image_url))
        except _UnavailableTaskImage:
            continue
        except Exception as exc:
            failures.append(exc)
    if errors is None and failures and not images:
        raise failures[0]
    return images


def _encode_image_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _next_judge_image_index(result_dir: Path) -> int:
    max_idx = 0
    for path in result_dir.glob("judge_image_*.png"):
        match = re.fullmatch(r"judge_image_(\d+)\.png", path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _save_judge_images(images: list[Any], judge_log_path: Path | None) -> list[str]:
    if judge_log_path is None:
        return []
    result_dir = judge_log_path.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    next_idx = _next_judge_image_index(result_dir)
    paths: list[str] = []
    for image in images:
        filename = f"judge_image_{next_idx}.png"
        image.save(result_dir / filename, format="PNG")
        paths.append(filename)
        next_idx += 1
    return paths


def _append_image_match_log(
    judge_log_path: Path | None,
    *,
    reference: str,
    image_files: list[str],
    ssim_scores: list[float],
    threshold: float,
    found_match: bool,
) -> None:
    _append_judge_log(
        judge_log_path,
        judge_type="visualwebarena_fuzzy_image_match",
        question="eval_fuzzy_image_match",
        reference=reference,
        prediction=json.dumps(
            {
                "image_files": image_files,
                "ssim_scores": ssim_scores,
                "threshold": threshold,
                "found_match": found_match,
            },
            ensure_ascii=True,
        ),
        messages=[],
        response="correct" if found_match else "incorrect",
    )


def _vlm_vqa_score(
    images: list[Any],
    question: str,
    expected_answer: str,
    judge_log_path: Path | None,
    image_log_paths: list[str] | None = None,
) -> float:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are grading VisualWebArena image evidence. "
                "Answer whether at least one provided image satisfies the question "
                f"with the expected answer.\nQuestion: {question}\n"
                f"Expected answer: {expected_answer}\n"
                "Respond with compact JSON only, using this schema: "
                '{"verdict":"correct|incorrect","answer":"yes|no|unknown",'
                '"rationale":"one short sentence grounded in the image"}.'
            ),
        }
    ]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(image)}})
    messages = [
        {"role": "system", "content": "You are a careful visual web task evaluator."},
        {"role": "user", "content": content},
    ]
    response_raw = _judge_chat(messages)  # type: ignore[arg-type]
    log_image_paths = image_log_paths if image_log_paths is not None else _save_judge_images(images, judge_log_path)
    log_messages = [
        {"role": "system", "content": "You are a careful visual web task evaluator."},
        {
            "role": "user",
            "content": [
                content[0],
                *[{"type": "image_file", "image_file": {"path": image_path}} for image_path in log_image_paths],
            ],
        },
    ]
    _append_judge_log(
        judge_log_path,
        judge_type="visualwebarena_page_image_vqa",
        question=question,
        reference=expected_answer,
        prediction=json.dumps({"image_files": log_image_paths}, ensure_ascii=True),
        messages=log_messages,  # type: ignore[arg-type]
        response=response_raw,
    )
    verdict = _parse_judge_json_label(
        response_raw,
        {"correct", "incorrect"},
        label_keys=("verdict",),
    )
    return float(verdict == "correct")


def _score_page_image_query_images(
    query: dict,
    images: list[Any],
    image_log_paths: list[str],
    judge_log_path: Path | None,
) -> float:
    score = 1.0
    for qa in query.get("eval_vqa", []):
        score *= _vlm_vqa_score(
            images=images,
            question=qa["question"],
            expected_answer=qa["answer"],
            judge_log_path=judge_log_path,
            image_log_paths=image_log_paths,
        )

    if "eval_fuzzy_image_match" in query:
        threshold = query.get("ssim_threshold", 0.8)
        refs = query["eval_fuzzy_image_match"].split(" |OR| ")

        ref_images = [_load_reference_image(ref) for ref in refs]
        found_match = False
        ssim_scores: list[float] = []
        for ref_image in ref_images:
            for image in images:
                ssim = _image_ssim(image, ref_image)
                ssim_scores.append(ssim)
                if ssim > threshold:
                    found_match = True
                    break
            if found_match:
                break
        score *= float(found_match)
        _append_image_match_log(
            judge_log_path,
            reference=query["eval_fuzzy_image_match"],
            image_files=image_log_paths,
            ssim_scores=ssim_scores,
            threshold=threshold,
            found_match=found_match,
        )
    return score


async def _page_image_query(
    task_config: dict,
    page,
    judge_log_path: Path | None,
    eval_context: dict[str, Any] | None = None,
) -> float:
    score = 1.0
    for query in task_config["eval"].get("page_image_query") or []:
        snapshot_urls = _page_image_snapshot_urls(query, eval_context)
        if snapshot_urls is not None:
            images, errors = [], []
            for image_url in snapshot_urls:
                try:
                    images.append(_fetch_image(image_url))
                except _UnavailableTaskImage:
                    continue
                except Exception as exc:
                    errors.append(exc)
            query_score = 0.0
            if images:
                image_log_paths = _save_judge_images(images, judge_log_path)
                query_score = _score_page_image_query_images(query, images, image_log_paths, judge_log_path)
            if errors and query_score < 1.0:
                raise errors[0]
            score *= query_score
            continue

        target_url = query["eval_image_url"]
        if isinstance(target_url, str) and target_url.startswith("func:"):
            target_url = await _eval_helper_expression(target_url, page)

        candidate_pages = _open_context_pages(page)
        if not candidate_pages:
            raise RuntimeError("No open page available for image evaluation")
        if target_url != "last":
            await goto(candidate_pages[0], target_url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)
            candidate_pages = [candidate_pages[0]]

        query_score = 0.0
        errors: list[Exception] = []
        for candidate in candidate_pages:
            try:
                images = await _collect_page_images(candidate, query["eval_image_class"], errors=errors)
                if images:
                    image_log_paths = _save_judge_images(images, judge_log_path)
                    query_score = max(
                        query_score,
                        _score_page_image_query_images(query, images, image_log_paths, judge_log_path),
                    )
            except Exception as exc:
                errors.append(exc)
        if errors and query_score < 1.0:
            raise errors[0]
        score *= query_score
    return score


async def evaluate_visualwebarena_task(
    task_config: dict,
    agent_result: dict,
    page,
    judge_log_path: Path | None = None,
    eval_context: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Evaluate a VisualWebArena task against the live final page."""
    try:
        score = 1.0
        messages: list[str] = []
        for eval_type in task_config["eval"]["eval_types"]:
            if eval_type == "string_match":
                cur_score = _string_match(task_config, agent_result.get("answer"), judge_log_path)
            elif eval_type == "url_match":
                candidate_urls = await _get_open_page_urls(page)
                cur_score, current_url, candidate_urls = _score_url_match_candidates(
                    task_config,
                    candidate_urls,
                )
            elif eval_type == "program_html":
                cur_score = await _program_html(task_config, page, judge_log_path, eval_context)
            elif eval_type == "page_image_query":
                cur_score = await _page_image_query(task_config, page, judge_log_path, eval_context)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported")
            score *= cur_score
            if eval_type == "url_match":
                messages.append(_url_match_message(task_config, current_url, cur_score, candidate_urls))
            else:
                messages.append(f"{eval_type}: score={cur_score}")
        return score, "; ".join(messages)
    except Exception as e:
        return 0.0, f"VisualWebArena evaluation error: {e}"


def _eval_helper_expression_sync(expr: str, page) -> Any:
    helper_expr = expr.split("func:", 1)[1] if expr.startswith("func:") else expr
    helper_expr = helper_expr.replace("__last_url__", page.url)
    allowed = {
        "shopping_get_latest_order_url": shopping_get_latest_order_url,
        "shopping_get_sku_latest_review_author": shopping_get_sku_latest_review_author,
        "shopping_get_sku_latest_review_rating": shopping_get_sku_latest_review_rating,
        "shopping_get_sku_latest_review_text": _shopping_get_sku_latest_review_text,
        "shopping_get_order_product_name_list": _shopping_get_order_product_name_list_sync,
        "shopping_get_order_product_quantity": _shopping_get_order_product_quantity_sync,
        "shopping_get_order_product_option": _shopping_get_order_product_option_sync,
        "shopping_get_product_attributes": _shopping_get_product_attributes_sync,
        "shopping_get_product_price": _shopping_get_product_price_sync,
        "shopping_get_num_reviews": _shopping_get_num_reviews_sync,
        "shopping_get_rating_as_percentage": _shopping_get_rating_as_percentage_sync,
        "get_query_text": _get_query_text_sync,
        "get_query_text_lowercase": _get_query_text_lowercase_sync,
        "reddit_get_post_url": reddit_get_post_url,
        "reddit_get_latest_comment_content_by_username": _reddit_get_latest_comment_content_by_username_sync,
        "reddit_get_latest_comment_obj_by_username": _reddit_get_latest_comment_obj_by_username_sync,
        "reddit_get_parent_comment_username_of_latest_comment_by_username": _reddit_get_parent_comment_username_of_latest_comment_by_username_sync,
        "gitlab_get_project_memeber_role": _gitlab_get_project_memeber_role_sync,
        "__page__": page,
    }
    try:
        return eval(helper_expr, {"__builtins__": {}}, allowed)
    except Exception as e:
        raise RuntimeError(f"VisualWebArena helper failed for {expr}: {e}") from e


def _shopping_get_all_product_order_sync(page) -> list[dict[str, Any]]:
    result = page.evaluate(
        """
(() => {

    const table = document.querySelector("#my-orders-table");
    if (!table) return [];
    return [...table.getElementsByTagName('tbody')].map((x) => {
        return [...x.getElementsByTagName('td')].reduce(function(obj, y) {
            const key = y.className.split(' ')[1];
            obj[key] = y.outerText;
            if (key === 'name' && y.querySelector('dl')) {
                var option_dict = {};
                const options = [...y.querySelector('dl').children];
                for (let i = 0; i < options.length; i += 2) {
                    option_dict[options[i].outerText] = options[i + 1].outerText;
                }
                obj['options'] = option_dict;
            }
            return obj;
        }, {});
    });

})();
        """
    )
    return result or []


def _shopping_get_order_product_name_list_sync(page) -> str:
    return " |OR| ".join([p.get("name", "") for p in _shopping_get_all_product_order_sync(page)])


def _shopping_get_order_product_quantity_sync(page, sku: str) -> int:
    skus = sku.split(" |OR| ") if "|OR|" in sku else [sku]
    for product in _shopping_get_all_product_order_sync(page):
        if str(product.get("sku", "")).strip() in skus:
            try:
                return int(str(product.get("qty", ""))[7:])
            except ValueError:
                return 0
    return 0


def _shopping_get_order_product_option_sync(page, sku: str, option_name: str) -> str:
    for product in _shopping_get_all_product_order_sync(page):
        if str(product.get("sku", "")).strip() == sku:
            return str(product.get("options", {}).get(option_name, ""))
    return ""


def _shopping_get_product_attributes_sync(page, attribute: str) -> str:
    return page.evaluate(
        """
        (attribute) => {
            const searchTerms = attribute.toLowerCase().split(' |or| ');
            const rows = document.querySelector('#productDetails_detailBullets_sections1 > tbody')?.children || [];
            return Array.from(rows)
                .filter(x => searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term)))
                .map(x => x.children[1].outerText)
                .join(', ');
        }
        """,
        attribute,
    )


def _shopping_get_product_price_sync(page) -> float:
    return page.evaluate(
        """
        () => {
            const el = document.querySelector("#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span");
            if (!el) return 0;
            const res = parseFloat(el.outerText.substr(1));
            return res ? res : 0;
        }
        """
    )


def _shopping_get_num_reviews_sync(page) -> int:
    return page.evaluate(
        """
        () => {
            const el = document.querySelector("#tab-label-reviews-title");
            if (!el) return 0;
            const res = parseInt(el.outerText.split(' ')[1]);
            return res ? res : 0;
        }
        """
    )


def _shopping_get_rating_as_percentage_sync(page) -> int:
    return page.evaluate(
        """
        () => {
            const el = document.querySelector('.rating-result');
            if (!el) return 0;
            const rating = parseFloat(el.title.replace('%', ''));
            return rating ? rating : 0;
        }
        """
    )


def _get_query_text_sync(page, selector: str) -> str:
    return page.evaluate(
        """
        (selector) => {
            const el = document.querySelector(selector);
            return el ? el.textContent : '';
        }
        """,
        selector,
    )


def _get_query_text_lowercase_sync(page, selector: str) -> str:
    return _get_query_text_sync(page, selector).lower()


def _reddit_get_post_comment_tree_sync(page) -> dict[str, Any]:
    return page.evaluate(
        """
        (function buildCommentTree(node, data_level) {
            if (!node) return {};
            let tree = {
                "username": (node.querySelector(".fg-inherit")?.outerText || ""),
                "net_score": parseInt((node.querySelector(".vote__net-score")?.outerText || "0")),
                "content": (node.querySelector(".comment__content")?.outerText || ""),
                "time": (node.querySelector('.comment__main > header > h1 > span > time')?.dateTime || ""),
                "children": []
            };
            node.querySelectorAll(".comment").forEach((child) => {
                if (parseInt(child.getAttribute('data-level')) === data_level + 1) {
                    tree.children.push(buildCommentTree(child, data_level + 1));
                }
            });
            return tree;
        })(document.querySelector("#main"), 0)
        """
    )


def _reddit_get_latest_comment_obj_by_username_sync(page, username: str) -> dict[str, Any]:
    comment_tree = _reddit_get_post_comment_tree_sync(page)
    latest_time = datetime.min.replace(tzinfo=timezone.utc)
    comment: dict[str, Any] = {}

    def dfs(node: dict[str, Any]) -> None:
        nonlocal latest_time, comment
        if node.get("username") == username:
            node_time = datetime.fromisoformat(node["time"].replace("Z", "+00:00"))
            if node_time > latest_time:
                comment = {**node, "time": node_time}
                latest_time = node_time
        for child in node.get("children", []):
            dfs(child)

    dfs(comment_tree)
    return comment


def _reddit_get_latest_comment_content_by_username_sync(page, username: str) -> str:
    return str(_reddit_get_latest_comment_obj_by_username_sync(page, username).get("content", ""))


def _reddit_get_parent_comment_username_of_latest_comment_by_username_sync(page, username: str) -> str:
    comment_tree = _reddit_get_post_comment_tree_sync(page)
    latest_time = datetime.min.replace(tzinfo=timezone.utc)
    parent_username = ""

    def dfs(node: dict[str, Any]) -> None:
        nonlocal latest_time, parent_username
        for child in node.get("children", []):
            if child.get("username") == username:
                child_time = datetime.fromisoformat(child["time"].replace("Z", "+00:00"))
                if child_time > latest_time:
                    parent_username = str(node.get("username", ""))
                    latest_time = child_time
            else:
                dfs(child)

    dfs(comment_tree)
    return parent_username


def _gitlab_get_project_memeber_role_sync(page, account_name: str) -> str:
    account_idx = page.evaluate(
        """
        (accountName) => {
            const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
            for (let i = 0; i < elements.length; i++) {
                if (elements[i].outerText === `@${accountName}`) return i;
            }
            return -1;
        }
        """,
        account_name,
    )
    return str(
        page.evaluate(
            """(idx) => document.querySelectorAll("td.col-max-role span")[idx]?.outerText || "" """,
            account_idx,
        )
    )


def _select_content_sync(target: dict, page) -> Any:
    locator = target["locator"]
    if not locator.strip():
        return page.content()
    if locator.startswith("document.") or locator.startswith("[...document."):
        for prep_action in target.get("prep_actions", []):
            page.evaluate(f"() => {prep_action}")
        return str(page.evaluate(_program_html_expression(locator)) or "")
    if locator.startswith("lambda:"):
        selected = page.evaluate(locator.removeprefix("lambda:"))
        # Zero is an observed numeric value; empty selections remain misses.
        if isinstance(selected, (int, float)) and not isinstance(selected, bool):
            return selected
        return selected if selected else None
    if locator.startswith("func:"):
        return _eval_helper_expression_sync(locator, page)
    raise ValueError(f"Unknown program_html locator: {locator}")


def _program_html_target_sync(
    target: dict,
    page,
    judge_log_path: Path | None,
    eval_context: dict[str, Any] | None = None,
) -> float:
    target_url = target["url"]
    snapshot_records = _program_html_snapshot_records(target, eval_context)
    if snapshot_records is not None:
        snapshot_score = _score_program_html_snapshot_records(
            target["required_contents"],
            snapshot_records,
            judge_log_path,
        )
        return snapshot_score

    if any(call["name"] == "shopping_get_latest_order_url" for call in extract_helper_calls(target_url)):
        candidate_urls = _order_delta_urls(eval_context)
        if candidate_urls is not None:
            target_score = 0.0
            for candidate_url in candidate_urls:
                candidate_target = {**target, "url": candidate_url}
                target_score = max(
                    target_score,
                    _program_html_target_sync(candidate_target, page, judge_log_path, None),
                )
            return target_score

    review_values = _review_delta_values(target.get("locator"), eval_context)
    if review_values is not None:
        return max(
            (
                _score_program_html_required(target["required_contents"], value, judge_log_path)
                for value in review_values
            ),
            default=0.0,
        )

    if isinstance(target_url, str) and target_url.startswith("func:"):
        target_url = _eval_helper_expression_sync(target_url, page)
    if target_url != "last":
        page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(3)

    selected_element = _select_content_sync(target, page)
    if selected_element is None:
        return 0.0
    selected_element = html.unescape(str(selected_element))
    required = target["required_contents"]
    return _score_program_html_required(required, selected_element, judge_log_path)


def _program_html_sync(
    task_config: dict,
    page,
    judge_log_path: Path | None,
    eval_context: dict[str, Any] | None = None,
) -> float:
    score = 1.0
    for target in task_config["eval"].get("program_html") or []:
        if target["url"] == "last":
            pages = list(page.context.pages)
            target_score = 0.0
            candidate_errors: list[Exception] = []
            for candidate in pages or [page]:
                try:
                    target_score = max(
                        target_score,
                        _program_html_target_sync(target, candidate, judge_log_path, eval_context),
                    )
                except Exception as exc:
                    candidate_errors.append(exc)
                    logger.info("VisualWebArena program_html tab check failed", exc_info=True)
            if candidate_errors and target_score < 1.0:
                raise candidate_errors[0]
            score *= target_score
        else:
            score *= _program_html_target_sync(target, page, judge_log_path, eval_context)
    return score


def _collect_page_images_sync(page, locator: str, *, errors: list[Exception] | None = None) -> list[Any]:
    if not locator.strip():
        elements = page.query_selector_all("img")
    elif locator.startswith("."):
        elements = []
        matched = page.query_selector_all(locator)
        if not matched and locator == ".products-grid .wishlist .product-image-photo":
            matched = page.query_selector_all(".products-grid.wishlist .product-image-photo")
        if not matched and locator == ".submission__image":
            matched = page.query_selector_all(".submission__link")
        for element in matched:
            is_img = element.evaluate("element => element.tagName === 'IMG'")
            if is_img:
                elements.append(element)
            else:
                elements.extend(element.query_selector_all("img"))
    else:
        raise ValueError(f"Unknown image locator: {locator}")

    failures = errors if errors is not None else []
    images: list[Any] = []
    for element in elements:
        try:
            href = element.get_attribute("href")
            if _looks_like_image_url(href):
                images.append(_fetch_image(href))
                continue
            image_url = element.get_attribute("src")
            if not image_url:
                continue
            if not image_url.startswith(("http://", "https://", "www.")):
                image_url = urllib.parse.urljoin(page.url, image_url)
            images.append(_fetch_image(image_url))
        except _UnavailableTaskImage:
            continue
        except Exception as exc:
            failures.append(exc)
    if errors is None and failures and not images:
        raise failures[0]
    return images


def _page_image_query_sync(
    task_config: dict,
    page,
    judge_log_path: Path | None,
    eval_context: dict[str, Any] | None = None,
) -> float:
    score = 1.0
    for query in task_config["eval"].get("page_image_query") or []:
        snapshot_urls = _page_image_snapshot_urls(query, eval_context)
        if snapshot_urls is not None:
            images, errors = [], []
            for image_url in snapshot_urls:
                try:
                    images.append(_fetch_image(image_url))
                except _UnavailableTaskImage:
                    continue
                except Exception as exc:
                    errors.append(exc)
            query_score = 0.0
            if images:
                image_log_paths = _save_judge_images(images, judge_log_path)
                query_score = _score_page_image_query_images(query, images, image_log_paths, judge_log_path)
            if errors and query_score < 1.0:
                raise errors[0]
            score *= query_score
            continue

        target_url = query["eval_image_url"]
        if isinstance(target_url, str) and target_url.startswith("func:"):
            target_url = _eval_helper_expression_sync(target_url, page)

        candidate_pages = _open_context_pages(page)
        if not candidate_pages:
            raise RuntimeError("No open page available for image evaluation")
        if target_url != "last":
            candidate_pages[0].goto(target_url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3)
            candidate_pages = [candidate_pages[0]]

        query_score = 0.0
        errors: list[Exception] = []
        for candidate in candidate_pages:
            try:
                images = _collect_page_images_sync(candidate, query["eval_image_class"], errors=errors)
                if images:
                    image_log_paths = _save_judge_images(images, judge_log_path)
                    query_score = max(
                        query_score,
                        _score_page_image_query_images(query, images, image_log_paths, judge_log_path),
                    )
            except Exception as exc:
                errors.append(exc)
        if errors and query_score < 1.0:
            raise errors[0]
        score *= query_score
    return score


def evaluate_visualwebarena_task_sync(
    task_config: dict,
    agent_result: dict,
    page,
    judge_log_path: Path | None = None,
    eval_context: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Synchronous variant for the human runner's sync Playwright page."""
    try:
        score = 1.0
        messages: list[str] = []
        for eval_type in task_config["eval"]["eval_types"]:
            if eval_type == "string_match":
                cur_score = _string_match(task_config, agent_result.get("answer"), judge_log_path)
            elif eval_type == "url_match":
                candidate_urls = _get_open_page_urls_sync(page)
                cur_score, current_url, candidate_urls = _score_url_match_candidates(
                    task_config,
                    candidate_urls,
                )
            elif eval_type == "program_html":
                cur_score = _program_html_sync(task_config, page, judge_log_path, eval_context)
            elif eval_type == "page_image_query":
                cur_score = _page_image_query_sync(task_config, page, judge_log_path, eval_context)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported")
            score *= cur_score
            if eval_type == "url_match":
                messages.append(_url_match_message(task_config, current_url, cur_score, candidate_urls))
            else:
                messages.append(f"{eval_type}: score={cur_score}")
        return score, "; ".join(messages)
    except Exception as e:
        return 0.0, f"VisualWebArena evaluation error: {e}"

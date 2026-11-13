# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Original source provenance: https://github.com/NVIDIA-NeMo/Gym/blob/d46f2f60d2f6eda11e94deb48e6a16b21b5389fc/resources_servers/webarena_browser/reference_evaluation/PROVENANCE.md
# Revision: 3b775dc538931ead0cb6b4922349da9c6d493dab
# Original file: webarena/common/eval_snapshots.py
# Original license: Apache-2.0; Copyright 2024 XLANG NLP Lab.
# Adapted for Gym; see resources_servers/webarena_browser/reference_evaluation/PROVENANCE.md.
"""Structured before/after snapshots for collision-prone evaluators."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any
from urllib.parse import SplitResult, parse_qsl, urlsplit

import httpx

from nemo_gym.web import EvaluatorInfrastructureError

from .classic_evaluation import (
    _program_html_expression,
    _shopping_get_auth_token,
    _site_url,
    gitlab_get_project_memeber_role,
    gitlab_get_project_memeber_role_sync,
)
from .navigation import goto


logger = logging.getLogger(__name__)
_SNAPSHOT_LOGIN_FORM = """() => document.querySelector(
    'form#login-form input[type="password"], form#new_user input[type="password"], '
    + 'form[action$="/login"] input[type="password"], form[action$="/sign_in"] input[type="password"]'
) !== null"""


def _same_snapshot_origin(requested: SplitResult, landed: SplitResult) -> bool:
    if requested.scheme not in {"http", "https"} or landed.scheme not in {requested.scheme, "https"}:
        return False
    ports = [None if url.port == {"http": 80, "https": 443}[url.scheme] else url.port for url in (requested, landed)]
    return bool(requested.hostname) and (requested.hostname, ports[0]) == (landed.hostname, ports[1])


def _snapshot_url_matches(requested_url: str, landed_url: str) -> bool:
    """Recognize the same resource, including bounded local-site canonical redirects."""
    requested, landed = urlsplit(requested_url), urlsplit(landed_url)
    if not _same_snapshot_origin(requested, landed):
        return False
    paths = [url.path.rstrip("/") or "/" for url in (requested, landed)]
    for env_name in ("WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT"):
        site_url = os.environ.get(env_name)
        if not site_url:
            continue
        site = urlsplit(site_url)
        prefix = site.path.rstrip("/")
        if not _same_snapshot_origin(site, requested) or not all(
            path == prefix or path.startswith(prefix + "/") for path in paths
        ):
            continue
        if env_name == "WA_REDDIT":
            # A changed post title changes the slug, not the forum/post identity.
            paths = [re.sub(r"^(/f/[^/]+/\d+)(?:/[^/]+)?$", r"\1", path[len(prefix) :]) for path in paths]
        else:
            paths = [re.sub(r"^/index\.php(?:/|$)", "/", path[len(prefix) :]).rstrip("/") or "/" for path in paths]
        break
    return paths[0] == paths[1] and sorted(parse_qsl(requested.query, keep_blank_values=True)) == sorted(
        parse_qsl(landed.query, keep_blank_values=True)
    )


def _validate_snapshot_navigation(record: dict[str, Any], phase: str) -> None:
    if "navigation" not in record:
        return  # Retained structured captures predate navigation metadata.
    navigation = record["navigation"]
    if not isinstance(navigation, dict):
        raise EvaluatorInfrastructureError("Malformed snapshot navigation")
    status, landed_url = navigation.get("status"), navigation.get("url")
    if type(status) is not int or not 200 <= status <= 599 or not isinstance(landed_url, str):
        raise EvaluatorInfrastructureError("Missing snapshot navigation response")
    requested, landed = urlsplit(record["url"]), urlsplit(landed_url)
    if not _same_snapshot_origin(requested, landed):
        raise EvaluatorInfrastructureError("Snapshot navigation left the target site")
    login_path = re.search(r"(?:^|/)(?:login|log_in|sign_in|signin)(?:/|$)", landed.path, re.IGNORECASE)
    login_query = ("page", "login") in parse_qsl(landed.query)
    if status >= 500 or status in {401, 403, 408, 425, 429} or login_path or login_query:
        raise EvaluatorInfrastructureError("Snapshot navigation reached an unavailable or unauthenticated page")
    if 300 <= status < 400 and status != 304:
        raise EvaluatorInfrastructureError("Snapshot navigation did not complete")
    # A missing file before creation is observed absence. After the agent acts,
    # a 404/410 or same-site redirect can be the task outcome (e.g. deletion).
    # Neither may be silently converted to an invalid/retried rollout.
    if phase == "before" and (
        (status >= 400 and status not in {404, 410}) or not _snapshot_url_matches(record["url"], landed_url)
    ):
        raise EvaluatorInfrastructureError("Snapshot baseline did not reach the requested resource")


def _snapshot_navigation(page, response, requested_url: str) -> dict[str, Any]:
    if response is None:
        raise EvaluatorInfrastructureError("Missing snapshot navigation response")
    navigation = {"status": response.status, "url": page.url}
    # Reject service/auth failures before extraction in all four collectors.
    # Baseline resource identity is checked once the capture phase is known.
    _validate_snapshot_navigation({"url": requested_url, "navigation": navigation}, "after")
    return navigation


def _snapshot_has_target_document(navigation: dict[str, Any], requested_url: str) -> bool:
    status = navigation["status"]
    return (200 <= status < 300 or status == 304) and _snapshot_url_matches(requested_url, navigation["url"])


def _stable_key(record: dict[str, Any], preferred_keys: tuple[str, ...]) -> str:
    for key in preferred_keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    payload = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def diff_records(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    before_by_key = {_stable_key(record, key_fields): record for record in before}
    after_by_key = {_stable_key(record, key_fields): record for record in after}

    added = [{**record, "_snapshot_key": key} for key, record in after_by_key.items() if key not in before_by_key]
    changed = [
        {**record, "_snapshot_key": key}
        for key, record in after_by_key.items()
        if key in before_by_key and record != before_by_key[key]
    ]
    removed = [{**record, "_snapshot_key": key} for key, record in before_by_key.items() if key not in after_by_key]
    return {"added": added, "changed": changed, "removed": removed}


def _added_text(before: Any, after: Any) -> str:
    before_lines = str(before or "").splitlines()
    after_lines = str(after or "").splitlines()
    added = [
        line[2:] for line in difflib.ndiff(before_lines, after_lines) if line.startswith("+ ") and line[2:].strip()
    ]
    return "\n".join(added)


def _shopping_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }


def _order_url(order: dict[str, Any]) -> str:
    order_id = order.get("increment_id") or order.get("entity_id") or order.get("order_id")
    return f"{_site_url('shopping')}/sales/order/view/order_id/{int(order_id)}/" if order_id else ""


def fetch_shopping_orders(page_size: int = 50) -> list[dict[str, Any]]:
    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": str(page_size),
    }
    response = httpx.get(
        f"{_site_url('shopping')}/rest/V1/orders",
        params=params,
        headers=_shopping_headers(),
        timeout=60,
    )
    response.raise_for_status()
    records = []
    for order in _record_list(response.json()["items"], "shopping orders"):
        items = order.get("items") or []
        records.append(
            {
                "entity_id": order.get("entity_id"),
                "increment_id": order.get("increment_id"),
                "created_at": order.get("created_at"),
                "status": order.get("status"),
                "grand_total": order.get("grand_total"),
                "url": _order_url(order),
                "items": [
                    {
                        "name": item.get("name"),
                        "sku": item.get("sku"),
                        "qty_ordered": item.get("qty_ordered"),
                    }
                    for item in items
                ],
            }
        )
    return records


def fetch_shopping_reviews(skus: list[str]) -> dict[str, list[dict[str, Any]]]:
    reviews_by_sku: dict[str, list[dict[str, Any]]] = {}
    headers = _shopping_headers()
    for sku in skus:
        response = httpx.get(
            f"{_site_url('shopping')}/rest/V1/products/{sku}/reviews",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        records = []
        for review in _record_list(response.json(), "shopping reviews"):
            ratings = review.get("ratings") or []
            first_rating = ratings[0] if ratings else {}
            records.append(
                {
                    "id": review.get("id") or review.get("review_id"),
                    "sku": sku,
                    "nickname": review.get("nickname"),
                    "title": review.get("title"),
                    "detail": review.get("detail"),
                    "created_at": review.get("created_at"),
                    "rating_percent": first_rating.get("percent"),
                    "rating_value": first_rating.get("value"),
                }
            )
        reviews_by_sku[sku] = records
    return reviews_by_sku


def collect_snapshots(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Collect requested API snapshots, propagating unavailable baselines."""
    snapshots: dict[str, Any] = {}
    adapters = (plan or {}).get("snapshot_adapters") or {}

    if "shopping_orders" in adapters:
        try:
            snapshots["shopping_orders"] = fetch_shopping_orders()
        except Exception as exc:
            raise EvaluatorInfrastructureError("Failed to collect shopping order snapshot") from exc

    review_params = adapters.get("shopping_reviews")
    if review_params:
        skus = [str(sku) for sku in review_params.get("skus", []) if sku]
        try:
            snapshots["shopping_reviews"] = fetch_shopping_reviews(skus)
        except Exception as exc:
            raise EvaluatorInfrastructureError("Failed to collect shopping review snapshot") from exc

    return snapshots


def _parse_func_args(expr: str) -> tuple[str, list[Any]]:
    helper_expr = expr.split("func:", 1)[1] if expr.startswith("func:") else expr
    tree = ast.parse(helper_expr, mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"Unsupported snapshot helper expression: {expr}")
    args = []
    for arg in call.args:
        if isinstance(arg, ast.Name) and arg.id == "__page__":
            args.append("__page__")
        elif isinstance(arg, ast.Constant):
            args.append(arg.value)
        else:
            raise ValueError(f"Unsupported snapshot helper argument in: {expr}")
    return call.func.id, args


async def _get_query_text(page, selector: str) -> str:
    return str(
        await page.evaluate(
            """
        (selector) => {
            const el = document.querySelector(selector);
            return el ? el.textContent : '';
        }
        """,
            selector,
        )
        or ""
    )


async def _gitlab_member_role(page, account_name: str) -> str:
    try:
        return await gitlab_get_project_memeber_role(page, account_name)
    except Exception as exc:
        raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc


async def _reddit_comment_tree(page) -> dict[str, Any]:
    try:
        return (
            await page.evaluate(
                """
            (function buildCommentTree(node, data_level) {
                let tree = {
                    "username": node.querySelector(".fg-inherit").outerText,
                    "content": node.querySelector(".comment__content").outerText,
                    "time": node.querySelector('.comment__main > header > h1 > span > time').dateTime,
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
            or {}
        )
    except Exception as exc:
        raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc


def _latest_comment_by_username(tree: dict[str, Any], username: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from datetime import datetime, timezone

    latest_time = datetime.min.replace(tzinfo=timezone.utc)
    latest: dict[str, Any] = {}
    latest_parent: dict[str, Any] | None = None

    def visit(node: dict[str, Any], parent: dict[str, Any] | None = None) -> None:
        nonlocal latest, latest_parent, latest_time
        if node.get("username") == username:
            try:
                node_time = datetime.fromisoformat(str(node["time"]).replace("Z", "+00:00"))
            except Exception:
                node_time = datetime.min.replace(tzinfo=timezone.utc)
            if node_time > latest_time:
                latest = node
                latest_parent = parent
                latest_time = node_time
        for child in node.get("children", []):
            visit(child, node)

    visit(tree)
    return latest, latest_parent


async def _select_program_html_snapshot(target: dict[str, Any], page) -> str:
    locator = str(target.get("locator", ""))
    if not locator.strip():
        return str(await page.content())
    if locator.startswith("document.") or locator.startswith("[...document."):
        try:
            return str(await page.evaluate(_program_html_expression(locator)) or "")
        except Exception as exc:
            raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc
    if locator.startswith("lambda:"):
        try:
            return str(await page.evaluate(locator.removeprefix("lambda:")) or "")
        except Exception as exc:
            raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc
    if locator.startswith("func:"):
        helper, args = _parse_func_args(locator)
        if helper in {"get_query_text", "get_query_text_lowercase"} and len(args) >= 2:
            text = await _get_query_text(page, str(args[1]))
            return text.lower() if helper == "get_query_text_lowercase" else text
        if helper == "gitlab_get_project_memeber_role" and len(args) >= 2:
            return await _gitlab_member_role(page, str(args[1]))
        if (
            helper
            in {
                "reddit_get_latest_comment_content_by_username",
                "reddit_get_parent_comment_username_of_latest_comment_by_username",
            }
            and len(args) >= 2
        ):
            tree = await _reddit_comment_tree(page)
            latest, parent = _latest_comment_by_username(tree, str(args[1]))
            if helper == "reddit_get_latest_comment_content_by_username":
                return str(latest.get("content", ""))
            return str((parent or {}).get("username", ""))
    raise ValueError(f"Unsupported program_html snapshot locator: {locator}")


async def _collect_program_html_snapshots(page, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in targets:
        specs_by_url[str(spec["target"].get("url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = await page.context.new_page()
        try:
            response = await goto(snapshot_page, url, wait_until="domcontentloaded", timeout=60000)
            navigation = _snapshot_navigation(snapshot_page, response, url)
            target_document = _snapshot_has_target_document(navigation, url)
            if await snapshot_page.evaluate(_SNAPSHOT_LOGIN_FORM):
                raise EvaluatorInfrastructureError("Snapshot navigation rendered a login form")
            for spec in specs:
                target = spec["target"]
                try:
                    value = await _select_program_html_snapshot(target, snapshot_page) if target_document else ""
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": value,
                            "navigation": navigation,
                            "target_document": target_document,
                        }
                    )
                except Exception:
                    logger.warning("Failed to extract program_html snapshot for %s", target.get("url"), exc_info=True)
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": "",
                            "error": True,
                        }
                    )
        except Exception:
            logger.warning("Failed to navigate program_html snapshot URL %s", url, exc_info=True)
            for spec in specs:
                target = spec["target"]
                snapshots.append(
                    {
                        "key": spec["key"],
                        "url": target.get("url"),
                        "locator": target.get("locator"),
                        "value": "",
                        "error": True,
                    }
                )
        finally:
            await snapshot_page.close()
    return snapshots


async def _collect_image_urls(page, locator: str) -> list[str]:
    if not locator.strip():
        selector = "img"
    else:
        selector = locator
    try:
        return (
            await page.evaluate(
                """
            (selector) => {
                const root = selector ? document.querySelector(selector) : document;
                if (!root) return [];
                const elements = root.matches && root.matches('img,a') ? [root] : Array.from(root.querySelectorAll('img,a'));
                return elements.map((el) => {
                    if (el.tagName.toLowerCase() === 'a') return el.href || '';
                    return el.currentSrc || el.src || '';
                }).filter(Boolean);
            }
            """,
                selector,
            )
            or []
        )
    except Exception as exc:
        raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc


async def _collect_page_image_snapshots(page, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in queries:
        specs_by_url[str(spec["query"].get("eval_image_url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = await page.context.new_page()
        try:
            response = await goto(snapshot_page, url, wait_until="domcontentloaded", timeout=60000)
            navigation = _snapshot_navigation(snapshot_page, response, url)
            target_document = _snapshot_has_target_document(navigation, url)
            if await snapshot_page.evaluate(_SNAPSHOT_LOGIN_FORM):
                raise EvaluatorInfrastructureError("Snapshot navigation rendered a login form")
            for spec in specs:
                query = spec["query"]
                try:
                    urls = (
                        await _collect_image_urls(snapshot_page, str(query.get("eval_image_class", "")))
                        if target_document
                        else []
                    )
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": urls,
                            "navigation": navigation,
                            "target_document": target_document,
                        }
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract page_image_query snapshot for %s",
                        query.get("eval_image_url"),
                        exc_info=True,
                    )
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": [],
                            "error": True,
                        }
                    )
        except Exception:
            logger.warning("Failed to navigate page_image_query snapshot URL %s", url, exc_info=True)
            for spec in specs:
                query = spec["query"]
                snapshots.append(
                    {
                        "key": spec["key"],
                        "url": query.get("eval_image_url"),
                        "locator": query.get("eval_image_class"),
                        "image_urls": [],
                        "error": True,
                    }
                )
        finally:
            await snapshot_page.close()
    return snapshots


async def collect_browser_snapshots(page, plan: dict[str, Any] | None) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    adapters = (plan or {}).get("snapshot_adapters") or {}
    program_targets = (adapters.get("program_html") or {}).get("targets") or []
    if program_targets:
        snapshots["program_html"] = await _collect_program_html_snapshots(page, program_targets)
    image_queries = (adapters.get("page_image_query") or {}).get("queries") or []
    if image_queries:
        snapshots["page_image_query"] = await _collect_page_image_snapshots(page, image_queries)
    return snapshots


def _get_query_text_sync(page, selector: str) -> str:
    return str(
        page.evaluate(
            """
        (selector) => {
            const el = document.querySelector(selector);
            return el ? el.textContent : '';
        }
        """,
            selector,
        )
        or ""
    )


def _gitlab_member_role_sync(page, account_name: str) -> str:
    try:
        return gitlab_get_project_memeber_role_sync(page, account_name)
    except Exception as exc:
        raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc


def _reddit_comment_tree_sync(page) -> dict[str, Any]:
    try:
        return (
            page.evaluate(
                """
            (function buildCommentTree(node, data_level) {
                let tree = {
                    "username": node.querySelector(".fg-inherit").outerText,
                    "content": node.querySelector(".comment__content").outerText,
                    "time": node.querySelector('.comment__main > header > h1 > span > time').dateTime,
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
            or {}
        )
    except Exception as exc:
        raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc


def _select_program_html_snapshot_sync(target: dict[str, Any], page) -> str:
    locator = str(target.get("locator", ""))
    if not locator.strip():
        return str(page.content())
    if locator.startswith("document.") or locator.startswith("[...document."):
        try:
            return str(page.evaluate(_program_html_expression(locator)) or "")
        except Exception as exc:
            raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc
    if locator.startswith("lambda:"):
        try:
            return str(page.evaluate(locator.removeprefix("lambda:")) or "")
        except Exception as exc:
            raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc
    if locator.startswith("func:"):
        helper, args = _parse_func_args(locator)
        if helper in {"get_query_text", "get_query_text_lowercase"} and len(args) >= 2:
            text = _get_query_text_sync(page, str(args[1]))
            return text.lower() if helper == "get_query_text_lowercase" else text
        if helper == "gitlab_get_project_memeber_role" and len(args) >= 2:
            return _gitlab_member_role_sync(page, str(args[1]))
        if (
            helper
            in {
                "reddit_get_latest_comment_content_by_username",
                "reddit_get_parent_comment_username_of_latest_comment_by_username",
            }
            and len(args) >= 2
        ):
            tree = _reddit_comment_tree_sync(page)
            latest, parent = _latest_comment_by_username(tree, str(args[1]))
            if helper == "reddit_get_latest_comment_content_by_username":
                return str(latest.get("content", ""))
            return str((parent or {}).get("username", ""))
    raise ValueError(f"Unsupported program_html snapshot locator: {locator}")


def _collect_program_html_snapshots_sync(page, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in targets:
        specs_by_url[str(spec["target"].get("url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = page.context.new_page()
        try:
            response = snapshot_page.goto(url, wait_until="domcontentloaded", timeout=60000)
            navigation = _snapshot_navigation(snapshot_page, response, url)
            target_document = _snapshot_has_target_document(navigation, url)
            if snapshot_page.evaluate(_SNAPSHOT_LOGIN_FORM):
                raise EvaluatorInfrastructureError("Snapshot navigation rendered a login form")
            for spec in specs:
                target = spec["target"]
                try:
                    value = _select_program_html_snapshot_sync(target, snapshot_page) if target_document else ""
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": value,
                            "navigation": navigation,
                            "target_document": target_document,
                        }
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract sync program_html snapshot for %s", target.get("url"), exc_info=True
                    )
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": "",
                            "error": True,
                        }
                    )
        except Exception:
            logger.warning("Failed to navigate sync program_html snapshot URL %s", url, exc_info=True)
            for spec in specs:
                target = spec["target"]
                snapshots.append(
                    {
                        "key": spec["key"],
                        "url": target.get("url"),
                        "locator": target.get("locator"),
                        "value": "",
                        "error": True,
                    }
                )
        finally:
            snapshot_page.close()
    return snapshots


def _collect_image_urls_sync(page, locator: str) -> list[str]:
    selector = "img" if not locator.strip() else locator
    try:
        return (
            page.evaluate(
                """
            (selector) => {
                const root = selector ? document.querySelector(selector) : document;
                if (!root) return [];
                const elements = root.matches && root.matches('img,a') ? [root] : Array.from(root.querySelectorAll('img,a'));
                return elements.map((el) => {
                    if (el.tagName.toLowerCase() === 'a') return el.href || '';
                    return el.currentSrc || el.src || '';
                }).filter(Boolean);
            }
            """,
                selector,
            )
            or []
        )
    except Exception as exc:
        raise EvaluatorInfrastructureError("Browser snapshot capture failed") from exc


def _collect_page_image_snapshots_sync(page, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in queries:
        specs_by_url[str(spec["query"].get("eval_image_url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = page.context.new_page()
        try:
            response = snapshot_page.goto(url, wait_until="domcontentloaded", timeout=60000)
            navigation = _snapshot_navigation(snapshot_page, response, url)
            target_document = _snapshot_has_target_document(navigation, url)
            if snapshot_page.evaluate(_SNAPSHOT_LOGIN_FORM):
                raise EvaluatorInfrastructureError("Snapshot navigation rendered a login form")
            for spec in specs:
                query = spec["query"]
                try:
                    urls = (
                        _collect_image_urls_sync(snapshot_page, str(query.get("eval_image_class", "")))
                        if target_document
                        else []
                    )
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": urls,
                            "navigation": navigation,
                            "target_document": target_document,
                        }
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract sync page_image_query snapshot for %s",
                        query.get("eval_image_url"),
                        exc_info=True,
                    )
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": [],
                            "error": True,
                        }
                    )
        except Exception:
            logger.warning("Failed to navigate sync page_image_query snapshot URL %s", url, exc_info=True)
            for spec in specs:
                query = spec["query"]
                snapshots.append(
                    {
                        "key": spec["key"],
                        "url": query.get("eval_image_url"),
                        "locator": query.get("eval_image_class"),
                        "image_urls": [],
                        "error": True,
                    }
                )
        finally:
            snapshot_page.close()
    return snapshots


def collect_browser_snapshots_sync(page, plan: dict[str, Any] | None) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    adapters = (plan or {}).get("snapshot_adapters") or {}
    program_targets = (adapters.get("program_html") or {}).get("targets") or []
    if program_targets:
        snapshots["program_html"] = _collect_program_html_snapshots_sync(page, program_targets)
    image_queries = (adapters.get("page_image_query") or {}).get("queries") or []
    if image_queries:
        snapshots["page_image_query"] = _collect_page_image_snapshots_sync(page, image_queries)
    return snapshots


def merge_snapshots(*snapshots: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for snapshot in snapshots:
        for key, value in snapshot.items():
            merged[key] = value
    return merged


def _record_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(record, dict) for record in value):
        raise EvaluatorInfrastructureError(f"Malformed {label} snapshot")
    if any(record.get("error") for record in value):
        raise EvaluatorInfrastructureError(f"Failed {label} snapshot")
    return value


def _validate_snapshots(plan: dict[str, Any], snapshots: dict[str, Any], phase: str) -> None:
    adapters = plan.get("snapshot_adapters") or {}
    for adapter in ("shopping_orders", "shopping_reviews", "program_html", "page_image_query"):
        if adapter not in adapters and adapter not in snapshots:
            continue
        params = adapters.get(adapter) or {}
        if adapter == "shopping_reviews":
            skus = params.get("skus", [])
            value = snapshots.get(adapter)
            if not isinstance(value, dict) or value.get("error") or not set(skus).issubset(value):
                raise EvaluatorInfrastructureError(f"Missing or failed {phase} {adapter} snapshot")
            for sku, records in value.items():
                _record_list(records, f"{phase} {adapter}/{sku}")
            continue
        specs = params.get("targets" if adapter == "program_html" else "queries", [])
        records = _record_list(snapshots.get(adapter), f"{phase} {adapter}")
        if adapter == "shopping_orders":
            continue
        keys = [record.get("key") for record in records]
        expected = {spec["key"] for spec in specs}
        if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys) or not expected.issubset(keys):
            raise EvaluatorInfrastructureError(f"Missing or malformed {phase} {adapter} target keys")
        for record in records:
            _validate_snapshot_navigation(record, phase)
            if adapter == "program_html":
                valid = isinstance(record.get("value"), str)
            else:
                urls = record.get("image_urls")
                valid = isinstance(urls, list) and all(isinstance(url, str) for url in urls)
            if not valid:
                raise EvaluatorInfrastructureError(f"Malformed {phase} {adapter} capture")


def _target_document_records(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Normalize retained navigation-aware captures without mutating raw evidence."""
    normalized = []
    for record in records:
        if "navigation" not in record:
            normalized.append(record)
            continue
        navigation = record["navigation"]
        target_document = _snapshot_has_target_document(navigation, record["url"])
        normalized.append(
            {
                **record,
                field: record[field] if target_document else "" if field == "value" else [],
                "target_document": target_document,
                "target_absent": navigation["status"] in {404, 410},
            }
        )
    return normalized


def build_snapshot_context(
    plan: dict[str, Any] | None,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    # Validate both captures before deriving any positive mutation. An empty
    # successful capture is valid; missing/failed/malformed data is not empty.
    required = dict((plan or {}).get("snapshot_adapters") or {})
    for adapter, collection in (
        ("shopping_reviews", "skus"),
        ("program_html", "targets"),
        ("page_image_query", "queries"),
    ):
        if not (required.get(adapter) or {}).get(collection) and adapter not in before and adapter not in after:
            required.pop(adapter, None)
    for adapter in set(before) | set(after):
        required.setdefault(adapter, {})
    for phase, captures in (("before", before), ("after", after)):
        _validate_snapshots({"snapshot_adapters": required}, captures, phase)
    deltas: dict[str, Any] = {}

    if "shopping_orders" in before or "shopping_orders" in after:
        deltas["shopping_orders"] = diff_records(
            before.get("shopping_orders", []),
            after.get("shopping_orders", []),
            key_fields=("increment_id", "entity_id"),
        )

    if "shopping_reviews" in before or "shopping_reviews" in after:
        review_deltas: dict[str, Any] = {}
        before_reviews = before.get("shopping_reviews", {})
        after_reviews = after.get("shopping_reviews", {})
        for sku in sorted(set(before_reviews) | set(after_reviews)):
            review_deltas[sku] = diff_records(
                before_reviews.get(sku, []),
                after_reviews.get(sku, []),
                key_fields=("id", "created_at", "nickname", "detail"),
            )
        deltas["shopping_reviews"] = review_deltas

    if "program_html" in before or "program_html" in after:
        before_records = _target_document_records(before.get("program_html", []), "value")
        after_records = _target_document_records(after.get("program_html", []), "value")
        program_delta = diff_records(before_records, after_records, key_fields=("key",))
        before_by_key = {record.get("key"): record for record in before_records}
        for bucket in ("added", "changed"):
            for record in program_delta[bucket]:
                before_value = before_by_key.get(record.get("key"), {}).get("value", "")
                record["before_value"] = before_value
                record["delta_value"] = _added_text(before_value, record.get("value", ""))
        deltas["program_html"] = program_delta

    if "page_image_query" in before or "page_image_query" in after:
        before_records = _target_document_records(before.get("page_image_query", []), "image_urls")
        after_records = _target_document_records(after.get("page_image_query", []), "image_urls")
        image_delta = diff_records(before_records, after_records, key_fields=("key",))
        before_by_key = {record.get("key"): set(record.get("image_urls") or []) for record in before_records}
        for bucket in ("added", "changed"):
            for record in image_delta[bucket]:
                before_urls = before_by_key.get(record.get("key"), set())
                record["delta_image_urls"] = [url for url in record.get("image_urls", []) if url not in before_urls]
        deltas["page_image_query"] = image_delta

    return {
        "collision_plan": plan or {},
        "snapshots": {"before": before, "after": after},
        "deltas": deltas,
    }

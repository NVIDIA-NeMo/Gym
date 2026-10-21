# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured before/after snapshots for collision-prone evaluators."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import logging
from collections import defaultdict
from typing import Any

import httpx

from .classic_evaluation import _shopping_get_auth_token, _site_url
from .navigation import goto


logger = logging.getLogger(__name__)


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
    for order in response.json().get("items", []):
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
        for review in response.json():
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
    """Collect all requested snapshots. Failures are logged and skipped."""
    snapshots: dict[str, Any] = {}
    adapters = (plan or {}).get("snapshot_adapters") or {}

    if "shopping_orders" in adapters:
        try:
            snapshots["shopping_orders"] = fetch_shopping_orders()
        except Exception:
            logger.warning("Failed to collect shopping order snapshot", exc_info=True)

    review_params = adapters.get("shopping_reviews")
    if review_params:
        skus = [str(sku) for sku in review_params.get("skus", []) if sku]
        try:
            snapshots["shopping_reviews"] = fetch_shopping_reviews(skus)
        except Exception:
            logger.warning("Failed to collect shopping review snapshot", exc_info=True)

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
            try {
                const el = document.querySelector(selector);
                return el ? el.textContent : '';
            } catch (e) {
                return '';
            }
        }
        """,
            selector,
        )
        or ""
    )


async def _gitlab_member_role(page, account_name: str) -> str:
    try:
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
                """(idx) => document.querySelectorAll("td.col-max-role span")[idx]?.outerText || """,
                account_idx,
            )
        )
    except Exception:
        return ""


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
    except Exception:
        return {}


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
            return str(await page.evaluate(f"() => {locator}") or "")
        except Exception:
            return ""
    if locator.startswith("lambda:"):
        try:
            return str(await page.evaluate(locator.removeprefix("lambda:")) or "")
        except Exception:
            return ""
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
            await goto(snapshot_page, url, wait_until="domcontentloaded", timeout=60000)
            for spec in specs:
                target = spec["target"]
                try:
                    value = await _select_program_html_snapshot(target, snapshot_page)
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": value,
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
    except Exception:
        return []


async def _collect_page_image_snapshots(page, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in queries:
        specs_by_url[str(spec["query"].get("eval_image_url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = await page.context.new_page()
        try:
            await goto(snapshot_page, url, wait_until="domcontentloaded", timeout=60000)
            for spec in specs:
                query = spec["query"]
                try:
                    urls = await _collect_image_urls(snapshot_page, str(query.get("eval_image_class", "")))
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": urls,
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
            try {
                const el = document.querySelector(selector);
                return el ? el.textContent : '';
            } catch (e) {
                return '';
            }
        }
        """,
            selector,
        )
        or ""
    )


def _gitlab_member_role_sync(page, account_name: str) -> str:
    try:
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
                """(idx) => document.querySelectorAll("td.col-max-role span")[idx]?.outerText || """,
                account_idx,
            )
        )
    except Exception:
        return ""


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
    except Exception:
        return {}


def _select_program_html_snapshot_sync(target: dict[str, Any], page) -> str:
    locator = str(target.get("locator", ""))
    if not locator.strip():
        return str(page.content())
    if locator.startswith("document.") or locator.startswith("[...document."):
        try:
            return str(page.evaluate(f"() => {locator}") or "")
        except Exception:
            return ""
    if locator.startswith("lambda:"):
        try:
            return str(page.evaluate(locator.removeprefix("lambda:")) or "")
        except Exception:
            return ""
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
            snapshot_page.goto(url, wait_until="domcontentloaded", timeout=60000)
            for spec in specs:
                target = spec["target"]
                try:
                    value = _select_program_html_snapshot_sync(target, snapshot_page)
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": value,
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
    except Exception:
        return []


def _collect_page_image_snapshots_sync(page, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in queries:
        specs_by_url[str(spec["query"].get("eval_image_url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = page.context.new_page()
        try:
            snapshot_page.goto(url, wait_until="domcontentloaded", timeout=60000)
            for spec in specs:
                query = spec["query"]
                try:
                    urls = _collect_image_urls_sync(snapshot_page, str(query.get("eval_image_class", "")))
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": urls,
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


def build_snapshot_context(
    plan: dict[str, Any] | None,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
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
        before_records = before.get("program_html", [])
        after_records = after.get("program_html", [])
        program_delta = diff_records(before_records, after_records, key_fields=("key",))
        before_by_key = {record.get("key"): record for record in before_records}
        for bucket in ("added", "changed"):
            for record in program_delta[bucket]:
                before_value = before_by_key.get(record.get("key"), {}).get("value", "")
                record["before_value"] = before_value
                record["delta_value"] = _added_text(before_value, record.get("value", ""))
        deltas["program_html"] = program_delta

    if "page_image_query" in before or "page_image_query" in after:
        before_records = before.get("page_image_query", [])
        after_records = after.get("page_image_query", [])
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

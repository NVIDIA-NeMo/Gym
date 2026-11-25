# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured before/after snapshots for collision-prone evaluations."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import inspect
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from .evaluation import (
    EvaluationConfig,
    HelperConfig,
    Request,
    _eval_helper_expression,
    _navigate,
    _request_helper_json,
    _shopping_auth_token,
    _site_url,
    default_evaluation_config,
)


logger = logging.getLogger(__name__)

OrderFetcher = Callable[[], list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]]
ReviewFetcher = Callable[
    [list[str]],
    dict[str, list[dict[str, Any]]] | Awaitable[dict[str, list[dict[str, Any]]]],
]


def _stable_key(record: dict[str, Any], preferred_keys: tuple[str, ...]) -> str:
    for key in preferred_keys:
        if (value := record.get(key)) not in (None, ""):
            return str(value)
    payload = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def diff_records(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    before_by_key = {_stable_key(record, key_fields): record for record in before}
    after_by_key = {_stable_key(record, key_fields): record for record in after}
    return {
        "added": [
            {**record, "_snapshot_key": key} for key, record in after_by_key.items() if key not in before_by_key
        ],
        "changed": [
            {**record, "_snapshot_key": key}
            for key, record in after_by_key.items()
            if key in before_by_key and record != before_by_key[key]
        ],
        "removed": [
            {**record, "_snapshot_key": key} for key, record in before_by_key.items() if key not in after_by_key
        ],
    }


def _added_text(before: Any, after: Any) -> str:
    return "\n".join(
        line[2:]
        for line in difflib.ndiff(str(before or "").splitlines(), str(after or "").splitlines())
        if line.startswith("+ ") and line[2:].strip()
    )


async def _shopping_headers(config: HelperConfig, request_fn: Request) -> dict[str, str]:
    token = await _shopping_auth_token(config, request_fn)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _order_url(order: dict[str, Any], config: HelperConfig) -> str:
    order_id = order.get("increment_id") or order.get("entity_id") or order.get("order_id")
    if not order_id:
        return ""
    return f"{_site_url(config, 'shopping')}/sales/order/view/order_id/{int(order_id)}/"


async def fetch_shopping_orders(
    config: HelperConfig,
    request_fn: Request,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    data = await _request_helper_json(
        "GET",
        f"{_site_url(config, 'shopping')}/rest/V1/orders",
        config=config,
        request_fn=request_fn,
        headers=await _shopping_headers(config, request_fn),
        params={
            "searchCriteria[sortOrders][0][field]": "created_at",
            "searchCriteria[sortOrders][0][direction]": "DESC",
            "searchCriteria[pageSize]": str(page_size),
        },
    )
    return [
        {
            "entity_id": order.get("entity_id"),
            "increment_id": order.get("increment_id"),
            "created_at": order.get("created_at"),
            "status": order.get("status"),
            "grand_total": order.get("grand_total"),
            "url": _order_url(order, config),
            "items": [
                {
                    "name": item.get("name"),
                    "sku": item.get("sku"),
                    "qty_ordered": item.get("qty_ordered"),
                }
                for item in order.get("items") or []
            ],
        }
        for order in data.get("items", [])
    ]


async def fetch_shopping_reviews(
    skus: list[str],
    config: HelperConfig,
    request_fn: Request,
) -> dict[str, list[dict[str, Any]]]:
    headers = await _shopping_headers(config, request_fn)
    reviews_by_sku: dict[str, list[dict[str, Any]]] = {}
    for sku in skus:
        reviews = await _request_helper_json(
            "GET",
            f"{_site_url(config, 'shopping')}/rest/V1/products/{sku}/reviews",
            config=config,
            request_fn=request_fn,
            headers=headers,
        )
        records = []
        for review in reviews:
            first_rating = (review.get("ratings") or [{}])[0]
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


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _collect_snapshots_async(
    plan: dict[str, Any] | None,
    *,
    config: EvaluationConfig,
    fetch_orders: OrderFetcher | None = None,
    fetch_reviews: ReviewFetcher | None = None,
) -> dict[str, Any]:
    """Collect API snapshots; failures are logged and omitted."""
    snapshots: dict[str, Any] = {}
    adapters = (plan or {}).get("snapshot_adapters") or {}
    if "shopping_orders" in adapters:
        try:
            if fetch_orders is not None:
                snapshots["shopping_orders"] = await _resolve(fetch_orders())
            else:
                if config.helper_config is None:
                    raise RuntimeError("helper_config is required for shopping snapshots")
                snapshots["shopping_orders"] = await fetch_shopping_orders(
                    config.helper_config,
                    config.request_fn,
                )
        except Exception:
            logger.warning("Failed to collect shopping order snapshot", exc_info=True)

    if review_params := adapters.get("shopping_reviews"):
        skus = [str(sku) for sku in review_params.get("skus", []) if sku]
        try:
            if fetch_reviews is not None:
                snapshots["shopping_reviews"] = await _resolve(fetch_reviews(skus))
            else:
                if config.helper_config is None:
                    raise RuntimeError("helper_config is required for shopping snapshots")
                snapshots["shopping_reviews"] = await fetch_shopping_reviews(
                    skus,
                    config.helper_config,
                    config.request_fn,
                )
        except Exception:
            logger.warning("Failed to collect shopping review snapshot", exc_info=True)
    return snapshots


def collect_snapshots(
    plan: dict[str, Any] | None,
    *,
    config: EvaluationConfig | None = None,
    fetch_orders: OrderFetcher | None = None,
    fetch_reviews: ReviewFetcher | None = None,
) -> Any:
    """Collect API snapshots from sync workers or async callers.

    Browser workers call this function synchronously between event-loop turns;
    async tests and integrations may await the returned coroutine.
    """
    operation = _collect_snapshots_async(
        plan,
        config=config or default_evaluation_config(),
        fetch_orders=fetch_orders,
        fetch_reviews=fetch_reviews,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.run(operation)
        return loop.run_until_complete(operation)
    return operation


async def _select_program_html_snapshot(
    target: dict[str, Any],
    page: Any,
    config: EvaluationConfig,
) -> str:
    locator = str(target.get("locator", ""))
    if not locator.strip():
        return str(await _resolve(page.content()))
    if locator.startswith(("document.", "[...document.")):
        try:
            return str(await _resolve(page.evaluate(f"() => {locator}")) or "")
        except Exception:
            return ""
    if locator.startswith("lambda:"):
        try:
            return str(await _resolve(page.evaluate(locator.removeprefix("lambda:"))) or "")
        except Exception:
            return ""
    if locator.startswith("func:"):
        return str(await _eval_helper_expression(locator, page, config))
    raise ValueError(f"Unsupported program_html snapshot locator: {locator}")


async def _collect_program_html_snapshots(
    page: Any,
    targets: list[dict[str, Any]],
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in targets:
        specs_by_url[str(spec["target"].get("url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = await _resolve(page.context.new_page())
        try:
            await _navigate(snapshot_page, url, config)
            for spec in specs:
                target = spec["target"]
                try:
                    value = await _select_program_html_snapshot(target, snapshot_page, config)
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": target.get("url"),
                            "locator": target.get("locator"),
                            "value": value,
                        }
                    )
                except Exception:
                    logger.warning("Failed to extract program_html snapshot for %s", url, exc_info=True)
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
            snapshots.extend(
                {
                    "key": spec["key"],
                    "url": spec["target"].get("url"),
                    "locator": spec["target"].get("locator"),
                    "value": "",
                    "error": True,
                }
                for spec in specs
            )
        finally:
            await _resolve(snapshot_page.close())
    return snapshots


async def _collect_image_urls(page: Any, locator: str) -> list[str]:
    selector = locator if locator.strip() else "img"
    try:
        result = page.evaluate(
            """
            (selector) => {
                const root = selector ? document.querySelector(selector) : document;
                if (!root) return [];
                const elements = root.matches && root.matches("img,a")
                    ? [root] : Array.from(root.querySelectorAll("img,a"));
                return elements.map((el) => {
                    if (el.tagName.toLowerCase() === "a") return el.href || "";
                    return el.currentSrc || el.src || "";
                }).filter(Boolean);
            }
            """,
            selector,
        )
        return list(await _resolve(result) or [])
    except Exception:
        return []


async def _collect_page_image_snapshots(
    page: Any,
    queries: list[dict[str, Any]],
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    snapshots = []
    specs_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in queries:
        specs_by_url[str(spec["query"].get("eval_image_url", ""))].append(spec)

    for url, specs in specs_by_url.items():
        snapshot_page = await _resolve(page.context.new_page())
        try:
            await _navigate(snapshot_page, url, config)
            for spec in specs:
                query = spec["query"]
                try:
                    image_urls = await _collect_image_urls(
                        snapshot_page,
                        str(query.get("eval_image_class", "")),
                    )
                    snapshots.append(
                        {
                            "key": spec["key"],
                            "url": query.get("eval_image_url"),
                            "locator": query.get("eval_image_class"),
                            "image_urls": image_urls,
                        }
                    )
                except Exception:
                    logger.warning("Failed to extract page image snapshot for %s", url, exc_info=True)
        except Exception:
            logger.warning("Failed to navigate page image snapshot URL %s", url, exc_info=True)
            snapshots.extend(
                {
                    "key": spec["key"],
                    "url": spec["query"].get("eval_image_url"),
                    "locator": spec["query"].get("eval_image_class"),
                    "image_urls": [],
                    "error": True,
                }
                for spec in specs
            )
        finally:
            await _resolve(snapshot_page.close())
    return snapshots


async def collect_browser_snapshots(
    page: Any,
    plan: dict[str, Any] | None,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    config = config or default_evaluation_config()
    snapshots: dict[str, Any] = {}
    adapters = (plan or {}).get("snapshot_adapters") or {}
    if targets := (adapters.get("program_html") or {}).get("targets"):
        snapshots["program_html"] = await _collect_program_html_snapshots(page, targets, config)
    if queries := (adapters.get("page_image_query") or {}).get("queries"):
        snapshots["page_image_query"] = await _collect_page_image_snapshots(page, queries, config)
    return snapshots


def merge_snapshots(*snapshots: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for snapshot in snapshots:
        merged.update(snapshot)
    return merged


def build_snapshot_context(
    first: dict[str, Any] | None,
    second: dict[str, Any],
    third: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build deltas, accepting both source and worker argument ordering."""
    if first and "snapshot_adapters" in first:
        plan, before, after = first, second, third or {}
    else:
        before, after, plan = first or {}, second, third
    deltas: dict[str, Any] = {}
    if "shopping_orders" in before or "shopping_orders" in after:
        deltas["shopping_orders"] = diff_records(
            before.get("shopping_orders", []),
            after.get("shopping_orders", []),
            key_fields=("increment_id", "entity_id"),
        )

    if "shopping_reviews" in before or "shopping_reviews" in after:
        review_deltas = {}
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
        delta = diff_records(before_records, after.get("program_html", []), key_fields=("key",))
        before_by_key = {record.get("key"): record for record in before_records}
        for bucket in ("added", "changed"):
            for record in delta[bucket]:
                before_value = before_by_key.get(record.get("key"), {}).get("value", "")
                record["before_value"] = before_value
                record["delta_value"] = _added_text(before_value, record.get("value", ""))
        deltas["program_html"] = delta

    if "page_image_query" in before or "page_image_query" in after:
        before_records = before.get("page_image_query", [])
        delta = diff_records(before_records, after.get("page_image_query", []), key_fields=("key",))
        before_urls = {record.get("key"): set(record.get("image_urls") or []) for record in before_records}
        for bucket in ("added", "changed"):
            for record in delta[bucket]:
                old_urls = before_urls.get(record.get("key"), set())
                record["delta_image_urls"] = [url for url in record.get("image_urls", []) if url not in old_urls]
        deltas["page_image_query"] = delta

    return {
        "collision_plan": plan or {},
        "snapshots": {"before": before, "after": after},
        "deltas": deltas,
    }

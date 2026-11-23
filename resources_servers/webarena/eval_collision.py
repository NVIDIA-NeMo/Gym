"""Collision planning for concurrent Classic WebArena evaluations."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from typing import Any


HIGH_RISK_HELPERS = {
    "shopping_get_latest_order_url",
    "shopping_get_sku_latest_review_author",
    "shopping_get_sku_latest_review_rating",
    "shopping_admin_get_cart_price_rule",
}

REVIEW_HELPERS = {
    "shopping_get_sku_latest_review_author",
    "shopping_get_sku_latest_review_rating",
}


def _literal_arg(node: ast.AST) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def extract_helper_calls(expr: Any) -> list[dict[str, Any]]:
    """Return calls in a ``func:`` expression without evaluating the expression."""
    if not isinstance(expr, str) or not expr.startswith("func:"):
        return []
    try:
        tree = ast.parse(expr.removeprefix("func:"), mode="eval")
    except SyntaxError:
        return []

    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        calls.append({"name": name, "args": [_literal_arg(arg) for arg in node.args]})
    return calls


def _add_snapshot(plan: dict[str, Any], adapter: str, **params: Any) -> None:
    existing = plan.setdefault("snapshot_adapters", {}).setdefault(adapter, {})
    for key, value in params.items():
        if not isinstance(value, list):
            existing[key] = value
            continue
        bucket = existing.setdefault(key, [])
        for item in value:
            if item is not None and item not in bucket:
                bucket.append(item)


def _plan_helper_call(plan: dict[str, Any], call: dict[str, Any]) -> None:
    name = call["name"]
    args = call.get("args") or []
    if name == "shopping_get_latest_order_url":
        _add_snapshot(plan, "shopping_orders")
    elif name in REVIEW_HELPERS:
        sku = str(args[0]) if args and args[0] is not None else None
        _add_snapshot(plan, "shopping_reviews", skus=[sku] if sku else [])
    # Cart price rules intentionally have no snapshot adapter.


def snapshot_target_key(target: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "url": target.get("url"),
            "locator": target.get("locator"),
            "required_contents": target.get("required_contents"),
            "eval_image_url": target.get("eval_image_url"),
            "eval_image_class": target.get("eval_image_class"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _is_fixed_url(value: Any) -> bool:
    return isinstance(value, str) and value != "last" and not value.startswith("func:")


def _supports_program_html_snapshot(target: dict[str, Any]) -> bool:
    if not _is_fixed_url(target.get("url")):
        return False
    locator = str(target.get("locator", ""))
    return (
        not locator.strip()
        or locator.startswith(("document.", "[...document.", "lambda:"))
        or locator.startswith("func:get_query_text(")
        or locator.startswith("func:get_query_text_lowercase(")
        or locator.startswith("func:gitlab_get_project_memeber_role(")
        or locator.startswith("func:reddit_get_latest_comment_content_by_username(")
        or locator.startswith("func:reddit_get_parent_comment_username_of_latest_comment_by_username(")
    )


def _supports_page_image_snapshot(query: dict[str, Any]) -> bool:
    return _is_fixed_url(query.get("eval_image_url"))


def _add_program_html_snapshot(plan: dict[str, Any], target: dict[str, Any]) -> None:
    spec = {"key": snapshot_target_key(target), "target": target}
    targets = plan.setdefault("snapshot_adapters", {}).setdefault("program_html", {}).setdefault("targets", [])
    if all(item["key"] != spec["key"] for item in targets):
        targets.append(spec)


def _add_page_image_snapshot(plan: dict[str, Any], query: dict[str, Any]) -> None:
    spec = {"key": snapshot_target_key(query), "query": query}
    queries = plan.setdefault("snapshot_adapters", {}).setdefault("page_image_query", {}).setdefault("queries", [])
    if all(item["key"] != spec["key"] for item in queries):
        queries.append(spec)


def _program_html_collision_key(target: dict[str, Any]) -> str | None:
    if not _supports_program_html_snapshot(target):
        return None
    url = str(target.get("url") or "")
    match = re.match(r"([^/]+://[^/]+|__GITLAB__)/([^/]+/[^/]+)/-/(?:raw|blob)/([^/]+)/", url)
    if match:
        _, repo, branch = match.groups()
        return f"gitlab-file-branch:{repo}:{branch}"
    if re.match(r"([^/]+://[^/]+|__GITLAB__)/byteblaze/?$", url):
        locator = str(target.get("locator") or "")
        if "cover-status" in locator:
            return "gitlab-profile:byteblaze:status"
        if 'itemprop="url"' in locator or "itemprop='url'" in locator:
            return "gitlab-profile:byteblaze:homepage"
    return f"url:{url}"


def build_collision_plan(
    task_config: dict[str, Any],
    *,
    program_html_collision_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Derive snapshot adapters from a task's evaluator config."""
    plan: dict[str, Any] = {"snapshot_adapters": {}, "target_overrides": {}}
    eval_config = task_config.get("eval") or {}

    for target in eval_config.get("program_html") or []:
        for field in ("url", "locator"):
            for call in extract_helper_calls(target.get(field)):
                if call["name"] in HIGH_RISK_HELPERS:
                    _plan_helper_call(plan, call)
        key = _program_html_collision_key(target)
        if key and program_html_collision_keys and key in program_html_collision_keys:
            _add_program_html_snapshot(plan, target)

    for query in eval_config.get("page_image_query") or []:
        if _supports_page_image_snapshot(query):
            _add_page_image_snapshot(plan, query)

    for params in plan["snapshot_adapters"].values():
        if isinstance(params.get("skus"), list):
            params["skus"].sort()
    return plan


def build_collision_plans(task_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build plans using the selected tasks to identify shared mutable targets."""
    task_ids_by_key: dict[str, set[Any]] = defaultdict(set)
    for task in task_configs:
        task_id = task.get("id", task.get("task_id"))
        for target in (task.get("eval") or {}).get("program_html") or []:
            if key := _program_html_collision_key(target):
                task_ids_by_key[key].add(task_id)
    collisions = {key for key, task_ids in task_ids_by_key.items() if len(task_ids) > 1}
    return [build_collision_plan(task, program_html_collision_keys=collisions) for task in task_configs]


def has_collision_mitigation(plan: dict[str, Any] | None) -> bool:
    return bool(plan and plan.get("snapshot_adapters"))

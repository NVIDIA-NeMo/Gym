# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure collision planning shared by WebArena preparation and scoring."""

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
    "shopping_get_sku_latest_review_text",
    "shopping_admin_get_cart_price_rule",
}

REVIEW_HELPERS = {
    "shopping_get_sku_latest_review_author",
    "shopping_get_sku_latest_review_rating",
    "shopping_get_sku_latest_review_text",
}


def _literal_arg(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def extract_helper_calls(expr: Any) -> list[dict[str, Any]]:
    """Return helper calls from a `func:` expression without evaluating it."""
    if not isinstance(expr, str) or not expr.startswith("func:"):
        return []
    source = expr.split("func:", 1)[1]
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return []

    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        calls.append(
            {
                "name": name,
                "args": [_literal_arg(arg) for arg in node.args],
            }
        )
    return calls


def _add_snapshot(plan: dict[str, Any], adapter: str, **params: Any) -> None:
    adapters = plan.setdefault("snapshot_adapters", {})
    existing = adapters.setdefault(adapter, {})
    for key, value in params.items():
        if isinstance(value, list):
            existing.setdefault(key, [])
            for item in value:
                if item is not None and item not in existing[key]:
                    existing[key].append(item)
        else:
            existing[key] = value


def _plan_helper_call(plan: dict[str, Any], call: dict[str, Any]) -> None:
    name = call["name"]
    args = call.get("args") or []
    if name == "shopping_get_latest_order_url":
        _add_snapshot(plan, "shopping_orders")
    elif name in REVIEW_HELPERS:
        sku = str(args[0]) if args and args[0] is not None else None
        _add_snapshot(plan, "shopping_reviews", skus=[sku] if sku else [])
    elif name == "shopping_admin_get_cart_price_rule":
        # No snapshot adapter exists for cart price rules yet, so do not
        # add snapshot mitigation. Existing evaluator behavior is preserved.
        return


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
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    target_spec = {
        "key": snapshot_target_key(target),
        "target": target,
    }
    existing = plan.setdefault("snapshot_adapters", {}).setdefault("program_html", {}).setdefault("targets", [])
    if all(item["key"] != target_spec["key"] for item in existing):
        existing.append(target_spec)


def _add_page_image_snapshot(plan: dict[str, Any], query: dict[str, Any]) -> None:
    query_spec = {
        "key": snapshot_target_key(query),
        "query": query,
    }
    existing = plan.setdefault("snapshot_adapters", {}).setdefault("page_image_query", {}).setdefault("queries", [])
    if all(item["key"] != query_spec["key"] for item in existing):
        existing.append(query_spec)


def _program_html_collision_key(target: dict[str, Any]) -> str | None:
    if not _supports_program_html_snapshot(target):
        return None
    url = str(target.get("url") or "")
    gitlab_file_match = re.match(
        r"([^/]+://[^/]+|__GITLAB__)/([^/]+/[^/]+)/-/(?:raw|blob)/([^/]+)/",
        url,
    )
    if gitlab_file_match:
        _, repo, branch = gitlab_file_match.groups()
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
    plan: dict[str, Any] = {
        "snapshot_adapters": {},
        "target_overrides": {},
    }
    eval_config = task_config.get("eval") or {}

    for target in eval_config.get("program_html") or []:
        for field in ("url", "locator"):
            for call in extract_helper_calls(target.get(field)):
                if call["name"] in HIGH_RISK_HELPERS:
                    _plan_helper_call(plan, call)
        collision_key = _program_html_collision_key(target)
        if collision_key and program_html_collision_keys and collision_key in program_html_collision_keys:
            _add_program_html_snapshot(plan, target)

    for query in eval_config.get("page_image_query") or []:
        if _supports_page_image_snapshot(query):
            _add_page_image_snapshot(plan, query)

    for adapter_params in plan["snapshot_adapters"].values():
        if isinstance(adapter_params.get("skus"), list):
            adapter_params["skus"].sort()
    return plan


def build_collision_plans(task_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build collision plans using the selected benchmark tasks as context.

    Fixed-URL program_html snapshots are only enabled when more than one task in
    the selected set checks the same mutable target. Most targets use the URL as
    their collision key; GitLab raw/blob file targets share a key by repo and
    branch because commits to different files on the same branch can conflict.
    This avoids snapshot overhead and false negatives for unique mutation tasks
    while keeping protection for genuinely shared mutable surfaces.
    """
    task_ids_by_program_key: dict[str, set[Any]] = defaultdict(set)
    for task_config in task_configs:
        task_id = task_config.get("id", task_config.get("task_id"))
        eval_config = task_config.get("eval") or {}
        for target in eval_config.get("program_html") or []:
            collision_key = _program_html_collision_key(target)
            if collision_key:
                task_ids_by_program_key[collision_key].add(task_id)

    program_html_collision_keys = {key for key, task_ids in task_ids_by_program_key.items() if len(task_ids) > 1}

    return [
        build_collision_plan(
            task_config,
            program_html_collision_keys=program_html_collision_keys,
        )
        for task_config in task_configs
    ]


def has_collision_mitigation(plan: dict[str, Any] | None) -> bool:
    if not plan:
        return False
    return bool(plan.get("snapshot_adapters"))

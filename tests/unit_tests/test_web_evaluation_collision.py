# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.web.evaluation_collision import (
    _add_snapshot,
    _program_html_collision_key,
    _supports_page_image_snapshot,
    _supports_program_html_snapshot,
    build_collision_plan,
    build_collision_plans,
    extract_helper_calls,
    has_collision_mitigation,
    snapshot_target_key,
)


def test_extract_helper_calls_is_safe_and_preserves_only_literal_args() -> None:
    assert extract_helper_calls(None) == []
    assert extract_helper_calls("plain text") == []
    assert extract_helper_calls("func:not valid Python (") == []
    assert extract_helper_calls("func:(lambda: 1)()") == []

    calls = extract_helper_calls("func:helpers.shopping_get_sku_latest_review_text('sku-1', dynamic_value)")
    assert calls == [
        {
            "name": "shopping_get_sku_latest_review_text",
            "args": ["sku-1", None],
        }
    ]


def test_add_snapshot_merges_lists_without_duplicates_or_none() -> None:
    plan = {}
    _add_snapshot(plan, "shopping_reviews", skus=["b", None, "a"])
    _add_snapshot(plan, "shopping_reviews", skus=["a", "c"], mode="before")

    assert plan == {
        "snapshot_adapters": {
            "shopping_reviews": {
                "skus": ["b", "a", "c"],
                "mode": "before",
            }
        }
    }


def test_snapshot_target_key_is_stable_and_uses_only_evaluator_identity_fields() -> None:
    target = {
        "url": "https://example.test/page",
        "locator": "document.body.innerText",
        "required_contents": {"must_include": ["saved"]},
        "eval_image_url": None,
        "eval_image_class": None,
        "ignored": "first",
    }
    first = snapshot_target_key(target)
    second = snapshot_target_key(target | {"ignored": "second"})

    assert first == second
    assert len(first) == 16


def test_snapshot_support_requires_fixed_urls_and_supported_locators() -> None:
    assert not _supports_program_html_snapshot({"url": "last"})
    assert not _supports_program_html_snapshot({"url": "func:latest_url()"})
    assert not _supports_program_html_snapshot({"url": "https://example.test", "locator": "unsupported()"})
    for locator in (
        "",
        "document.body.innerText",
        "[...document.querySelectorAll('a')]",
        "lambda: document.title",
        "func:get_query_text('x')",
        "func:get_query_text_lowercase('x')",
        "func:gitlab_get_project_memeber_role('x')",
        "func:reddit_get_latest_comment_content_by_username('x')",
        "func:reddit_get_parent_comment_username_of_latest_comment_by_username('x')",
    ):
        assert _supports_program_html_snapshot({"url": "https://example.test", "locator": locator})

    assert _supports_page_image_snapshot({"eval_image_url": "https://example.test/a.png"})
    assert not _supports_page_image_snapshot({"eval_image_url": "last"})


def test_program_html_collision_keys_group_mutable_surfaces() -> None:
    assert (
        _program_html_collision_key(
            {
                "url": "__GITLAB__/group/repo/-/blob/main/README.md",
                "locator": "document.body.innerText",
            }
        )
        == "gitlab-file-branch:group/repo:main"
    )
    assert (
        _program_html_collision_key(
            {
                "url": "https://gitlab.test/group/repo/-/raw/main/config.yml",
                "locator": "document.body.innerText",
            }
        )
        == "gitlab-file-branch:group/repo:main"
    )
    assert (
        _program_html_collision_key(
            {"url": "__GITLAB__/byteblaze", "locator": "document.querySelector('.cover-status')"}
        )
        == "gitlab-profile:byteblaze:status"
    )
    assert (
        _program_html_collision_key(
            {"url": "__GITLAB__/byteblaze/", "locator": "document.querySelector('[itemprop=\"url\"]')"}
        )
        == "gitlab-profile:byteblaze:homepage"
    )
    assert (
        _program_html_collision_key({"url": "https://example.test/settings", "locator": "document.body.innerText"})
        == "url:https://example.test/settings"
    )
    assert _program_html_collision_key({"url": "last"}) is None


def test_build_collision_plan_collects_helper_and_image_snapshots() -> None:
    shared_target = {
        "url": "https://shop.test/orders",
        "locator": "document.body.innerText",
        "required_contents": {"must_include": ["order"]},
    }
    task = {
        "id": 1,
        "eval": {
            "program_html": [
                {
                    "url": "func:shopping_get_latest_order_url()",
                    "locator": "document.body.innerText",
                },
                {
                    "url": "https://shop.test/reviews",
                    "locator": (
                        "func:shopping_get_sku_latest_review_text('sku-z') + "
                        "shopping_get_sku_latest_review_rating('sku-a') + "
                        "shopping_get_sku_latest_review_author('sku-z')"
                    ),
                },
                {
                    "url": "func:shopping_admin_get_cart_price_rule('rule')",
                    "locator": "document.body.innerText",
                },
                shared_target,
            ],
            "page_image_query": [
                {"eval_image_url": "https://example.test/reference.png", "eval_image_class": "hero"},
                {"eval_image_url": "last"},
            ],
        },
    }

    plan = build_collision_plan(task, program_html_collision_keys={"url:https://shop.test/orders"})

    adapters = plan["snapshot_adapters"]
    assert adapters["shopping_orders"] == {}
    assert adapters["shopping_reviews"]["skus"] == ["sku-a", "sku-z"]
    assert len(adapters["program_html"]["targets"]) == 1
    assert adapters["program_html"]["targets"][0]["target"] == shared_target
    assert len(adapters["page_image_query"]["queries"]) == 1
    assert plan["target_overrides"] == {}
    assert has_collision_mitigation(plan)


def test_build_collision_plan_deduplicates_snapshot_specs() -> None:
    target = {"url": "https://example.test/shared", "locator": "document.body.innerText"}
    query = {"eval_image_url": "https://example.test/reference.png"}
    plan = build_collision_plan(
        {"eval": {"program_html": [target, target], "page_image_query": [query, query]}},
        program_html_collision_keys={"url:https://example.test/shared"},
    )

    assert len(plan["snapshot_adapters"]["program_html"]["targets"]) == 1
    assert len(plan["snapshot_adapters"]["page_image_query"]["queries"]) == 1


def test_build_collision_plans_only_snapshots_cross_task_collisions() -> None:
    shared = {"url": "https://example.test/shared", "locator": "document.body.innerText"}
    unique = {"url": "https://example.test/unique", "locator": "document.body.innerText"}
    plans = build_collision_plans(
        [
            {"id": "a", "eval": {"program_html": [shared, unique]}},
            {"task_id": "b", "eval": {"program_html": [shared]}},
            {"id": "a", "eval": {"program_html": [unique]}},
        ]
    )

    assert [
        [entry["target"]["url"] for entry in plan["snapshot_adapters"].get("program_html", {}).get("targets", [])]
        for plan in plans
    ] == [
        ["https://example.test/shared"],
        ["https://example.test/shared"],
        [],
    ]


def test_has_collision_mitigation_handles_empty_plans() -> None:
    assert not has_collision_mitigation(None)
    assert not has_collision_mitigation({})
    assert not has_collision_mitigation({"snapshot_adapters": {}})

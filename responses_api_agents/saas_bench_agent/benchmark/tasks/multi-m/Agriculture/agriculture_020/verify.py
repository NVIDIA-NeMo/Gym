#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_020/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Verifier for agriculture_020: exact Recipya prerequisite -> Grocy plan."""

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile


GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
RECIPYA_CONTAINER = os.getenv("RECIPYA_CONTAINER")
GROCY_DB = "/config/data/grocy.db"
RECIPYA_DB = "/root/.config/Recipya/Database/recipya.db"
RECIPE_NAME = "Layered Zucchini Casserole"
TASK_NOTE = "Bistrot Provençal menu expansion"
TARGET_STOCK = 5.0
TARGETS = ("zucchini", "eggplant", "onion", "mushrooms", "fresh tomatoes")

for name, value in [
    ("GROCY_CONTAINER", GROCY_CONTAINER),
    ("RECIPYA_CONTAINER", RECIPYA_CONTAINER),
]:
    if not value:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)


_checks: list[tuple[str, int, bool, str]] = []
_recipe_ok = False
_selection_ok = False
_selected_products: dict[str, dict] = {}
_expected: dict[int, float] = {}
_relevant_rows: list[dict] = []
_recipya_db_copy: str | None = None


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


def docker_exec(container: str, *args: str, timeout: int = 20) -> tuple[int, str, str]:
    result = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def recipya_query(sql: str) -> list[dict]:
    global _recipya_db_copy
    if _recipya_db_copy is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        _recipya_db_copy = tmp.name
        tmp.close()
        result = subprocess.run(
            ["docker", "cp", f"{RECIPYA_CONTAINER}:{RECIPYA_DB}", _recipya_db_copy],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Recipya DB copy failed: {result.stderr.strip()}")
        for suffix in ("-wal", "-shm"):
            subprocess.run(
                ["docker", "cp", f"{RECIPYA_CONTAINER}:{RECIPYA_DB}{suffix}",
                 _recipya_db_copy + suffix],
                capture_output=True,
                text=True,
                timeout=30,
            )
    connection = sqlite3.connect(_recipya_db_copy)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def grocy_query(sql: str) -> list[dict]:
    php = (
        '$db=new PDO("sqlite:' + GROCY_DB + '");'
        "$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);"
        "$rows=$db->query(" + json.dumps(sql) + ")->fetchAll(PDO::FETCH_ASSOC);"
        "echo json_encode($rows);"
    )
    rc, stdout, stderr = docker_exec(GROCY_CONTAINER, "php", "-r", php)
    if rc != 0:
        raise RuntimeError(f"Grocy query failed: {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def direct_target(value: str) -> str | None:
    words = set(re.findall(r"[a-z]+", (value or "").lower()))
    prepared = {
        "balsamic", "bread", "casserole", "chip", "chips", "chutney", "cracker",
        "crackers", "crisp", "crisps", "cruncher", "dip", "dish", "fried",
        "hummus", "meal", "mix",
        "noodle", "noodles", "pasta", "pickle", "pickled", "pizza", "risotto",
        "sauce", "snack", "soup", "spread", "stir", "wok",
    }
    if words & prepared:
        return None
    if words & {"zucchini", "zucchinis", "courgette", "courgettes"}:
        return "zucchini"
    if words & {"eggplant", "eggplants", "aubergine", "aubergines"}:
        return "eggplant"
    if words & {"onion", "onions"}:
        if not words & {
            "powder", "ring", "rings", "scallion", "scallions", "seasoning", "spring",
        }:
            return "onion"
    if words & {"mushroom", "mushrooms"}:
        if not words & {"cream", "canned"}:
            return "mushrooms"
    if words & {"tomato", "tomatoes"}:
        if not words & {
            "paste", "sauce", "juice", "ketchup", "canned", "diced", "powder",
            "soup", "stewed", "puree", "pureed",
        }:
            return "fresh tomatoes"
    return None


def check_1_complete_recipya_prerequisite() -> None:
    global _recipe_ok
    try:
        recipes = recipya_query(
            "SELECT r.id,r.name FROM recipes r "
            "JOIN user_recipe ur ON ur.recipe_id=r.id "
            "JOIN users u ON u.id=ur.user_id "
            "WHERE r.name='Layered Zucchini Casserole' "
            "AND LOWER(u.email)='admin@recipya.com' ORDER BY r.id"
        )
        if len(recipes) != 1:
            check("1. complete exact Recipya recipe prerequisite", 5, False,
                  f"expected one exact admin recipe, found {len(recipes)}")
            return
        recipe_id = int(recipes[0]["id"])
        ingredients = recipya_query(
            "SELECT i.name FROM ingredient_recipe ir "
            "JOIN ingredients i ON i.id=ir.ingredient_id "
            f"WHERE ir.recipe_id={recipe_id} ORDER BY ir.ingredient_order"
        )
        found = {direct_target(row.get("name") or "") for row in ingredients}
        found.discard(None)
        missing = sorted(set(TARGETS) - found)
        _recipe_ok = not missing
        check(
            "1. complete exact Recipya recipe prerequisite",
            5,
            _recipe_ok,
            f"recipe_id={recipe_id}; all five direct vegetables present"
            if _recipe_ok else "missing direct vegetables: " + ", ".join(missing),
        )
    except Exception as exc:
        check("1. complete exact Recipya recipe prerequisite", 5, False,
              f"exception: {exc}")


def load_expected_plan() -> tuple[dict[str, dict], dict[int, float]]:
    products = grocy_query(
        "SELECT p.id,p.name,"
        "COALESCE((SELECT SUM(s.amount) FROM stock s WHERE s.product_id=p.id),0) AS stock "
        "FROM products p ORDER BY p.id"
    )
    candidates: dict[str, list[dict]] = {target: [] for target in TARGETS}
    for row in products:
        target = direct_target(row.get("name") or "")
        if not target:
            continue
        try:
            row["stock"] = float(row.get("stock") or 0)
        except (TypeError, ValueError):
            row["stock"] = 0.0
        candidates[target].append(row)

    missing = [target for target, rows in candidates.items() if not rows]
    if missing:
        raise RuntimeError("no direct Grocy product for: " + ", ".join(missing))
    selected = {
        target: sorted(rows, key=lambda row: (-float(row["stock"]), int(row["id"])))[0]
        for target, rows in candidates.items()
    }
    expected = {
        int(row["id"]): TARGET_STOCK - float(row["stock"])
        for row in selected.values()
        if float(row["stock"]) < TARGET_STOCK
    }
    return selected, expected


def load_relevant_rows() -> list[dict]:
    return grocy_query(
        "SELECT sl.id,sl.product_id,sl.amount,COALESCE(sl.note,'') AS note,"
        "COALESCE(p.name,'') AS product_name "
        "FROM shopping_list sl LEFT JOIN products p ON p.id=sl.product_id "
        "WHERE COALESCE(sl.note,'')='Bistrot Provençal menu expansion' "
        "OR LOWER(COALESCE(sl.note,'')) LIKE '%menu expansion%' "
        "ORDER BY sl.id"
    )


def check_2_exact_conditional_product_set() -> None:
    global _selected_products, _expected, _selection_ok
    if not _recipe_ok:
        check("2. exact deterministic Grocy deficit product set", 5, False,
              "gated: complete Recipya recipe is invalid")
        return
    try:
        _selected_products, _expected = load_expected_plan()
        _selection_ok = len(_selected_products) == len(TARGETS)
        selection = ", ".join(
            f"{target}=#{row['id']}@{float(row['stock']):g}"
            for target, row in _selected_products.items()
        )
        check(
            "2. exact deterministic Grocy deficit product set",
            5,
            _selection_ok,
            selection if _selection_ok else "deterministic selection is incomplete",
        )
    except Exception as exc:
        check("2. exact deterministic Grocy deficit product set", 5, False,
              f"exception: {exc}")


def check_3_exact_amounts_notes_and_negatives() -> None:
    global _relevant_rows
    if not _recipe_ok or not _selection_ok:
        check("3. exact Grocy amounts, uniqueness, and negative constraints", 10, False,
              "gated: Recipya recipe or deterministic product selection is invalid")
        return
    try:
        selected, expected = _selected_products, _expected
        _relevant_rows = load_relevant_rows()
        rows = _relevant_rows
        problems = []
        by_product: dict[int, list[dict]] = {}
        for row in rows:
            product_id = int(row.get("product_id") or 0)
            by_product.setdefault(product_id, []).append(row)
            if row.get("note") != TASK_NOTE:
                problems.append(f"entry {row.get('id')} has inexact note")
        if set(by_product) != set(expected):
            problems.append(
                f"expected only deficient IDs {sorted(expected)}, found {sorted(by_product)}"
            )
        for product_id, deficit in expected.items():
            entries = by_product.get(product_id, [])
            if len(entries) != 1:
                problems.append(f"product {product_id} has {len(entries)} entries, expected 1")
                continue
            try:
                amount = float(entries[0].get("amount") or 0)
            except (TypeError, ValueError):
                amount = 0.0
            if abs(amount - deficit) > 1e-6:
                problems.append(
                    f"product {product_id} amount={amount:g}, expected={deficit:g}"
                )
        selected_ids = {int(row["id"]) for row in selected.values()}
        overstocked = selected_ids - set(expected)
        if overstocked & set(by_product):
            problems.append(
                f"adequately stocked selected IDs were added: {sorted(overstocked & set(by_product))}"
            )
        check(
            "3. exact Grocy amounts, uniqueness, and negative constraints",
            10,
            not problems,
            "exact empty plan" if not problems and not expected else (
                "all deficits exact" if not problems else "; ".join(problems)
            ),
        )
    except Exception as exc:
        check("3. exact Grocy amounts, uniqueness, and negative constraints", 10, False,
              f"exception: {exc}")


def main() -> None:
    check_1_complete_recipya_prerequisite()
    check_2_exact_conditional_product_set()
    check_3_exact_amounts_notes_and_negatives()
    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    print(f"SCORE: {earned / total if total else 0:.3f}  PASS: {all_pass}  ({earned}/{total})",
          file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_013/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Verifier for agriculture_013: visual recipe selection -> Grocy stock plan."""

import base64
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request


GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
RECIPYA_CONTAINER = os.getenv("RECIPYA_CONTAINER")
GROCY_DB = "/config/data/grocy.db"
RECIPYA_DB = "/root/.config/Recipya/Database/recipya.db"
SOURCE_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs",
    "recipya_recipe_545.jpg",
)
EXPECTED_SOURCE_SHA256 = "a26e2ae865112384d040deae9e156a9d5c978422126c0b6f45dd4e354f697099"
TARGET_STOCK = 5.0

for name, value in [
    ("GROCY_CONTAINER", GROCY_CONTAINER),
    ("RECIPYA_CONTAINER", RECIPYA_CONTAINER),
]:
    if not value:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)


_checks: list[tuple[str, int, bool, str]] = []


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


_recipya_db_copy: str | None = None


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
        '$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);'
        '$rows=$db->query(' + json.dumps(sql) + ')->fetchAll(PDO::FETCH_ASSOC);'
        'echo json_encode($rows);'
    )
    rc, stdout, stderr = docker_exec(GROCY_CONTAINER, "php", "-r", php)
    if rc != 0:
        raise RuntimeError(f"Grocy query failed: {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def canonical_vegetable(value: str) -> str | None:
    words = set(re.findall(r"[a-z]+", value.lower()))
    if words & {
        "balsamic", "bread", "casserole", "chip", "chips", "chutney", "cracker",
        "crackers", "crisp", "crisps", "cruncher", "dip", "dish", "fried",
        "hummus", "meal", "mix",
        "noodle", "noodles", "pasta", "pickle", "pickled", "pizza", "risotto",
        "sauce", "snack", "soup", "spread", "stir", "wok",
    }:
        return None
    if words & {"zucchini", "courgette", "courgettes"}:
        return "zucchini"
    if words & {"eggplant", "eggplants", "aubergine", "aubergines"}:
        return "eggplant"
    if words & {"onion", "onions"} and not words & {
        "powder", "scallion", "scallions", "seasoning", "spring",
    }:
        return "onion"
    if words & {"tomato", "tomatoes"} and not words & {
        "paste", "sauce", "juice", "ketchup", "canned", "diced", "powder",
    }:
        return "tomato"
    if "capsicum" in words or (
        words & {"pepper", "peppers"} and not words & {"black", "white", "ground", "powder"}
    ):
        return "pepper"
    return None


def read_grocy_recipe_picture(filename: str) -> bytes | None:
    if not filename or os.path.basename(filename) != filename:
        return None
    candidates = [
        f"/config/data/storage/recipepictures/{filename}",
        f"/var/www/data/storage/recipepictures/{filename}",
    ]
    for path in candidates:
        rc, stdout, _ = docker_exec(GROCY_CONTAINER, "base64", path)
        if rc == 0 and stdout.strip():
            return base64.b64decode(stdout)
    rc, stdout, _ = docker_exec(
        GROCY_CONTAINER, "find", "/config/data/storage", "-type", "f",
        "-name", filename,
    )
    if rc == 0 and stdout.strip():
        path = stdout.splitlines()[0]
        rc2, encoded, _ = docker_exec(GROCY_CONTAINER, "base64", path)
        if rc2 == 0 and encoded.strip():
            return base64.b64decode(encoded)
    return None


def image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def llm_compare_images(
    source: bytes,
    actual: bytes,
    recipe_summary: str,
    timeout: int = 45,
) -> tuple[bool, str]:
    prompt = (
        "Image 1 is the supplied restaurant photo. Image 2 is the image actually attached "
        "to the target Grocy recipe. Determine whether they depict the same dish and whether "
        "the attached image supports the recipe summary. The visible dish must be a rustic "
        "cooked vegetable preparation centered on tomato, zucchini/courgette, eggplant/"
        "aubergine, onion, and pepper. Generic or different dish images must be NO.\n\n"
        f"Recipe summary: {recipe_summary}\nAnswer only YES or NO."
    )
    payload = {
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{image_mime(source)};base64," + base64.b64encode(source).decode()
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:{image_mime(actual)};base64," + base64.b64encode(actual).decode()
                }},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 64,
    }
    try:
        request = urllib.request.Request(
            os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1")
            + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.getenv('MINDRA_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        message = body.get("choices", [{}])[0].get("message", {})
        answer = (message.get("content") or "").strip().upper()
        if not answer:
            return False, "vision judge returned no message content"
        return answer.startswith("YES"), answer
    except Exception as exc:
        return False, f"vision judge error: {exc}"


_recipya_recipe: dict | None = None
_grocy_recipe: dict | None = None
_grocy_products: dict[str, dict] = {}
_recipya_candidates: list[dict] = []
_candidate_chains: list[dict] = []
_recipya_ok = False
_grocy_ok = False
_picture_ok = False


def check_1_recipya_visual_recipe() -> None:
    global _recipya_candidates, _recipya_ok
    try:
        rows = recipya_query(
            "SELECT r.id, r.name FROM recipes r "
            "JOIN user_recipe ur ON ur.recipe_id=r.id "
            "JOIN users u ON u.id=ur.user_id "
            "WHERE LOWER(u.email)='admin@recipya.com' ORDER BY r.id DESC"
        )
        candidates = []
        for row in rows:
            recipe_id = int(row["id"])
            ingredients = [r["name"] for r in recipya_query(
                "SELECT i.name FROM ingredient_recipe ir "
                "JOIN ingredients i ON i.id=ir.ingredient_id "
                f"WHERE ir.recipe_id={recipe_id} ORDER BY ir.ingredient_order"
            )]
            instructions = [r["name"] for r in recipya_query(
                "SELECT i.name FROM instruction_recipe ir "
                "JOIN instructions i ON i.id=ir.instruction_id "
                f"WHERE ir.recipe_id={recipe_id} ORDER BY ir.instruction_order"
            )]
            targets = {canonical_vegetable(value) for value in ingredients}
            targets.discard(None)
            if len(ingredients) >= 5 and len(instructions) >= 4 and len(targets) >= 4:
                candidates.append({
                    "id": recipe_id,
                    "name": row["name"],
                    "ingredients": ingredients,
                    "instructions": instructions,
                    "targets": targets,
                })
        if not candidates:
            check("1. Recipya photo-derived recipe is complete", 4, False,
                  "no admin recipe has >=5 ingredients, >=4 instructions, and >=4 visual vegetables")
            return
        _recipya_candidates = candidates
        _recipya_ok = True
        check("1. Recipya photo-derived recipe is complete", 4, True,
              f"{len(candidates)} qualifying candidate(s)")
    except Exception as exc:
        check("1. Recipya photo-derived recipe is complete", 4, False, f"exception: {exc}")


def check_2_grocy_recipe_matches_upstream() -> None:
    global _candidate_chains, _grocy_ok
    if not _recipya_ok or not _recipya_candidates:
        check("2. Grocy recipe exactly mirrors upstream visual ingredients", 3, False,
              "gated: Recipya visual recipe is invalid")
        return
    try:
        chains = []
        for upstream in _recipya_candidates:
            rows = grocy_query(
                "SELECT id,name,picture_file_name FROM recipes WHERE name="
                + sql_quote(upstream["name"])
            )
            if len(rows) != 1 or rows[0]["name"] != upstream["name"]:
                continue
            recipe = rows[0]
            positions = grocy_query(
                "SELECT rp.product_id,rp.amount,p.name AS product_name "
                "FROM recipes_pos rp JOIN products p ON p.id=rp.product_id "
                f"WHERE rp.recipe_id={int(recipe['id'])}"
            )
            by_target: dict[str, list[dict]] = {}
            for position in positions:
                target = canonical_vegetable(position.get("product_name") or "")
                if target:
                    by_target.setdefault(target, []).append(position)
            expected = set(upstream["targets"])
            if set(by_target) != expected:
                continue
            products = {}
            for target in expected:
                matches = by_target.get(target, [])
                if len(matches) != 1 or float(matches[0].get("amount") or 0) <= 0:
                    break
                products[target] = matches[0]
            else:
                chains.append({
                    "recipya": upstream,
                    "grocy": recipe,
                    "products": products,
                })
        if not chains:
            check("2. Grocy recipe exactly mirrors upstream visual ingredients", 3, False,
                  "no qualifying Recipya candidate has one exact complete same-name Grocy recipe")
            return
        _candidate_chains = chains
        _grocy_ok = True
        check("2. Grocy recipe exactly mirrors upstream visual ingredients", 3, True,
              f"{len(chains)} complete same-name association(s)")
    except Exception as exc:
        check("2. Grocy recipe exactly mirrors upstream visual ingredients", 3, False,
              f"exception: {exc}")


def check_3_actual_grocy_picture_matches() -> None:
    global _recipya_recipe, _grocy_recipe, _grocy_products, _picture_ok
    if not _recipya_ok or not _grocy_ok or not _candidate_chains:
        check("3. target Grocy recipe has the exact supplied dish image", 3, False,
              "gated: upstream recipe association is incomplete")
        return
    try:
        if not os.path.isfile(SOURCE_IMAGE):
            check("3. target Grocy recipe has the exact supplied dish image", 3, False,
                  "source image missing")
            return
        with open(SOURCE_IMAGE, "rb") as source_file:
            source = source_file.read()
        source_digest = hashlib.sha256(source).hexdigest()
        if source_digest != EXPECTED_SOURCE_SHA256:
            check("3. target Grocy recipe has the exact supplied dish image", 3, False,
                  f"source image hash={source_digest}, expected={EXPECTED_SOURCE_SHA256}")
            return

        matches = []
        failures = []
        for chain in _candidate_chains:
            upstream = chain["recipya"]
            recipe = chain["grocy"]
            filename = (recipe.get("picture_file_name") or "").strip()
            actual = read_grocy_recipe_picture(filename)
            if not actual:
                failures.append(f"recipe_id={recipe['id']}: missing '{filename}'")
                continue
            actual_digest = hashlib.sha256(actual).hexdigest()
            if actual_digest != source_digest:
                failures.append(
                    f"recipe_id={recipe['id']}: image hash={actual_digest}, "
                    f"expected exact source hash={source_digest}"
                )
                continue
            summary = f"{upstream['name']}; ingredients: " + ", ".join(upstream["ingredients"])
            passed, detail = llm_compare_images(source, actual, summary)
            if passed:
                matches.append((chain, filename, actual_digest, detail))
            else:
                failures.append(f"recipe_id={recipe['id']}: {detail}")
        if len(matches) != 1:
            detail = f"expected one visually matching association, found {len(matches)}"
            if failures:
                detail += "; " + "; ".join(failures[:3])
            check("3. target Grocy recipe has the exact supplied dish image", 3, False,
                  detail)
            return
        selected, filename, actual_digest, detail = matches[0]
        _recipya_recipe = selected["recipya"]
        _grocy_recipe = selected["grocy"]
        _grocy_products = selected["products"]
        _picture_ok = True
        check("3. target Grocy recipe has the exact supplied dish image", 3, True,
              f"recipe_id={_grocy_recipe['id']}; file='{filename}'; "
              f"sha256={actual_digest}; {detail}")
    except Exception as exc:
        check("3. target Grocy recipe has the exact supplied dish image", 3, False,
              f"exception: {exc}")


def check_4_exact_conditional_shopping_plan() -> None:
    if not _recipya_ok or not _grocy_ok or not _picture_ok or not _recipya_recipe:
        check("4. Grocy shopping plan exactly covers visual-ingredient deficits", 10, False,
              "gated: Recipya/Grocy visual recipe chain is incomplete")
        return
    try:
        expected: dict[int, float] = {}
        for position in _grocy_products.values():
            product_id = int(position["product_id"])
            stock_rows = grocy_query(
                f"SELECT COALESCE(SUM(amount),0) AS amount FROM stock WHERE product_id={product_id}"
            )
            stock = float(stock_rows[0]["amount"] or 0) if stock_rows else 0.0
            if stock < TARGET_STOCK:
                expected[product_id] = TARGET_STOCK - stock

        exact_note = f"Bistrot Provençal — {_recipya_recipe['name']}"
        rows = grocy_query(
            "SELECT sl.id,sl.product_id,sl.amount,COALESCE(sl.note,'') AS note "
            "FROM shopping_list sl "
            "WHERE COALESCE(sl.note,'') LIKE 'Bistrot Provençal%'"
        )
        actual: dict[int, list[dict]] = {}
        for row in rows:
            if row["note"] != exact_note:
                actual.setdefault(int(row.get("product_id") or 0), []).append(row)
                continue
            actual.setdefault(int(row.get("product_id") or 0), []).append(row)

        problems = []
        if set(actual) != set(expected):
            problems.append(f"product_ids expected={sorted(expected)} actual={sorted(actual)}")
        for product_id, needed in expected.items():
            entries = actual.get(product_id, [])
            if len(entries) != 1:
                problems.append(f"product {product_id}: {len(entries)} entries")
                continue
            entry = entries[0]
            if entry["note"] != exact_note:
                problems.append(f"product {product_id}: inexact note")
            try:
                amount = float(entry.get("amount") or 0)
            except (TypeError, ValueError):
                amount = 0.0
            if abs(amount - needed) > 1e-6:
                problems.append(f"product {product_id}: amount={amount:g}, needed={needed:g}")
        check("4. Grocy shopping plan exactly covers visual-ingredient deficits", 10,
              not problems, "exact conditional set" if not problems else "; ".join(problems))
    except Exception as exc:
        check("4. Grocy shopping plan exactly covers visual-ingredient deficits", 10, False,
              f"exception: {exc}")


def main() -> None:
    check_1_recipya_visual_recipe()
    check_2_grocy_recipe_matches_upstream()
    check_3_actual_grocy_picture_matches()
    check_4_exact_conditional_shopping_plan()
    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    print(f"SCORE: {earned / total if total else 0:.3f}  PASS: {all_pass}  ({earned}/{total})",
          file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

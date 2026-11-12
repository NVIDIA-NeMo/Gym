#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_031/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Adaptation: register the shipped Drupal collation for database reads; task checks unchanged.
"""Verifier for agriculture_031: visual recipe -> stock -> farm traceability."""

import base64
import hashlib
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request


RECIPYA_CONTAINER = os.getenv("RECIPYA_CONTAINER")
GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
FARMOS_CONTAINER = os.getenv("FARMOS_CONTAINER")
RECIPYA_DB = "/root/.config/Recipya/Database/recipya.db"
GROCY_DB = "/config/data/grocy.db"
FARMOS_DB = "/opt/drupal/web/sites/default/files/.ht.sqlite"

SOURCE_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs",
    "recipya_recipe_006.jpg",
)
EXPECTED_RECIPE = "Beef and Broccoli Stir-Fry"
EXPECTED_CUISINE = "Chinese"
EXPECTED_PRODUCT = "Broccoli"
TARGET_LOG_NAME = "2024 Broccoli Harvest — North Field East Bed (Side Shoots)"
TARGET_ASSET_NAME = "North Field — East Bed"
EXPECTED_OMRI = "OMRI-ORG-2024-1187"
EXPECTED_SOURCE_SHA256 = "951f50e302e49b45f60d3186b0c1c5860ede1cca2065e72893848b86dd4f02b6"
# Expected WebP bytes produced by the deployed Recipya image pipeline for this source.
EXPECTED_STORED_SHA256 = "4b69b0cb9fb0a62dbaa8967a3b795d8a99f16f0759247924e7d653ae2496b1e5"
EXPECTED_FARMOS_NOTES = f"OMRI certification: {EXPECTED_OMRI}"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

for name, value in [
    ("RECIPYA_CONTAINER", RECIPYA_CONTAINER),
    ("GROCY_CONTAINER", GROCY_CONTAINER),
    ("FARMOS_CONTAINER", FARMOS_CONTAINER),
]:
    if not value:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)


_checks: list[tuple[str, int, bool, str]] = []
_db_copies: dict[tuple[str, str], str] = {}
_recipe: dict | None = None
_product: dict | None = None
_target_log: dict | None = None
_recipe_ok = False
_grocy_ok = False
_farmos_ok = False


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


def sqlite_copy_query(container: str, db_path: str, sql: str) -> list[dict]:
    key = (container, db_path)
    local_db = _db_copies.get(key)
    if local_db is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        local_db = tmp.name
        tmp.close()
        result = subprocess.run(
            ["docker", "cp", f"{container}:{db_path}", local_db],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"DB copy failed for {db_path}: {result.stderr.strip()}")
        for suffix in ("-wal", "-shm"):
            subprocess.run(
                ["docker", "cp", f"{container}:{db_path}{suffix}", local_db + suffix],
                capture_output=True,
                text=True,
                timeout=30,
            )
        _db_copies[key] = local_db
    connection = sqlite3.connect(local_db)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def recipya_query(sql: str) -> list[dict]:
    return sqlite_copy_query(RECIPYA_CONTAINER, RECIPYA_DB, sql)


def grocy_query(sql: str) -> list[dict]:
    return sqlite_copy_query(GROCY_CONTAINER, GROCY_DB, sql)


def farmos_query(sql: str) -> list[dict]:
    encoded = base64.b64encode(sql.encode()).decode()
    php = (
        "require '/opt/drupal/vendor/autoload.php';"
        "$sql=base64_decode('" + encoded + "');"
        "$db=new SQLite3('" + FARMOS_DB + "');"
        "$db->createCollation('NOCASE_UTF8',['Drupal\\Component\\Utility\\Unicode','strcasecmp']);"
        "$result=$db->query($sql);"
        "if($result===false){fwrite(STDERR,$db->lastErrorMsg());exit(2);}"
        "$rows=[];while($row=$result->fetchArray(SQLITE3_ASSOC)){$rows[]=$row;}"
        "echo json_encode($rows);"
    )
    rc, stdout, stderr = docker_exec(FARMOS_CONTAINER, "php", "-d", "display_errors=stderr", "-r", php)
    if rc != 0:
        raise RuntimeError(f"FarmOS query failed: {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def description_lines(value: str) -> list[str]:
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", value or "")
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return [line.strip() for line in text.splitlines() if line.strip()]


def image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def read_recipya_image(image_id: str) -> bytes | None:
    image_id = (image_id or "").strip()
    if not UUID_RE.fullmatch(image_id):
        return None
    path = f"/root/.config/Recipya/Images/{image_id}.webp"
    rc, stdout, _ = docker_exec(RECIPYA_CONTAINER, "base64", path)
    if rc == 0 and stdout.strip():
        try:
            return base64.b64decode(stdout)
        except ValueError:
            return None
    return None


def compare_attached_image(source: bytes, actual: bytes, summary: str) -> tuple[bool, str]:
    prompt = (
        "Image 1 is the supplied customer dish photo. Image 2 is the image actually "
        "attached to the candidate Recipya recipe. Are they the same dish, and do both "
        "support the recipe summary as a Chinese beef-and-broccoli stir-fry? A generic, "
        "unrelated, or merely similar image must be NO.\n\n"
        f"Recipe summary: {summary}\nAnswer only YES or NO."
    )
    payload = {
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url":
                    f"data:{image_mime(source)};base64,{base64.b64encode(source).decode()}"}},
                {"type": "image_url", "image_url": {"url":
                    f"data:{image_mime(actual)};base64,{base64.b64encode(actual).decode()}"}},
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
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read())
        message = body.get("choices", [{}])[0].get("message", {})
        answer = (message.get("content") or "").strip().upper()
        if not answer:
            return False, "vision judge returned no message content"
        return answer.startswith("YES"), answer
    except Exception as exc:
        return False, f"vision judge error: {exc}"


def quantified(value: str) -> bool:
    return re.search(r"\d|[¼½¾⅓⅔⅛]", value or "") is not None


def check_1_complete_visually_linked_recipe() -> None:
    global _recipe, _recipe_ok
    try:
        rows = recipya_query(
            "SELECT r.id,r.name,r.image FROM recipes r "
            "JOIN user_recipe ur ON ur.recipe_id=r.id "
            "JOIN users u ON u.id=ur.user_id "
            "WHERE r.name='Beef and Broccoli Stir-Fry' "
            "AND LOWER(u.email)='admin@recipya.com' ORDER BY r.id"
        )
        if len(rows) != 1:
            check("1. complete visually linked Recipya recipe", 4, False,
                  f"expected one exact admin recipe, found {len(rows)}")
            return
        _recipe = rows[0]
        recipe_id = int(_recipe["id"])
        ingredients = [row["name"] for row in recipya_query(
            "SELECT i.name FROM ingredient_recipe ir "
            "JOIN ingredients i ON i.id=ir.ingredient_id "
            f"WHERE ir.recipe_id={recipe_id} ORDER BY ir.ingredient_order"
        )]
        instructions = [row["name"] for row in recipya_query(
            "SELECT i.name FROM instruction_recipe ir "
            "JOIN instructions i ON i.id=ir.instruction_id "
            f"WHERE ir.recipe_id={recipe_id} ORDER BY ir.instruction_order"
        )]
        cuisines = [row["name"] for row in recipya_query(
            "SELECT c.name FROM cuisine_recipe cr "
            "JOIN cuisines c ON c.id=cr.cuisine_id "
            f"WHERE cr.recipe_id={recipe_id} ORDER BY c.name"
        )]
        broccoli = []
        protein = []
        for value in ingredients:
            words = set(re.findall(r"[a-z]+", value.lower()))
            if "broccoli" in words and not words & {"soup", "powder", "extract"}:
                broccoli.append(value)
            if words & {"beef", "steak", "sirloin", "flank"} and not words & {
                "bouillon", "broth", "flavor", "flavour", "sauce", "seasoning", "stock",
            }:
                protein.append(value)
        problems = []
        if EXPECTED_CUISINE not in cuisines:
            problems.append(f"cuisine={cuisines}, expected Chinese")
        if not any(quantified(value) for value in broccoli):
            problems.append("no quantified broccoli ingredient")
        if not any(quantified(value) for value in protein):
            problems.append("no quantified beef ingredient")
        if len(instructions) < 4:
            problems.append(f"only {len(instructions)} instructions")
        source_image = b""
        if not os.path.isfile(SOURCE_IMAGE):
            problems.append("source image missing")
        else:
            with open(SOURCE_IMAGE, "rb") as source_file:
                source_image = source_file.read()
            source_digest = hashlib.sha256(source_image).hexdigest()
            if source_digest != EXPECTED_SOURCE_SHA256:
                problems.append(
                    f"source image hash={source_digest}, expected={EXPECTED_SOURCE_SHA256}"
                )
        actual_image = read_recipya_image(_recipe.get("image") or "")
        if not actual_image:
            problems.append(f"associated image file missing for image='{_recipe.get('image')}'")
        elif source_image and hashlib.sha256(source_image).hexdigest() == EXPECTED_SOURCE_SHA256:
            actual_digest = hashlib.sha256(actual_image).hexdigest()
            if actual_digest != EXPECTED_STORED_SHA256:
                problems.append(
                    f"stored image hash={actual_digest}, expected Recipya WebP hash="
                    f"{EXPECTED_STORED_SHA256}"
                )
            else:
                summary = f"{_recipe['name']}; cuisine={','.join(cuisines)}; ingredients=" \
                          + ", ".join(ingredients)
                image_ok, detail = compare_attached_image(source_image, actual_image, summary)
                if not image_ok:
                    problems.append(detail)
        _recipe_ok = not problems
        check(
            "1. complete visually linked Recipya recipe",
            4,
            _recipe_ok,
            f"recipe_id={recipe_id}; image={_recipe.get('image')}; "
            f"stored_sha256={EXPECTED_STORED_SHA256}"
            if _recipe_ok else "; ".join(problems),
        )
    except Exception as exc:
        check("1. complete visually linked Recipya recipe", 4, False,
              f"exception: {exc}")


def check_2_exact_grocy_product_with_stock() -> None:
    global _product, _grocy_ok
    try:
        rows = grocy_query(
            "SELECT id,name,COALESCE(description,'') AS description FROM products "
            "WHERE name='Broccoli' ORDER BY id"
        )
        if len(rows) != 1:
            check("2. exact Grocy vegetable product has positive stock", 3, False,
                  f"expected one exact Broccoli product, found {len(rows)}")
            return
        _product = rows[0]
        product_id = int(_product["id"])
        stock_rows = grocy_query(
            f"SELECT COALESCE(SUM(amount),0) AS amount FROM stock WHERE product_id={product_id}"
        )
        stock = float(stock_rows[0]["amount"] or 0) if stock_rows else 0.0
        _grocy_ok = stock > 0
        check(
            "2. exact Grocy vegetable product has positive stock",
            3,
            _grocy_ok,
            f"product_id={product_id}; stock={stock:g}",
        )
    except Exception as exc:
        check("2. exact Grocy vegetable product has positive stock", 3, False,
              f"exception: {exc}")


def check_3_exact_farmos_log_annotation() -> None:
    global _target_log, _farmos_ok
    try:
        exact_rows = farmos_query(
            "SELECT id,name,timestamp,notes__value FROM log_field_data "
            "WHERE type='harvest' "
            "AND name COLLATE BINARY='2024 Broccoli Harvest — North Field East Bed (Side Shoots)' "
            "ORDER BY timestamp DESC,id DESC"
        )
        if len(exact_rows) != 1:
            check("3. unique FarmOS harvest log has exact final notes and asset", 3, False,
                  f"expected one exact target harvest log, found {len(exact_rows)}")
            return
        _target_log = exact_rows[0]
        target_id = int(_target_log["id"])
        notes = strip_html(_target_log.get("notes__value") or "")
        problems = []
        asset_rows = farmos_query(
            "SELECT id FROM asset_field_data WHERE type='land' "
            "AND name='North Field — East Bed' ORDER BY id"
        )
        if len(asset_rows) != 1:
            problems.append(f"expected one exact East Bed land asset, found {len(asset_rows)}")
        else:
            asset_id = int(asset_rows[0]["id"])
            linked_rows = farmos_query(
                "SELECT la.asset_target_id,a.type FROM log__asset la "
                "LEFT JOIN asset_field_data a ON a.id=la.asset_target_id "
                f"WHERE la.entity_id={target_id} AND la.deleted=0 ORDER BY la.asset_target_id"
            )
            linked_assets = [
                (int(row["asset_target_id"]), row.get("type")) for row in linked_rows
            ]
            if linked_assets != [(asset_id, "land")]:
                problems.append(
                    f"asset links={linked_assets}, expected only [({asset_id}, 'land')]"
                )
        if notes != EXPECTED_FARMOS_NOTES:
            problems.append(
                f"notes={notes!r}, expected exact {EXPECTED_FARMOS_NOTES!r}"
            )
        other_rows = farmos_query(
            "SELECT id,name,type FROM log_field_data "
            f"WHERE id<>{target_id} AND notes__value LIKE '%OMRI-ORG-2024-1187%' "
            "ORDER BY id"
        )
        if other_rows:
            problems.append(
                "OMRI number also appears on non-target log IDs "
                + str([int(row["id"]) for row in other_rows])
            )
        _farmos_ok = not problems
        check(
            "3. unique FarmOS harvest log has exact final notes and asset",
            3,
            _farmos_ok,
            f"log_id={target_id}" if _farmos_ok else "; ".join(problems),
        )
    except Exception as exc:
        check("3. unique FarmOS harvest log has exact final notes and asset", 3, False,
              f"exception: {exc}")


def check_4_gated_exact_traceability_description() -> None:
    if not _recipe_ok or not _recipe or not _grocy_ok or not _product or not _farmos_ok:
        check("4. Grocy description exactly traces validated recipe and harvest", 10, False,
              "gated: visual recipe, Grocy stock, or exact FarmOS update is invalid")
        return
    try:
        rows = grocy_query(
            f"SELECT COALESCE(description,'') AS description FROM products "
            f"WHERE id={int(_product['id'])}"
        )
        description = rows[0]["description"] if rows else ""
        lines = description_lines(description)
        recipe_line = f"Recipya recipe ID: {int(_recipe['id'])}"
        omri_line = f"OMRI certification: {EXPECTED_OMRI}"
        problems = []
        expected_lines = [recipe_line, omri_line]
        if lines != expected_lines:
            problems.append(f"description lines={lines!r}, expected={expected_lines!r}")
        check(
            "4. Grocy description exactly traces validated recipe and harvest",
            10,
            not problems,
            "exact two-line final description" if not problems else "; ".join(problems),
        )
    except Exception as exc:
        check("4. Grocy description exactly traces validated recipe and harvest", 10, False,
              f"exception: {exc}")


def main() -> None:
    check_1_complete_visually_linked_recipe()
    check_2_exact_grocy_product_with_stock()
    check_3_exact_farmos_log_annotation()
    check_4_gated_exact_traceability_description()
    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    print(f"SCORE: {earned / total if total else 0:.3f}  PASS: {all_pass}  ({earned}/{total})",
          file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

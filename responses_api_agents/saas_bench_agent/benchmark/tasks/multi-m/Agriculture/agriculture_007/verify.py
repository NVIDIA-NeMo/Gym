#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_007/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Verify Recipya visual selection -> exact Grocy deficit plan for agriculture_007."""

import atexit
import base64
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime


GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
RECIPYA_CONTAINER = os.getenv("RECIPYA_CONTAINER")
RECIPYA_DB = "/root/.config/Recipya/Database/recipya.db"
RECIPYA_IMAGES = "/root/.config/Recipya/Images"
GROCY_DB_CANDIDATES = [
    "/config/data/grocy.db",
    "/config/data/data/grocy.db",
    "/var/www/data/grocy.db",
]
SELECTION_RE = re.compile(r"^AG007 SELECTED \| Recipya #(\d+) \| (.+)$")

for _name, _value in [
    ("GROCY_CONTAINER", GROCY_CONTAINER),
    ("RECIPYA_CONTAINER", RECIPYA_CONTAINER),
]:
    if not _value:
        print(f"FATAL: {_name} not set", file=sys.stderr)
        sys.exit(1)


_checks: list[tuple[str, int, bool, str]] = []
_recipya_db_copy = ""
_grocy_db_path = ""
_marked_rows: list[dict] = []
_recipe: dict = {}
_ingredients: list[str] = []
_image_bytes = b""
_grocy_started_at = ""
_grocy_started_epoch: int | None = None
_recipya_started_at = ""
_recipya_started_epoch: int | None = None
_recipe_ok = False
_visual_ok = False
_marker_ok = False
_stock_untouched = False


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{suffix}", file=sys.stderr)


def docker_exec(container: str, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    result = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _cleanup() -> None:
    if _recipya_db_copy:
        try:
            os.unlink(_recipya_db_copy)
        except OSError:
            pass


atexit.register(_cleanup)


def recipya_query(sql: str) -> list[dict]:
    global _recipya_db_copy
    if not _recipya_db_copy:
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
    connection = sqlite3.connect(_recipya_db_copy)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def _find_grocy_db() -> str:
    global _grocy_db_path
    if _grocy_db_path:
        return _grocy_db_path
    for candidate in GROCY_DB_CANDIDATES:
        rc, _, _ = docker_exec(GROCY_CONTAINER, "test", "-f", candidate, timeout=5)
        if rc == 0:
            _grocy_db_path = candidate
            return candidate
    return GROCY_DB_CANDIDATES[0]


def grocy_query(sql: str) -> list[dict]:
    php = (
        '$db=new PDO("sqlite:' + _find_grocy_db() + '");'
        '$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);'
        '$s=$db->query($argv[1]);'
        'echo json_encode($s->fetchAll(PDO::FETCH_ASSOC));'
    )
    rc, out, err = docker_exec(GROCY_CONTAINER, "php", "-r", php, "--", sql)
    if rc != 0:
        raise RuntimeError(f"Grocy query failed: {err.strip()[:300]}")
    return json.loads(out) if out.strip() else []


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def container_started_epoch(container: str) -> tuple[str, int]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.StartedAt}}", container],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"container start audit failed: {result.stderr.strip()}")
    started_at = result.stdout.strip()
    try:
        normalized = re.sub(r"(\.\d{6})\d+(?=Z|[+-]\d{2}:\d{2}$)", r"\1", started_at)
        started_epoch = int(
            datetime.fromisoformat(normalized.replace("Z", "+00:00")).timestamp()
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid container StartedAt={started_at!r}") from exc
    return started_at, started_epoch


def description_lines(value: str) -> list[str]:
    value = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", value or "")
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    return [line.strip() for line in value.splitlines() if line.strip()]


def read_recipe_image(image_id: str) -> bytes:
    image_id = (image_id or "").strip()
    if not image_id or os.path.basename(image_id) != image_id:
        return b""
    names = [image_id]
    if not image_id.lower().endswith((".webp", ".jpg", ".jpeg", ".png", ".gif")):
        names.insert(0, image_id + ".webp")
    for name in names:
        rc, encoded, _ = docker_exec(
            RECIPYA_CONTAINER,
            "base64",
            f"{RECIPYA_IMAGES}/{name}",
        )
        if rc != 0 or not encoded.strip():
            continue
        try:
            data = base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
        except ValueError:
            continue
        if len(data) >= 1024:
            return data
    return b""


def image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def mindra_yes_no(prompt: str, image_bytes: bytes | None = None, timeout: int = 60) -> tuple[bool, str]:
    content: list[dict] = []
    if image_bytes:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_mime(image_bytes)};base64,"
                + base64.b64encode(image_bytes).decode("ascii")
            },
        })
    content.append({"type": "text", "text": prompt + "\nAnswer only YES or NO."})
    payload = {
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
    }
    try:
        request = urllib.request.Request(
            os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.getenv('MINDRA_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False, "judge response missing choices"
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return False, "judge response missing message"
        content_value = message.get("content")
        if not isinstance(content_value, str) or not content_value.strip():
            return False, "judge response missing message content"
        answer = content_value.strip().upper()
        return answer.startswith("YES"), answer[:120]
    except Exception as exc:
        return False, f"judge error: {exc}"


def load_context() -> None:
    global _marked_rows, _recipe, _ingredients, _image_bytes
    global _grocy_started_at, _grocy_started_epoch
    global _recipya_started_at, _recipya_started_epoch
    _grocy_started_at, _grocy_started_epoch = container_started_epoch(GROCY_CONTAINER)
    _recipya_started_at, _recipya_started_epoch = container_started_epoch(RECIPYA_CONTAINER)
    admin_recipes = recipya_query(
        "SELECT DISTINCT r.id,r.name,COALESCE(r.description,'') AS description,"
        "COALESCE(r.image,'') AS image,COALESCE(r.created_at,'') AS created_at,"
        "CAST(strftime('%s',r.created_at) AS INTEGER) AS created_epoch "
        "FROM recipes r JOIN user_recipe ur ON ur.recipe_id=r.id "
        "JOIN users u ON u.id=ur.user_id "
        "WHERE LOWER(u.email)='admin@recipya.com' ORDER BY r.id"
    )
    selections: list[tuple[dict, re.Match[str]]] = []
    selection_lines = 0
    for row in admin_recipes:
        for line in description_lines(row.get("description") or ""):
            if "AG007 SELECTED" not in line:
                continue
            selection_lines += 1
            match = SELECTION_RE.fullmatch(line)
            if match is not None:
                selections.append((row, match))

    if selection_lines == 1 and len(selections) == 1:
        row, match = selections[0]
        recipe_name = html.unescape(row.get("name") or "")
        try:
            created_epoch = int(row["created_epoch"])
        except (KeyError, TypeError, ValueError):
            created_epoch = 0
        if (int(match.group(1)) == int(row["id"])
                and html.unescape(match.group(2)) == recipe_name
                and _recipya_started_epoch is not None
                and 0 < created_epoch < _recipya_started_epoch):
            _recipe = row
            _recipe["name"] = recipe_name
            recipe_id = int(row["id"])
            _ingredients = [html.unescape(ingredient["name"]) for ingredient in recipya_query(
                "SELECT i.name FROM ingredient_recipe ir "
                "JOIN ingredients i ON i.id=ir.ingredient_id "
                f"WHERE ir.recipe_id={recipe_id} ORDER BY ir.ingredient_order"
            )]
            _image_bytes = read_recipe_image(_recipe.get("image") or "")

    _marked_rows = grocy_query(
        "SELECT sl.id,sl.product_id,sl.amount,COALESCE(sl.note,'') AS note,"
        "COALESCE(p.name,'') AS product_name "
        "FROM shopping_list sl LEFT JOIN products p ON p.id=sl.product_id "
        "WHERE sl.note LIKE 'AG007%';"
    )


def check_1_exact_admin_recipe() -> None:
    global _recipe_ok
    has_tomato = any(re.search(r"\btomato(?:es)?\b", value, re.IGNORECASE) for value in _ingredients)
    _recipe_ok = bool(
        _recipe and _ingredients and has_tomato and _image_bytes
    )
    check(
        "1. exact admin-owned Recipya recipe with real cover",
        4,
        _recipe_ok,
        f"id={_recipe.get('id')} name='{_recipe.get('name')}' ingredients={len(_ingredients)} "
        f"image_bytes={len(_image_bytes)} recipya_started={_recipya_started_at}"
        if _recipe else "one exact standalone Recipya selection line did not resolve uniquely",
    )


def check_2_actual_cover_is_tomato_dominant() -> None:
    global _visual_ok
    if not _recipe_ok:
        check("2. actual Recipya cover is tomato-dominant", 5, False, "gated: recipe invalid")
        return
    prompt = (
        "Inspect the actual cover image stored for this Recipya recipe. Does the visible dish "
        "clearly use tomato as a dominant ingredient or sauce base, rather than as a minor "
        "garnish or incidental component? The ingredient list must also support that reading.\n"
        f"Recipe: {_recipe['name']}\nIngredients: " + " | ".join(_ingredients)
    )
    _visual_ok, detail = mindra_yes_no(prompt, _image_bytes)
    check("2. actual Recipya cover is tomato-dominant", 5, _visual_ok, detail)


def check_3_exact_unique_marker_rows() -> None:
    global _marker_ok, _stock_untouched
    if not _visual_ok:
        check("3. exact task entries and untouched task-start stock", 1, False,
              "gated: visual recipe invalid")
        return
    expected_note = f"AG007 | Recipya #{int(_recipe['id'])} | {_recipe['name']}"
    product_ids = [row.get("product_id") for row in _marked_rows]
    problems = []
    if any((row.get("note") or "") != expected_note for row in _marked_rows):
        problems.append("inexact or mixed task marker")
    if any(product_id is None for product_id in product_ids):
        problems.append("entry without a Grocy product")
    if len(product_ids) != len(set(product_ids)):
        problems.append("duplicate product entries")
    if any(abs(float(row.get("amount") or 0) - 1.0) > 1e-6 for row in _marked_rows):
        problems.append("every amount must equal 1")
    if any(re.search(r"\btomato(?:es)?\b", row.get("product_name") or "", re.IGNORECASE)
           for row in _marked_rows):
        problems.append("tomato appears in task shopping entries")

    stock_query_ok = True
    try:
        stock_logs = grocy_query(
            "SELECT id,COALESCE(row_created_timestamp,'') AS created_at,"
            "CAST(strftime('%s',row_created_timestamp,'utc') AS INTEGER) AS created_epoch,"
            "COALESCE(transaction_type,'') AS transaction_type,COALESCE(undone,0) AS undone,"
            "COALESCE(undone_timestamp,'') AS undone_at,"
            "CASE WHEN undone_timestamp IS NULL OR TRIM(undone_timestamp) = '' THEN NULL "
            "ELSE CAST(strftime('%s',undone_timestamp,'utc') AS INTEGER) END AS undone_epoch "
            "FROM stock_log ORDER BY id"
        )
    except Exception as exc:
        stock_query_ok = False
        stock_logs = []
        problems.append(f"stock-log audit unavailable: {exc}")
    late_created_logs = []
    late_undone_logs = []
    invalid_timestamps = []
    invalid_undo_states = []
    for row in stock_logs:
        try:
            created_epoch = int(row["created_epoch"])
        except (KeyError, TypeError, ValueError):
            invalid_timestamps.append(f"created:{row.get('id')}")
        else:
            if (_grocy_started_epoch is not None
                    and created_epoch >= _grocy_started_epoch):
                late_created_logs.append(row)

        try:
            undone = int(row.get("undone") or 0)
        except (TypeError, ValueError):
            invalid_undo_states.append(row.get("id"))
            continue
        undone_at = str(row.get("undone_at") or "").strip()
        undone_epoch_raw = row.get("undone_epoch")
        if undone not in (0, 1) or (undone == 0 and (undone_at or undone_epoch_raw is not None)):
            invalid_undo_states.append(row.get("id"))
            continue
        if undone == 1:
            try:
                undone_epoch = int(undone_epoch_raw) if undone_at else None
            except (TypeError, ValueError):
                undone_epoch = None
            if undone_epoch is None:
                invalid_timestamps.append(f"undone:{row.get('id')}")
            elif (_grocy_started_epoch is not None
                  and undone_epoch >= _grocy_started_epoch):
                late_undone_logs.append(row)
    if _grocy_started_epoch is None:
        problems.append("invalid Grocy container start baseline")
    if invalid_timestamps:
        problems.append(f"unparseable stock-log timestamps={invalid_timestamps[:5]}")
    if invalid_undo_states:
        problems.append(f"inconsistent stock-log undo states={invalid_undo_states[:5]}")
    if late_created_logs:
        problems.append(
            "Grocy stock changed during the task "
            f"(new log ids={[row.get('id') for row in late_created_logs[:5]]})"
        )
    if late_undone_logs:
        problems.append(
            "Grocy stock booking was undone during the task "
            f"(log ids={[row.get('id') for row in late_undone_logs[:5]]})"
        )
    _stock_untouched = (
        stock_query_ok and _grocy_started_epoch is not None
        and not invalid_timestamps and not invalid_undo_states
        and not late_created_logs and not late_undone_logs
    )
    _marker_ok = not problems
    check(
        "3. exact task entries and untouched task-start stock",
        1,
        _marker_ok,
        f"entries={len(_marked_rows)}, container_started={_grocy_started_at}" if _marker_ok
        else "; ".join(problems),
    )


def check_4_exact_inventory_deficit_set() -> None:
    if not (_recipe_ok and _visual_ok and _marker_ok and _stock_untouched):
        check("4. exact auxiliary-ingredient deficit set", 10, False,
              "gated: selection/visual/stock/row chain invalid")
        return
    try:
        products = grocy_query(
            "SELECT p.id,p.name,"
            "COALESCE((SELECT SUM(s.amount) FROM stock s WHERE s.product_id=p.id),0) AS stock "
            "FROM products p ORDER BY p.name,p.id"
        )
    except Exception as exc:
        check("4. exact auxiliary-ingredient deficit set", 10, False,
              f"inventory read failed: {exc}")
        return
    shopping_products = [
        f"#{row.get('product_id')} {row.get('product_name') or ''}"
        for row in _marked_rows
    ]
    prompt = (
        "Evaluate an inventory comparison. Ignore recipe quantities and preparation adjectives. "
        "Treat singular/plural and ordinary culinary synonyms as the same ingredient. Exclude all "
        "tomato forms from the shopping requirement. A positive-stock product supplies an ingredient "
        "only if it is a direct usable source; a flavored/composite product that merely mentions the "
        "ingredient does not count. The marked row must reuse an existing suitable Grocy product when "
        "one exists; the final product list must not contain a newly created semantic duplicate. If no "
        "suitable product exists, the marked product name must equal the displayed Recipya ingredient "
        "text. After deduplicating ingredient identities, are the marked Grocy shopping products exactly "
        "the complete set of non-tomato recipe ingredients that lack a direct positive-stock source, "
        "with no stocked or unrelated extras, duplicates, or omissions?\n\n"
        "RECIPE INGREDIENTS:\n- " + "\n- ".join(_ingredients) +
        "\n\nALL GROCY PRODUCTS (ID | NAME | CURRENT STOCK):\n- " +
        "\n- ".join(
            f"#{row['id']} | {row['name']} | {float(row.get('stock') or 0):g}"
            for row in products
        ) +
        "\n\nMARKED SHOPPING PRODUCTS:\n" + (
            "- " + "\n- ".join(shopping_products) if shopping_products else "(none)"
        )
    )
    passed, detail = mindra_yes_no(prompt, timeout=75)
    check("4. exact auxiliary-ingredient deficit set", 10, passed, detail)


def main() -> None:
    try:
        load_context()
    except Exception as exc:
        print(f"WARNING: context load failed: {exc}", file=sys.stderr)
    check_1_exact_admin_recipe()
    check_2_actual_cover_is_tomato_dominant()
    check_3_exact_unique_marker_rows()
    check_4_exact_inventory_deficit_set()
    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    print(f"SCORE: {earned / total if total else 0:.3f}  PASS: {all_pass}  ({earned}/{total})", file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/multi-m/Agriculture/agriculture_003/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged.
# Adaptation: register the shipped Drupal collation for database reads; task checks unchanged.
"""Verify the FarmOS-to-e-label traceability chain for agriculture_003."""

import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request


HOST = os.getenv("SERVER_HOSTNAME", "localhost")
FARMOS_PORT = os.getenv("FARMOS_PORT")
FARMOS_CONTAINER = os.getenv("FARMOS_CONTAINER")
E_LABEL_PORT = os.getenv("E_LABEL_PORT")
E_LABEL_CONTAINER = os.getenv("E_LABEL_CONTAINER")

for _name, _value in [
    ("FARMOS_PORT", FARMOS_PORT),
    ("FARMOS_CONTAINER", FARMOS_CONTAINER),
    ("E_LABEL_PORT", E_LABEL_PORT),
    ("E_LABEL_CONTAINER", E_LABEL_CONTAINER),
]:
    if not _value:
        print(f"FATAL: {_name} not set", file=sys.stderr)
        sys.exit(1)

FARMOS_DB = "/opt/drupal/web/sites/default/files/.ht.sqlite"
BASE_URL = f"http://{HOST}:{E_LABEL_PORT}"
TASK_LOG_NAME = "AG003 - 2024 Pinot Noir Harvest Intake"
TASK_PRODUCT_NAME = "Estate Pinot Noir"
TASK_LOG_NOTE = "Harvested Pinot Noir grapes for the 2024 Burgundy lot."
TASK_ASSET_NAME = "AG003 - Pinot Noir Vineyard Lot"

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


def farmos_query(sql: str) -> list[dict]:
    php = (
        "require '/opt/drupal/vendor/autoload.php';"
        "$db=new PDO('sqlite:" + FARMOS_DB + "');"
        "$db->sqliteCreateCollation('NOCASE_UTF8',['Drupal\\Component\\Utility\\Unicode','strcasecmp']);"
        "$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);"
        "$s=$db->query($argv[1]);"
        "echo json_encode($s->fetchAll(PDO::FETCH_ASSOC));"
    )
    rc, out, err = docker_exec(FARMOS_CONTAINER, "php", "-d", "display_errors=stderr", "-r", php, "--", sql)
    if rc != 0:
        raise RuntimeError(f"FarmOS query failed: {err.strip()[:300]}")
    return json.loads(out) if out.strip() else []


def _resolve_elabel_db() -> str:
    candidates = [
        os.getenv("E_LABEL_DB_CONTAINER", ""),
        E_LABEL_CONTAINER.replace("-app", "-db") if "-app" in E_LABEL_CONTAINER else "",
        E_LABEL_CONTAINER + "-db",
        E_LABEL_CONTAINER,
        "elabel-db",
    ]
    for candidate in dict.fromkeys(c for c in candidates if c):
        try:
            rc, _, _ = docker_exec(candidate, "echo", "ok", timeout=5)
            if rc == 0:
                return candidate
        except Exception:
            continue
    return E_LABEL_CONTAINER


E_LABEL_DB = _resolve_elabel_db()


def elabel_rows(query: str) -> list[list[str]]:
    last_error = "sqlcmd not found"
    for binary in ["/opt/mssql-tools18/bin/sqlcmd", "/opt/mssql-tools/bin/sqlcmd"]:
        args = [
            binary, "-S", "localhost", "-U", "sa", "-P", "Elabel2024!Strong",
            "-d", "elabel", "-h", "-1", "-W", "-s", "|",
            "-Q", "SET NOCOUNT ON; " + query,
        ]
        if "mssql-tools18" in binary:
            args.insert(9, "-C")
        rc, out, err = docker_exec(E_LABEL_DB, *args)
        if rc == 0:
            rows = []
            for line in out.splitlines():
                line = line.strip()
                if not line or "rows affected" in line.lower():
                    continue
                rows.append([part.strip() for part in line.split("|")])
            return rows
        last_error = err.strip() or out.strip()
        if "not found" not in last_error.lower():
            break
    raise RuntimeError(last_error)


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def http_get(path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(BASE_URL + path, timeout=15) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception:
        return 0, ""


_asset_rows: list[dict] = []
_log_rows: list[dict] = []
_product_rows: list[list[str]] = []
_asset_id = 0
_log_id = 0
_trace_code = ""
_product: dict[str, str] = {}
_public_html = ""
_source_ok = False
_product_ok = False
_public_ok = False
_trace_ok = False
_compliance_ok = False


def load_context() -> None:
    global _asset_rows, _log_rows, _product_rows, _asset_id, _log_id
    global _trace_code, _product, _public_html, _public_ok

    _asset_rows = farmos_query(
        "SELECT id, name, type FROM asset_field_data "
        "WHERE name = 'AG003 - Pinot Noir Vineyard Lot' ORDER BY id"
    )
    if len(_asset_rows) == 1:
        _asset_id = int(_asset_rows[0]["id"])

    safe_log_name = TASK_LOG_NAME.replace("'", "''")
    _log_rows = farmos_query(
        "SELECT id, name, type, timestamp, notes__value FROM log_field_data "
        f"WHERE type = 'harvest' AND name = '{safe_log_name}' ORDER BY id"
    )
    if len(_log_rows) == 1:
        _log_id = int(_log_rows[0]["id"])
    if _asset_id and _log_id:
        _trace_code = f"PN24-A{_asset_id}-H{_log_id}"

    _product_rows = elabel_rows(
        "SELECT CAST(Id AS NVARCHAR(36)), Name, ISNULL(FBOName,''), "
        "CAST(WineVintage AS NVARCHAR(10)), ISNULL(WineAppellation,''), "
        "CAST(WineAlcohol AS NVARCHAR(20)), CAST(Volume AS NVARCHAR(20)), "
        "CAST(WineType AS NVARCHAR(10)), ISNULL(Brand,''), ISNULL(Sku,'') "
        "FROM Product WHERE LOWER(LTRIM(RTRIM(Name))) = 'estate pinot noir' "
        "ORDER BY CreatedOn"
    )
    if len(_product_rows) == 1 and len(_product_rows[0]) >= 10:
        row = _product_rows[0]
        _product = {
            "id": row[0], "name": row[1], "fbo": row[2], "vintage": row[3],
            "appellation": row[4], "alcohol": row[5], "volume": row[6],
            "type": row[7], "brand": row[8], "sku": row[9],
        }
        codes = [_product["sku"], _product["id"]]
        for code in dict.fromkeys(code for code in codes if code and code.upper() != "NULL"):
            status, body = http_get(f"/l/{code}")
            if status == 200 and len(body) > 100:
                _public_html = body
                text = strip_html(body)
                _public_ok = (
                    TASK_PRODUCT_NAME in text
                    and "2024" in text
                    and re.search(r"13[.,]5\s*%?\s*vol", text) is not None
                )
                if _public_ok:
                    break


def check_1_farmos_trace_source() -> None:
    global _source_ok
    problems = []
    if len(_asset_rows) != 1:
        problems.append(f"expected one exact task vineyard asset, found {len(_asset_rows)}")
    elif _asset_rows[0].get("type") != "land":
        problems.append(f"asset type={_asset_rows[0].get('type')}, expected land")
    if len(_log_rows) != 1:
        problems.append(f"expected one exact task harvest log, found {len(_log_rows)}")
    if _asset_id and _log_id:
        links = farmos_query(
            "SELECT asset_target_id FROM log__asset "
            f"WHERE entity_id = {_log_id} AND deleted = 0 ORDER BY asset_target_id"
        )
        linked_ids = [int(row["asset_target_id"]) for row in links]
        if linked_ids != [_asset_id]:
            problems.append(f"linked asset ids={linked_ids}, expected [{_asset_id}]")
        notes = strip_html(_log_rows[0].get("notes__value") or "")
        if notes != TASK_LOG_NOTE:
            problems.append(f"notes must equal the required sentence, got '{notes[:100]}'")
        try:
            log_date = dt.datetime.fromtimestamp(int(_log_rows[0]["timestamp"])).date()
            if log_date != dt.date.today():
                problems.append(f"harvest log date={log_date}, expected {dt.date.today()}")
        except (TypeError, ValueError, OSError):
            problems.append("harvest log timestamp is invalid")
    _source_ok = not problems
    check(
        "1. exact FarmOS trace source",
        4,
        _source_ok,
        f"asset={_asset_id}, log={_log_id}, trace={_trace_code}" if _source_ok
        else "; ".join(problems),
    )


def check_2_elabel_core_fields() -> None:
    global _product_ok
    if not _source_ok:
        check("2. exact e-label core fields", 3, False, "gated: FarmOS trace source invalid")
        return
    problems = []
    if len(_product_rows) != 1 or not _product:
        problems.append(f"expected one exact product, found {len(_product_rows)}")
    else:
        if _product.get("name") != TASK_PRODUCT_NAME:
            problems.append(f"name='{_product.get('name', '')}'")
        try:
            if int(_product["vintage"]) != 2024:
                problems.append(f"vintage={_product['vintage']}")
            if abs(float(_product["alcohol"]) - 13.5) >= 0.05:
                problems.append(f"alcohol={_product['alcohol']}")
            volume = float(_product["volume"])
            if not (abs(volume - 0.75) < 0.01 or abs(volume - 750) < 1):
                problems.append(f"volume={volume}")
            if int(_product["type"]) != 2:
                problems.append(f"WineType={_product['type']} (expected Red=2)")
        except (TypeError, ValueError) as exc:
            problems.append(f"numeric field error: {exc}")
        if _product.get("appellation", "").strip().casefold() != "burgundy":
            problems.append(f"appellation='{_product.get('appellation', '')}'")
    _product_ok = not problems
    check("2. exact e-label core fields", 3, _product_ok, "; ".join(problems))


def check_3_dynamic_trace_brand() -> None:
    global _trace_ok
    if not (_source_ok and _product_ok and _public_ok):
        check(
            "3. FarmOS-derived Brand trace",
            3,
            False,
            "gated: source, product, and public label must all be valid",
        )
        return
    other_rows = elabel_rows(
        "SELECT CAST(Id AS NVARCHAR(36)) FROM Product "
        f"WHERE Brand = '{_trace_code}' AND Id <> '{_product['id']}'"
    )
    _trace_ok = _product.get("brand") == _trace_code and not other_rows
    detail = f"Brand='{_product.get('brand', '')}', expected '{_trace_code}', other={len(other_rows)}"
    check("3. FarmOS-derived Brand trace", 3, _trace_ok, detail)


def check_4_exact_compliance_associations() -> None:
    global _compliance_ok
    if not (_source_ok and _product_ok and _trace_ok and _public_ok):
        check(
            "4. exact producer and allergen association",
            2,
            False,
            "gated: source, product, and public label must all be valid",
        )
        return
    ingredients = elabel_rows(
        "SELECT i.Name, CAST(ISNULL(i.Allergen,0) AS NVARCHAR(5)) "
        "FROM ProductIngredient pi JOIN Ingredient i ON i.Id = pi.IngredientId "
        f"WHERE pi.ProductId = '{_product['id']}' ORDER BY i.Name"
    )
    names = [row[0].strip() for row in ingredients if row]
    sulphites = [
        row for row in ingredients
        if row and row[0].strip().casefold() == "sulphites"
        and len(row) > 1 and row[1] in ("1", "True", "true")
    ]
    _compliance_ok = (
        _product.get("fbo", "").strip() == "Boutique Organic Farm"
        and len(ingredients) == 1
        and len(sulphites) == 1
    )
    check(
        "4. exact producer and allergen association",
        2,
        _compliance_ok,
        f"FBO='{_product.get('fbo', '')}', ingredients={names}",
    )


def check_5_public_label_chain() -> None:
    if not (_source_ok and _product_ok and _trace_ok and _compliance_ok):
        check("5. public label renders traced wine", 8, False,
              "gated: source, product, trace, or compliance invalid")
        return
    check(
        "5. public label renders traced wine",
        8,
        _public_ok,
        "exact name, vintage, and 13.5 % vol found" if _public_ok
        else "public page missing or does not render the exact traced wine",
    )


def main() -> None:
    try:
        load_context()
    except Exception as exc:
        print(f"WARNING: context load failed: {exc}", file=sys.stderr)
    check_1_farmos_trace_source()
    check_2_elabel_core_fields()
    check_3_dynamic_trace_brand()
    check_4_exact_compliance_associations()
    check_5_public_label_chain()

    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    score = earned / total if total else 0.0
    print(f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})", file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
